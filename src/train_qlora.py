import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

from eval_fight import limit_pixels
from fight_data import messages_for, read_jsonl


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def language_target_pattern(model):
    names = [name for name, _ in model.named_modules() if name.endswith("q_proj")]
    language_names = [
        name
        for name in names
        if "visual" not in name and "vision" not in name
    ]
    if not language_names:
        raise RuntimeError(f"No language q_proj modules found. Sample modules: {names[:20]}")
    prefix = language_names[0].split(".layers.")[0]
    suffixes = "|".join(TARGET_SUFFIXES)
    pattern = rf"{prefix}\..*\.({suffixes})$"
    print(f"lora_target_pattern={pattern}", flush=True)
    return pattern


def encode(processor, image, label):
    full = processor.apply_chat_template(
        messages_for(image, label),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = processor.apply_chat_template(
        messages_for(image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_length = prompt["input_ids"].shape[1]
    full_ids = full["input_ids"]
    if prompt_length >= full_ids.shape[1]:
        raise RuntimeError("Prompt is not shorter than the full training example.")
    if not torch.equal(full_ids[:, :prompt_length], prompt["input_ids"]):
        raise RuntimeError("Prompt tokens are not a prefix of the full example.")
    labels = full_ids.clone()
    labels[:, :prompt_length] = -100
    full["labels"] = labels
    return full


def answer_length(processor, label):
    """Tokens after the generation prompt: '<label><|im_end|>\\n' (what encode() supervises)."""
    return len(processor.tokenizer.encode(f"{label}<|im_end|>\n", add_special_tokens=False))


def encode_batch(processor, images, labels):
    """Right-padded batch; only the assistant answer tokens are supervised."""
    processor.tokenizer.padding_side = "right"
    batch = processor.apply_chat_template(
        [messages_for(image, label) for image, label in zip(images, labels)],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    target = torch.full_like(batch["input_ids"], -100)
    lengths = batch["attention_mask"].sum(dim=1)
    for row, (length, label) in enumerate(zip(lengths.tolist(), labels)):
        span = answer_length(processor, label)
        target[row, length - span : length] = batch["input_ids"][row, length - span : length]
    batch["labels"] = target
    return batch


def check_batch_encoding(processor, rows):
    """encode_batch must supervise exactly the tokens encode() supervises, row by row."""
    batch = encode_batch(processor, [row["image"] for row in rows], [row["label"] for row in rows])
    for index, row in enumerate(rows):
        single = encode(processor, row["image"], row["label"])
        expected = single["labels"][0]
        length = int(batch["attention_mask"][index].sum())
        if not torch.equal(batch["labels"][index, :length], expected):
            raise RuntimeError(f"Batched labels differ from single-example labels for {row['image']}")
        if not torch.equal(batch["input_ids"][index, :length], single["input_ids"][0]):
            raise RuntimeError(f"Batched input_ids differ for {row['image']}")
    print(f"batch_encoding_check=ok rows={len(rows)}", flush=True)


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1, help="effective batch = batch-size * accum")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    set_seed(args.seed)

    root = Path(args.root).resolve()
    rows = read_jsonl(root / "data/train.jsonl")
    for row in rows:
        image = Path(row["image"])
        if not image.is_absolute():
            row["image"] = str(root / image)
    output = Path(args.output) if args.output else root / "outputs/fight-lora"
    output.mkdir(parents=True, exist_ok=True)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=language_target_pattern(model),
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable={trainable} total={total} pct={100 * trainable / total:.3f}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    limit_pixels(processor)

    first = encode(processor, rows[0]["image"], rows[0]["label"])
    visible = first["labels"][0]
    visible = visible[visible != -100]
    decoded = processor.tokenizer.decode(visible, skip_special_tokens=False)
    print(f"first_label_tokens={decoded!r} image_grid_thw={first['image_grid_thw'].tolist()}", flush=True)

    check_batch_encoding(processor, rows[: args.batch_size])

    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    batches_per_epoch = (len(rows) + args.batch_size - 1) // args.batch_size
    updates_per_epoch = (batches_per_epoch + args.accum - 1) // args.accum
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.05)),
        num_training_steps=total_updates,
    )

    model.train()
    device = model.device
    step = 0
    history = []
    order_rng = random.Random(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(rows)
        order_rng.shuffle(order)
        epoch_loss = 0.0
        seen = 0
        chunks = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for index, chunk in enumerate(chunks, start=1):
            batch = encode_batch(processor, [row["image"] for row in chunk], [row["label"] for row in chunk])
            loss = model(**move_to_device(batch, device)).loss / args.accum
            loss.backward()
            epoch_loss += loss.item() * args.accum * len(chunk)
            seen += len(chunk)
            if index % args.accum == 0 or index == len(chunks):
                torch.nn.utils.clip_grad_norm_(
                    (param for param in model.parameters() if param.requires_grad),
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or index == len(chunks):
                    print(
                        f"epoch={epoch + 1} step={step} sample={seen}/{len(order)} "
                        f"loss={epoch_loss / seen:.4f}",
                        flush=True,
                    )
        history.append({"epoch": epoch + 1, "mean_loss": epoch_loss / seen})
        print(f"epoch={epoch + 1} mean_loss={epoch_loss / seen:.4f}", flush=True)
    elapsed = time.perf_counter() - started

    model.save_pretrained(output)
    processor.save_pretrained(output)
    meta = {
        "args": vars(args),
        "train_images": len(rows),
        "optimizer_steps": step,
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "loss_history": history,
    }
    (output / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved={output} train_seconds={elapsed:.0f}", flush=True)


if __name__ == "__main__":
    main()
