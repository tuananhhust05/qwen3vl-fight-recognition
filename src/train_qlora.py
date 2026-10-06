import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

from fight_data import messages_for, read_jsonl


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def limit_pixels(processor):
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = 512 * 28 * 28
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = 128 * 28 * 28


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


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

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
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=language_target_pattern(model),
        ),
    )
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(args.model)
    limit_pixels(processor)

    first = encode(processor, rows[0]["image"], rows[0]["label"])
    visible = first["labels"][0]
    visible = visible[visible != -100]
    decoded = processor.tokenizer.decode(visible, skip_special_tokens=False)
    print(f"first_label_tokens={decoded!r}", flush=True)

    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    updates_per_epoch = (len(rows) + args.accum - 1) // args.accum
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.05)),
        num_training_steps=total_updates,
    )

    model.train()
    device = model.device
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        seen = 0
        for index, row in enumerate(rows, start=1):
            batch = move_to_device(encode(processor, row["image"], row["label"]), device)
            loss = model(**batch).loss / args.accum
            loss.backward()
            epoch_loss += loss.item() * args.accum
            seen += 1
            if index % args.accum == 0 or index == len(rows):
                torch.nn.utils.clip_grad_norm_(
                    (param for param in model.parameters() if param.requires_grad),
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or index == len(rows):
                    print(
                        f"epoch={epoch + 1} step={step} sample={index}/{len(rows)} "
                        f"loss={epoch_loss / seen:.4f}",
                        flush=True,
                    )
        print(f"epoch={epoch + 1} mean_loss={epoch_loss / seen:.4f}", flush=True)

    model.save_pretrained(output)
    processor.save_pretrained(output)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
