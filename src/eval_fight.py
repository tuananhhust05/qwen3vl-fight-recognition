"""Evaluate the base model or a LoRA adapter on a manifest split.

The answer is read from the logits of the first generated token: p(fight) is the
softmax over the two label tokens {"fight", "none"}. This is equivalent to greedy
decoding restricted to the label set, gives a calibrated score for ROC/PR curves,
and lets us check format compliance (is the unconstrained argmax token a label?).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from fight_data import LABELS, messages_for, parse_label, read_jsonl
from metrics import full_report

MAX_PIXELS = 512 * 28 * 28
MIN_PIXELS = 128 * 28 * 28


def limit_pixels(processor):
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = MAX_PIXELS
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = MIN_PIXELS
    # recent transformers read the pixel budget from `size` (a dict or SizeDict)
    size = getattr(image_processor, "size", None)
    if size is not None and (getattr(size, "longest_edge", None) or (isinstance(size, dict) and "longest_edge" in size)):
        image_processor.size = {"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}


def load_model(model_name, adapter, quantize=True):
    kwargs = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    limit_pixels(processor)
    return model, processor


def label_token_ids(tokenizer):
    ids = {}
    for label in LABELS:
        pieces = tokenizer.encode(label, add_special_tokens=False)
        ids[label] = pieces[0]
    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"Label tokens collide: {ids}")
    return ids


@torch.inference_mode()
def score_batch(model, processor, images, token_ids):
    inputs = processor.apply_chat_template(
        [messages_for(image) for image in images],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    logits = model(**inputs).logits[:, -1, :].float()
    pair = torch.stack([logits[:, token_ids["fight"]], logits[:, token_ids["none"]]], dim=-1)
    p_fight = torch.softmax(pair, dim=-1)[:, 0]
    free_tokens = logits.argmax(dim=-1)
    free_text = processor.batch_decode(free_tokens[:, None], skip_special_tokens=True)
    return p_fight.tolist(), free_text


def resolve(root, image):
    image = Path(image)
    return image if image.is_absolute() else root / image


def measure_latency(model, processor, images, token_ids, repeats):
    """Single-image end-to-end latency (preprocess + forward), after warm-up."""
    for image in images[:3]:
        score_batch(model, processor, [image], token_ids)
    torch.cuda.synchronize()
    timings = []
    for image in images[:repeats]:
        start = time.perf_counter()
        score_batch(model, processor, [image], token_ids)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return {
        "batch_size": 1,
        "images": len(timings),
        "p50_ms": timings[len(timings) // 2],
        "p95_ms": timings[min(len(timings) - 1, int(len(timings) * 0.95))],
        "mean_ms": sum(timings) / len(timings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-quant", action="store_true", help="load the base model in bf16")
    parser.add_argument("--latency-images", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = read_jsonl(root / "data" / f"{args.split}.jsonl")
    model, processor = load_model(args.model, args.adapter or None, quantize=not args.no_quant)
    token_ids = label_token_ids(processor.tokenizer)
    images = [str(resolve(root, row["image"])) for row in rows]

    # warm-up so the throughput number excludes CUDA/cuBLAS initialisation
    score_batch(model, processor, images[: args.batch_size], token_ids)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predictions = []
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset : offset + args.batch_size]
        batch_images = images[offset : offset + args.batch_size]
        probabilities, free_text = score_batch(model, processor, batch_images, token_ids)
        for row, probability, text in zip(batch_rows, probabilities, free_text):
            predictions.append(
                {
                    "image": row["image"],
                    "source": row.get("source", "unknown"),
                    "label": row["label"],
                    "p_fight": probability,
                    "free_token": text.strip(),
                    "free_prediction": parse_label(text),
                }
            )
        print(f"{len(predictions)}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    report = full_report(predictions, args.threshold)
    report["throughput"] = {
        "batch_size": args.batch_size,
        "images": len(rows),
        "seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if args.latency_images > 0:
        report["latency"] = measure_latency(
            model, processor, images, token_ids, args.latency_images
        )
    output = {
        "model": args.model,
        "adapter": args.adapter or None,
        "quantization": None if args.no_quant else "nf4-double-quant",
        "split": args.split,
        "gpu": torch.cuda.get_device_name(0),
        "metrics": report,
        "predictions": predictions,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "by_source"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
