import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from fight_data import messages_for, parse_label, read_jsonl, score


def load_model(model_name, adapter):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    limit_pixels(processor)
    return model, processor


def limit_pixels(processor):
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = 512 * 28 * 28
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = 128 * 28 * 28


def predict(model, processor, image):
    inputs = processor.apply_chat_template(
        messages_for(image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return text, parse_label(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = read_jsonl(root / "data" / f"{args.split}.jsonl")
    model, processor = load_model(args.model, args.adapter or None)
    predictions = []
    for index, row in enumerate(rows, start=1):
        image = Path(row["image"])
        if not image.is_absolute():
            image = root / image
        text, label = predict(model, processor, str(image))
        predictions.append(
            {
                "image": str(image.relative_to(root)) if image.is_relative_to(root) else str(image),
                "label": row["label"],
                "prediction": label,
                "raw": text.strip(),
            }
        )
        if index % 20 == 0 or index == len(rows):
            running = score(predictions)
            print(
                f"{index}/{len(rows)} accuracy={running['accuracy']:.3f}",
                flush=True,
            )
    metrics = score(predictions)
    output = {"adapter": args.adapter or None, "metrics": metrics, "predictions": predictions}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
