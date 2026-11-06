import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Must match training (src/fight_data.py and src/eval_fight.py).
PROMPT = (
    "Does this image show people fighting, including a punch, kick, slap, or grapple? "
    "Reply with only one word: fight or none."
)
MAX_PIXELS = 512 * 28 * 28
MIN_PIXELS = 128 * 28 * 28


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def limit_pixels(processor):
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = MAX_PIXELS
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = MIN_PIXELS
    size = getattr(image_processor, "size", None)
    if size is not None and (getattr(size, "longest_edge", None) or (isinstance(size, dict) and "longest_edge" in size)):
        image_processor.size = {"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}


def load_model(base_model, adapter, device):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float16 if device == "mps" else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(base_model)
    limit_pixels(processor)
    return model, processor


def predict(model, processor, image, device, threshold):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    tokenizer = processor.tokenizer
    fight_id = tokenizer.encode("fight", add_special_tokens=False)[0]
    none_id = tokenizer.encode("none", add_special_tokens=False)[0]
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    p_fight = torch.softmax(logits[[fight_id, none_id]], dim=-1)[0].item()
    return {"label": "fight" if p_fight >= threshold else "none", "p_fight": round(p_fight, 4)}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=str(root / "sample" / "fight.jpg"))
    parser.add_argument("--adapter", default=str(root / "adapter"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not (Path(args.adapter) / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    device = pick_device()
    model, processor = load_model(args.base_model, args.adapter, device)
    print(json.dumps(predict(model, processor, image, device, args.threshold)))


if __name__ == "__main__":
    main()
