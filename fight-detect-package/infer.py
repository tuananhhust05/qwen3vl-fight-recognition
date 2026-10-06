import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROMPT = (
    "Does this image show people fighting, including a punch, kick, slap, or grapple? "
    "Reply with only one word: fight or none."
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(base_model, adapter, device):
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(base_model)
    return model, processor


def predict(model, processor, image, device):
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
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    text = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )[0]
    return text.strip()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=str(root / "sample" / "fight.jpg"))
    parser.add_argument("--adapter", default=str(root / "adapter"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not Path(args.adapter).exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    device = pick_device()
    print(f"device={device}", flush=True)
    model, processor = load_model(args.base_model, args.adapter, device)
    answer = predict(model, processor, image, device)
    print(answer)


if __name__ == "__main__":
    main()
