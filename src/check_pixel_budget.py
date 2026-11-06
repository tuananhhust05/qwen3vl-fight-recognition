"""Show how many visual tokens an image gets with and without limit_pixels().

    python src/check_pixel_budget.py --model Qwen/Qwen3-VL-4B-Instruct --image path.jpg
"""

import argparse

from PIL import Image
from transformers import AutoProcessor

from eval_fight import limit_pixels
from fight_data import messages_for


def legacy_limit_pixels(processor):
    """The v1 implementation, verbatim: only sets attributes that may not exist."""
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = 512 * 28 * 28
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = 128 * 28 * 28


def visual_tokens(processor, image):
    batch = processor.apply_chat_template(
        messages_for(image), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    t, h, w = batch["image_grid_thw"][0].tolist()
    merge = processor.image_processor.merge_size
    return (t, h, w), t * h * w // (merge * merge)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"image_size={Image.open(args.image).size}")
    print(f"default size={dict(processor.image_processor.size)}")
    grid, tokens = visual_tokens(processor, args.image)
    print(f"default grid_thw={grid} visual_tokens={tokens}")
    legacy_limit_pixels(processor)
    grid, tokens = visual_tokens(processor, args.image)
    print(f"v1 limit_pixels grid_thw={grid} visual_tokens={tokens}")
    limit_pixels(processor)
    print(f"limited size={dict(processor.image_processor.size)}")
    grid, tokens = visual_tokens(processor, args.image)
    print(f"limited grid_thw={grid} visual_tokens={tokens}")


if __name__ == "__main__":
    main()
