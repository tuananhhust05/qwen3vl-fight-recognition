# Fight Detect Package

LoRA adapter for `Qwen/Qwen3-VL-4B-Instruct` (training seed 42, see the main README for evaluation).
Input: one image. Output: JSON with `label` (`fight` / `none`) and `p_fight`.

## Folder

```
fight-detect-package/
  adapter/          # LoRA weights (r=16, 132 MB) + processor files
  sample/fight.jpg  # sample image (Action-Net, test split)
  infer.py
  requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the base model from Hugging Face (~9 GB).

## Run

```bash
python infer.py sample/fight.jpg            # -> {"label": "fight", "p_fight": 0.9985}
python infer.py /path/to/image.jpg --threshold 0.5
```

## Notes

* The prompt and the image pixel budget in `infer.py` must stay identical to training.
* `p_fight` is the softmax over the `fight` / `none` answer tokens; 0.5 is the operating point used in the evaluation.
* The adapter was trained on a 4-bit (NF4) base. `infer.py` loads the base in bf16/fp16, which needs about 10-11 GB of GPU memory (the bf16 evaluation run peaked at 11.3 GB with batch 16 and had a p50 latency of 97 ms per image on an RTX 4090).
* CUDA is preferred. Apple MPS works. CPU works but is slow.
