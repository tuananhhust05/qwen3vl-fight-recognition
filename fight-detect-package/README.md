# Fight Detect Package

Fine-tuned LoRA adapter for `Qwen/Qwen3-VL-4B-Instruct`.
Input: one image. Output: `fight` or `none`.

## Folder

```
fight-detect-package/
  adapter/          # LoRA weights
  sample/fight.jpg  # sample image
  infer.py
  requirements.txt
  README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads the base model from Hugging Face (~9GB).

## Run

```bash
python infer.py sample/fight.jpg
python infer.py /path/to/your/image.jpg
```

## Notes

- Keep the prompt inside `infer.py` unchanged.
- GPU (CUDA) is preferred. Apple MPS works. CPU works but is slow.
- This package does not include the base model. Internet is needed for the first download.
