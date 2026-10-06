# Fine-Tuning Qwen3-VL-4B for Still-Image Fight Behavior Recognition

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Research-lightgrey.svg)](#license-and-data-attribution)

**Supervised adaptation of a vision–language model (VLM) for binary recognition of physical fight behaviors from a single RGB image.**

This repository provides a complete pipeline for dataset curation, QLoRA fine-tuning of [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), and held-out evaluation that compares the base model against the adapted model under an identical protocol.

| Setting | Accuracy | Fight Recall | None Precision |
| --- | ---: | ---: | ---: |
| Base Qwen3-VL-4B-Instruct | 93.26% | 87.74% | 87.59% |
| **QLoRA fine-tuned (this work)** | **99.29%** | **98.71%** | **98.45%** |

Evaluation set: **282** images (**155** fight / **127** none), never used for gradient updates.

---

## Abstract

Detecting interpersonal violence from still images is a practical computer-vision problem for content moderation and surveillance triage. Temporal cues available in video are absent in a single frame, so appearance-only recognition must rely on body configuration, contact, and scene context. We fine-tune a 4B-parameter multimodal instruction model with **4-bit QLoRA**, adapting only language-side projection layers while freezing the vision encoder. On a curated held-out split, the adapted model reduces false negatives on fight images from **19 → 2** while preserving zero false positives on non-fight hard negatives (hugging, high-five, handshaking).

---

## Method

### Base model

- **Architecture:** Qwen3-VL (dense VLM)
- **Checkpoint:** `Qwen/Qwen3-VL-4B-Instruct`
- **Task format:** image + fixed English instruction → one-token class label (`fight` / `none`)

### Prompt (fixed at train and inference)

```text
Does this image show people fighting, including a punch, kick, slap, or grapple?
Reply with only one word: fight or none.
```

### Parameter-efficient adaptation

| Hyperparameter | Value |
| --- | --- |
| Method | QLoRA (NF4, double quant) |
| Compute dtype | bfloat16 |
| LoRA rank \(r\) | 16 |
| LoRA \(\alpha\) | 32 |
| LoRA dropout | 0.05 |
| Target modules | language `q/k/v/o_proj`, `gate/up/down_proj` |
| Trainable parameters | 33.0M (**0.74%** of 4.47B) |
| Epochs | 2 |
| Effective batch size | 8 (batch 1 × accum 8) |
| Learning rate | \(1\times10^{-4}\) (cosine, 5% warmup) |
| Max image tokens | \(\approx 512\times28\times28\) pixels |
| Hardware | 1× NVIDIA RTX A5000 (24 GB) |

Vision modules remain frozen. Labels mask the user prompt so only the assistant answer contributes to the loss.

---

## Dataset

### Positive class (`fight`)

| Source | Role | Count (approx.) |
| --- | --- | ---: |
| Action-Net (`fighting`) | street / sports / brawl stills, manually filtered | 677 |
| HII Boxing-Punching | punch-centric interactions | 193 |
| HII Kicking | kick-centric interactions | 151 |

Action-Net `fighting` was visually audited; unrelated frames (standing talk, aftermath, protests without contact, UI overlays, games) were removed before training.

### Negative class (`none`)

Hard negatives from the Human Interaction Images (HII) dataset, stored under `datasets/negatives/`:

- Hugging
- HighFive
- Handshaking

These classes are visually adjacent to grappling and are used to stress-test false-positive behavior.

### Split protocol

| Split | Fight | None | Total |
| --- | ---: | ---: | ---: |
| Train | 866 | 510 | 1,376 |
| Test (held-out) | 155 | 127 | 282 |

JSONL manifests are stored under [`data/`](data/). Image paths in the manifests are absolute on the training host; regenerate locally with:

```bash
python src/prepare_data.py --root .
```

---

## Results

### Aggregate metrics (held-out test)

| Model | Accuracy | Fight Recall | Fight Precision | None Recall | None Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.9326 | 0.8774 | 1.0000 | 1.0000 | 0.8759 |
| Fine-tuned | **0.9929** | **0.9871** | **1.0000** | **1.0000** | **0.9845** |

### Confusion matrices

**Baseline**

|  | Pred `fight` | Pred `none` | Other |
| --- | ---: | ---: | ---: |
| True `fight` | 136 | 18 | 1 |
| True `none` | 0 | 127 | 0 |

**Fine-tuned**

|  | Pred `fight` | Pred `none` | Other |
| --- | ---: | ---: | ---: |
| True `fight` | 153 | 2 | 0 |
| True `none` | 0 | 127 | 0 |

Full per-image predictions: [`results/baseline.json`](results/baseline.json), [`results/finetuned.json`](results/finetuned.json). Summary: [`results/comparison.json`](results/comparison.json).

### Interpretation

1. The base VLM already separates many fight vs non-fight cases (high none-recall).
2. Fine-tuning primarily **recovers missed fight cases** (recall +11.0 pp) without introducing false alarms on the hard-negative set used here.
3. Remaining errors (2 images) are ambiguous punch frames from HII Boxing-Punching.

---

## Repository layout

```text
vlm-finetune/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes          # Git LFS for *.safetensors
├── src/
│   ├── fight_data.py       # split construction + metrics
│   ├── prepare_data.py
│   ├── train_qlora.py      # QLoRA training loop
│   └── eval_fight.py       # baseline / adapter evaluation
├── data/                   # train/test JSONL + summary
├── datasets/               # curated fight / negative images
├── results/                # baseline vs fine-tuned reports
└── fight-detect-package/   # portable inference bundle
    ├── adapter/            # LoRA weights (Git LFS)
    ├── infer.py
    ├── requirements.txt
    └── sample/fight.jpg
```

---

## Quick start (inference)

```bash
git clone <your-repo-url>
cd vlm-finetune
git lfs pull
python3 -m venv venv
source venv/bin/activate
pip install -r fight-detect-package/requirements.txt
python fight-detect-package/infer.py fight-detect-package/sample/fight.jpg
```

Expected output for a fight sample:

```text
fight
```

The first run downloads the base checkpoint from Hugging Face (~9 GB). The LoRA adapter is loaded from `fight-detect-package/adapter/`.

---

## Reproducing training

Requires a CUDA GPU with ≥16 GB VRAM (24 GB recommended).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py --root .
python src/eval_fight.py --root . --out results/baseline.json
python src/train_qlora.py --root . --output outputs/fight-lora
python src/eval_fight.py --root . --adapter outputs/fight-lora --out results/finetuned.json
```

---

## Limitations

- Labels are **binary** (`fight` / `none`). The adapter does not emit fine-grained subclasses (punch / kick / slap / grapple).
- Images are mostly web / consumer photography; CCTV domain shift is not guaranteed.
- Non-fight coverage in this release focuses on hugging, high-five, and handshaking; other hard negatives (sports contact, dancing, police restraint) may still confuse the model.
- Still-image recognition cannot recover actions that are only identifiable over time.

---

## Ethical use

This project is intended for research, dataset curation studies, and defensive moderation prototypes. Do not deploy for unlawful surveillance or discriminatory enforcement. Training data includes depictions of violence; handle and redistribute responsibly under the licenses of the original dataset providers.

---

## License and data attribution

Code in this repository is provided for research use. Upstream assets retain their original licenses:

- **Qwen3-VL-4B-Instruct** — [Qwen / Alibaba](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- **Action-Net** — [OlafenwaMoses/Action-Net](https://github.com/OlafenwaMoses/Action-Net)
- **Human Interaction Images (HII)** — Tanisik et al., *Pattern Recognition Letters*, 2016 ([project page](https://vision.cs.hacettepe.edu.tr/interaction_images/))

---

## Citation

If you use this repository, please cite the base model and source datasets, and reference this work as:

```bibtex
@misc{vlm-fight-qlora-2026,
  title        = {Fine-Tuning Qwen3-VL-4B for Still-Image Fight Behavior Recognition},
  year         = {2026},
  note         = {QLoRA adaptation with held-out fight / none evaluation},
  howpublished = {\url{https://github.com/<your-username>/<your-repo>}}
}
```

---

## Acknowledgments

Experiments were conducted on a cloud NVIDIA RTX A5000 instance. Model implementation relies on Hugging Face `transformers`, `peft`, and `bitsandbytes`.
# qwen3vl-fight-recognition
