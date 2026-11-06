# Fine-Tuning Qwen3-VL-4B for Still-Image Fight Behavior Recognition

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.12-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-Research-lightgrey.svg)](#license-and-data-attribution)

**4-bit QLoRA adaptation of a vision–language model for binary `fight` / `none` recognition from a single RGB image, evaluated on a leakage-audited held-out split over 5 training seeds.**

| Setting (held-out test, 322 images) | Accuracy | Macro-F1 | Fight recall | False positives | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3-VL-4B-Instruct, zero-shot | 88.20% | 88.08% | 80.61% | 0 / 126 | 0.9946 |
| Base + decision threshold tuned on val | 94.72% | 94.57% | 91.33% | 0 / 126 | 0.9946 |
| **QLoRA fine-tuned, mean ± std over 5 seeds** | **98.76 ± 0.58%** | **98.70 ± 0.60%** | **98.27 ± 1.06%** | 0.6 / 126 (mean) | **0.9997** |

* Errors per 322 test images: **38 → 4.0** (mean over seeds) versus the zero-shot model, **17 → 4.0** versus the threshold-tuned baseline.
* Every number above is produced by [`scripts/run_experiments.sh`](scripts/run_experiments.sh) and stored, with per-image predictions, under [`results/`](results/). Training/eval logs are in [`logs/`](logs/).

---

## What changed relative to the first version

The first release ([`results/legacy/`](results/legacy/)) reported 99.29% accuracy on 282 images. Re-auditing that setup found issues that are fixed here:

| Issue in v1 | Evidence | Fix |
| --- | --- | --- |
| **Train/test leakage**: 8 of 282 test images (2.8%) are re-uploads or crops of training images, including cross-source copies (Action-Net ↔ HII) | [`results/audit_legacy_split.json`](results/audit_legacy_split.json) | Split by near-duplicate *groups* (MD5 + 64-bit pHash, Hamming ≤ 6, union-find). Audit of the new split: **0** leaked images in test and val ([`audit_new_split.json`](results/audit_new_split.json)) |
| No validation split; one training run | — | 70/10/20 train/val/test per source; 5 seeds; mean ± std and bootstrap 95% CIs |
| The 512×28×28 pixel budget was silently ignored by recent `transformers` (it reads `size`, a `SizeDict`), so images were fed at native resolution (e.g. 1,457 visual tokens for a 1500×1000 photo) | [`results/pixel_budget_check.txt`](results/pixel_budget_check.txt) (`src/check_pixel_budget.py`) | Budget applied through `size`; ≤ 0.4 MP per image, ~100–390 visual tokens instead of ~1,450 |
| Inference package used a different pixel budget than training | `fight-detect-package/infer.py` | Same budget; outputs `p(fight)` |
| Sample-by-sample training loop (batch 1 × 8 accumulation) | — | True batches of 8 (same effective batch). A check asserts the batched labels equal the single-example labels token-for-token. ~10 min per run on one RTX 4090 |
| Answer parsed from free-form generation | — | `p(fight)` = softmax over the two label tokens at the first answer position. This enables ROC/PR analysis and threshold selection; format compliance of free generation is still reported |

The headline accuracy is slightly lower than v1 (98.76% vs 99.29%), but it is now measured on a test set that is larger (322 vs 282) and free of duplicates, averaged over seeds instead of one run.

---

## Method

| | |
| --- | --- |
| Base model | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (4.47B params) |
| Prompt (train = inference) | `Does this image show people fighting, including a punch, kick, slap, or grapple? Reply with only one word: fight or none.` |
| Quantization | NF4, double quantization, bf16 compute |
| LoRA | r = 16, α = 32, dropout 0.05, on language-model `q/k/v/o/gate/up/down_proj`; vision tower frozen |
| Trainable parameters | 33.0M (**0.74%** of 4.47B) |
| Loss | Cross-entropy on the assistant answer tokens only (`<label><|im_end|>\n`) |
| Optimisation | AdamW, lr 1e-4, cosine schedule with 5% warm-up, grad-clip 1.0, 2 epochs, batch 8 (284 steps) |
| Image budget | 100k – 401k pixels (Qwen3-VL patch 16, merge 2) |
| Seeds | 42, 43, 44, 45, 46 (LoRA init + data order) |
| Hardware | 1× NVIDIA RTX 4090 24 GB per run (board power-capped at 250 W). Peak training VRAM **11.0 GB**. Wall-clock 539–762 s per run |
| Software | torch 2.11.0+cu129, transformers 5.12.0, peft 0.21.2, bitsandbytes 0.49.2 |

## Dataset

| Source | Label | Images after de-duplication | Train / Val / Test |
| --- | --- | ---: | ---: |
| Action-Net `fighting` (train + test folders pooled) | fight | 645 | 452 / 64 / 129 |
| HII Boxing-Punching | fight | 187 | 131 / 19 / 37 |
| HII Kicking | fight | 151 | 106 / 15 / 30 |
| HII Hugging (hard negative) | none | 209 | 146 / 21 / 42 |
| HII HighFive (hard negative) | none | 196 | 137 / 20 / 39 |
| HII Handshaking (hard negative) | none | 226 | 158 / 23 / 45 |
| **Total** | | **1,614** (from 1,658 raw; 44 duplicates removed) | **1,130 / 162 / 322** |

Manifests: [`data/train.jsonl`](data/train.jsonl), [`data/val.jsonl`](data/val.jsonl), [`data/test.jsonl`](data/test.jsonl). Summary and audit counters: [`data/summary.json`](data/summary.json). The v1 manifests are kept in [`data/legacy/`](data/legacy/) for comparison.

---

## Results

### Per seed (threshold 0.5, test set)

| Seed | Accuracy | 95% bootstrap CI | Missed fights (of 196) | False alarms (of 126) |
| --- | ---: | --- | ---: | ---: |
| 42 | 99.38% | [98.45, 100.0] | 2 | 0 |
| 43 | 98.14% | [96.58, 99.38] | 6 | 0 |
| 44 | 99.38% | [98.45, 100.0] | 1 | 1 |
| 45 | 98.45% | [97.20, 99.69] | 5 | 0 |
| 46 | 98.45% | [96.89, 99.69] | 3 | 2 |
| **Mean ± std** | **98.76 ± 0.58%** | | **3.4** | **0.6** |

Choosing the threshold on the validation split instead of 0.5 gives 98.39 ± 0.67%. Validation accuracy is already 100% for every seed, so it carries no signal for threshold choice, and 0.5 is kept as the reported operating point. Source data: [`results/summary.json`](results/summary.json).

### Accuracy by source (zero-shot → fine-tuned seed 42)

| Source | Zero-shot | Fine-tuned |
| --- | ---: | ---: |
| Action-Net fighting | 80.6% | 100.0% |
| HII Boxing-Punching | 78.4% | 97.3% |
| HII Kicking | 83.3% | 96.7% |
| HII Hugging / HighFive / Handshaking | 100% / 100% / 100% | 100% / 100% / 100% |

The zero-shot model already rejects every hard negative. Its errors are all missed fights, which is why its ROC-AUC (0.995) is much higher than its accuracy: the ranking is good but the 0.5 operating point is badly calibrated. Fine-tuning mainly fixes this recall gap.

### Error analysis

Across 5 seeds, 10 distinct test images are ever misclassified. The most persistent ones are ambiguous at the label level:

* `hii/Boxing-Punching/093.jpg` (wrong in 5/5 seeds): two infants on a sofa, one resting a fist on the other's cheek.
* `hii/Kicking/147.jpg` (4/5): two footballers in training, one playfully kicking toward the other, both smiling.
* `negatives/HighFive/165.jpg` (2/5, false alarm): a raised-arm high five partly occluded by an umbrella.

These are "playful contact" cases that the binary label set cannot express. They point at the dataset's label definition rather than at a model failure.

### Inference cost

Measured with [`scripts/benchmark.sh`](scripts/benchmark.sh) on one otherwise idle RTX 4090 over the 322 test images. Throughput is batch 16. Latency is batch 1, end-to-end (pre-processing + forward pass for `p(fight)`), p50/p95 over 50 images after warm-up.

| Configuration | Test accuracy | Throughput | Latency p50 / p95 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Base, NF4 | 88.20% | 18.4 img/s | 135 / 146 ms | 5.7 GB |
| Fine-tuned (seed 42), NF4 base + LoRA (evaluation setting) | 99.38% | 13.6 img/s | 178 / 191 ms | 5.8 GB |
| Fine-tuned (seed 42), bf16 base + LoRA (`fight-detect-package` setting) | 99.38% | 13.8 img/s | 97 / 108 ms | 11.3 GB |

The 4-bit and bf16 runs make the same predictions on all 322 test images (2 missed fights, 0 false alarms). The adapter needs no 4-bit kernels at serving time. With bf16 it runs at < 0.11 s per image at p95, and with NF4 it fits in under 6 GB. Raw files: `results/bench_*_test.json`.

---

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_experiments.sh          # audit -> split -> baseline -> 3 seeds -> report
SEEDS="42 43 44 45 46" bash scripts/run_experiments.sh   # the 5-seed setting reported above
pytest -q                                # unit tests (CPU only)
```

Single steps:

```bash
python src/audit_duplicates.py --root . --train data/train.jsonl --test data/test.jsonl
python src/prepare_data.py --root .
python src/eval_fight.py --root . --split test --out results/baseline_test.json
python src/train_qlora.py --root . --seed 42 --output outputs/fight-lora-seed42
python src/eval_fight.py --root . --adapter outputs/fight-lora-seed42 --split test --out results/finetuned_seed42_test.json
python src/report.py --results results
```

The experiments were run with model weights downloaded once to local disk (`--model /path/to/Qwen3-VL-4B-Instruct`, SHA-256 checked against the Hub). This is why the `model` field in the result files holds a local path.

## Inference package

[`fight-detect-package/`](fight-detect-package/) contains the seed-42 adapter (seed 42 is the script default and was fixed before results were seen) and a standalone `infer.py`:

```bash
pip install -r fight-detect-package/requirements.txt
python fight-detect-package/infer.py fight-detect-package/sample/fight.jpg
# {"label": "fight", "p_fight": 0.9985}     (actual output; the bf16 evaluation run gives 0.998499 for this image)
```

## Repository layout

```text
src/
  fight_data.py        sources, near-duplicate grouping, leakage-free split, metrics
  audit_duplicates.py  exact + pHash duplicate audit between two manifests
  train_qlora.py       batched QLoRA training loop
  eval_fight.py        logit-based evaluation, throughput and latency
  metrics.py           ROC/PR, bootstrap CIs, threshold selection
  report.py            seed aggregation -> results/summary.json
scripts/run_experiments.sh   full pipeline
scripts/benchmark.sh         inference-cost benchmark
tests/                 CPU unit tests (split has no cross-split duplicates, metrics, parsing)
data/  results/  logs/
fight-detect-package/
```

## Limitations

* Binary labels only. Playful contact, sparring and sport are not separate classes (see error analysis).
* All negatives come from HII and are interaction poses (hug, high-five, handshake). Other hard negatives such as contact sports, dancing or police restraint are not covered, so the near-zero false-alarm rate should not be extrapolated beyond these classes.
* Web and consumer photos. CCTV viewpoints, low light and motion blur are untested.
* The test set is small (322 images). One error moves accuracy by 0.31 pp, so compare runs through the confidence intervals.
* Still images cannot capture actions that are only identifiable over time.

## Ethical use

Intended for research and defensive moderation prototypes. Do not deploy for unlawful surveillance or discriminatory enforcement. The training data depicts violence; redistribute only under the source licenses.

## License and data attribution

Code is provided for research use. Upstream assets keep their licenses:

* **Qwen3-VL-4B-Instruct**: [Qwen / Alibaba](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
* **Action-Net**: [OlafenwaMoses/Action-Net](https://github.com/OlafenwaMoses/Action-Net)
* **Human Interaction Images (HII)**: Tanisik et al., *Pattern Recognition Letters*, 2016 ([project page](https://vision.cs.hacettepe.edu.tr/interaction_images/))
