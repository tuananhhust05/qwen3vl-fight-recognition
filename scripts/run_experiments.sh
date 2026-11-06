#!/usr/bin/env bash
# End-to-end reproduction of every number in README.md:
#   leakage audit -> leakage-free split -> zero-shot baseline -> QLoRA x N seeds -> eval -> report
# Two GPUs: seeds are distributed round-robin over $GPUS.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
SEEDS=${SEEDS:-"42 43 44"}
GPUS=(${GPUS:-0 1})
mkdir -p results logs outputs

# 1. audit the original (legacy) split, then build and audit the new one
python src/audit_duplicates.py --root . --train data/legacy/train.jsonl --test data/legacy/test.jsonl \
  --out results/audit_legacy_split.json
python src/prepare_data.py --root .
python src/audit_duplicates.py --root . --train data/train.jsonl --test data/test.jsonl \
  --out results/audit_new_split.json
python src/audit_duplicates.py --root . --train data/train.jsonl --test data/val.jsonl \
  --out results/audit_new_split_val.json

# 2. zero-shot baseline (same 4-bit loading as the fine-tuned runs)
CUDA_VISIBLE_DEVICES=${GPUS[0]} python src/eval_fight.py --model "$MODEL" --split val --latency-images 0 \
  --out results/baseline_val.json
CUDA_VISIBLE_DEVICES=${GPUS[0]} python src/eval_fight.py --model "$MODEL" --split test \
  --out results/baseline_test.json

# 3. QLoRA fine-tuning, one seed per GPU in parallel
run_seed() {
  local seed=$1 gpu=$2 out=outputs/fight-lora-seed$1
  CUDA_VISIBLE_DEVICES=$gpu python src/train_qlora.py --model "$MODEL" --seed "$seed" --output "$out" \
    > "logs/train_seed$seed.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python src/eval_fight.py --model "$MODEL" --adapter "$out" --split val \
    --latency-images 0 --out "results/finetuned_seed${seed}_val.json" > "logs/eval_seed${seed}_val.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu python src/eval_fight.py --model "$MODEL" --adapter "$out" --split test \
    --out "results/finetuned_seed${seed}_test.json" > "logs/eval_seed${seed}_test.log" 2>&1
  cp "$out/train_meta.json" "results/train_meta_seed$seed.json"
}
index=0
for seed in $SEEDS; do
  run_seed "$seed" "${GPUS[$((index % ${#GPUS[@]}))]}" &
  index=$((index + 1))
  if (( index % ${#GPUS[@]} == 0 )); then wait; fi
done
wait

# 4. aggregate
python src/report.py --results results
