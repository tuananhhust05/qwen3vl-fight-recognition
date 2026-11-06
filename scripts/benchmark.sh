#!/usr/bin/env bash
# Inference-cost benchmark on the test split (run on an otherwise idle GPU).
#   MODEL=/path/to/Qwen3-VL-4B-Instruct ADAPTER=outputs/fight-lora-seed42 bash scripts/benchmark.sh
# Writes results/bench_{base_nf4,seed42_nf4,seed42_bf16}_test.json (metrics, throughput at batch 16,
# single-image latency over 50 images, peak VRAM).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
ADAPTER=${ADAPTER:-outputs/fight-lora-seed42}
mkdir -p logs results
cd src
python eval_fight.py --root .. --model "$MODEL" --split test \
  --out ../results/bench_base_nf4_test.json > ../logs/bench_base_nf4.log 2>&1
python eval_fight.py --root .. --model "$MODEL" --adapter "../$ADAPTER" --split test \
  --out ../results/bench_seed42_nf4_test.json > ../logs/bench_seed42_nf4.log 2>&1
python eval_fight.py --root .. --model "$MODEL" --adapter "../$ADAPTER" --split test --no-quant \
  --out ../results/bench_seed42_bf16_test.json > ../logs/bench_seed42_bf16.log 2>&1
echo "benchmark done"
