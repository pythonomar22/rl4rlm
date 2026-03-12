#!/bin/bash
# Quick eval: NIAH + doc_classify + multi_niah (3 benchmarks, ~15 minutes)
# Usage: ./scripts/quick_eval.sh <model-path> <experiment-name>
# Example: ./scripts/quick_eval.sh "tinker://session:train:0/weights/state-0018" "sft_v1_ep1"

MODEL_PATH="${1:?Usage: $0 <model-path> <experiment-name>}"
EXPERIMENT="${2:?Usage: $0 <model-path> <experiment-name>}"

echo "Quick eval: $EXPERIMENT"
echo "Model: $MODEL_PATH"
echo "Benchmarks: niah, multi_niah, doc_classify (10 tasks each)"

uv run python eval/run_eval.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --model-path "$MODEL_PATH" \
  --benchmark niah multi_niah doc_classify \
  --n-tasks 10 \
  --seed-offset 20000 \
  --experiment-name "quick_${EXPERIMENT}"
