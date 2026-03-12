#!/bin/bash
# Evaluate SFT V8 checkpoints
# Run after SFT V8 training completes
#
# Usage: bash scripts/eval_sft_v8.sh

set -e

SFT_V8_SESSION="5c203420-1891-5999-8cef-23cbb0be8402"
BASE_MODEL="Qwen/Qwen3.5-35B-A3B"
SEED_OFFSET=10000
N_TASKS=20

eval_checkpoint() {
    local name="$1"
    local checkpoint="$2"
    local extra="$3"

    echo "===== Evaluating: $name ====="
    mkdir -p "results/${name}"

    nohup uv run python eval/run_eval.py \
        --model "$BASE_MODEL" \
        --model-path "tinker://${SFT_V8_SESSION}:train:0/weights/${checkpoint}" \
        --benchmark all \
        --n-tasks "$N_TASKS" \
        --seed-offset "$SEED_OFFSET" \
        $extra \
        --experiment-name "$name" \
        > "results/${name}/log.txt" 2>&1 &
    echo "PID: $!"
}

echo "SFT V8 Evaluation Suite"
echo "======================="
echo "Session: $SFT_V8_SESSION"
echo ""

# Evaluate key checkpoints
# Epoch 1 end (~step 148)
eval_checkpoint "sft_v8_ep1" "checkpoint-0150" ""
echo "Waiting 5s between launches..."
sleep 5

# Epoch 2 end (~step 296)
eval_checkpoint "sft_v8_ep2" "checkpoint-0300" ""
sleep 5

# Final (epoch 3)
eval_checkpoint "sft_v8_final" "final" ""
sleep 5

# Best checkpoint with strategies
eval_checkpoint "sft_v8_final_strategy" "final" "--eval-strategy"

echo ""
echo "All evaluations launched. Check results/ for progress."
echo "Each eval takes ~3-5 hours for all 14 benchmarks."
