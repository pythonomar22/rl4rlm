#!/bin/bash
# Comprehensive evaluation of all promising checkpoints
# Runs all 14 benchmarks with fixed seeds for fair comparison
#
# Usage: bash scripts/eval_comprehensive.sh [checkpoint_name] [model_path]

set -e

SEED_OFFSET=10000
N_TASKS=20
BASE_MODEL="Qwen/Qwen3.5-35B-A3B"

# V13 session (completed or nearly so)
V13_SESSION="7e983e2b-9ee6-53a6-85d1-7de6c73a3c94"

# SFT V8 session (launched just now)
SFT_V8_SESSION="5c203420-1891-5999-8cef-23cbb0be8402"

# V11-s5 weights (best GRPO-only model)
V11_S5="tinker://bae6fabb-fcd7-58a6-b0af-101131f8e6a6:train:0/weights/state-0005"

eval_model() {
    local name="$1"
    local model_path="$2"
    local extra_args="$3"

    echo "===== Evaluating: $name ====="
    echo "Model path: $model_path"

    mkdir -p "results/comprehensive_${name}"

    cmd="uv run python eval/run_eval.py \
        --model $BASE_MODEL \
        --benchmark all \
        --n-tasks $N_TASKS \
        --seed-offset $SEED_OFFSET \
        --experiment-name comprehensive_${name}"

    if [ -n "$model_path" ]; then
        cmd="$cmd --model-path \"$model_path\""
    fi

    if [ -n "$extra_args" ]; then
        cmd="$cmd $extra_args"
    fi

    echo "Running: $cmd"
    eval "nohup $cmd > results/comprehensive_${name}/log.txt 2>&1 &"
    echo "PID: $!"
}

echo "Comprehensive Evaluation Suite"
echo "=============================="
echo "Seed offset: $SEED_OFFSET"
echo "N tasks: $N_TASKS"
echo ""

# 1. Base model (reference)
# eval_model "base" "" ""

# 2. Base + strategies
# eval_model "base_strategy" "" "--eval-strategy"

# 3. V13 checkpoints (from current training)
# eval_model "v13_s10" "tinker://$V13_SESSION:train:0/weights/checkpoint-0010" ""
# eval_model "v13_s14" "tinker://$V13_SESSION:train:0/weights/checkpoint-0014" ""

# 4. SFT V8 (after training completes)
# eval_model "sft_v8_ep1" "tinker://$SFT_V8_SESSION:train:0/weights/checkpoint-0148" ""
# eval_model "sft_v8_ep2" "tinker://$SFT_V8_SESSION:train:0/weights/checkpoint-0296" ""
# eval_model "sft_v8_final" "tinker://$SFT_V8_SESSION:train:0/weights/final" ""

# 5. SFT V8 + strategies
# eval_model "sft_v8_final_strategy" "tinker://$SFT_V8_SESSION:train:0/weights/final" "--eval-strategy"

echo "Edit this script to uncomment the evaluations you want to run."
echo "Each eval takes ~2-4 hours for all 14 benchmarks."
