#!/bin/bash
# Run all benchmarks on a checkpoint in parallel
# Usage: ./scripts/eval_checkpoint.sh <model-path> <experiment-name>
#
# Example:
#   ./scripts/eval_checkpoint.sh \
#     "tinker://de5a5059-ed71-5661-acad-de7fdae4f048:train:0/weights/state-0005" \
#     grpo_v3_step5

MODEL_PATH="$1"
EXPERIMENT_NAME="$2"

if [ -z "$MODEL_PATH" ] || [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <model-path> <experiment-name>"
    exit 1
fi

MODEL="Qwen/Qwen3.5-35B-A3B"

echo "Running all benchmarks for: $EXPERIMENT_NAME"
echo "Model path: $MODEL_PATH"
echo "---"

# Core benchmarks (fast, 10-20 tasks each)
for BENCH in niah multi_niah doc_classify multi_hop_qa notebook_qa hard_niah verbatim_copy; do
    echo "Launching: $BENCH"
    uv run python eval/run_eval.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --benchmark "$BENCH" \
        --n-tasks 10 \
        --experiment-name "${EXPERIMENT_NAME}_${BENCH}" \
        >> "results/${EXPERIMENT_NAME}_${BENCH}.log" 2>&1 &
done

# Heavier benchmarks (separate due to resource use)
for BENCH in dataframe_qa code_debug; do
    echo "Launching: $BENCH"
    uv run python eval/run_eval.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --benchmark "$BENCH" \
        --n-tasks 8 \
        --experiment-name "${EXPERIMENT_NAME}_${BENCH}" \
        >> "results/${EXPERIMENT_NAME}_${BENCH}.log" 2>&1 &
done

# New hard benchmark
echo "Launching: hard_multi_hop"
uv run python eval/run_eval.py \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --benchmark hard_multi_hop \
    --n-tasks 10 \
    --experiment-name "${EXPERIMENT_NAME}_hard_multihop" \
    >> "results/${EXPERIMENT_NAME}_hard_multihop.log" 2>&1 &

# OOLONG (external, slower)
echo "Launching: oolong"
uv run python eval/run_eval.py \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --benchmark oolong \
    --n-tasks 10 \
    --experiment-name "${EXPERIMENT_NAME}_oolong" \
    >> "results/${EXPERIMENT_NAME}_oolong.log" 2>&1 &

echo "All benchmarks launched! Monitor with:"
echo "  tail -f results/${EXPERIMENT_NAME}_*.log"
echo ""
echo "Check completion with:"
echo "  grep 'COMPLETE' results/${EXPERIMENT_NAME}_*.log"
