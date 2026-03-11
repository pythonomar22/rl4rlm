#!/bin/bash
# Run remaining V4-s5 benchmarks sequentially
# Benchmarks 3-11 (niah and multi_niah already done)

set -e
cd /root/rlm

MODEL="Qwen/Qwen3.5-35B-A3B"
MODEL_PATH="tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005"
N_TASKS=20
EXP_NAME="v4s5_headtohead"
LOGDIR="/root/rlm/results/v4s5_headtohead"

for BENCH in doc_classify multi_hop_qa code_debug dataframe_qa notebook_qa hard_niah verbatim_copy event_counting hard_multi_hop; do
    echo "=========================================="
    echo "Starting benchmark: $BENCH at $(date)"
    echo "=========================================="
    uv run python eval/run_eval.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --benchmark "$BENCH" \
        --n-tasks $N_TASKS \
        --experiment-name "$EXP_NAME" 2>&1 | tee "$LOGDIR/log_${BENCH}.txt"
    echo "Finished $BENCH at $(date)"
    echo ""
done

echo "ALL BENCHMARKS COMPLETE at $(date)"
