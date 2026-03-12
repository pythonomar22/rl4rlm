#!/bin/bash
# Quick regression eval — runs 5 regression benchmarks only
# Usage: bash scripts/eval_regression.sh <session_id> <state_name> <experiment_name> [--eval-strategy]

SESSION_ID="${1:?Usage: bash scripts/eval_regression.sh <session_id> <state_name> <experiment_name>}"
STATE="${2:?Provide state name, e.g. state-0010}"
EXPERIMENT="${3:?Provide experiment name, e.g. v9_s10_regression}"
STRATEGY_FLAG="${4:-}"  # Optional: --eval-strategy
MODEL="Qwen/Qwen3.5-35B-A3B"
MODEL_PATH="tinker://${SESSION_ID}:train:0/weights/${STATE}"

echo "=== Regression Eval ==="
echo "Model path: $MODEL_PATH"
echo "Experiment: $EXPERIMENT"
echo "Strategy: ${STRATEGY_FLAG:-none}"

BENCHMARKS=(cross_doc_compare event_counting dataframe_qa notebook_qa key_value_retrieval)

for bench in "${BENCHMARKS[@]}"; do
    echo "Launching: $bench"
    nohup uv run python eval/run_eval.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --benchmark "$bench" \
        --n-tasks 12 \
        $STRATEGY_FLAG \
        --experiment-name "$EXPERIMENT" \
        > "results/${EXPERIMENT}_${bench}.log" 2>&1 &
    echo "  PID: $!"
    sleep 1
done

echo ""
echo "Monitor: tail -f results/${EXPERIMENT}_*.log"
echo "Check: grep 'Accuracy' results/${EXPERIMENT}_*.log"
