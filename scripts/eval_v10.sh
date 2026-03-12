#!/bin/bash
# Comprehensive evaluation of V10 checkpoint
# Usage: bash scripts/eval_v10.sh <session_id> <state_name> <experiment_name>
# Example: bash scripts/eval_v10.sh 39e17e9d-7f3b-530b-be1b-d54a823d8224 state-0005 v10_s5

SESSION_ID="${1:?Usage: bash scripts/eval_v10.sh <session_id> <state_name> <experiment_name>}"
STATE="${2:?Provide state name, e.g. state-0005}"
EXPERIMENT="${3:?Provide experiment name, e.g. v10_s5}"
MODEL="Qwen/Qwen3.5-35B-A3B"
MODEL_PATH="tinker://${SESSION_ID}:train:0/weights/${STATE}"
N_TASKS="${4:-20}"

echo "=== V10 Comprehensive Evaluation ==="
echo "Session: $SESSION_ID"
echo "State: $STATE"
echo "Experiment: $EXPERIMENT"
echo "Model path: $MODEL_PATH"
echo "Tasks per benchmark: $N_TASKS"
echo ""

# Create output directory
mkdir -p "results/${EXPERIMENT}"

# Run all 14 benchmarks in parallel (4 at a time to avoid API overload)
BENCHMARKS=(
    niah multi_niah doc_classify dataframe_qa
    code_debug multi_hop_qa notebook_qa hard_niah
    verbatim_copy oolong hard_multi_hop event_counting
    cross_doc_compare key_value_retrieval
)

for bench in "${BENCHMARKS[@]}"; do
    echo "Launching: $bench"
    nohup uv run python eval/run_eval.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        --benchmark "$bench" \
        --n-tasks "$N_TASKS" \
        --eval-strategy \
        --experiment-name "$EXPERIMENT" \
        > "results/${EXPERIMENT}_${bench}.log" 2>&1 &
    echo "  PID: $!"

    # Stagger launches slightly to avoid thundering herd on API
    sleep 2
done

echo ""
echo "All benchmarks launched. Monitor with:"
echo "  tail -f results/${EXPERIMENT}_*.log"
echo ""
echo "Check completion with:"
echo "  grep -l 'ALL EVALUATIONS COMPLETE' results/${EXPERIMENT}_*.log | wc -l"
