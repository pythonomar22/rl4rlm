#!/bin/bash
# Launch GRPO V5 training with all improvements:
# 1. Per-trajectory temperature scaling [0.8-1.5]
# 2. Intermediate decomposition reward for hard_multi_hop
# 3. mixed_v5 task type (25% hard_multi_hop, reduced easy tasks)
# 4. Decomposition example in system prompt
# 5. Improved timeout (90s base + 60s per 100K chars)
#
# Usage: bash scripts/launch_v5.sh <model-path>
# Example: bash scripts/launch_v5.sh "tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005"

MODEL_PATH="${1:-tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005}"

echo "Launching GRPO V5 training"
echo "  Model path: $MODEL_PATH"
echo "  Task type: mixed_v5"
echo "  Key improvements: temperature scaling, decomposition reward, hard task focus"
echo "---"

mkdir -p data/rl/grpo_35b_v5

uv run python training/rl_tinker.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "$MODEL_PATH" \
    --steps 30 --K 8 --batch-size 4 --lr 2e-6 \
    --task-type mixed_v5 --save-every 5 \
    --experiment-name grpo_35b_v5 >> data/rl/grpo_35b_v5/grpo_v5_stdout.log 2>&1 &

echo "V5 training launched (PID: $!)"
echo "Monitor: tail -f data/rl/grpo_35b_v5/grpo_v5_stdout.log"
