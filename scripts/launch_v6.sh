#!/bin/bash
# Launch GRPO V6 training with all novel improvements:
# 1. Gradient accumulation (single optim_step per step)
# 2. Adaptive task difficulty (auto-increase when saturated)
# 3. Anti-shortcut training (minimum 50K context, subcall reward bonus)
# 4. Multi-turn persistence bonus
# 5. Code diversity bonus (anti-mode-collapse)
# 6. KL penalty via reward shaping
# 7. Linear warmup + cosine decay
# 8. Narrower temperature schedule [0.7-1.2]
# 9. Intermediate decomposition reward for hard_multi_hop
#
# Usage: bash scripts/launch_v6.sh <model-path>
# Example: bash scripts/launch_v6.sh "tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005"

MODEL_PATH="${1:-tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005}"

echo "Launching GRPO V6 training"
echo "  Model path: $MODEL_PATH"
echo "  Task type: mixed_v6 (anti-shortcut, min 50K contexts)"
echo "  Key innovations:"
echo "    - Gradient accumulation (clean full-batch gradients)"
echo "    - Adaptive task difficulty (auto-increase when saturated)"
echo "    - Anti-shortcut training (min 50K context forces genuine chunking)"
echo "    - Multi-turn persistence bonus"
echo "    - Code diversity bonus"
echo "    - KL penalty via reward shaping"
echo "---"

mkdir -p data/rl/grpo_35b_v6

uv run python training/rl_tinker_v6.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "$MODEL_PATH" \
    --steps 30 --K 8 --batch-size 4 --lr 2e-6 \
    --kl-coeff 0.01 \
    --task-type mixed_v6 --save-every 5 \
    --warmup-steps 2 --grad-accum-batch 4 \
    --experiment-name grpo_35b_v6 >> data/rl/grpo_35b_v6/grpo_v6_stdout.log 2>&1 &

echo "V6 training launched (PID: $!)"
echo "Monitor: tail -f data/rl/grpo_35b_v6/grpo_v6_stdout.log"
