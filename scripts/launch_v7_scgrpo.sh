#!/bin/bash
# Launch GRPO V7 with Strategy-Conditioned training (SC-GRPO).
#
# Novel contribution: each trajectory gets a randomly assigned strategy prompt,
# forcing structurally diverse code patterns that combat mode collapse.
#
# All V6 improvements are preserved (gradient accumulation, adaptive difficulty,
# anti-shortcut training, multi-turn persistence bonus, KL penalty).
#
# Usage: bash scripts/launch_v7_scgrpo.sh <model-path>
# Example: bash scripts/launch_v7_scgrpo.sh "tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005"

if [ -z "$1" ]; then
    echo "Usage: bash scripts/launch_v7_scgrpo.sh <model-path>"
    echo "  model-path: tinker:// path to base or checkpoint state"
    exit 1
fi
MODEL_PATH="$1"

echo "Launching GRPO V7 (SC-GRPO) training"
echo "  Model path: $MODEL_PATH"
echo "  Key innovation: Strategy-Conditioned GRPO"
echo "    - Each trajectory gets a random strategy prompt"
echo "    - 6 strategies: standard, extract_compute, binary_search, map_reduce, two_pass, small_chunks"
echo "    - Strategy selection weighted by task type"
echo "    - Wider temperature range [0.6-1.5] for maximum diversity"
echo "  Also includes all V6 improvements"
echo "---"

mkdir -p data/rl/grpo_35b_v7

uv run python training/rl_tinker_v6.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "$MODEL_PATH" \
    --steps 30 --K 8 --batch-size 4 --lr 2e-6 \
    --kl-coeff 0.01 \
    --task-type mixed_v6 --save-every 5 \
    --warmup-steps 2 --grad-accum-batch 4 \
    --strategy-conditioning \
    --experiment-name grpo_35b_v7 >> data/rl/grpo_35b_v7/grpo_v7_stdout.log 2>&1 &

echo "V7 training launched (PID: $!)"
echo "Monitor: tail -f data/rl/grpo_35b_v7/grpo_v7_stdout.log"
