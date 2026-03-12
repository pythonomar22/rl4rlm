#!/bin/bash
# Launch GRPO V15: Light refinement from SFT V9 checkpoint
#
# Two-phase training: SFT warmstart → light GRPO
# Uses V12 conservative distribution (only proven-improvement tasks)
#
# Usage: bash scripts/launch_grpo_v15.sh <sft_state_path>
# Example: bash scripts/launch_grpo_v15.sh tinker://SESSION:train:0/weights/state-0296

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/launch_grpo_v15.sh <sft_state_path>"
    echo "  sft_state_path: tinker:// path to SFT checkpoint state"
    exit 1
fi

SFT_PATH="$1"
BASE_MODEL="Qwen/Qwen3.5-35B-A3B"

echo "GRPO V15: Light refinement from SFT"
echo "====================================="
echo "SFT checkpoint: $SFT_PATH"
echo "Base model: $BASE_MODEL"
echo ""

mkdir -p /root/rlm/data/rl/grpo_35b_v15

# Conservative GRPO settings:
# - Low LR (5e-7) to avoid catastrophic forgetting
# - Only 10 steps
# - V12 conservative task distribution
# - Strategy conditioning ON
# - Save every step for granular checkpoint selection
nohup uv run python training/rl_tinker_v6.py \
    --model "$BASE_MODEL" \
    --model-path "$SFT_PATH" \
    --steps 10 \
    --K 8 \
    --batch-size 4 \
    --lr 5e-7 \
    --kl-coeff 0.005 \
    --task-type mixed_v12 \
    --save-every 1 \
    --warmup-steps 2 \
    --grad-accum-batch 4 \
    --strategy-conditioning \
    --ngrpo-virtual-reward \
    --credit-assignment \
    --experiment-name grpo_35b_v15 \
    > /root/rlm/data/rl/grpo_35b_v15/grpo_v15_stdout.log 2>&1 &

echo "V15 PID: $!"
echo "Log: /root/rlm/data/rl/grpo_35b_v15/grpo_v15_stdout.log"
echo ""
echo "Monitor: tail -f /root/rlm/data/rl/grpo_35b_v15/grpo_v15_stdout.log"
