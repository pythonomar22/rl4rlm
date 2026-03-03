#!/bin/bash
# Comprehensive evaluation: all models × all benchmarks
# Run after RL v2 training completes.
#
# Models: base, SFT v2, RL v1, RL v2
# Benchmarks: NIAH (100 tasks), Multi-NIAH (24 tasks), DocClassify (20 tasks)
#
# Usage: CUDA_VISIBLE_DEVICES=7 bash scripts/run_comprehensive_eval.sh

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set (6, 7, or 6,7)"
    exit 1
fi

echo "============================================"
echo "COMPREHENSIVE EVALUATION"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "============================================"

# --- BASE MODEL ---
echo ""
echo ">>> Evaluating: BASE (Qwen3-1.7B)"
uv run python eval/run_eval.py \
    --model Qwen/Qwen3-1.7B \
    --benchmark all \
    --n-tasks 100 \
    --max-iterations 8 \
    --experiment-name comprehensive_base \
    --seed-offset 20000 \
    2>&1 | tee results/comprehensive_base_stdout.log

# --- SFT v2 ---
echo ""
echo ">>> Evaluating: SFT v2"
uv run python eval/run_eval.py \
    --model Qwen/Qwen3-1.7B \
    --adapter data/sft/lora_v2/final \
    --benchmark all \
    --n-tasks 100 \
    --max-iterations 8 \
    --experiment-name comprehensive_sft_v2 \
    --seed-offset 20000 \
    2>&1 | tee results/comprehensive_sft_v2_stdout.log

# --- RL v1 ---
echo ""
echo ">>> Evaluating: RL v1 (trained on 5K-20K only)"
uv run python eval/run_eval.py \
    --model Qwen/Qwen3-1.7B \
    --adapter data/rl/grpo_v1/final \
    --benchmark all \
    --n-tasks 100 \
    --max-iterations 8 \
    --experiment-name comprehensive_rl_v1 \
    --seed-offset 20000 \
    2>&1 | tee results/comprehensive_rl_v1_stdout.log

# --- RL v2 ---
if [ -d "data/rl/grpo_v2/final" ]; then
    echo ""
    echo ">>> Evaluating: RL v2 (trained on all lengths)"
    uv run python eval/run_eval.py \
        --model Qwen/Qwen3-1.7B \
        --adapter data/rl/grpo_v2/final \
        --benchmark all \
        --n-tasks 100 \
        --max-iterations 8 \
        --experiment-name comprehensive_rl_v2 \
        --seed-offset 20000 \
        2>&1 | tee results/comprehensive_rl_v2_stdout.log
else
    echo "WARNING: data/rl/grpo_v2/final not found, skipping RL v2 eval"
fi

echo ""
echo "============================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Date: $(date)"
echo "============================================"
