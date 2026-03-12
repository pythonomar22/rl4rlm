#!/bin/bash
# Wait for V10/V10h training to complete, then launch evaluations
# Usage: bash scripts/eval_when_done.sh

set -e
source /root/rlm/.env
export TINKER_API_KEY

V10_PID=1290657
V10H_PID=1294534

V10_SESSION="39e17e9d-7f3b-530b-be1b-d54a823d8224"
V10H_SESSION="5e74f8e5-d1d5-5529-9c8c-fb0e4e1530ad"

echo "Waiting for V10 (PID $V10_PID) and V10h (PID $V10H_PID) to complete..."

# Wait for V10
while kill -0 $V10_PID 2>/dev/null; do
    sleep 30
done
echo "V10 completed!"

# Launch V10 final eval (last checkpoint should be state-0040)
echo "Launching V10 s40 evaluation..."
mkdir -p /root/rlm/results/quick_v10_s40
nohup uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "tinker://${V10_SESSION}:train:0/weights/state-0040" \
    --benchmark niah multi_niah doc_classify dataframe_qa code_debug multi_hop_qa \
    --n-tasks 20 \
    --seed-offset 10000 \
    --experiment-name quick_v10_s40 \
    > /root/rlm/results/quick_v10_s40/log.txt 2>&1 &
echo "V10 s40 eval PID: $!"

# Also eval V10 s35 in case final checkpoint is worse
mkdir -p /root/rlm/results/quick_v10_s35
nohup uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "tinker://${V10_SESSION}:train:0/weights/state-0035" \
    --benchmark niah multi_niah doc_classify dataframe_qa \
    --n-tasks 20 \
    --seed-offset 10000 \
    --experiment-name quick_v10_s35 \
    > /root/rlm/results/quick_v10_s35/log.txt 2>&1 &
echo "V10 s35 eval PID: $!"

# Wait for V10h
while kill -0 $V10H_PID 2>/dev/null; do
    sleep 30
done
echo "V10h completed!"

# Launch V10h final eval
mkdir -p /root/rlm/results/quick_v10h_s40
nohup uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "tinker://${V10H_SESSION}:train:0/weights/state-0040" \
    --benchmark niah multi_niah doc_classify dataframe_qa code_debug multi_hop_qa \
    --n-tasks 20 \
    --seed-offset 10000 \
    --experiment-name quick_v10h_s40 \
    > /root/rlm/results/quick_v10h_s40/log.txt 2>&1 &
echo "V10h s40 eval PID: $!"

echo "All evaluations launched. Monitor with:"
echo "  tail -f results/quick_v10_s40/log.txt"
echo "  tail -f results/quick_v10h_s40/log.txt"
