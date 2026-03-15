# RLM-Qwen3.5-35B: Training Natively Recursive Language Models

Training [Recursive Language Models](https://arxiv.org/abs/2502.14155) (RLMs) via reinforcement learning on the Tinker API.

**[Paper (PDF)](icmltemplate/finalpaper.pdf)** | **[Model on HuggingFace](https://huggingface.co/omar81939/rlm-qwen35-35b-a3b)**

## Results

+21.7pp average improvement across 14 benchmarks via RS-SFT on 3,644 self-mined trajectories. 13 wins, 1 loss vs base.

## Structure

```
scaffold/     # RLM runtime (repl.py, rlm.py, llm_query.py)
eval/          # Evaluation harness (14 benchmarks)
training/      # Training scripts (GRPO, RS-SFT)
scripts/       # Data pipeline & utilities
```
