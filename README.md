# Reinforcement Learning for Recursive Language Models

Training [Recursive Language Models](https://arxiv.org/abs/2502.14155) (RLMs) via reinforcement learning. We train Qwen3-1.7B to solve long-context tasks by writing Python code in a REPL, decomposing inputs programmatically via `llm_query()` sub-calls.

**Key result:** RL (DPO, GRPO) improves multi-needle retrieval from 58% to 88% over SFT — but only after fixing two subtle implementation bugs. Naive GRPO catastrophically regresses to 41%.

## Results

| Model | NIAH (100) | Multi-NIAH (24) | DocCls (20) | Avg |
|-------|-----------|----------------|------------|-----|
| Base (Qwen3-1.7B) | 72.0 | 38.3 | 80.3 | 63.5 |
| SFT | 90.0 | 57.9 | 82.4 | 76.8 |
| STaR (iterative SFT) | 87.0 | 58.4 | 83.4 | 76.3 |
| RL-v3 (buggy GRPO) | 90.0 | 41.4 | 83.9 | 71.8 |
| **DPO** | 83.0 | **87.9** | 82.6 | **84.5** |
| **GRPO-v4** | 82.0 | 85.1 | 83.2 | 83.4 |

Total compute: ~11 GPU-hours on a single H200.

## What is an RLM?

An RLM places the full input in a Python REPL as the variable `context`, then the model writes code to decompose, search, and aggregate over it. The model never sees the full input in its context window — it interacts with it programmatically:

```python
# Model generates code like this:
chunk_size = 20000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
results = []
for chunk in chunks:
    answer = llm_query(f"Find the secret code in:\n{chunk}")
    if "not found" not in answer.lower():
        results.append(answer)
FINAL_VAR("results")
```

## Why RL Failed (and What Fixed It)

Our initial GRPO implementations (v1-v3) all failed. We traced the failures to two bugs:

1. **Unconditional log-probabilities.** Policy loss was computed on raw code text without the conversation context (system prompt, task metadata, REPL feedback). This trained the model to produce/avoid code patterns unconditionally, causing degenerate repetitive outputs.

2. **Weight-space KL proxy.** L2 penalty `||θ - θ_ref||² / d` normalized by 17.4M LoRA parameters produced ~zero regularization. Token-level KL via a frozen reference model is needed.

After fixing both — conditioning log-probs on the full conversation and using token-level KL — both DPO and GRPO-v4 produce dramatic improvements with no degenerate outputs.

## Project Structure

```
scaffold/          # RLM infrastructure
  repl.py          # Python REPL environment
  rlm.py           # Main RLM loop
  llm_query.py     # Sub-call model wrapper
  prompts/         # System prompts

training/          # Training methods
  sft.py           # LoRA supervised fine-tuning
  rl.py            # GRPO v1-v3 (original, buggy)
  rl_v4.py         # GRPO v4 (fixed: conditioned log-probs + token-level KL)
  dpo.py           # Direct Preference Optimization
  logprobs.py      # Shared conditioned log-prob utilities
  rewards.py       # Reward functions

eval/              # Evaluation
  run_eval.py      # Evaluation harness
  benchmarks/      # NIAH, Multi-NIAH, Document Classification

scripts/           # Data collection & analysis
  collect_trajectories.py   # Self-bootstrap trajectory collection
  collect_star_v2.py        # STaR iterative collection
  filter_trajectories.py    # Trajectory cleaning
  fix_templates.py          # Fix FINAL/FINAL_VAR template errors
  analyze_results.py        # Results analysis

data/              # Checkpoints, trajectories, training data
  sft/             # All model checkpoints (LoRA adapters)
  trajectories/    # Raw collected trajectories
  filtered/        # Cleaned SFT training samples

results/           # Evaluation results (JSON)

ideas/             # Experiment records (hypothesis → results → conclusion)
```

## Setup

```bash
# Requires Python 3.11+, CUDA GPU
uv sync

# Or with pip
pip install -e .
```

Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (downloaded automatically).

## Training Pipeline

### 1. Trajectory Collection (self-bootstrap)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/collect_trajectories.py \
  --model Qwen/Qwen3-1.7B --n-tasks 150
```

### 2. Filter & Clean
```bash
uv run python scripts/filter_trajectories.py \
  --input data/trajectories/<run_dir> --output data/filtered/sft_samples.jsonl
```

### 3. SFT
```bash
CUDA_VISIBLE_DEVICES=0 uv run python training/sft.py \
  --model Qwen/Qwen3-1.7B --data data/filtered/sft_samples.jsonl \
  --output data/sft/lora_v2
```

### 4. STaR (iterative SFT on harder tasks)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/collect_star_v2.py \
  --model data/sft/lora_v2/final --base-model Qwen/Qwen3-1.7B

uv run python training/sft.py \
  --model Qwen/Qwen3-1.7B --data data/filtered/sft_samples_v3_combined.jsonl \
  --output data/sft/lora_v3
```

### 5a. DPO
```bash
CUDA_VISIBLE_DEVICES=0 uv run python training/dpo.py \
  --model data/sft/lora_v3/final --base-model Qwen/Qwen3-1.7B \
  --output data/sft/dpo_v1 --k 8 --beta 0.1 --lr 5e-6 --epochs 3
```

### 5b. GRPO-v4
```bash
CUDA_VISIBLE_DEVICES=0 uv run python training/rl_v4.py \
  --model data/sft/lora_v3/final --base-model Qwen/Qwen3-1.7B \
  --output data/sft/rl_v4 --k 8 --kl-coeff 0.05 --steps 30
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 uv run python eval/run_eval.py \
  --model data/sft/dpo_v1/final --base-model Qwen/Qwen3-1.7B \
  --benchmark all --experiment-name eval_dpo_v1
```

## Checkpoints

All checkpoints are LoRA adapters (67MB each) on top of `Qwen/Qwen3-1.7B`:

| Checkpoint | Description | Avg |
|-----------|-------------|-----|
| `data/sft/lora_v2/final` | SFT on 87 self-bootstrap trajectories | 76.8 |
| `data/sft/lora_v3/final` | STaR (iterative SFT, 132 trajectories) | 76.3 |
| `data/sft/rl_v3/final` | GRPO-v3 (buggy, degenerate outputs) | 71.8 |
| `data/sft/dpo_v1/final` | DPO (best model) | **84.5** |
| `data/sft/rl_v4/final` | GRPO-v4 (fixed) | 83.4 |

## Benchmarks

- **NIAH-100**: Single needle-in-a-haystack retrieval. 5 document lengths (5K-100K chars) x 5 needle positions x 4 seeds.
- **Multi-NIAH-24**: Retrieve K needles from one document. K in {3, 5, 8, 10}.
- **DocCls-20**: Classify N documents into 6 categories. N in {5, 10, 15, 20}.

## Citation

```bibtex
@misc{omar2026rlrlm,
  title={Reinforcement Learning for Training Natively Recursive Language Models on Long-Context Tasks},
  author={Simon Omar},
  year={2026},
  howpublished={CS234 Final Project, Stanford University}
}
```

## References

- [Recursive Language Models (Zhang, Kraska, Khattab 2026)](https://arxiv.org/abs/2502.14155)
- [GRPO / DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DPO (Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290)
- [STaR (Zelikman et al. 2022)](https://arxiv.org/abs/2203.14465)
- [LoRA (Hu et al. 2022)](https://arxiv.org/abs/2106.09685)
