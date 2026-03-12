# RLM-Qwen3.5-35B: A Natively Recursive Language Model

Training [Recursive Language Models](https://arxiv.org/abs/2502.14155) (RLMs) via reinforcement learning. We train **Qwen3.5-35B-A3B** (MoE, 35B total / 3B active) to solve long-context tasks by writing Python code in a persistent REPL, recursively decomposing inputs via `llm_query()` sub-calls.

## Key Results

| Benchmark (N) | Base | RLM-V11 | Delta |
|----------------|------|---------|-------|
| NIAH (20) | 60.0% | **80.0%** | +20.0 |
| Multi-NIAH (20) | 91.5% | 87.8% | -3.7 |
| Doc-Classify (20) | 81.6% | **99.2%** | +17.6 |
| DataFrame QA (20) | **54.0%** | 40.0% | -14.0 |
| Code Debug (15) | 25.6% | 25.6% | 0.0 |
| Multi-Hop QA (20) | 85.0% | 80.0% | -5.0 |
| Notebook QA (15) | 70.0% | 66.7% | -3.3 |
| Hard NIAH (15) | 93.3% | **100.0%** | +6.7 |
| Verbatim Copy (10) | 100.0% | 100.0% | 0.0 |
| OOLONG (10) | 0.0% | **20.0%** | +20.0 |
| Hard Multi-Hop (10) | 40.0% | **50.0%** | +10.0 |
| Event Counting (20) | 57.2% | **72.9%** | +15.7 |
| Cross-Doc Compare (12) | 43.0% | 24.4% | -18.6 |
| Key-Value Retrieval (12) | 51.3% | 36.1% | -15.2 |
| **Average (14)** | **60.9%** | **63.0%** | **+2.1** |

**RLM-V11**: GRPO RL training (11 steps from base, LoRA rank 32).
Trained on Tinker API. Evaluated on 14 diverse benchmarks spanning search, extraction, comparison, and counting.

### Key Findings

1. **+20pp on needle-in-haystack search** (NIAH 60% -> 80%, OOLONG 0% -> 20%) — RL teaches better chunking and extraction strategies.

2. **Specialization-generalization tradeoff** — RL improves search/classification tasks but regresses structured extraction (DataFrame QA -14pp, Cross-Doc -19pp). Training teaches format-rigid parsing that breaks on diverse output formats.

3. **Strategy-Conditioned GRPO (SC-GRPO)** — novel training method that assigns random strategy prompts per trajectory, eliminating mode collapse (0% degenerate outputs vs 60% with standard GRPO).

4. **Strategy amplification** — RL training's primary value is making models responsive to prompt-level strategy guidance. Strategy prompts alone provide +5.5pp average, while training alone provides +0.9pp — but training enables strategy-specific gains impossible without it.

## What is an RLM?

The full input **never enters the LLM's context window**. Instead:
1. Input stored as `context` variable in a Python REPL
2. LLM sees only metadata (length, prefix, available functions)
3. LLM writes code: chunk context, call `llm_query()` on chunks, aggregate
4. `llm_query(text)` invokes a fresh LLM call on a substring — the recursive call
5. LLM calls `FINAL(answer)` when done

This lets a model with a 32K context window process **millions of tokens** with no degradation.

```python
# Model generates code like this:
chunk_size = 15000
overlap = 2000
results = []
for i in range(0, len(context), chunk_size - overlap):
    chunk = context[i:i + chunk_size]
    answer = llm_query(f"Find the secret code in this text:\n{chunk}")
    if "not found" not in answer.lower():
        results.append(answer)
FINAL(results[0] if results else "Not found")
```

## Setup

```bash
# Requires Python 3.11+
uv sync

# Set Tinker API key
cp .env.example .env
# Edit .env with your Tinker API key
```

## Project Structure

```
scaffold/              # RLM runtime (model-agnostic)
  repl.py              # Persistent Python REPL with sandboxing
  rlm.py               # Main RLM loop (Algorithm 1 from the paper)
  llm_query.py         # Model wrappers (Tinker API)
  prompts/             # System prompts per model

eval/                  # Evaluation harness
  run_eval.py          # Eval runner (14 benchmarks)
  benchmarks/          # Benchmark generators (pure Python)

training/              # Training scripts (Tinker API)
  rl_tinker_v6.py      # GRPO / SC-GRPO training
  sft_tinker.py        # Supervised fine-tuning
  dpo_tinker.py        # Direct Preference Optimization
  rewards.py           # Reward functions

scripts/               # Data pipeline & utilities
ideas/                 # Experiment notes & research documents
```

## Training Pipeline (Tinker API)

### 1. Evaluate Base Model
```bash
uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --benchmark all --n-tasks 20 \
    --experiment-name baseline
```

### 2. GRPO Training (SC-GRPO)
```bash
uv run python training/rl_tinker_v6.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --steps 10 --K 8 --batch-size 4 --lr 1e-6 \
    --strategy-conditioning \
    --experiment-name grpo_v11
```

### 3. Evaluate Trained Model
```bash
uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "tinker://SESSION:train:0/weights/state-0005" \
    --benchmark all --n-tasks 20 \
    --experiment-name eval_v11_s5
```

## Benchmarks (14 total)

| Benchmark | Type | Complexity | Description |
|-----------|------|------------|-------------|
| NIAH | Search | O(1) | Single needle in 5K-100K chars |
| Multi-NIAH | Search | O(K) | K=3-10 needles in 10K-100K chars |
| Hard NIAH | Search | O(1) | Adversarial distractors |
| Doc-Classify | Classification | O(N) | N=5-20 articles into 6 categories |
| DataFrame QA | Extraction | O(N) | Tabular data analysis on CSV |
| Code Debug | Extraction | O(N) | Bug finding in codebases |
| Multi-Hop QA | Reasoning | O(K) | 2-3 hop cross-reference chains |
| Hard Multi-Hop | Reasoning | O(K) | 3-5 hop chains across 10+ entities |
| Notebook QA | Extraction | O(N) | Jupyter notebook analysis |
| Event Counting | Counting | O(N) | Count/aggregate events in logs |
| Cross-Doc Compare | Comparison | O(N^2) | Cross-document entity comparison |
| Key-Value Retrieval | Search | O(1) | Exact key lookup in large maps |
| Verbatim Copy | Reproduction | O(N) | Faithfully reproduce context segments |
| OOLONG | External | O(N) | Real D&D transcript aggregation |

## Citation

```bibtex
@misc{omar2026rlm35b,
  title={Training Natively Recursive Language Models: Strategy-Conditioned GRPO for Long-Context Code Generation},
  author={Simon Omar},
  year={2026},
  howpublished={CS234 Final Project, Stanford University}
}
```

## References

- [Recursive Language Models (Zhang, Kraska, Khattab 2026)](https://arxiv.org/abs/2502.14155)
- [GRPO / DeepSeek-R1 (Guo et al. 2025)](https://arxiv.org/abs/2501.12948)
- [Tinker Training API](https://github.com/thinking-machines-lab/tinker-cookbook)
- [OOLONG Benchmark (Beltagy et al. 2025)](https://arxiv.org/abs/2511.02817)

## License

MIT
