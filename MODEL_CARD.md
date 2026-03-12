# RLM-Qwen3.5-35B-A3B

A natively recursive language model based on Qwen3.5-35B-A3B, trained with Strategy-Conditioned GRPO (SC-GRPO) to solve long-context tasks by writing Python code in a persistent REPL.

## Model Description

This model is a LoRA fine-tune (rank 32) of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) trained with reinforcement learning to generate code that decomposes long-context tasks into manageable sub-problems.

**Architecture:** Mixture-of-Experts, 35B total parameters, 3B active per token
**Training:** SC-GRPO (Strategy-Conditioned GRPO), 40 steps from V4-s5 checkpoint
**Training API:** [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook) (remote GPU training)

### How It Works

The model operates within an RLM (Recursive Language Model) scaffold:
1. The full input is stored as a `context` variable in a Python REPL — it never enters the model's context window
2. The model sees only metadata about the context (length, prefix, available functions)
3. The model writes Python code to chunk, search, and aggregate over the context
4. `llm_query(text)` recursively invokes the model on substrings
5. `FINAL(answer)` terminates with the answer

This enables processing of **millions of tokens** with a 32K context window.

## Training Details

### SC-GRPO (Strategy-Conditioned GRPO)

Standard GRPO for code generation suffers from mode collapse — the model converges to a single code template regardless of task type. We address this with **Strategy-Conditioned GRPO**: each training trajectory receives a randomly assigned strategy prompt (e.g., "use binary search", "process in two passes", "extract then compute in Python"). This forces structurally diverse code generation and eliminates mode collapse (0% degenerate outputs vs 60% with standard GRPO).

### Training Configuration

- **Base model:** Qwen/Qwen3.5-35B-A3B
- **LoRA rank:** 32 (train MLP + attention + unembed)
- **Learning rate:** 1e-6 (with cosine decay)
- **K trajectories:** 8 per task
- **Batch size:** 4 tasks per step
- **Gradient accumulation:** 4 micro-batches
- **KL coefficient:** 0.005
- **Steps:** 40 (from V4-s5 checkpoint, which was 5 GRPO steps from base)
- **Strategy conditioning:** Enabled (11 strategy types)
- **Credit assignment:** Per-turn (FINAL turns weighted 1.5x, error turns 0.3x)
- **Training time:** ~20 hours on Tinker API

### Task Distribution

Training used a regression-targeted task distribution (V10):
- Cross-Doc Compare (20%), Key-Value Retrieval (14%)
- DataFrame QA (14%), Event Counting (12%)
- NIAH (10%), Doc-Classify (10%)
- Hard Multi-Hop (10%), Code Debug (10%)

## Evaluation Results

Evaluated on 14 benchmarks spanning search, extraction, comparison, and counting. Results confirmed across 2 independent evaluation sets with different random seeds.

| Benchmark | Base | RLM-V10 | Delta |
|-----------|------|---------|-------|
| NIAH | 65.0% | **75.0%** | +10.0 |
| Multi-NIAH | **99.4%** | 90.0% | -9.4 |
| Hard NIAH | 83.3% | **93.3%** | +10.0 |
| Doc-Classify | 56.3% | **76.6%** | +20.3 |
| DataFrame QA | 75.0% | **85.0%** | +10.0 |
| Code Debug | **50.0%** | 43.3% | -6.7 |
| Multi-Hop QA | 55.0% | **60.0%** | +5.0 |
| Hard Multi-Hop | 30.0% | 30.0% | 0.0 |
| Notebook QA | 46.7% | **63.3%** | +16.6 |
| Event Counting | **46.4%** | 41.7% | -4.7 |
| Cross-Doc Compare | 42.2% | **42.9%** | +0.7 |
| Key-Value Retrieval | 29.2% | **75.0%** | +45.8 |
| Verbatim Copy | 20.0% | **60.0%** | +40.0 |
| OOLONG | 0.0% | **20.0%** | +20.0 |
| **Average** | **49.9%** | **61.1%** | **+11.3** |

Cross-evaluation on a second independent seed set confirms **+9.7pp** average improvement (6 benchmarks).

### Key Findings

1. **+11.3pp average improvement** across 14 benchmarks (9 wins, 3 losses, 2 ties)
2. **Massive gains on retrieval:** Key-Value +46pp, Verbatim Copy +40pp, Doc-Classify +20pp
3. **Minor regressions:** Multi-NIAH -9pp, Code Debug -7pp, Event Counting -5pp
4. **Per-benchmark strategy prompts** can further boost specific tasks at evaluation time

## Limitations

- **Requires RLM scaffold:** The model is designed to run within the RLM loop with REPL access — direct prompting will not produce RLM behavior
- **Temperature sensitivity:** Evaluated at temperature 0.7; results may vary at other temperatures
- **REPL security:** The model executes arbitrary Python code in a sandboxed REPL — not suitable for untrusted inputs without additional sandboxing
- **Small evaluation sets:** N=10-20 per benchmark; individual benchmark deltas may not be statistically significant

## Usage

```python
from scaffold.rlm import run_rlm
from scaffold.llm_query import TinkerModel

model = TinkerModel("Qwen/Qwen3.5-35B-A3B", model_path="path/to/checkpoint")
result = run_rlm(model, question="Find the secret code", context=long_document)
```

See the [repository](https://github.com/simonomar/rlm) for full setup instructions.

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

- Zhang, Kraska, Khattab (2026). [Recursive Language Models](https://arxiv.org/abs/2502.14155). arXiv:2502.14155
- Guo et al. (2025). [DeepSeek-R1](https://arxiv.org/abs/2501.12948). arXiv:2501.12948
