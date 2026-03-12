# RLM-Qwen3.5-35B-A3B

A natively recursive language model based on Qwen3.5-35B-A3B, trained with Strategy-Conditioned GRPO (SC-GRPO) to solve long-context tasks by writing Python code in a persistent REPL.

## Model Description

This model is a LoRA fine-tune (rank 32) of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) trained with reinforcement learning to generate code that decomposes long-context tasks into manageable sub-problems.

**Architecture:** Mixture-of-Experts, 35B total parameters, 3B active per token
**Training:** GRPO with strategy conditioning (SC-GRPO), 11 steps from base model
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
- **Steps:** 11 (from V9-s10 checkpoint which was 10 steps from base)
- **Strategy conditioning:** Enabled (6 strategy types)
- **Credit assignment:** Per-turn (FINAL turns weighted 1.5x, error turns 0.3x)

### Task Distribution

Training used a mixed task distribution:
- NIAH (20%), Multi-NIAH (2%), Doc-Classify (10%)
- Cross-Doc Compare (20%), Event Counting (15%)
- Hard Multi-Hop (15%), Key-Value Retrieval (14%)
- Code Debug (4%)

## Evaluation Results

Evaluated on 14 benchmarks spanning search, extraction, comparison, and counting:

| Benchmark | Base | RLM-V11 | Delta |
|-----------|------|---------|-------|
| NIAH | 60.0% | **80.0%** | +20.0 |
| Multi-NIAH | 91.5% | 87.8% | -3.7 |
| Doc-Classify | 81.6% | **99.2%** | +17.6 |
| DataFrame QA | **54.0%** | 40.0% | -14.0 |
| Code Debug | 25.6% | 25.6% | 0.0 |
| Multi-Hop QA | **85.0%** | 80.0% | -5.0 |
| Notebook QA | 70.0% | 66.7% | -3.3 |
| Hard NIAH | 93.3% | **100.0%** | +6.7 |
| Verbatim Copy | 100.0% | 100.0% | 0.0 |
| OOLONG | 0.0% | **20.0%** | +20.0 |
| Hard Multi-Hop | 40.0% | **50.0%** | +10.0 |
| Event Counting | 57.2% | **72.9%** | +15.7 |
| Cross-Doc Compare | **43.0%** | 24.4% | -18.6 |
| Key-Value Retrieval | **51.3%** | 36.1% | -15.2 |
| **Average** | **60.9%** | **63.0%** | **+2.1** |

### Key Findings

1. **Strong on search tasks:** NIAH +20pp, OOLONG +20pp, Doc-Classify +17.6pp, Event Counting +15.7pp
2. **Regressions on extraction:** Cross-Doc -18.6pp, KV-Retrieval -15.2pp, DataFrame QA -14pp
3. **Root cause:** RL trains format-rigid parsing in sub-call code that breaks on diverse output formats
4. **Strategy amplification:** Training enables the model to follow strategy prompts effectively, providing additional gains on specific benchmarks

## Limitations

- **Specialization-generalization tradeoff:** Improvements on search tasks come at the cost of extraction tasks
- **Requires RLM scaffold:** The model is designed to run within the RLM loop with REPL access — direct prompting will not produce RLM behavior
- **Temperature sensitivity:** Evaluated at temperature 0.7; results may vary at other temperatures
- **REPL security:** The model executes arbitrary Python code in a sandboxed REPL — not suitable for untrusted inputs without additional sandboxing

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
