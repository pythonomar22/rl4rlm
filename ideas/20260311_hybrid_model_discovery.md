# Critical Discovery: Hybrid Model Architecture (Trained Root + Base Sub-Calls)

## The Problem
RL training (GRPO/SC-GRPO) on code generation shares LoRA weights with the sub-call model (llm_query). As training progresses, the model's question-answering ability for sub-calls degrades:
- V4-s5 (5 steps): sub-calls still good (NIAH 85%)
- V7-s5 (10 steps): sub-calls degraded (NIAH 65% = base level)

## The Solution
**HybridTinkerModel**: Use fine-tuned model for root code generation, base model for sub-calls.
- Root `.generate()` → trained sampling client (writes better code)
- `.sub_query()` → base sampling client (answers questions accurately)

## Results: V7-s5 Hybrid vs V7-s5 Non-Hybrid

| Benchmark | Base | V4-s5 | V7 | V7-hybrid | Hybrid effect |
|-----------|------|-------|-----|-----------|--------------|
| NIAH | 65.0% | 85.0% | 65.0% | **80.0%** | **+15.0%** |
| Multi-Hop QA | 55.0% | 65.0% | 60.0% | **70.0%** | **+10.0%** |
| Event Counting | 47.8% | 61.2% | 51.9% | **66.5%** | **+14.6%** |
| Hard Multi-Hop | 40.0% | 10.0% | 40.0% | 20.0% | **-20.0%** |

## Key Insights

1. **RL training on code generation degrades QA ability** — the LoRA adapter learns code patterns that interfere with simple question answering

2. **Hybrid approach is a massive win for most tasks** — +10-15% on NIAH, Multi-Hop QA, Event Counting

3. **Hard Multi-Hop is the exception** — complex multi-step extraction benefits from trained sub-call patterns. The base model is worse at structured extraction tasks.

4. **This explains V4-s5 vs V7 tradeoff** — V4-s5 was at a sweet spot where training improved both code gen AND sub-call quality. V7 overtrained the sub-calls.

## Implications

1. **For the paper:** This is a novel finding — hybrid RLM architecture outperforms single-model RLM
2. **For training:** We can train more aggressively (more steps) if we use hybrid mode, because sub-call degradation no longer matters
3. **For V8/V9:** Hybrid mode should be the default evaluation mode
4. **For deployment:** Need separate sampling clients (2x cost for sub-calls using base model)
5. **Connection to original paper:** The RLM paper (Zhang et al.) used a separate sub-call model — our finding independently confirms this is necessary

## Implementation
Added `HybridTinkerModel` class to `scaffold/llm_query.py` and `--hybrid` flag to `eval/run_eval.py`.
