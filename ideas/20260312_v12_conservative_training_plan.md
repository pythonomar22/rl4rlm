# V12: Conservative Training for No-Regression RLM

**Date:** 2026-03-12

## Goal

Train a model that beats base on ALL 14 benchmarks, not just 5. The specialization-generalization tradeoff must be broken.

## Current Situation (Clean H2H, seed-offset 10000)

| Benchmark | Base | V4-s5 | Best of Any | Delta |
|-----------|------|-------|-------------|-------|
| NIAH | 60.0 | 70.0 | 80.0 (V11-s5) | +20.0 |
| Multi-NIAH | 91.5 | 95.5 | 99.4 (V4-H) | +7.9 |
| Doc-Classify | 81.6 | 98.8 | 99.4 (V9-s10) | +17.8 |
| DFQA | **54.0** | 47.0 | 75.0 (Base+strat) | -7.0 (trained) |
| Code Debug | **25.6** | 25.6 | 25.6 | 0.0 |
| Multi-Hop QA | **85.0** | 85.0 | 85.0 | 0.0 |
| Notebook QA | **70.0** | 60.0 | 70.8 (V4+strat) | -10.0 (trained) |
| Hard NIAH | **93.3** | 93.3 | 93.3 | 0.0 |
| Verbatim Copy | **100.0** | 100.0 | 100.0 | 0.0 |
| OOLONG | 0.0 | 10.0 | 10.0 (V4-s5) | +10.0 |
| Hard Multi-Hop | 40.0 | 50.0 | 50.0 (V4-s5) | +10.0 |
| Event Counting | **57.2** | 50.4 | 66.4 (V9-s10) | -6.8 (V4), +9.2 (V9) |
| Cross-Doc | **43.0** | 28.6 | 43.0 (Base) | -14.4 |
| Key-Value | **51.3** | 45.3 | 61.7 (V9+strat) | -6.0 (V4) |

**Problem**: 5 benchmarks consistently regress with ANY trained model. No single model beats base on everything.

## Root Causes of Regression

1. **DFQA**: RL teaches chunk+llm_query for CSV → breaks at scale (context overflow). Fix: reward penalty for llm_query on structured data.
2. **Cross-Doc**: cross_doc_separate strategy teaches rigid extract-then-compare. Fix: V11 already removed it.
3. **Notebook QA**: RL trains sequential rigid parsing instead of adaptive. Fix: lower training weight or exclude.
4. **Event Counting**: Some sub-types regress (ratio, count_value). Only V9-s10 with longer training improves.
5. **Key-Value**: V4-s5 regresses (-6pp) but V9-s10 improves (+4.8pp). More training on this task helps.

## V12 Design Options

### Option A: Ultra-Conservative (KL=0.02, 3 steps)

Start from BASE model (not V4-s5). Very high KL to stay close to base.
- Pros: Minimal damage to base capabilities
- Cons: May not learn enough
- Tasks: Only tasks where base is weak (NIAH, Doc-Classify, Event Counting, Hard Multi-Hop)
- Skip: DFQA, Cross-Doc, Notebook QA, Multi-NIAH, Verbatim Copy, Hard NIAH

### Option B: Selective Task Distribution

Start from BASE model. Normal KL (0.005) but carefully chosen tasks.
- Include: NIAH (25%), Doc-Classify (20%), Event Counting (20%), Hard Multi-Hop (15%), Key-Value (10%), OOLONG (10%)
- Exclude: DFQA, Cross-Doc, Notebook QA, Multi-NIAH, Multi-Hop QA, Code Debug, Verbatim Copy, Hard NIAH
- Rationale: Only train on tasks where we've seen improvement. Never touch tasks where training hurts.

### Option C: DFQA-Aware Training

Include DFQA but with reward shaping:
- For DFQA tasks with >10K char context: bonus +0.2 for code that does NOT use llm_query
- For DFQA tasks with <10K char context: normal reward
- This teaches the model to choose Python for structured data

### Option D: Progressive Training with Early Stopping

Start from BASE model. Train for 20 steps but eval after EVERY step.
- If any benchmark drops below base by >3pp, stop.
- Use the checkpoint with the best average across all 14 benchmarks.
- Most compute-intensive but most likely to find the sweet spot.

### Option E: Reward-Weighted KL

Per-task KL coefficients:
- NIAH/Doc-Classify: KL=0.002 (allow more drift for search tasks)
- DFQA/Cross-Doc/Notebook QA: KL=0.05 (very conservative for extraction tasks)
- This prevents the model from drifting on extraction tasks while allowing improvement on search tasks
- Requires modifying rl_tinker_v6.py

## Recommendation

**Option B** is the most practical:
1. Start from base model
2. Only train on 6 tasks where improvement is proven
3. Use SC-GRPO with generic strategies (no task-specific harmful patterns)
4. 5-10 steps, save every step
5. Eval checkpoint-0001 through 0010 on all 14 benchmarks
6. Select the checkpoint that beats base on the most benchmarks

If any trained checkpoint beats base on all 14 benchmarks → we have our model.
If not → fall back to best-of-all with oracle selection (current +7.3pp).

## Implementation

```python
# V12 task distribution
TASK_WEIGHTS_V12 = {
    'niah': 0.25,
    'doc_classify': 0.20,
    'event_counting': 0.20,
    'hard_multi_hop': 0.15,
    'key_value_retrieval': 0.10,
    'oolong': 0.10,
}

# Strategy weights: only generic strategies
TASK_STRATEGY_WEIGHTS_V12 = {
    # All tasks get standard + extract_compute + two_pass
    # No task-specific strategies that could teach bad patterns
}
```

## Dependencies

- Wait for V10-s5, V11-s5 clean h2h to complete (in progress)
- Especially V11-s5 cross_doc_compare result
- If V11-s5 cross_doc improves, add cross_doc to V12 with V11's strategy set
