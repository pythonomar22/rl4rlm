# V7 SC-GRPO Step 5 Analysis

## Three-Way Comparison: Base vs V4-s5 (GRPO) vs V7-s5 (SC-GRPO)

| Benchmark | Base | V4-s5 | V7-s5 | V7 vs V4 |
|-----------|------|-------|-------|----------|
| NIAH | 65.0% | **85.0%** | 65.0% | -20.0% |
| Multi-NIAH | 83.9% | 99.4% | 99.4% | 0.0% |
| Doc-Classify | 88.4% | 96.8% | **97.2%** | +0.4% |
| Multi-Hop QA | 55.0% | **65.0%** | 60.0% | -5.0% |
| Code Debug | 18.9% | 25.6% | **27.8%** | +2.2% |
| Notebook QA | 80.0% | 73.3% | 73.3% | 0.0% |
| Hard NIAH | 100.0% | 93.3% | 93.3% | 0.0% |
| Verbatim | 87.5% | 100.0% | 100.0% | 0.0% |
| Event Counting | 47.8% | **61.2%** | 51.9% | -9.3% |
| Hard Multi-Hop | 40.0% | 10.0% | **40.0%** | **+30.0%** |
| OOLONG | 20.0% | 0.0% | 0.0% | 0.0% |
| **Average** | **62.4%** | **64.5%** | **64.4%** | **-0.2%** |

## Key Insight: Specialization vs Generalization Tradeoff

V4-s5 (standard GRPO) and V7-s5 (SC-GRPO) achieve nearly identical overall performance (64.5% vs 64.4%), but distribute it differently:

### V4-s5 wins (specialization)
- NIAH: 85% vs 65% (-20%) — V4-s5 learned optimal single-pass chunking pattern
- Event Counting: 61.2% vs 51.9% (-9.3%) — V4-s5 learned entity extraction patterns
- Multi-Hop QA: 65% vs 60% (-5%) — V4-s5 has stronger single-strategy lookup

### V7-s5 wins (generalization)
- Hard Multi-Hop: 40% vs 10% (+30%) — SC-GRPO's strategy diversity enables multi-step reasoning
- Code Debug: 27.8% vs 25.6% (+2.2%) — different strategies help find bugs
- Doc-Classify: 97.2% vs 96.8% (+0.4%) — marginal improvement

### Why This Happens
Standard GRPO converges on a single "best" strategy for each task type. This is great for in-distribution tasks but catastrophic for novel tasks. SC-GRPO forces the model to maintain multiple strategies, which:
1. Prevents over-specialization (no single pattern dominates)
2. Preserves ability to do multi-step reasoning (Hard Multi-Hop)
3. But loses peak performance on tasks where one strategy is clearly optimal (NIAH)

## Training Dynamics

### Mode Collapse Rate
| Step | Standard GRPO (V6) | SC-GRPO (V7) |
|------|-------------------|--------------|
| 1 | ~50% | 50% |
| 2-3 | ~60% | **0%** |
| 4 | ~60% | 75% |
| 5 | N/A (killed) | **0%** |
| Overall | ~60% | **28%** |

SC-GRPO reduces mode collapse from 60% to 28%. But step 4 shows it can still collapse (3/4 groups). This motivates V8's NGRPO + asymmetric advantages.

## Implications for V8

1. **Need BOTH specialization AND generalization** — the best model would combine V4-s5's pattern learning with V7-s5's strategy diversity
2. **NGRPO virtual max-reward** would help the all-fail groups in step 4 (S4G3: mean=0.011)
3. **Asymmetric advantages** would amplify diversity further
4. **MaxRL** would ensure hard tasks with rare successes still get gradient signal
5. **Consider adaptive SC-GRPO:** stronger strategy conditioning early, relaxed later

## Model Paths
- V4-s5: `tinker://74615872-6b0b-50ba-bcbc-7c0b6a92abe3:train:0/weights/state-0005`
- V7-s5: `tinker://2e48210b-f605-566d-86c6-f8c3c0f8a95f:train:0/weights/state-0005`
