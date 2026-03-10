# GRPO v1 Step 10 Evaluation Results
**Date:** 2026-03-10
**Checkpoint:** tinker://fb9497f3-...:train:0/weights/state-0010

## Results Comparison

| Benchmark | Base Model | GRPO Step 10 | Delta |
|-----------|-----------|-------------|-------|
| NIAH-20 | 81.0% | 75.0% | -6.0% |
| Multi-NIAH-12 | 97.8% | 100.0% | +2.2% |
| Doc-Classify-10 | 53.6% | 92.0% | +38.4% |
| **Average** | **77.5%** | **89.0%** | **+11.5%** |

## Analysis

### Doc-Classify: Massive Improvement (+38.4%)
- The model learned to classify ALL documents, not just the first one
- 9/10 tasks had >80% accuracy
- The RL signal worked: trajectories classifying all docs got high reward

### Multi-NIAH: Maintained at Ceiling (+2.2%)
- 12/12 tasks perfect recall (100%)
- Base model was already at 97.8%, so this is ceiling performance
- No regression from RL training

### NIAH: Slight Regression (-6.0%)
- 15/20 = 75% (vs baseline 81%)
- Failures on positions 0.5 and 0.8 at 20K chars
- Small sample size (20 vs 100 for baseline) means high variance
- Need larger eval to confirm regression

## Key Observations

1. **GRPO from base model works!** No SFT warmup needed.
2. **Task mixing prevented catastrophic forgetting** (unlike SFT v1/v2)
3. **Doc-classify benefited most** — largest room for improvement
4. **Only 10 RL steps** — more training could help further

## NIAH Failure Analysis (5 failures)
- niah_10004_10000_0.5: Not found
- niah_10008_10000_0.5: Not found  (position 0.5 weakness!)
- niah_10011_20000_0.2: Not found
- niah_10012_20000_0.5: Not found  (position 0.5 again)
- niah_10016_50000_0.2: Not found
- Position 0.5 at 10K: 0/2 (worst)
- Position 0.2 at 20K+ : 1/2

## Next Steps

1. Continue GRPO training to step 20, 30 — reward may still improve
2. Run full NIAH eval (100 tasks) to get reliable comparison
3. Consider reducing LR if NIAH regression worsens
4. Evaluate at step 20 checkpoint for comparison
