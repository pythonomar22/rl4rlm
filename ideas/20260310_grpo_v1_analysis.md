# GRPO v1 Full Analysis
**Date:** 2026-03-10

## Results Summary

| Benchmark | Base | Step 10 | Step 20 | Delta (Step 20) |
|-----------|------|---------|---------|-----------------|
| NIAH-20 | 81.0% | 75.0% | 70.0% | -11.0% |
| Multi-NIAH-12 | 97.8% | 100.0% | 100.0% | +2.2% |
| Doc-Classify-10 | 53.6% | 92.0% | 97.0% | +43.4% |
| **Average** | **77.5%** | **89.0%** | **89.0%** | **+11.5%** |

## New Benchmark Baselines

| Benchmark | Base Model |
|-----------|-----------|
| DataFrame QA (10 tasks) | 80.0% |
| Code Debug (8 tasks) | 25.0% |

## Training Reward Trajectory

Steps 1-22 (ongoing):
- Steps 1-3: Reward 0.51 → 0.78 (rapid initial learning)
- Steps 4-10: Reward ~0.65-0.80 (stabilizing)
- Steps 11-14: Reward 0.80-0.89 (model becoming very consistent)
- Steps 15-22: Reward ~0.75-0.90 (plateau, some steps with 0 updates)

## Key Observations

### 1. Doc-Classify: Massive Success (+43.4%)
- Base model: classifies only 1 doc then stops
- GRPO model: consistently classifies ALL documents
- 9/10 tasks with 100% accuracy at step 20
- The RL signal was clear and the model learned it

### 2. NIAH: Concerning Regression (-11.0%)
- Failures concentrated at positions 0.5 and 0.8 in 20K-50K docs
- Model may be using doc-classify chunking patterns (larger chunks, less overlap)
- For NIAH, you need smaller chunks with more overlap to find needles at arbitrary positions
- The RL training may be overwriting NIAH-specific code patterns

### 3. Multi-NIAH: Perfect Maintenance (+2.2%)
- 100% recall at both step 10 and step 20
- Multi-NIAH is similar enough to NIAH that the patterns transfer
- But it's much easier (multiple needles = multiple chances to find them)

### 4. Code Debug: Base Model is Bad (25%)
- Model finds bugs in filler code (false positives) instead of real bugs
- Gets distracted by syntax issues in auto-generated code
- Only reliably finds count_words bug (obvious logic error)
- Great target for RL training

### 5. DataFrame QA: Base Model Already Good (80%)
- Base model can parse CSV, compute statistics, rank
- Failures on harder tasks (larger datasets, more complex aggregation)
- Need harder tasks (200K+ char datasets) to show RLM advantage

## Root Cause of NIAH Regression

The mixed training (1/3 each task) means doc-classify gets 1/3 of all gradient updates.
Doc-classify rewards a pattern of "process each document separately, classify, aggregate."
NIAH rewards "scan with overlap, find needle, return it."

These are subtly different strategies:
- Doc-classify: large chunks (whole documents), no overlap needed
- NIAH: small overlapping chunks, careful search

The model is drifting toward doc-classify patterns because that's where the largest reward signal was.

## Plan for GRPO v2

1. **Increase NIAH weight in task mix** — 40% NIAH, 20% multi-NIAH, 20% doc-classify, 20% new
2. **Lower LR** — current 5e-5 effective is too high, causing drift
3. **Add new tasks** — DataFrame QA and Code Debug for diversity
4. **Resume from best checkpoint** — step 10 had best NIAH/overall balance
5. **True importance_sampling** — both positive and negative advantages
