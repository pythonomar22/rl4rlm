# Cross-Doc Strategy Conditioning Problem

**Date:** 2026-03-11

## The Problem

V9-s5 (trained without strategy conditioning) scores 51% on cross_doc_compare.
V9-s10 (trained WITH strategy conditioning, including cross_doc_separate) drops to 24.2%.
More training with the wrong strategy makes things WORSE.

## Root Cause

The `cross_doc_separate` strategy teaches:
1. Extract from Document A
2. Extract from Document B
3. Compare in Python

This works for overlap_entities (find common names) and timeline_conflict (find conflicting events).
But it FAILS for metric_comparison (compare performance metrics across docs) because:
- Metric comparison requires understanding what to compare BEFORE extracting
- The "separate extraction" pattern extracts generic data, then can't compare
- V9-s5's natural approach compared metrics directly in a single pass — more flexible

## Evidence

| Cross-Doc Subtype | V9-s5 | V9-s10 | Delta |
|-------------------|-------|--------|-------|
| metric_comparison | 100% (3/3) | 33% (1/3) | -67pp |
| overlap_entities | 55% | 26% | -29pp |
| timeline_conflict | 22% | 11% | -11pp |
| budget_diff | 27% | 27% | 0pp |

The catastrophic metric_comparison failure (-67pp) drives most of the regression.

## Possible Solutions

### 1. Remove cross_doc_separate from strategy pool
- Let the model discover its own approach via reward signal
- V9-s5 proved this works better (51% vs 24.2%)
- Risk: model might converge to single-pass pattern that worked in V4-s5 (28.6%)

### 2. Subtask-specific strategies
- Use `cross_doc_separate` only for overlap_entities
- Use a `cross_doc_metric_compare` strategy for metric tasks
- Use generic strategy for budget_diff and timeline_conflict
- Requires modifying benchmark to report subtask type at training time

### 3. Mixed strategy training for cross_doc
- 50% no strategy (let model discover)
- 25% cross_doc_separate (for entity overlap)
- 25% cross_doc_metric (for metric comparison)

### 4. Increase cross_doc weight without strategy
- Use V10's 18% weight for cross_doc
- But with standard/generic strategies only (not cross_doc_separate)
- This combines V9-s5's flexibility with more training signal

### 5. Hybrid approach for cross_doc
- Trained root writes comparison code
- Base model sub-calls do the extraction (more flexible)
- V10-hybrid might solve this naturally

## Recommendation

Option 4 is the most promising: train with high cross_doc weight but WITHOUT the explicit cross_doc_separate strategy. The model needs freedom to develop its own comparison approach rather than being forced into a rigid "extract-then-compare" pipeline.

For V11 if needed: modify TASK_STRATEGY_WEIGHTS to set cross_doc_compare strategies to standard/generic only.

## Updated: Cross-Doc Subtype Analysis (Clean H2H, seed-offset 10000)

| Subtype (3 tasks each) | Base | V4-s5 | V4-H | Delta V4-s5 |
|------------------------|------|-------|------|-------------|
| metric_comparison | **100%** | 33% | 67% | **-67pp** |
| overlap_entities | 19% | **35%** | 22% | +16pp |
| common_projects | **53%** | 27% | 27% | **-26pp** |
| timeline_conflict | 0% | **19%** | 0% | +19pp |
| **Total** | **43%** | **28.6%** | **28.7%** | **-14.4pp** |

Training IMPROVES entity-finding subtypes (+16pp overlap, +19pp timeline) but
DESTROYS comparison/matching subtypes (-67pp metric, -26pp projects).

The net result is -14.4pp because the regressions outweigh the gains.

V11 prediction: if removing cross_doc_separate preserves metric_comparison (base=100%), 
V11 could achieve ~55% on cross_doc (keeping improvements + preserving comparisons).
