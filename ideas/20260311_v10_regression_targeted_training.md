# V10: Regression-Targeted Training

Date: 2026-03-11

## Motivation

Clean head-to-head evaluation shows V4-s5 (our best model) only beats base by +0.5pp average.
5 benchmarks improve, 5 regress, 4 tied. We need to fix the regressions without losing gains.

## Root Cause Analysis (from trajectory inspection)

### 1. Single-Pass Convergence (Cross-Doc Compare: -14.4pp)
V4-s5 processes both documents in a single combined query instead of separately.
Base model uses two-pass: extract from Doc A, extract from Doc B, compare in Python.
**Fix**: New `cross_doc_separate` strategy that enforces separate extraction.

### 2. Chunk Size Drift (Key-Value Retrieval: -6pp)
V4-s5 learned to use larger chunks (20K) for efficiency, misses entries at boundaries.
Base uses smaller chunks (15K) with more overlap, catches all entries.
**Fix**: New `lookup_thorough` strategy with 5-8K chunks and mandatory exhaustive search.

### 3. Format Precision Loss (Notebook QA: -10pp, DataFrame QA: -7pp)
V4-s5 drops format info: "87.0%" → "0.870", wrong decimal representations.
**Fix**: New `precision_extract` and `table_preserve` strategies.

### 4. Missing Training Task (Key-Value Retrieval: -6pp)
Key-value retrieval was NOT in V9 training mix at all! Model never trained on it.
**Fix**: Added to V10 training at 14% weight.

## V10 Design

### 5 New Strategies (added to STRATEGY_SUFFIXES)
1. `cross_doc_separate` - Process documents separately, then compare
2. `table_preserve` - Keep tabular structure intact, parse headers first
3. `precision_extract` - Preserve exact format (%, decimals, units)
4. `lookup_thorough` - Small chunks, exhaustive search, second pass if needed
5. `notebook_sequential` - Process cells in order, track variable state

### Task Distribution (V10 vs V9)
| Task | V9 | V10 | Reason |
|------|-----|-----|--------|
| cross_doc_compare | 10% | **18%** | Biggest regression (-14.4pp) |
| key_value_retrieval | **0%** | **14%** | Missing entirely! (-6pp) |
| notebook_qa | 10% | **14%** | -10pp regression |
| dataframe_qa | 10% | **12%** | -7pp regression |
| event_counting | 15% | **12%** | -6.8pp regression |
| hard_multi_hop | 15% | 8% | Already +10pp |
| multi_hop_qa | 15% | 6% | Already 0pp |
| code_debug | 10% | 6% | Already 0pp |
| doc_classify | 5% | 4% | Already +17.2pp |
| niah | 5% | 3% | Already +10pp |
| multi_niah | 5% | 3% | Already +4pp |

### Training Config
- **Start from**: V4-s5 weights (preserve search gains)
- **LR**: 1e-6 (lower than V9's 2e-6 to avoid forgetting)
- **Steps**: 40 (more steps since lower LR)
- **K**: 8 (same as V9)
- **Batch**: 4 (same as V9)
- **SC-GRPO**: ON (with new strategies)
- **NGRPO virtual reward**: ON
- **KL coeff**: 0.005 (same as V9)

### Key Hypothesis
Training heavily on regression tasks with task-specific strategy prompts will:
1. Teach the model correct patterns for structured extraction
2. Fix cross-document comparison via separate-then-compare strategy
3. Add key-value retrieval capability (currently untrained)
4. Maintain search gains since V4-s5 weights already encode them + 25% search task weight remains
