# GRPO v2 Step 5 Evaluation Analysis

## Evaluation Results (v2 Step 5 vs Base Model)

| Benchmark | Base Model | v1 Step 10 | v2 Step 5 | Delta (v2 vs base) |
|-----------|------------|------------|-----------|---------------------|
| NIAH (20 tasks) | 81.0% | 75.0% | 80.0% | -1.0% |
| Multi-NIAH (12 tasks) | 97.8% | 100.0% | 90.0% | -7.8% |
| Doc-Classify (10 tasks) | 53.6% | 92.0% | 65.0% | +11.4% |
| Multi-Hop QA (10 tasks) | 50.0% | N/A | 50.0% | 0% |
| Code Debug (8 tasks) | 25.0% | N/A | 50.0%* | +25.0%* |
| DataFrame QA (10 tasks) | 80.0% | N/A | 48.0% | -32.0% |
| **Average** | **64.6%** | N/A | **63.8%** | **-0.8%** |

*Code Debug 50% is inflated — 5/8 tasks had `count_words` bug (easy), model found 4/5 of those but 0/3 other bugs. Benchmark diversity fixed.

## Key Findings

### 1. NIAH: Roughly Unchanged (80% vs 81%)
- By doc length: 5K=100%, 10K=80%, 20K=80%, 50K=60%
- Base model: 5K=100%, 10K~100%, 20K~80%, 50K~50%
- **50K improved** (60% vs ~50%), **10K regressed** (80% vs ~100%)
- Failure mode: all 4 failures returned "Not found" — chunking worked but sub-calls missed the needle

### 2. Multi-NIAH: Regressed (-7.8%)
- By doc length: 10K=100%, 20K=95%, 50K=75%
- Base model was nearly perfect at 97.8%
- Regressions at 50K: task 9 only found 1/5 needles (20% recall), task 11 found 4/5 (80%)
- The model is losing its ability to systematically scan all chunks at longer contexts

### 3. Doc-Classify: Improved (+11.4%)
- 65.0% vs 53.6% base
- By size: 5 docs=76%, 10 docs=54%
- Task 7 scored 0% due to formatting (returned Python list instead of newlines) — actual classification was 100% correct
- If formatting bug fixed, true accuracy would be ~72%

### 4. Multi-Hop QA: Unchanged (50%)
- 5/10 correct
- **Critical failure pattern**: All failures use single-pass chunking — the model asks each chunk for the final answer instead of decomposing the multi-hop chain
- Successful cases happen when the chain links are in the same chunk
- This is the #1 training target for v3 — model needs to learn multi-step decomposition

### 5. Code Debug: Unreliable (50%, inflated)
- Found `count_words` bug 4/5 times (easy: `counts.get(word, 1)` → should be `counts.get(word, 0) + 1`)
- Failed on ALL other bug types: `flatten_list` (0/2), `lru_cache` (0/1)
- Benchmark diversity issue FIXED: now uses round-robin bug assignment
- Need to re-evaluate with diverse bugs to get true accuracy

### 6. DataFrame QA: Regressed Significantly (-32%)
- ~48% (3.8/8 so far) vs 80% base
- Failures include wrong ticker ranking, incorrect aggregation sums
- Large DataFrames (54K chars) are harder — model's chunking may lose table structure

## Training Reward Trend (Concerning)

Surviving process rewards (Steps 6-8):
- Step 6: 0.741
- Step 7: 0.665
- Step 8: 0.480

Declining reward trend. Possible causes:
1. Model encountering harder task types in mixed_v2 (code_debug, dataframe_qa)
2. LR might still be too high for later steps
3. The model is losing capability on some tasks while improving on others

## NIAH Failure Analysis

All 4 NIAH failures returned "Not found":
- `niah_10008_10000_0.8`: Sub-query asked for "time/date" but answer was "Professor James Chen"
- `niah_10011_20000_0.2`: Similar narrow sub-query interpretation
- The model reinterprets the question too narrowly in sub-calls

## Decisions for v3

1. **Multi-Hop QA at 15% weight**: The key training target — model needs to learn multi-step decomposition
2. **Hard NIAH at 5%**: 100K+ docs for long-context generalization
3. **NIAH at 30%** (down from 40%): Base model already good, free up weight for multi-hop
4. **Code-debug and DFQA**: Keep in training mix but fix benchmarks for evaluation
5. **Resume from best checkpoint**: Wait for step 10 eval to determine
6. **Consider LR schedule**: Decay LR in later steps to prevent reward degradation
