# Failure Mode Analysis — What RL Does and Doesn't Fix
**Date:** 2026-03-10

## What RL Fixes (RLM-Specific Skills)

### 1. Systematic Scanning (+41.4% Doc-Classify)
- **Before RL:** Model classifies 1 document then stops → 53.6%
- **After RL:** Model iterates through ALL documents → 95%
- **Mechanism:** RL reward signal is clear — full classifications get high reward
- **Code pattern:** `while i < len(context): chunk = ...; classify(chunk); i += chunk_size`

### 2. Improved Search Patterns (+4% NIAH, +10% Hard NIAH)
- RL teaches better chunk sizes and overlap strategies
- Model learns to handle boundary positions (0.01, 0.98)
- At 1M chars, RL-trained model is perfect vs 90% base

### 3. Better Text Reproduction (+10% Verbatim Copy)
- Base model: 90% (occasionally can't find target paragraph)
- RL model: 100% (never misses)
- RL improves the model's ability to systematically search and extract

### 4. Multi-Hop Chain Reasoning (+20% Multi-Hop QA)
- Base model: 50% (fails on documents ≥50K chars)
- RL model: 70% (improved on longer documents)
- BUT: still using compound queries, not true decomposition

## What RL Doesn't Fix (Fundamental Limitations)

### 1. Counting/Aggregation (OOLONG: 30% → ~10%)
- OOLONG requires precise counting across 150K+ char D&D transcripts
- "How many rolls of value 13?" requires iterating through EVERY mention
- The model's chunking approach loses count accuracy
- Predicted 3 when answer was 4, predicted 6 when answer was 3
- **RL actually HURTS:** Text-search RL makes the model search-then-answer, but counting needs exhaustive scan

### 2. Numerical Reasoning (DataFrame QA: 80% → 50%)
- DataFrame QA requires analytical operations on CSV data
- Chunking disrupts table structure
- Rankings and aggregations on 50K+ datasets fail
- **Same issue:** Text-focused RL training optimizes for text search, not numerical analysis

### 3. True Multi-Step Decomposition (Hard Multi-Hop: ~20%)
- Tasks requiring sequential reasoning (find A → use A to find B → use B to find C)
- Model still does single-pass compound queries
- Even with 100K+ docs and distractor chains, model picks up distractor answers
- **Why:** RL reward is only on final answer — no intermediate credit for decomposition steps

### 4. Code Analysis (Code-Debug: 25% → 25%)
- Bug finding in large codebases
- Requires understanding code semantics, not just text patterns
- RL on text tasks doesn't transfer to code analysis
- **Needs:** Code-specific training signal with intermediate rewards

## Key Insight for Paper

The pattern is clear: **RL improves tasks that benefit from better text search patterns** (scan all, find needle, reproduce text) but **doesn't help tasks requiring fundamentally different reasoning** (counting, numerical analysis, sequential decomposition, code understanding).

This suggests a taxonomy of RLM skills:
1. **Scan skills** (O(N) iteration) — RL helps a lot ✓
2. **Search skills** (O(1) needle finding) — RL helps somewhat ✓
3. **Count skills** (O(N) exhaustive + count) — RL doesn't help ✗
4. **Decomposition skills** (multi-step chaining) — RL doesn't help yet ✗
5. **Analysis skills** (numerical/code reasoning) — RL hurts ✗

## Implications for v4 Training

To teach decomposition:
1. **Intermediate rewards** — give partial credit for finding bridge entities
2. **Curriculum learning** — start with 2-hop easy, progress to 4-hop hard
3. **Teacher distillation** — show the model what decomposition LOOKS like
4. **Hard multi-hop tasks** — 100K-200K docs with distractor chains (created!)

To prevent regression on counting/numerical:
1. **Higher weight on DFQA** in task mix
2. **Add OOLONG-style counting tasks** to training
3. **Separate reward signals** for different skill types
