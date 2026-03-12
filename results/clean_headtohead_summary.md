# Clean Head-to-Head Evaluation — Definitive Results

Date: 2026-03-11
Seed offset: 10000 (NIAH), defaults for others
All bug fixes applied (temperature, event_counting seeded RNG, config recording)

## Full Results Table

| Benchmark (N tasks) | Base | V4-s5 | V4-s5-hybrid | Best Delta |
|---------------------|------|-------|-------------|------------|
| NIAH (20) | 60.0% | 70.0% | **75.0%** | **+15.0** |
| Multi-NIAH (20) | 91.5% | 95.5% | **99.4%** | **+7.9** |
| Doc-Classify (20) | 81.6% | 98.8% | **99.2%** | **+17.6** |
| DataFrame QA (20) | **54.0%** | 47.0% | 20.0% | -34.0 |
| Code Debug (15) | **25.6%** | 25.6% | 22.2% | -3.4 |
| Multi-Hop QA (20) | 85.0% | 85.0% | 85.0% | 0.0 |
| Notebook QA (15) | **70.0%** | 60.0% | 63.3% | -6.7 |
| Hard NIAH (15) | 93.3% | 93.3% | 93.3% | 0.0 |
| Verbatim Copy (10) | 100.0% | 100.0% | 100.0% | 0.0 |
| OOLONG (10) | 0.0% | **10.0%** | 0.0% | **+10.0** |
| Hard Multi-Hop (10) | 40.0% | **50.0%** | 40.0% | **+10.0** |
| Event Counting (20) | **57.2%** | 50.4% | 55.6% | -1.6 |
| Cross-Doc Compare (12) | **43.0%** | 28.6% | 28.7% | -14.3 |
| Key-Value Retrieval (12) | **51.3%** | 45.3% | 52.1% | +0.8 |

## V10-s5 / V11-s5 Partial Results (running, seed-offset 10000)

| Benchmark | Base | V4-s5 | V4-H | V10-s5 | V11-s5 | Best |
|-----------|------|-------|------|--------|--------|------|
| NIAH | 60.0 | 70.0 | 75.0 | 75.0 | **80.0** | V11-s5 |
| Multi-NIAH | 91.5 | 95.5 | **99.4** | 90.4 | 87.8 | V4-H |
| Doc-Classify | 81.6 | 98.8 | 99.2 | 98.8 | **99.2** | V4-H/V11 |
| DFQA | **54.0** | 47.0 | 20.0 | 38.7%* | 41.7%* | Base |
| Code Debug | **25.6** | 25.6 | 22.2 | ... | ... | |
| Multi-Hop QA | 85.0 | 85.0 | 85.0 | ... | ... | |
| Notebook QA | **70.0** | 60.0 | 63.3 | ... | ... | |
| Hard NIAH | 93.3 | 93.3 | 93.3 | ... | ... | |
| Verbatim Copy | 100.0 | 100.0 | 100.0 | ... | ... | |
| OOLONG | 0.0 | **10.0** | 0.0 | ... | ... | |
| Hard Multi-Hop | 40.0 | **50.0** | 40.0 | ... | ... | |
| Event Counting | **57.2** | 50.4 | 55.6 | ... | ... | |
| Cross-Doc | **43.0** | 28.6 | 28.7 | ... | ... | |
| Key-Value | 51.3 | 45.3 | **52.1** | ... | ... | |

*partial — 15/20 (V10-s5) and 12/20 (V11-s5) DFQA tasks complete. Remaining are 50t tasks (expected ~0%).

DFQA by size (confirms scaling failure):
| Size | Base | V10-s5 | V11-s5 |
|------|------|--------|--------|
| 5t | ~40% | 76% | 60% |
| 15t | ~40% | 20% | 20% |
| 30t | ~80% | 20% | 50%* |
| 50t | ~56% | ... | ... |
*only 2/5 done

Critical to watch: V11-s5 cross_doc (validates strategy fix), V10-s5 hard_multi_hop (validates task weight)

### Training Reward Trends (last 5 steps avg)
| Run | Steps Done | Avg Reward | Trend |
|-----|-----------|------------|-------|
| V9 | 20/30 | 0.481 | volatile, declining from s10 peak |
| V10 | 12/40 | 0.508 | flat |
| V10h | 12/40 | 0.510 | flat |
| V11 | 9/20 | 0.620 | **increasing** (best of all) |

## Averages

| Config | Average (14 benchmarks) |
|--------|------------------------|
| Base | 60.9% |
| V4-s5 | 61.4% (+0.5pp) |
| V4-s5-hybrid | 59.6% (-1.3pp) |
| Best-of-trained | 63.2% (+2.3pp) |

## Win/Loss/Tie

| Config | Improved | Regressed | Tied |
|--------|----------|-----------|------|
| V4-s5 vs Base | 5 | 5 | 4 |
| V4-s5-hybrid vs Base | 4 | 5 | 5 |

## Key Findings

### Strong improvements (trained model excels)
1. **Doc-Classify**: +17.6pp (81.6% → 99.2%) — trained root generates better classification code
2. **NIAH**: +15.0pp (60.0% → 75.0%) — hybrid mode finds needles more reliably
3. **Hard Multi-Hop**: +10.0pp (40.0% → 50.0%) — V4-s5 non-hybrid decomposes better
4. **OOLONG**: +10.0pp (0.0% → 10.0%) — V4-s5 marginally better (still very low)
5. **Multi-NIAH**: +7.9pp (91.5% → 99.4%) — near-perfect with hybrid

### Clear regressions (base model is better)
1. **DataFrame QA**: -34.0pp (54.0% → 20.0%) for hybrid — trained root generates wrong CSV parsing code
2. **Cross-Doc Compare**: -14.3pp (43.0% → 28.7%) — trained model over-chunks cross-doc tasks
3. **Notebook QA**: -6.7pp (70.0% → 63.3%) — trained model worse at structured Jupyter parsing
4. **Event Counting**: -1.6pp (57.2% → 55.6%) — slight regression, counting strategy issues

### No change
- Multi-Hop QA: 85.0% (all same), Hard NIAH: 93.3% (all same), Verbatim Copy: 100.0% (all same)
- Code Debug: 25.6% (V4-s5 same as base), Key-Value Retrieval: ~52% (hybrid = base)

## Root Cause Analysis

The RL training creates a **specialization vs generalization tradeoff**:

1. **Training improves search-type tasks**: NIAH, Multi-NIAH, Doc-Classify are fundamentally
   "find X in context" — the model learns better chunking and search strategies for these.

2. **Training hurts structured extraction**: DataFrame QA requires parsing CSV tables carefully.
   The trained model applies aggressive chunking patterns learned from search tasks, which
   breaks structured data parsing.

3. **Hybrid mode amplifies both effects**: Better search (base sub-calls are more reliable for
   factual extraction) but worse for DataFrame QA (trained root generates wrong parsing code
   AND base sub-calls can't compensate for bad chunking).

4. **Cross-Doc Compare regression**: Requires comparing information across TWO documents.
   The trained model's single-pass chunking doesn't naturally handle cross-document comparison.

## Previous vs Current OOLONG
Previously reported: Base 20%, V4-s5 0% (OOLONG appeared as a training regression).
Current: Base 0%, V4-s5 10% (OOLONG is actually IMPROVED by training).
Root cause of discrepancy: Different seeds/task instances. OOLONG with 10 tasks has high variance.

## Previous vs Current Hard Multi-Hop
Previously reported: Base 40%, V4-s5 10% (severe regression).
Current: Base 40%, V4-s5 50% (IMPROVEMENT!).
Root cause of discrepancy: Different seeds. Hard Multi-Hop with 10 tasks has high variance.
Also: previous eval may have had inconsistent seeds (Bug #3 and #4).

## Implications for Training

To improve overall average beyond +2.3pp, need:
1. Add DataFrame QA, Cross-Doc Compare to training with HIGHER weight
2. Use task-specific strategy prompts that DON'T chunk CSVs aggressively
3. Consider separate LoRA adapters for search vs extraction tasks
4. Train on larger evaluation sets (20+ tasks) to reduce variance
