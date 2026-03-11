# Head-to-Head Analysis: V4-s5 vs Base Model

## Results Summary (11/11 benchmarks complete, excluding DataFrame QA)

| Benchmark | Base | V4-s5 | Delta | Category |
|-----------|------|-------|-------|----------|
| NIAH (20) | 65.0% | 85.0% | +20.0% | **Strong Gain** |
| Multi-NIAH (20) | 83.9% | 99.4% | +15.5% | **Strong Gain** |
| Event Counting (20) | 47.8% | 61.2% | +13.4% | **Strong Gain** |
| Verbatim Copy (10) | 87.5% | 100.0% | +12.5% | **Strong Gain** |
| Multi-Hop QA (20) | 55.0% | 65.0% | +10.0% | **Moderate Gain** |
| Doc-Classify (20) | 88.4% | 96.8% | +8.4% | **Moderate Gain** |
| Code Debug (15) | 18.9% | 25.6% | +6.7% | **Moderate Gain** |
| Notebook QA (15) | 80.0% | 73.3% | -6.7% | Mild Regression |
| Hard NIAH (15) | 100.0% | 93.3% | -6.7% | Mild Regression |
| Hard Multi-Hop (10) | 40.0% | 10.0% | -30.0% | **Severe Regression** |
| OOLONG (10) | 20.0% | 0.0% | -20.0% | **Severe Regression** |
| **Average (11)** | **62.4%** | **64.5%** | **+2.1%** | |
| **Average (9 in-dist)** | **69.6%** | **77.7%** | **+8.1%** | |

## Key Insight: Bimodal Improvement

The V4-s5 model shows a clear bimodal pattern:
1. **Tasks in the training distribution** (NIAH, multi-NIAH, doc-classify, multi-hop, verbatim): +8-20% improvement
2. **Hard tasks outside training distribution** (hard multi-hop, OOLONG, hard NIAH): -7 to -30% regression

## Why Hard Multi-Hop Regresses (-30%)

V4-s5 gets 1/10 on hard multi-hop (vs 4/10 for base). Analysis of failures:
- Model finds the first hop but returns wrong intermediate result
- Example: Q="budget for project led by Rachel Robinson" → V4-s5 returns "Project Atlas, $150,000" but expected "$4.2 million"
- The model's chunking strategy works for finding entities but doesn't chain multiple lookups reliably
- RL training may have reinforced "find first match" patterns that fail on multi-step reasoning

## Why OOLONG Regresses (-20%)

V4-s5 scores 0/10 on OOLONG (vs 2/10 for base). OOLONG requires:
- Counting specific events in D&D transcripts (152K chars)
- Exact numerical aggregation across the full document
- Both models are bad (base only 20%), but V4-s5 is even worse
- RL training may have biased the model toward "return first found result" instead of "aggregate across all chunks"

## Implications for V7/V8

1. **SC-GRPO (V7)** should help by forcing diverse strategies — the model won't converge on a single chunking pattern
2. **Hard task training** is critical — V6 added hard_multi_hop to the training mix, V7 should show improvement
3. **Need counting/aggregation training** — both event_counting and OOLONG require map-reduce patterns
4. **Notebook QA scoring artifact** — 2 tasks scored 0.0 due to "0.87" vs "87.0%" mismatch, actual performance ~86%

## DataFrame QA Excluded

Both models crash on task 15 (750K chars, 85K tokens > 65K context) for baseline and task 10 (216K chars, 185K tokens) for V4-s5. The V4-s5 RLM scaffold adds overhead that makes the context even larger. Not directly comparable.

## Scoring Notes

- Notebook QA has 2 artificial 0.0 scores where model returned correct decimal (0.87) instead of "87.0%"
- Event counting uses partial scoring (close/approximate/partial_entity) which gives fractional scores
- Hard multi-hop is strict exact match on the final answer
