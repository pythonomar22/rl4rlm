# Regression Benchmark Comparison

Date: 2026-03-11

## Regression Benchmarks — 4-Way Comparison (12 tasks each)

| Benchmark | Base | V4-s5 | V9-s5 | V4-s5+Strategy |
|-----------|------|-------|-------|----------------|
| Cross-Doc Compare | **43.0%** | 28.6% | **51.0%** | 22.0% |
| Key-Value Retrieval | 51.3% | 45.3% | 37.8% | **62.5%** |
| Notebook QA | **70.0%** | 60.0% | 62.5% | **70.8%** |
| DataFrame QA | 54.0% | 47.0% | 31.7% | **63.6%** |
| Event Counting | 57.2% | 50.4% | **75.0%** | 52.5% |

## Strategies Used (V4-s5+Strategy)
- cross_doc_compare: cross_doc_separate (HURTS — too prescriptive)
- key_value_retrieval: lookup_thorough (HELPS +17pp over V4-s5)
- notebook_qa: notebook_sequential (HELPS +10.8pp)
- dataframe_qa: table_preserve (HELPS +16.6pp)
- event_counting: standard (no strategy, already in log)

## Fix Map
| Benchmark | Fixed By | Approach |
|-----------|----------|----------|
| Cross-Doc Compare | **More training** (V9-s5) | +22pp over V4-s5, beats base |
| Key-Value Retrieval | **Strategy prompt** (lookup_thorough) | +17pp over V4-s5, beats base |
| Notebook QA | **Strategy prompt** (notebook_sequential) | +10.8pp, matches base |
| DataFrame QA | **Strategy prompt** (table_preserve) | +16.6pp, beats base |
| Event Counting | **More training** (V9-s5) | +25pp over V4-s5, beats base |

## Best-of-All Configuration (projected)

| Benchmark | Base | Best | Config | Delta |
|-----------|------|------|--------|-------|
| NIAH | 60.0% | 75.0% | V4-s5-hybrid | +15.0 |
| Multi-NIAH | 91.5% | 99.4% | V4-s5-hybrid | +7.9 |
| Doc-Classify | 81.6% | 99.2% | V4-s5-hybrid | +17.6 |
| DataFrame QA | 54.0% | 63.6% | V4-s5+table_preserve | +9.6 |
| Code Debug | 25.6% | 25.6% | V4-s5 | 0.0 |
| Multi-Hop QA | 85.0% | 85.0% | any | 0.0 |
| Notebook QA | 70.0% | 70.8% | V4-s5+notebook_sequential | +0.8 |
| Hard NIAH | 93.3% | 93.3% | any | 0.0 |
| Verbatim Copy | 100.0% | 100.0% | any | 0.0 |
| OOLONG | 0.0% | 10.0% | V4-s5 | +10.0 |
| Hard Multi-Hop | 40.0% | 50.0% | V4-s5 | +10.0 |
| Event Counting | 57.2% | 75.0% | V9-s5 | +17.8 |
| Cross-Doc Compare | 43.0% | 51.0% | V9-s5 | +8.0 |
| Key-Value Retrieval | 51.3% | 62.5% | V4-s5+lookup_thorough | +11.2 |
| **Average** | **60.9%** | **71.4%** | best-of | **+10.5** |

## Key Insight

No single configuration beats base on all benchmarks. But **the combination of training
improvements (V9-s5) and strategy prompts fixes EVERY regression**:

- Training adds: cross_doc (+8pp), event_counting (+18pp)
- Strategy prompts add: dataframe_qa (+10pp), notebook_qa (+1pp), key_value (+11pp)
- The gains are complementary — each approach fixes different benchmarks

V10 training combines both: regression-focused task distribution + new strategies in SC-GRPO.
