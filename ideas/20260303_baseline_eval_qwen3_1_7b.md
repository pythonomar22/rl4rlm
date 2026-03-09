# 20260303: Baseline Eval Qwen3-1.7B as RLM

## Hypothesis
Qwen3-1.7B with the v2 system prompt can solve NIAH tasks as an RLM at above-random accuracy without any training.

## Method
Ran 25 NIAH tasks across 5 document lengths (5K, 10K, 20K, 50K, 100K) at 5 needle positions (0.1, 0.2, 0.5, 0.8, 0.9). Full RLM scaffold.

## Expected Outcome (with numbers)
30-50% accuracy (paper's Qwen3-8B base scored near 0, but our prompt is better and NIAH is simpler than the paper's benchmarks).

## What Would Disprove This
<10% → model can't function as RLM even on simple tasks, need major intervention.

## Results
| Metric | Value |
|--------|-------|
| **Overall accuracy** | **68%** |
| Tasks | 25 |
| Time | 538.8s (21.6s/task) |

### By document length:
| Length | Accuracy |
|--------|----------|
| 5K | 80% |
| 10K | 80% |
| 20K | **100%** |
| 50K | 40% |
| 100K | 40% |

### By needle position:
| Position | Accuracy |
|----------|----------|
| 0.1 (near start) | 80% |
| 0.2 | 20% |
| 0.5 (middle) | 80% |
| 0.8 | 80% |
| 0.9 (near end) | 80% |

### Key observations:
1. **Strong at ≤20K**: 80-100% accuracy. The model's RLM pattern works well within its comfort zone.
2. **Drops at 50K+**: 40% accuracy. The model struggles with very long documents, likely because chunking is too coarse or sub-calls miss the needle.
3. **Position 0.2 anomaly**: 20% — only 1/5. May be a statistical artifact with small N, or a specific interaction between chunk boundaries and needle placement.
4. **100% at 20K**: Perfect score. This is the sweet spot — long enough to need chunking, short enough for reliable sub-calls.
5. **Self-bootstrap is viable**: 68% accuracy means ~100 correct trajectories from 150 attempts. Enough for SFT.

### Token costs:
- Root calls: 34 (1.36/task avg)
- Sub-calls: 42 (1.68/task avg)
- Input tokens: 124,871
- Output tokens: 27,513
- Total time: ~9 minutes

## Conclusion
**68% baseline is excellent.** This is already a strong starting point. The model reliably uses the RLM pattern (chunking → llm_query → aggregate → FINAL). SFT on correct trajectories should push this higher, and RL should improve robustness on longer documents.

The fact that 20K is 100% while 50K+ drops to 40% gives a clear training target: teach the model better chunking strategies for longer documents.

### Next steps
1. Collect trajectories (ongoing, 50 tasks x 3 attempts = 150)
2. Filter correct ones (~100 expected)
3. SFT warm-start
4. Eval after SFT → measure improvement
5. RL training → measure RL benefit over SFT
