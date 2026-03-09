# Baseline Evaluation: Qwen3.5-35B-A3B + RLM Scaffold (No Fine-Tuning)
**Date:** 2026-03-10
**Model:** Qwen3.5-35B-A3B (MoE, 35B total / 3B active params)
**Infrastructure:** Tinker API

## Summary

The base Qwen3.5-35B-A3B model with the RLM scaffold (no fine-tuning) achieves 77.5% average across three benchmarks. This is already higher than the 1.7B base (63.5%) and close to the 1.7B DPO (84.5%), with dramatically different strengths/weaknesses.

## Results

### NIAH-100 (Single Needle Retrieval): 81.0%
| Doc Length | Accuracy |
|-----------|----------|
| 5K | 70% |
| 10K | 90% |
| 20K | 95% |
| 50K | 70% |
| 100K | 80% |

| Position | Accuracy |
|----------|----------|
| 0.10 | 85% |
| 0.25 | 80% |
| 0.50 | 90% |
| 0.75 | 70% |
| 0.90 | 80% |

**Observations:**
- Position 0.75 is the weakest (70%) — this is at chunk boundary with 20K chunks
- 50K documents are oddly weak (70%) despite 100K being 80%
- Position 0.25 improved from 60% (1.7B base) to 80% — better chunking
- 5K unexpectedly low (70%) — model may be over-chunking short documents

### Multi-NIAH-24 (Multiple Needle Retrieval): 97.8% recall
| Needle Count | Recall |
|-------------|--------|
| 3 needles | 100.0% |
| 5 needles | 95.0% |
| 8 needles | 98.4% |
| 10 needles | 100.0% |

**Observations:**
- NEAR PERFECT. Already better than our best 1.7B model (DPO: 87.9%)
- Only weakness: 5-needle tasks at 95% (1 miss out of 20 needles total)
- The 35B model's sub-calls extract needles much more reliably than 1.7B
- This benchmark may be largely SOLVED by the base model

### Doc-Classify-20: 53.6% accuracy
| Doc Count | Accuracy |
|----------|----------|
| 5 docs | 68.0% |
| 10 docs | 62.0% |
| 15 docs | 49.3% |
| 20 docs | 35.0% |

**Observations:**
- Significantly WORSE than 1.7B base (80.3%)
- Scales inversely with n_docs: 68% → 35% as complexity grows
- Sub-call model classifications are often wrong or mis-formatted
- May be a prompt engineering issue — sub-call prompts not optimized for classification
- This is the clear area where SFT/RL can make the biggest impact

## Comparison with CS234 (Qwen3-1.7B)

| Model | NIAH-100 | Multi-NIAH | DocCls | Avg |
|-------|----------|-----------|--------|-----|
| 1.7B Base | 72.0% | 38.3% | 80.3% | 63.5% |
| 1.7B SFT | 90.0% | 57.9% | 82.4% | 76.8% |
| 1.7B DPO | 83.0% | 87.9% | 82.6% | 84.5% |
| **35B Base** | **81.0%** | **97.8%** | **53.6%** | **77.5%** |

**Key Insights:**
1. Multi-NIAH is essentially SOLVED by the base 35B model (+59.5pp over 1.7B base, +9.9pp over 1.7B DPO)
2. NIAH is comparable to 1.7B SFT (81% vs 90%) — SFT should push this to 95%+
3. DocClassify is the big gap — 53.6% vs 80.3% for 1.7B base
4. The 35B model is much stronger at extraction/retrieval but weaker at classification

## Hypothesis: Why DocClassify is Low

1. **Sub-call prompt mismatch:** The system prompt gives a classification example but sub-calls produce verbose/wrong categories
2. **Category naming:** The model may use slightly different category names than expected (e.g., "Tech" vs "Technology")
3. **Output format:** The model returns categories in various formats that the scorer can't parse
4. **The 1.7B model was trained on doc-classify tasks** — it learned the exact format

**Plan:** Inspect trajectories to confirm, then fix via:
- Better system prompt for classification tasks
- SFT on correct classification trajectories
- Possibly separate sub-call prompt for classification vs retrieval

## Training Priorities

Given these baselines, the training focus should be:
1. **DocClassify:** Biggest gap (53.6% → target 85%+). SFT on correct trajectories.
2. **NIAH:** Room to improve (81% → target 95%+). Particularly at 50K and position 0.75.
3. **Multi-NIAH:** Already near-perfect (97.8%). Maintain, don't degrade.

Total eval time: ~1140s (19 min) for NIAH-100, ~428s (7 min) for Multi-NIAH, ~525s (9 min) for DocClassify.
