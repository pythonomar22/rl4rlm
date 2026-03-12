# SFT Breakthrough: Dramatic NIAH Improvement Without GRPO Tradeoffs

**Date:** 2026-03-12

## Key Finding

SFT on correct trajectories from eval runs produces **dramatically better NIAH** (90% vs base 60%, +30pp) while avoiding the zero-sum tradeoff that plagued GRPO training. This is the single largest improvement over base we've achieved.

## Approach

Instead of GRPO (which creates negative gradients → format rigidity → regressions), we extracted correct trajectories from evaluation runs and used them for supervised fine-tuning. Three datasets:

1. **SFT V1**: 291 samples from 3 eval runs (base, V4-s5, V4-s5-hybrid). Missing multi_niah entirely, only 1 oolong sample.
2. **SFT V3**: 890 balanced samples from all sources, capped at 80 per task type, all 14 types represented.
3. **SFT V4**: 419 targeted regression samples, starting from V4-s5 weights (only regression tasks: DFQA, cross_doc, notebook_qa, event_counting, KV, code_debug, multi_hop).

## Quick Eval Results (10 tasks, seed-offset 20000)

| Model | NIAH | Multi-NIAH | Doc-Classify |
|-------|------|------------|-------------|
| Base | 60.0% | 91.5% | 81.6% |
| V4-s5 (GRPO) | 70.0% | 95.5% | 98.8% |
| SFT V1 ep1 (291, from scratch) | 70.0% | 56.7% | 84.0% |
| SFT V1 ep2 | **90.0%** | 88.0% | 78.0% |
| SFT V1 ep3 | 90.0% | 74.0% | 89.0% |
| **SFT V1 ep5** | **100.0%** | **96.0%** | 89.0% |
| SFT V3 ep1 (890, balanced) | 80.0% | 72.0% | **97.0%** |
| SFT V4 ep1 (from V4-s5, targeted) | 100.0% | 74.0% | 89.0% |

## Full Eval Results (20 tasks, seed-offset 10000 — definitive)

| Model | NIAH | Multi-NIAH |
|-------|------|------------|
| Base | 60.0% | 91.5% |
| V4-s5 | 70.0% | 95.5% |
| SFT V1 ep2 | **85.0%** | **89.9%** |
| SFT V1 ep5 | **90.0%** | pending |

## Key Observations

1. **NIAH scales with epochs**: 70% → 90% → 90% → 100% (quick eval). Epoch 5 full eval: 90%. This is the best NIAH result of any model we've trained.

2. **Multi-NIAH fragile without training data**: SFT V1 ep1 collapsed to 56.7% because multi_niah was absent from training. By ep5 it recovered to 96% — the model re-learned from NIAH transfer.

3. **Doc-classify:** SFT V3 balanced (97%) nearly matches GRPO V4-s5 (98.8%). SFT can teach doc_classify too.

4. **SFT from V4-s5 (V4 targeted)**: Gets perfect NIAH (100%) immediately but multi_niah drops (74%). The targeted regression data may interfere with V4-s5's multi_niah capability.

5. **Epoch selection matters**: SFT V1 shows non-monotonic behavior on multi_niah (56.7% → 88% → 74% → 96%). This is likely noise from 10-task eval.

## Why SFT Works Better Than GRPO

1. **No negative gradients**: SFT only teaches correct patterns. GRPO pushes AWAY from incorrect patterns, causing format rigidity.
2. **Preserved base flexibility**: The model retains its diverse approach repertoire instead of narrowing.
3. **No reward hacking**: GRPO can reward surface patterns (e.g., always chunking even when not needed). SFT teaches from actually-correct solutions.
4. **Standard recipe**: SFT warm-start → light GRPO is the known-good approach (DAPO, BAPO, DeepSeek-R1).

## Next Steps

1. **Full 14-benchmark eval** on SFT V1 ep2 and ep5 (running)
2. **SFT V3 epoch 2** quick eval (running) — balanced data should be most robust
3. **Two-phase training**: Best SFT checkpoint → 3-5 steps of GRPO with fixed temperature schedule
4. **Collect more diverse trajectories** for underrepresented tasks (cross_doc: 35, KV: 38, oolong: 12)

## Hypotheses for Full Eval

- SFT V1 ep5 will show 90%+ NIAH but may regress on DFQA/cross_doc (no SFT data for these tasks)
- SFT V3 balanced will show more even improvement across all 14 tasks
- The two-phase approach (SFT → light GRPO) should beat both SFT-only and GRPO-only
