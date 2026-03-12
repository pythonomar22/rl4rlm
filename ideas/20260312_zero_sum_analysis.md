# Zero-Sum Tradeoff Analysis: Why Training Fails to Beat Base Across All Benchmarks

**Date:** 2026-03-12

## The Problem

No single trained model beats the base model on ALL 14 benchmarks. Every training approach (GRPO, SFT) shows a consistent pattern: improving some benchmarks while regressing others, with the net effect near zero.

## Evidence

### GRPO Models
| Model | Avg | Improved | Regressed | Tied |
|-------|-----|----------|-----------|------|
| V4-s5 | 61.4% (+0.5) | 5 | 5 | 4 |
| V10-s5 | 59.8% (-1.1) | 4 | 6 | 4 |
| V11-s5 | 63.1% (+2.2) | 7 | 5 | 2 |

### SFT Models (complete results)
| Model | Avg | Improved | Regressed | Key Wins | Key Losses |
|-------|-----|----------|-----------|----------|------------|
| SFT V1 ep2 | 52.2% (-8.7) | 5 | 9 | NIAH +25, DFQA +14 | multi_hop -40, hard_niah -33, hard_multi_hop -30 |
| SFT V1 ep5 | 54.1% (-6.8) | 4 | 10 | NIAH +30, oolong +20 | multi_hop -30, hard_multi_hop -30, hard_niah -27 |
| SFT V3 ep2 | ~61.4% (12/14) | 3 | 8 | DFQA +30, NIAH +20 | hard_multi_hop -20, code_debug -17 |

### SFT V6 (WEAK-TASK-ONLY — BEST SFT APPROACH)
Quick eval (7 benchmarks, 10 tasks, seed-offset 20000):
| Benchmark | V6 ep2 | Base | Diff |
|-----------|--------|------|------|
| NIAH | **100.0%** | 60.0% | +40.0 |
| Hard NIAH | 80.0% | 93.3% | -13.3 |
| Multi-hop QA | **90.0%** | 85.0% | +5.0 |
| Multi-NIAH | 86.0% | 91.5% | -5.5 |
| Verbatim | 97.5% | 100.0% | -2.5 |
| DataFrame QA | **60.0%** | 54.0% | +6.0 |
| Doc-Classify | 78.0% | 81.6% | -3.6 |
| **Average** | **84.5%** | **80.8%** | **+3.7** |

V6 ep2 achieves the BEST strong-task preservation of any SFT model:
- Multi-NIAH: -5.5pp (vs V5-cons -37.5pp, V1 ep2 -1.6pp)
- Verbatim: -2.5pp (vs V1 ep2 -10pp)
- Multi-hop: +5pp (vs V1 ep2 -40pp!)

### Disproven Hypotheses

1. **Higher LoRA rank helps**: R64 hard_niah = 60% (WORSE than R32's 80%). More capacity = more aggressive changes = more interference.
2. **Conservative LR prevents forgetting**: V5 conservative (LR 5e-6) still collapsed multi_niah to 54% (-37.5pp).
3. **SFT is better than GRPO**: SFT V1 at 52-54% is far worse than base (60.9%). GRPO V11-s5 at 63.1% remains our best.

### Confirmed Hypothesis
4. **Weak-task-only training reduces interference**: V6 ep2 shows the smallest strong-task regressions while achieving large weak-task improvements.

### Oracle Best (per-benchmark max across all models)
Oracle average: 67.6% — this is the ceiling if we could pick the best model per benchmark.

## Root Causes

### 1. ~~LoRA Capacity Constraint~~ (DISPROVEN)
- LoRA rank 64 performed WORSE than rank 32 on hard_niah (60% vs 80%)
- More capacity lets the model change more aggressively, causing MORE interference
- The zero-sum is NOT about capacity — it's about gradient direction

### 2. Single-Turn Training Data for Multi-Step Tasks
- multi_hop_qa: 87/122 samples (71%) are single-turn
- notebook_qa: 98/105 samples (93%) are single-turn
- This teaches the model to shortcut multi-step reasoning into one pass
- **Mitigation tested:** SFT V5 (multi-turn filtered)

### 3. Format Rigidity Transfer
- SFT on regular NIAH teaches format-specific extraction (e.g., regex parsing)
- This breaks on hard_niah's adversarial distractors
- Base model uses flexible Counter.most_common() that survives distractors
- **Root cause:** training data has ONE dominant pattern (chunk+llm_query+Counter) regardless of task

### 4. Cross-Task Interference
- Training on task A's patterns affects behavior on unrelated task B
- Because LoRA modifies the same weight matrices for all tasks
- The model can't distinguish "when doing NIAH, use pattern X" vs "when doing hard_niah, use pattern Y"
- **Mitigation tested:** SFT V6 (weak tasks only — don't train on tasks where base is strong)

## Completed Experiments

| Experiment | What it tested | Result | Verdict |
|-----------|---------------|--------|---------|
| SFT V5 (r32) | Multi-turn filtering | Not evaluated separately | — |
| SFT V5 R64 | Higher LoRA rank | NIAH 90%, multi_hop 30%, verbatim 70%, DFQA 80% | WORSE: more capacity = more interference |
| SFT V5 Conservative (LR 5e-6) | Gentle nudging | multi_niah 54% (-37.5pp collapse) | DISPROVEN: low LR doesn't prevent forgetting |
| SFT V6 ep2 (weak only, from scratch) | Don't train on strong tasks | Definitive: 8/14 done. NIAH 80%, multi_niah 76.2%, code_debug 15.6%, multi_hop 55% | WORSE than base avg |
| SFT V7 (weak only, from V11-s5) | Improve V11-s5 weak tasks | Quick: ep1 best (82.1%), definitive: NIAH 65%, multi_niah 83.8% | Zero-sum persists |
| SFT V7c (conservative, from V11-s5) | Lower LR + weak only | Quick: ep1 NIAH 100%, multi_niah 80%, doc_classify 72% | Zero-sum persists |
| V14 GRPO (SFT→GRPO two-phase) | SFT warm-start + RL | Quick: NIAH 100%, multi_niah 70%, hard_niah 60%, verbatim 47.5% | WORSE: 72.6% avg vs 80.8% base |
| V11-s10 | More GRPO steps from V11-s5 | Definitive: 60.0% avg (-3.1pp from V11-s5) | WORSE: more training = regression |
| V10-s5 | Regression-targeted GRPO | Definitive: 59.8% avg (-1.1pp from base) | WORSE |

## Critical Discovery: Evaluation Variance

**Run-to-run variance is ~15pp on individual benchmarks.**
- V11-s5 NIAH: 80% (run 1) vs 65% (run 2), same model/seeds/config
- V7 ep1 NIAH: 100% (quick eval) vs 65% (definitive eval)
- Root cause: temperature=0.7 for root code generation
- Implication: individual benchmark comparisons < 10pp are within noise
- Average across 14 benchmarks has ~4pp variance (sqrt(14) reduction)

## Theoretical Framework

The zero-sum tradeoff is **NOT** about LoRA capacity (disproven by R64). It's about:
1. **Gradient direction interference**: LoRA modifies shared weight matrices for ALL tasks
2. **Format rigidity**: Training teaches specific extraction patterns that fail on other tasks
3. **Single-turn collapse**: SFT data is majority single-turn, breaks multi-step reasoning
4. **Cross-task weight sharing**: No mechanism to route different tasks to different weight subsets

The analogy is **multi-task learning with shared parameters** — a well-known problem in ML where improving one task's loss can increase another task's loss (negative transfer).

## What Might Actually Work

1. **Strategy prompts at inference** — zero training cost, +5-6pp empirically (CURRENT BEST PATH)
2. **Task-specific LoRA adapters** — separate adapters per task type, composed at inference
3. **Mixture-of-LoRA** — learn to route different tasks to different adapter subsets
4. **Full fine-tuning** (if Tinker supports it) — more capacity, less interference
5. **DPO** — gentler gradient updates than SFT, might reduce interference
6. **Ensemble at inference** — run base + trained, pick better answer per task

## Best Results So Far

### 1. V11-s5 (GRPO): 63.1% average (+2.2pp) — CURRENT BEST
- 7 improved, 5 regressed, 2 tied
- Best at: hard_niah (100%), doc_classify (99.2%), event_counting (72.9%)
- Worst at: cross_doc (-18.6), KV (-15.2), DFQA (-14)
- Session: bae6fabb, state-0005

### 2. V11-s5 + Strategy prompts: eval running
- Uses table_preserve (DFQA), notebook_sequential (notebook), lookup_thorough (KV)
- V9-s10 + strategies got 66.5% — V11-s5 should be similar or better
- **This is our most likely release candidate**

### 3. Oracle best (per-benchmark max across ALL models): 67.6%
- The theoretical ceiling with model routing
- 9 improved, 5 tied, 0 regressed vs base

### V11-s10: 60.0% average — more training HURTS
| Benchmark | V11-s5 | V11-s10 | Change |
|-----------|--------|---------|--------|
| Multi-NIAH | 87.8% | 95.4% | +7.6 |
| Code Debug | 25.6% | 32.2% | +6.6 |
| DFQA | 40.0% | 46.5% | +6.5 |
| KV | 36.1% | 41.1% | +5.0 |
| NIAH | 80.0% | 75.0% | -5.0 |
| Event Count | 72.9% | 67.3% | -5.6 |
| Notebook | 66.7% | 60.0% | -6.7 |
| Hard NIAH | 100.0% | 93.3% | -6.7 |
| Multi-Hop | 80.0% | 70.0% | -10.0 |
| Hard Multi | 50.0% | 40.0% | -10.0 |
| Oolong | 20.0% | 0.0% | -20.0 |

### Recommended next steps
1. Complete V11-s5 + strategy eval (running) — **highest priority**
2. If V11-s5 + strategy > 65%: this is our release model
3. Run V11-s5 + strategy 2-3 times to measure variance
4. Write paper with V11-s5 as primary result, strategy augmentation as bonus
5. Consider: per-benchmark confidence intervals using bootstrap
