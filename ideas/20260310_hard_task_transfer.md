# Hard Task Training Transfers to Easy Tasks
**Date:** 2026-03-10

## Finding

V4-s5 (trained with 20% hard_multi_hop in task mix) shows **dramatic improvements on DataFrame QA** (75% vs 35% at v3-s5), even though V4 had the same 10% DFQA in its mix as V3.

## Evidence

### DataFrame QA: Turn Count Comparison
| Metric | v3-s5 (no hard tasks) | v4-s5 (20% hard tasks) |
|--------|----------------------|----------------------|
| Score | 35% (2.8/8) | **75% (6/8)** |
| Avg turns | 1.1 | **2.1** |
| Max turns | 2 | **4** |
| Multi-turn tasks | 1/8 | **4/8** |

### What Changed
v3-s5 model: single-pass, gives wrong answers confidently
v4-s5 model: iterates, checks results, uses multiple chunks

### Specific Examples
- Task "ABT": v3-s5 = 1 turn, wrong (NVDA). v4-s5 = 3 turns, correct (ABT)
- Task "LLY": v3-s5 = 1 turn, wrong (JNJ). v4-s5 = 4 turns, correct (LLY)
- Task "2024-01-06": v3-s5 = 1 turn, wrong (01-02). v4-s5 = 2 turns, correct

## Why This Happens

Training on **150K-character hard_multi_hop** tasks teaches the model:
1. **Persistence**: scan all chunks, don't stop at first result
2. **Chunking strategies**: how to process documents too large for single-pass
3. **Result validation**: check if the first answer makes sense

These skills transfer to DataFrame QA because:
- Large CSV datasets (200K+ chars) also require chunking
- Financial data analysis requires iterating to find specific records
- Wrong-first-attempt recovery is the same skill regardless of task

## Implication

**Hard tasks are the best training signal.** Instead of training on a balanced mix
of easy-to-hard tasks, we should focus on the hardest tasks that the model can
sometimes solve (>5% reward). Easy tasks (doc_classify, multi_niah) saturate
quickly and provide no gradient, while hard tasks provide persistent signal AND
improve easy-task performance through skill transfer.

## For V5

- Increase hard task fraction: 25% hard_multi_hop, 15% multi_hop_qa
- Reduce easy tasks: 10% niah, 5% multi_niah, 5% doc_classify
- Add harder variants of DataFrame QA (30+ tickers, 120+ days)
- Watch for the "persistence effect": multi-turn trajectories on hard tasks → better results on all tasks

## For Paper

This is a strong contribution: **training on harder tasks improves performance on
all tasks through skill transfer**. This is counter-intuitive — one might expect
that training on hard_multi_hop only improves multi-hop performance. But the
learned skills (chunking, persistence, validation) are general.

Key narrative: "RLM training teaches transferable code-generation strategies, not
task-specific patterns. Training on harder tasks forces the model to learn more
robust strategies that generalize."
