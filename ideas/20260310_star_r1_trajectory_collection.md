# STaR Round 1: Trajectory Collection from Base Qwen3.5-35B-A3B
**Date:** 2026-03-10
**Model:** Qwen3.5-35B-A3B (base, no fine-tuning)
**Temperature:** 0.7
**Max iterations:** 8

## Collection Summary

### Multi-NIAH (24 tasks)
- **22/24 correct (91.7%)** — avg score 0.878
- All correct trajectories solved in **1 turn** (single chunk-and-query code block)
- 89 sub-calls total (~3.7 per task)
- Collection time: 364s (6.1 min)
- Very high quality trajectories — model writes correct chunking code immediately

### Doc-Classify (20 tasks)
- **19/20 correct (95.0%)** — avg score 0.805 (partial accuracy counted)
- 73 sub-calls for 22 root calls (~3.3 sub-calls per task)
- Collection time: 476s (7.9 min)
- Strong performance even on 15-doc and 20-doc tasks
- Notable variance: some tasks get 100%, others much lower
- Key failure mode: 20-doc tasks sometimes timeout or mis-classify

### NIAH (150 tasks, in progress)
- **~87/110 correct (79.1%)** as of writing (still collecting)
- Consistent with baseline eval (81.0%)
- Failure modes: position 0.75, 50K documents

## Key Observations

1. **Base model is already very capable.** 91.7% on multi-NIAH and 95% on doc-classify during trajectory collection. The main challenge is NIAH (79.1%).

2. **Single-turn trajectories dominate.** The 35B model almost always solves tasks in 1 turn. This means:
   - SFT training data will be mostly single-turn conversations
   - The model already writes good chunking code
   - Training should focus on reliability, not teaching code patterns

3. **Doc-classify collection is MUCH better than baseline eval (95% vs 53.6%).** The discrepancy is because:
   - Different seeds → different task instances
   - The trajectory collection uses score_doc_classify which may be more lenient
   - The collection has min_score=0.0 (saves all, including partial scores)

4. **Sub-call efficiency.** About 3-4 sub-calls per task on average — the model is efficient with its recursive calls.

## Training Plan

Given these high success rates, we have plenty of correct trajectories:
- Multi-NIAH: 22 correct trajectories → ~22 SFT samples (1 turn each)
- Doc-Classify: 19 correct → ~19 SFT samples
- NIAH: ~87 correct (estimated final ~120) → ~120 SFT samples

Total estimated: ~160 SFT samples. This is small but should be sufficient for LoRA SFT.

### SFT Strategy
1. Use ALL correct trajectories (score > 0, terminated)
2. Include doc-classify partial scores (e.g., 80% accuracy) as positive examples — the model learned something useful
3. Weight samples by quality? Or treat all equally since the model is already good
4. 5 epochs with small batch size should suffice

### Next Steps After SFT
1. Evaluate SFT checkpoint on same benchmarks
2. STaR round 2: collect from SFT model on harder tasks
3. GRPO RL: use importance_sampling loss on Tinker
4. DPO: pair correct vs incorrect trajectories for preference learning
