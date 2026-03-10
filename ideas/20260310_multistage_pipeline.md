# Multi-Stage Training Pipeline for Best Possible RLM

## Current Problem
- SC-GRPO (V7) is working well for mode collapse but is slow (each step ~1hr)
- Teacher distillation gives gold trajectories but only ~50% success rate
- Student RFT gives self-generated trajectories but limited diversity
- No mechanism to combine all these signals

## Proposed Pipeline

### Stage 1: SFT on Diverse Gold Trajectories
**Input:** Teacher (397B) + best student (V4-s5) trajectories
**Goal:** Teach the model diverse code patterns without mode collapse risk

Steps:
1. Collect teacher trajectories (batch 1 + batch 2: ~100+ trajectories)
2. Collect student RFT trajectories (V4-s5 on hard tasks: ~50+ trajectories)
3. Filter: score >= 0.5, terminated, ≤8 turns, ≤2 errors
4. Convert to SFT samples using filter_trajectories.py
5. SFT with low LR (1e-5) for 1-2 epochs on Tinker

Expected:
- Model learns diverse code patterns from teacher
- Model reinforces its own successful strategies
- Low risk of catastrophic forgetting (small data, low LR)

### Stage 2: SC-GRPO on SFT Checkpoint
**Input:** Stage 1 checkpoint
**Goal:** Refine strategies via RL, improve on hard tasks

Steps:
1. Resume V7-style SC-GRPO from Stage 1 checkpoint
2. Focus on weakest benchmarks: code_debug, hard_multi_hop, event_counting
3. Run for 10-15 steps (until improvement plateaus)
4. Use adaptive difficulty + anti-shortcut

Expected:
- Model improves on hard tasks
- SC-GRPO prevents mode collapse
- Gradient signal from diverse strategies

### Stage 3: Online DPO / Rejection Sampling
**Input:** Stage 2 best checkpoint
**Goal:** Final refinement via preference learning

Steps:
1. Run model on 200+ diverse tasks with K=4 samples each
2. For each task: best trajectory = "chosen", worst = "rejected"
3. DPO training with β=0.1 on Tinker (forward_backward_custom)
4. Alternatively: keep only best trajectories for another SFT round

Expected:
- Model learns to prefer correct strategies over incorrect ones
- More stable than GRPO (no mode collapse risk in DPO)
- Final polish on edge cases

### Stage 4: Evaluation & Selection
**Input:** Checkpoints from all stages
**Goal:** Select best model for public release

Steps:
1. Run all 12+ benchmarks on each checkpoint
2. Compare head-to-head on all tasks
3. Run qualitative analysis on 10 hard tasks (manual inspection)
4. Select best checkpoint based on both quantitative and qualitative metrics
5. Upload to HuggingFace

## Timeline Estimate
- Stage 1: 2-4 hours (data collection + SFT)
- Stage 2: 6-12 hours (10-15 GRPO steps)
- Stage 3: 2-4 hours (data collection + DPO/SFT)
- Stage 4: 2-4 hours (evaluation)

## Novel Contributions from This Pipeline
1. SC-GRPO for code generation (proven to eliminate mode collapse)
2. Multi-stage SFT→GRPO→DPO pipeline for RLM training
3. Teacher distillation for recursive models
4. Anti-shortcut training with minimum context lengths
5. Strategy-conditioned trajectory diversity
