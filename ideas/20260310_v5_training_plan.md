# V5 Training Plan: Addressing Mode Collapse and Decomposition
**Date:** 2026-03-10

## Summary of Lessons

### What Works
1. **GRPO from base model** (no SFT warmup) — avoids catastrophic forgetting
2. **Cosine LR** — delays mode collapse by ~3 steps vs constant LR
3. **Diverse task mix** — prevents over-specialization on easy tasks
4. **hard_multi_hop in training** — provides persistent learning signal (hard tasks never saturate)

### What Doesn't Work
1. **Fixed temperature** — all K=8 trajectories converge to identical code
2. **Sparse reward on decomposition tasks** — model learns compound queries instead of decomposition
3. **Easy-task saturation** — doc_classify, multi_niah, niah all reach 100% quickly, then provide 0 gradient
4. **Code Debug** — stuck at 25% regardless of training (may need targeted training or different approach)

### Key Metrics Across Checkpoints

| Benchmark | Base | Best | Which |
|-----------|------|------|-------|
| NIAH | 81% | 100% | v3-s10, v4-s5 |
| Multi-NIAH | 97.8% | 100% | v3-s5 |
| Doc-Classify | 53.6% | 100% | v3-s5, v3-s10 |
| Multi-Hop QA | 50% | 70% | v2-s10, v4-s5 |
| Code Debug | 25% | 25% | all same |
| DataFrame QA | 80% | 80% | base (!) |
| Notebook QA | 60% | 80% | v4-s5 |
| Hard NIAH | 90% | 100% | v2-s10, v3-s5 |
| Verbatim | 90% | 100% | v2-s10+ |
| OOLONG | 20% | 20% | base (!) |
| Hard Multi-Hop | 20% | 30% | v3-s5 |

## V5 Design

### Start Point
- **v4-s5** (once eval completes and if it's best overall)
- Or v3-s5 if v4-s5 shows regressions

### Key Changes

#### 1. Per-Trajectory Temperature Scaling (implemented)
```python
temp_schedule = [0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.5]
model.temperature = temp_schedule[k]  # Different T for each of K=8 trajectories
```
Forces diversity even when model is confident. Low-T samples should be the "correct" template, high-T samples should explore alternatives.

#### 2. Intermediate Decomposition Reward (implemented)
```python
# For hard_multi_hop: 60% final answer + 25% bridge entity discovery + 15% format
decomp_bonus = fraction of bridge entities found in trajectory stdout
reward = 0.60 * score + 0.25 * decomp_bonus + 0.15 * (format_bonus + 0.1)
```
Gives partial credit for finding intermediate entities even when final answer is wrong.

#### 3. Adaptive Task Difficulty (NOT YET implemented)
Track per-task-type skip rate. When a task type saturates (>50% skip rate over 3 steps), increase difficulty:
- NIAH: longer docs (200K → 500K → 1M)
- Doc-Classify: more categories or articles
- Multi-NIAH: more needles (K=8 → 10 → 15)
This prevents easy-task saturation from dominating the training mix.

#### 4. Increased Timeout (implemented)
```python
code_timeout = max(90, 90 + (len(prompt) // 100000) * 60)
```
150K docs get 150s timeout (vs 90s before). Prevents timeouts from corrupting the reward signal.

#### 5. Focus on Weakest Benchmarks
- **DataFrame QA**: Currently REGRESSES with RL. Need targeted training or different approach.
  - Idea: add DataFrame-specific reward that checks numerical accuracy
  - Idea: SFT on gold DataFrame QA trajectories from 397B teacher
- **OOLONG**: Also regresses. Counting/aggregation tasks need different signal.
- **Code Debug**: Stuck at 25%. May need code-specific reward model or demonstrations.

### Task Mix (V5)
```python
mixed_v5 = {
    "niah": 10%,           # saturated, keep minimal
    "multi_niah": 5%,      # saturated, keep minimal
    "doc_classify": 5%,    # saturated, keep minimal
    "hard_multi_hop": 25%, # key focus
    "multi_hop_qa": 15%,   # key focus
    "notebook_qa": 15%,    # showing improvement
    "dataframe_qa": 10%,   # needs recovery
    "code_debug": 10%,     # needs improvement
    "hard_niah": 5%,       # saturated but keep for robustness
}
```

### Training Config
- LR: 2e-6 (cosine decay to 10%)
- K=8 with temperature scaling [0.8-1.5]
- Batch: 4 tasks/step
- Steps: 30 (or until mode collapse)
- Save every 5 steps
- Checkpoint from best available (v4-s5 or v3-s5)

## Success Criteria

| Benchmark | Target | Rationale |
|-----------|--------|-----------|
| Hard Multi-Hop | 40%+ | Decomposition via intermediate reward |
| Multi-Hop QA | 75%+ | Build on v4-s5's 70% |
| Notebook QA | 85%+ | Build on v4-s5's 80% |
| DataFrame QA | 60%+ | Recovery from regression |
| NIAH | 100% | Maintain |
| All others | ≥ v3-s5 | No regressions |

## Timeline
1. Complete V4-s5 and V3-s10 evals
2. Determine best starting checkpoint
3. Implement adaptive difficulty
4. Launch V5 training with all fixes
5. Eval at step 5, 10, 15, 20
