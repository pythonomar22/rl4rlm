# GRPO v1 Mode Collapse Analysis
**Date:** 2026-03-10

## Diagnosis

### The Problem
After ~17 training steps, the model collapsed to a deterministic policy:
- All K trajectories within a group produce **identical** code (character-for-character)
- This means advantages = 0 for all trajectories → 0 gradient updates
- Steps 18, 19, 22, 24, 30 all had 0 updates

### Evidence
**Step 10 (healthy diversity):**
- traj0: `chunk_size=5000, overlap=500`, simple dedup
- traj1: `chunk_size=5000, overlap=500`, adds `final_answers[:3]` limit
- traj2: `chunk_size=10000, overlap=1000`, complex parsing + fallback llm_query

**Step 30 (collapsed):**
- All 3 trajectories: identical code, `chunk_size=2000, overlap=500`, same exact answer
- model_stats accumulating (718→719→720 root_calls — bug, not reset between trajectories)

### Timeline
| Steps | Reward | Updates | Status |
|-------|--------|---------|--------|
| 1-10 | 0.51→0.80 | 15-24 | Healthy learning |
| 11-17 | 0.80→0.90 | 8-16 | Converging |
| 18-30 | 0.90 | **0** (intermittent) | Mode collapsed |

### Root Causes

1. **Positive-only advantages** (`max(advantage, 0)`)
   - Only reinforces good trajectories, never pushes away from bad ones
   - Over time, the model converges to one "safe" template
   - No diversity pressure

2. **Too-high learning rate** (5e-6 base → 50e-6 effective)
   - With 10x LoRA scaling, effective LR was very high
   - Each gradient step causes large policy changes
   - Combined with positive-only REINFORCE → rapid mode collapse

3. **Temperature too low** (0.8)
   - At mode collapse, even temperature=0.8 produces identical outputs
   - Need higher temp OR entropy regularization

4. **Infrequent weight refresh** (every 3 steps)
   - Policy drifts from the sampling policy between refreshes
   - By the time we refresh, the policy has already narrowed

5. **cross_entropy ≠ true GRPO**
   - No importance sampling ratio (π_new/π_old)
   - No clipping to prevent too-large policy updates
   - No negative advantages for diversity

### Good News: NOT Reward Hacking
The model IS a genuine RLM:
- Writes proper Python code with `llm_query()`, `FINAL_VAR()`
- Uses overlapping chunks, aggregates results
- Gets correct answers (score=1.0)
- No shortcutting or gaming the reward

The model just found ONE good strategy and stuck with it.

## GRPO v2 Fixes

1. **Allow negative advantages** → push away from below-average trajectories
2. **Lower LR** → 2e-6 base (20e-6 effective), 2.5x lower
3. **Higher temperature** → 1.0 for exploration
4. **Refresh every step** → keep policy and sampling aligned
5. **Weighted task mix** → more NIAH to prevent regression
6. **True importance_sampling loss** → proper GRPO with ratio clipping
7. **Reset model stats** between trajectory collections
8. **Resume from step 10** → best checkpoint before collapse
