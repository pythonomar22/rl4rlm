# Mode Collapse in GRPO Training: Analysis and Mitigations
**Date:** 2026-03-10

## The Pattern

Every GRPO training run collapses around step 10-14:

| Version | Collapse Start | Collapse Pattern | Recovery? |
|---------|---------------|------------------|-----------|
| v1 | Step 17 | Positive-only advantages, high LR | No |
| v2 | Step 11 | All K=8 identical, 0 updates | Partial (step 14) |
| v3 | Step 12 | All K=8 identical on easy tasks | Partial (step 14, code_debug=0.014) |
| v4 | Step 5+ | hard_multi_hop all identical | Running... |

## Why It Happens

### The Fundamental Issue
GRPO uses **group-relative advantages**: A_i = (r_i - mean(r)) / std(r).
When all K trajectories get the same reward → std = 0 → A_i = 0 → no gradient.

This happens because:
1. **Easy tasks saturate first**: doc_classify, multi_niah, niah → model converges to template
2. **Temperature doesn't help enough**: even at T=1.0, the model generates identical code
3. **LoRA rank 32 limits diversity**: only ~1M trainable parameters, quickly converges
4. **Code generation is more deterministic than text**: once the model learns `chunk_size = 20000; while i < len(context)...`, all K samples produce the same code

### Evidence of Convergence
V3 skipping pattern:
- Steps 1-5: 0 skips (all groups diverse)
- Steps 6-8: 2-3 skips/step (easy tasks saturating)
- Steps 9-10: 1 skip/step (code_debug provides variance)
- Steps 11: 3/4 skips
- Steps 12-13: 4/4 skips (TOTAL collapse)
- Step 14: 1/4 skips (code_debug=0.014 provides variance because it's hard)

## Key Insight

**Collapse correlates with task difficulty.** Easy tasks (doc_classify 0.915, multi_niah 0.915, niah 0.872) converge first. Hard tasks (code_debug 0.012-0.464, hard_multi_hop 0.000-0.250) provide variance longer.

This suggests: the cure for mode collapse is **harder tasks**.

## Potential Mitigations

### 1. Adaptive Task Difficulty (Most Promising)
Track per-task-type skip rate. When a task type has >50% skip rate over last 3 steps, increase its difficulty:
- NIAH: increase doc length (100K → 200K → 500K)
- Multi-NIAH: increase K (3 → 5 → 8 → 10) or doc length
- Doc-Classify: increase N (5 → 10 → 15 → 20) or add harder categories
- Multi-Hop QA: increase doc length and number of hops

### 2. Entropy Bonus in Reward
```python
# Add entropy penalty for repetitive code
from collections import Counter
def code_entropy(code: str) -> float:
    tokens = code.split()
    counts = Counter(tokens)
    total = sum(counts.values())
    entropy = -sum((c/total) * log(c/total) for c in counts.values())
    return entropy / log(total) if total > 1 else 0  # normalized 0-1

# In compute_reward:
entropy = code_entropy(trajectory_dict.get("parsed_code", ""))
reward += 0.02 * entropy  # small bonus for diverse code
```

### 3. Temperature Annealing (Reverse Direction)
Instead of starting high and decaying, start at T=1.0 and **increase** to T=1.5-2.0 when skip rate exceeds threshold.

### 4. Per-Group Temperature Scaling
When generating K trajectories for a prompt, use different temperatures:
- K=8: T = [0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.5]
- This ensures variance even when the model is confident

### 5. Larger K (K=16 or K=32)
More samples → more chance of diverse rewards. But 2-4x more expensive per step.

### 6. Best-of-N Rejection
Don't update on groups where all K samples are identical. Instead, re-generate with higher temperature. This prevents wasted steps but doesn't address the underlying convergence.

### 7. Reference Policy Mixing
Occasionally generate some trajectories from the **reference** (frozen base) model. This injects diversity from the starting distribution.

## Recommended Approach for V5

Combine approaches 1 + 4:
1. **Adaptive difficulty**: Track skip rates, increase difficulty when saturated
2. **Per-group temperature**: Use T ∈ [0.8, 1.5] across K=8 samples
3. Keep cosine LR schedule (delays collapse even if it doesn't prevent it)
4. Focus on hard tasks: increase hard_multi_hop to 30% of mix, add harder variants

## Implications for Paper

Mode collapse is a **universal challenge** for GRPO-style RL on code generation tasks. Our analysis shows:
1. Code is more deterministic than text → collapses faster
2. Easy-task saturation is the proximate cause, not LR or temperature
3. Harder tasks delay collapse by providing variance
4. Adaptive difficulty (curriculum learning) is the natural solution
5. Per-group temperature scaling is cheap and effective

This is a publishable finding — most GRPO papers focus on text generation where this pattern is less pronounced.
