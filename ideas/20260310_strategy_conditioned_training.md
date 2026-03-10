# Strategy-Conditioned Training: Solving Mode Collapse in Code RL
**Date:** 2026-03-10

## The Core Problem

V4b trajectory analysis revealed devastating mode collapse:
- **100% structural identity** on long contexts (150K+): all K=8 trajectories use identical 20K/2K chunking template
- **Zero genuine multi-turn trajectories** — all multi-turn are error retries
- **No aggregation** — always `results[0]`, never merges information across chunks
- **No task adaptation** — same template regardless of task type (counting, search, classification)

GRPO requires variance in rewards across the K group. When all trajectories are structurally identical, `std(rewards) ≈ 0`, advantages are zero, and there's no gradient.

## Why Existing Fixes Aren't Enough

### Temperature (V6: [0.7-1.2])
Even at T=1.5, the model generates the same template with minor prompt rewording. The template is so deeply ingrained that temperature alone can't break it. At T=2.0+, outputs degrade (syntax errors, hallucinations) rather than producing novel strategies.

### Adaptive Difficulty (V6)
Increases doc length/complexity when tasks saturate. But this makes mode collapse WORSE — the model's single template (20K chunk, take first result) is MORE dominant on longer docs.

### Code Diversity Bonus (V6: +0.03 max)
Too weak. Even with the bonus, the template-matching strategy dominates because it gets the correctness reward (0.75-0.85 weight) reliably.

### Sub-call Count Bonus (V6: +0.10 max)
Rewards more sub-calls, but the template already makes sub-calls (8-12 on 150K+ docs). It doesn't reward DIFFERENT sub-call patterns.

## Proposed Solution: Strategy-Conditioned Training (SC-GRPO)

### The Insight
Mode collapse happens in **code space** because code generation is deterministic — there's usually one "obviously correct" template. But there are MANY valid strategies for processing long documents:

1. **Sequential chunking** — the template everyone learns
2. **Extract-then-compute** — extract raw data, compute in Python
3. **Binary search** — narrow down to relevant section
4. **Map-reduce** — map each chunk to structured output, reduce
5. **Two-pass verification** — scan, then re-read candidates
6. **Decomposition** — break question into sub-questions
7. **Parallel extraction** — extract different aspects simultaneously
8. **Sampling** — randomly sample sections, extrapolate

### How SC-GRPO Works

For each training prompt:
1. **Randomly assign a strategy to each trajectory** in the K-group
2. Each trajectory sees a **strategy-augmented system prompt** (base prompt + strategy description)
3. Different strategies → different code patterns → variance in rewards → learning signal
4. The model learns: "When asked to do X, code pattern Y works well"

Key difference from standard GRPO: diversity is injected via **prompt space**, not temperature space.

### Advantage Computation
Standard GRPO: A_i = (r_i - mean(r_all)) / std(r_all)
SC-GRPO: Same formula. The strategies are just different samples from the policy π(code | prompt, strategy). Better strategies get higher advantage.

### Test-Time Inference
Option A: Use the base system prompt (no strategy). Model has learned diverse patterns and picks the best one for the task.
Option B: Try N strategies in parallel, pick the best answer (Best-of-N). This is N× more expensive but more reliable.
Option C: Train a lightweight strategy selector that picks the best strategy based on task metadata.

## Implementation Plan

### V7 Training Script
Based on V6, add:

```python
STRATEGY_PROMPTS = {
    "standard": QWEN35_35B_SYSTEM_PROMPT,
    "extract_compute": QWEN35_35B_SYSTEM_PROMPT + "\n\nPrefer extracting raw data then computing in Python.",
    "binary_search": QWEN35_35B_SYSTEM_PROMPT + "\n\nPrefer binary search to narrow down location.",
    "map_reduce": QWEN35_35B_SYSTEM_PROMPT + "\n\nPrefer map-reduce: map chunks to structured output, reduce in Python.",
    "two_pass": QWEN35_35B_SYSTEM_PROMPT + "\n\nPrefer two-pass: scan all chunks first, then re-read candidates.",
}

# In trajectory generation:
for k in range(K):
    strategy = random.choice(list(STRATEGY_PROMPTS.keys()))
    system_prompt = STRATEGY_PROMPTS[strategy]
    traj = rlm(prompt, model, system_prompt, ...)
```

### Strategy-Task Compatibility Matrix
Some strategies work better for certain tasks:

| Strategy | Good for | Bad for |
|----------|----------|---------|
| Sequential chunking | NIAH, classification | Counting, ratio |
| Extract-then-compute | Counting, ratio, per-entity | NIAH |
| Binary search | NIAH, first/last | Classification, counting |
| Map-reduce | Classification, per-entity | NIAH |
| Two-pass | Multi-hop, hard_multi_hop | Simple search |
| Decomposition | Multi-hop, hard_multi_hop | Single-fact tasks |

### Expected Impact
1. **Mode collapse delayed or eliminated** — different strategies guarantee structural diversity
2. **Better task-specific performance** — model learns the right strategy for each task
3. **Publishable novelty** — no prior work on strategy-conditioned GRPO for code generation

## Alternative: Teacher Distillation + SC-GRPO

Best approach may be two-stage:
1. **SFT on teacher trajectories** from 397B-A17B (diverse strategies)
2. **SC-GRPO** on the SFT checkpoint (refine strategy selection)

The teacher provides initial strategy diversity. SC-GRPO refines which strategies work best for which tasks.

## Risk: Strategy Prompt Leakage

If the strategy prompt becomes a crutch, the model may ONLY generate good code when given a strategy hint. Mitigation:
- Include "no strategy" (standard prompt) in the strategy pool
- Gradually reduce strategy frequency during training (curriculum)
- Evaluate without strategy prompts to verify generalization

## For the Paper

This addresses a fundamental gap in code RL:
"Standard GRPO applied to code generation suffers from rapid mode collapse because code is more deterministic than natural language. We introduce Strategy-Conditioned GRPO (SC-GRPO), which injects diversity through the prompt space rather than the temperature space, enabling continued learning throughout training."
