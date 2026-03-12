# RS-SFT vs GRPO: Breaking the Zero-Sum Tradeoff

**Date:** 2026-03-12

## The Problem

GRPO training creates a zero-sum specialization-generalization tradeoff:
- V4-s5: +0.5pp average (5 improved, 5 regressed, 4 tied)
- V9-s10: -0.2pp average (5 improved, 5 regressed, 4 tied)
- Training gains and losses cancel to approximately ZERO

Root causes (from format_rigidity_analysis.md):
1. RL rewards speed over correctness -> model uses larger chunks, misses boundary entries
2. RL converges on rigid format-specific parsing templates that fail on format variation
3. Policy gradient pushes toward task-specific optima that are incompatible across tasks
4. Negative gradients push model AWAY from good base patterns (loose parsing, multi-step decomposition)

## Why RS-SFT Should Help

GRPO trains on ALL trajectories (good AND bad), weighting by advantage:
- Good trajectories get positive gradient (reinforce these patterns)
- Bad trajectories get NEGATIVE gradient (suppress these patterns)

The problem: negative gradients are where the damage happens. When the model produces a
"bad" trajectory that happens to use the SAME code pattern as the base model's good pattern
for a different task, the negative gradient suppresses that pattern globally.

**RS-SFT eliminates negative gradients entirely.** It only trains on the BEST trajectories
with standard cross-entropy loss. The model can only learn to reproduce exemplary behavior,
never to avoid anything. This should preserve base model patterns for task types not in the
training set.

## Why RS-SFT Specifically Addresses Format Rigidity

The format rigidity problem occurs because RL rewards code that "works on average" even if
it uses brittle format-specific parsing. RS-SFT with a quality filter can:

1. **Filter for format-robust code**: Only include trajectories where the code uses loose
   parsing (line splitting, regex) rather than strict format parsing ("ORG: Name")
2. **Filter for multi-step decomposition**: Only include trajectories that use 2+ turns
   for complex tasks (ensuring the model learns decomposition, not shortcuts)
3. **Balance across task types**: Equal-weight training data prevents any task type from
   dominating the loss landscape
4. **Higher quality threshold**: Only include trajectories with score >= 0.9 (near-perfect)
   rather than GRPO's implicit "better than group average" criterion

## Concrete Plan

### Phase 1: Large-Scale Trajectory Collection

**Goal:** Collect many trajectories from the BASE model across all 14 benchmarks.

**Scale calculation:**
- 14 benchmarks
- 30 tasks per benchmark = 420 unique tasks
- K=8 attempts per task = 3,360 total trajectories
- Expected success rate: ~60% (base model average) = ~2,016 correct trajectories
- After quality filtering (~50% pass rate): ~1,000 high-quality SFT samples

**Cost estimate:**
- Each trajectory: ~4 root turns x 1 root call + 3 sub-calls = ~4 API calls
- 3,360 trajectories x 4 calls = ~13,440 Tinker API calls
- At ~10s per call = ~37 hours (can be parallelized)
- With Tinker's async pipeline: ~8-12 hours wall clock

**Alternative: Teacher distillation**
Use Qwen3.5-397B-A17B to generate gold trajectories for tasks where the base model fails.
This gives us correct trajectories for the hardest tasks, which the base model cannot produce.

### Phase 2: Multi-Criteria Quality Filtering

Filter criteria (all must pass):
1. **Correct answer**: score >= 0.9 (not just > 0)
2. **Proper termination**: FINAL/FINAL_VAR called
3. **Multi-turn for complex tasks**: hard_multi_hop, cross_doc_compare, multi_hop_qa
   must have 2+ turns (prevents shortcut learning)
4. **Uses llm_query**: At least 1 sub-call (prevents memorization / direct-answer)
5. **Clean code**: No errors, no syntax issues
6. **Format-robust parsing**: Code does NOT contain strict format patterns
   (e.g., no `if "ORG: " in line: parts = line.split("ORG: ")`)
7. **Reasonable chunk size**: Chunks between 5K-25K chars (prevents oversized chunks)
8. **No f-string bugs**: No `llm_query("{context}")` without f-string

### Phase 3: Task-Balanced SFT on Tinker

**Architecture:**
- Use Tinker's `forward_backward` with `cross_entropy` loss
- Per-token weights from `renderer.build_supervised_example`
- Equal samples per task type (oversample rare tasks, undersample common ones)

**Hyperparameters:**
- LoRA rank: 32 (same as GRPO)
- LR: 2e-5 (10x GRPO LR -- SFT needs higher LR, see TINKER.md recommendation)
- Epochs: 3 (over the filtered dataset)
- Batch size: 4 with gradient accumulation of 4 = effective batch 16
- Linear warmup for 10% of steps, then cosine decay to 10% of peak LR

### Phase 4: Evaluation

Compare:
1. Base model (no training)
2. RS-SFT model (this plan)
3. V4-s5 GRPO model
4. RS-SFT -> GRPO (SFT warmstart then 5 steps GRPO)
5. RS-SFT + strategy prompts

The key metric: **does RS-SFT avoid the regression pattern?**
- We expect: improvements on NIAH, doc_classify, event_counting (same as GRPO)
- Critical test: does cross_doc, DFQA, notebook_qa, multi_hop ALSO stay at or above base?

## Approach Variants

### A. Pure RS-SFT (Recommended First Try)
Collect -> Filter -> SFT. Simple, predictable.

### B. Expert Iteration / Iterated RS-SFT (STaR)
1. Collect from base, filter, SFT -> checkpoint v1
2. Collect from v1 on HARDER tasks, filter, merge with round 1 data
3. SFT on merged data -> checkpoint v2
4. Repeat 2-3 rounds

This is the original STaR approach. Advantage: each round generates trajectories
for tasks the previous round couldn't solve.

### C. RS-SFT + Light RL (Best of Both Worlds)
1. RS-SFT first (Phase 1-3 above)
2. Then 3-5 steps of GRPO with very high KL penalty (0.05-0.1)
3. The SFT checkpoint starts much closer to correct behavior
4. GRPO fine-tunes the edges without the destructive mode collapse

### D. Online Rejection Sampling During RL
Modify GRPO to only update on POSITIVE-advantage trajectories:
- Generate K trajectories per task
- Compute advantages
- DISCARD all negative-advantage trajectories (don't backprop on them)
- Only train on above-average trajectories with cross_entropy loss
- This is GRPO minus the "push away from bad" component

### E. DPO on Trajectory Pairs
For each task, pair the best trajectory with the worst:
- Chosen: highest-reward trajectory (correct, clean code)
- Rejected: lowest-reward trajectory (incorrect or messy code)
- DPO loss learns to prefer good trajectories
- DPO is more stable than GRPO but still uses negative signal

## Why RS-SFT over DPO

DPO still uses negative signal (the rejected trajectory). The format rigidity problem
comes from the model learning to AVOID certain patterns. DPO would still push the model
away from base-model patterns when those patterns appear in rejected trajectories.

RS-SFT only learns to REPRODUCE good patterns. It never sees what to avoid. This makes
it theoretically impossible for RS-SFT to cause the format rigidity regression.

## Risk Assessment

**Risks of RS-SFT:**
1. Lower ceiling than RL -- SFT can only learn what the base model already does well
   (mitigated by teacher distillation for hard tasks)
2. Data quality is critical -- bad filtering ruins everything
3. May not improve on tasks where base model already scores >90%
   (ceiling effect -- but this is fine, we just want no regression)
4. Overfitting on small dataset -- mitigate with early stopping, low epoch count

**Risks of staying with GRPO:**
1. Zero-sum tradeoff is fundamental to the algorithm
2. More training steps = more divergence from base (V9-s15 declining)
3. Strategy prompts mask but don't fix the underlying regression
4. Format rigidity worsens with more RL training

## Implementation Files

- `training/sft_tinker.py` -- NEW: SFT training on Tinker API (replaces local sft.py)
- `scripts/collect_trajectories_tinker.py` -- NEW: Trajectory collection using Tinker
- `scripts/filter_rs_sft.py` -- NEW: Enhanced filtering for RS-SFT (code quality checks)
- Modifies: `training/rl_tinker_v6.py` (add online RS variant as flag)

## Recommendation

**Start with Approach A (pure RS-SFT)**. It's the simplest, most predictable, and directly
tests the hypothesis that negative gradients cause regressions.

If RS-SFT shows improvement without regression, proceed to Approach C (RS-SFT + light RL)
to push the ceiling higher.

If RS-SFT shows no improvement at all, the problem isn't negative gradients --
it's something else about the training distribution, and we should try Approach B
(iterated STaR with harder tasks).
