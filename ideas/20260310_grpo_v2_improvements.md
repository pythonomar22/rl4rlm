# GRPO v2 Improvements
**Date:** 2026-03-10

## Issues Observed in v1

### 1. Doc-Classify Format Problem
The model classifies only 1 document and calls FINAL immediately, instead of all N.
- Out of ~20 doc_classify attempts, only ~25% classify all documents
- Caused by: model sees first document, classifies it, and terminates
- RL signal is correct (low reward for partial classification)
- But need many steps for the model to learn "classify ALL documents"

### 2. Cross-Entropy ≠ True GRPO
Current approach uses cross_entropy with positive-advantage weighting.
This is essentially rejection sampling / REINFORCE with positive advantages only.
- Missing: negative advantages (down-weight bad trajectories)
- Missing: importance sampling ratio (old_logprobs / new_logprobs)
- Fix: Switch to `importance_sampling` loss with logprob capture

### 3. Slow Trajectory Collection
~16 min per step with batch_size=3, K=8. 30 steps = 8 hours.
- Doc-classify tasks are slowest (many sub-calls)
- Consider: parallelizing trajectory collection? Tinker supports async sampling.

## Proposed v2 Changes

### A. True Importance Sampling GRPO
1. Enable `capture_logprobs=True` in TinkerModel
2. Store per-token logprobs with each trajectory turn
3. Use `importance_sampling` loss function:
   ```python
   datum = tinker.Datum(
       model_input=...,
       loss_fn_inputs={
           "target_tokens": ...,  # Generated tokens
           "logprobs": ...,       # Logprobs from sampling time
           "advantages": ...,     # Can be negative!
       }
   )
   training_client.forward_backward([datum], "importance_sampling")
   ```
4. This allows both positive AND negative advantages

### B. Curriculum on Doc-Classify
1. Start with 5-doc tasks (easier to get right)
2. Only add 10-doc, 15-doc after model learns format
3. Or: add explicit format instructions to the system prompt

### C. Task-Specific Rewards
Adjust reward function:
- Doc-classify: Higher weight on "attempted all docs" (format reward)
- NIAH: bonus for finding in first chunk (efficiency)
- Multi-NIAH: bonus for completeness (recall)

### D. More Aggressive Weight Refresh
- Refresh sampling client every step (not every 3)
- Costs ~10s per refresh but ensures latest policy generates trajectories

### E. Parallel Trajectory Collection
- Use Tinker's async sampling to generate K trajectories in parallel
- Would reduce step time from ~16 min to ~2-3 min
- But: RLM scaffold runs sequentially (REPL execution is serial)
- Alternative: pre-generate tasks, run multiple GRPO workers

## Priority

1. **A (importance_sampling)** — most impactful for learning quality
2. **D (weight refresh)** — easy change, direct benefit
3. **B (curriculum)** — helps doc-classify specifically
4. **C (task rewards)** — fine-tuning
5. **E (parallel)** — optimization, lowest priority
