# Deep Audit: Critical Training Pipeline Issues

Date: 2026-03-11

## Executive Summary

After deep-diving into the training code, Tinker cookbook reference implementation, loss function documentation, training logs, and sample trajectories, we identified **7 critical issues** that collectively explain why our best model only achieves +4% overall (and regresses on 4/11 benchmarks). These are not incremental improvements — they are fundamental bugs and design flaws that, once fixed, should produce a dramatically better model.

## CRITICAL BUG 1: Wrong Loss Function (importance_sampling → ppo)

**Impact: HIGHEST — explains the unstable training**

Our V7 training uses `importance_sampling` loss (Tinker's unclipped IS loss). The Tinker docs confirm: **"No clipping is applied"** to the IS ratio π_new/π_old.

The Tinker cookbook's `rl_loop.py` uses `importance_sampling` because it resamples every single step — the sampling policy always matches the training policy. But our pipeline collects K=8 trajectories, computes advantages, then does `forward_backward` + `optim_step`. By the time we process the last datums in a step, the policy has already been updated by earlier `forward_backward` calls (gradient accumulation). The IS ratio diverges.

**Evidence:**
- V4 losses: 107, -62, 497, -35, -70 (reasonable, policy drift is small)
- V7 losses: 4,336,545, -933,989, 11,516,850 (**100,000x larger, policy drift is catastrophic**)
- V7 has more datums per step (48-55 vs 16-40 in V4), more gradient accumulation steps, and SC-GRPO produces more diverse code — all of which increase policy drift

**Fix:** Switch to `ppo` loss with clipping:
```python
training_client.forward_backward(datums, "ppo")
```
PPO clips the IS ratio to [1-ε, 1+ε], preventing gradient explosions. Tinker supports `clip_low_threshold` and `clip_high_threshold` for asymmetric clipping (DAPO-style).

## CRITICAL BUG 2: Sub-Query Temperature Leak

**Impact: HIGH — corrupts reward signal by making sub-calls noisy**

File: `scaffold/llm_query.py`, line 208

```python
def sub_query(self, prompt_str):
    params = self._tinker.SamplingParams(
        temperature=self.temperature,  # BUG: uses ROOT generation temperature
    )
```

During training, root generation temperature is set per-trajectory: `model.temperature = temp_schedule[k]` where `temp_schedule = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]`.

Sub-calls are factual extraction queries ("Find X in this chunk"). They need LOW temperature (0.1-0.3) for reliable extraction. At T=1.3, sub-calls hallucinate, producing noisy/wrong extractions. This causes correct RLM code to fail, teaching the model "this approach doesn't work" when the issue is sub-call quality, not code quality.

**Fix:** Use a fixed low temperature for sub-calls:
```python
def sub_query(self, prompt_str):
    sub_temp = getattr(self, 'sub_temperature', 0.3)
    params = self._tinker.SamplingParams(temperature=sub_temp, ...)
```

## CRITICAL BUG 3: Advantage Std Normalization (Not in Cookbook)

**Impact: HIGH — amplifies noisy gradients**

Our code (rl_tinker_v6.py:1025):
```python
advantage = (reward - mean_r) / (std_r + 1e-8)
```

Tinker cookbook (rl_loop.py:192):
```python
advantages_G = [reward - mean_reward for reward in rewards_G]
```

The cookbook does NOT divide by std. Our normalization amplifies small differences. For a group with rewards [0.72, 0.73, 0.71, ...] (std=0.007), the normalized advantage for 0.73 is (0.73-0.72)/0.007 ≈ 1.43, vs raw 0.01. Combined with the unclipped IS loss, this produces enormous gradients.

**Fix:** Remove std normalization:
```python
advantage = reward - mean_r
```

Or use the Dr. GRPO approach (arXiv:2503.20783): remove both std normalization AND length normalization.

## CRITICAL ISSUE 4: System Prompt Teaches Single-Pass Pattern

**Impact: HIGH — root cause of hard task regression**

4 out of 5 examples in the system prompt use `results[0]` (take first match). The model learns "find the first hit and return it" instead of "decompose, search iteratively, aggregate."

Evidence from V7 step 5 trajectories: on hard_multi_hop tasks, the model asks compound questions ("Find the project led by Rachel Robinson and its headquarters") as single llm_query calls. No chunk contains both facts, so it gets "Not found" or picks up a distractor.

**Fix:** Rewrite system prompt:
1. Remove `results[0]` from Example 1 — use `results[-1]` or aggregation
2. Make Example 4 (decomposition) more prominent — show it as the DEFAULT approach
3. Add Example 6: aggregation with Python counting (no llm_query delegation)

## IMPORTANT BUG 5: Training/Eval Seed Overlap

Training uses `seed_offset = step * 1000`, producing seeds 0-29000.
Evaluation uses `seed_offset = 10000`.
Steps 10-19 overlap with evaluation seeds. Some eval tasks may have been trained on.

**Fix:** Use `seed_offset = step * 1000 + 50000` for training (separate namespace).

## IMPORTANT ISSUE 6: OOLONG is OOD — Never Seen in Training

All post-trained models score 0% on OOLONG (base: 20%). The model has never trained on unstructured aggregation from natural language. Event counting uses structured `[EVENT #N]` markers, which is a completely different extraction challenge from D&D transcript analysis.

**Fix:** Add OOLONG-style training tasks with unstructured text (no markers). Or at minimum, include OOLONG in the training mix at 5% weight.

## IMPORTANT ISSUE 7: Reward/Scoring Mismatches

- `binary_reward` (rewards.py:23) uses `expected.lower() in predicted.lower()` — substring match. Expected "3" matches "13", "300", etc.
- Event counting gives 0.7 partial credit for being within 10% — rewards "almost right" counting
- Code debug scoring only checks function name presence, not actual bug identification
- Notebook QA format mismatch: "0.87" vs "87.0%" scored as wrong

These are less critical because `score_trajectory` dispatches to per-benchmark scorers, not `binary_reward`. But the partial credit in event counting inflates training rewards.

## Recommended V9 Training Plan

1. **Switch to `ppo` loss** with clip_high=0.28, clip_low=0.2 (DAPO-style asymmetric)
2. **Fix sub-query temperature** to 0.3
3. **Remove advantage std normalization**
4. **Rewrite system prompt** with decomposition as default pattern
5. **Add seed separation** (training seeds offset +50000)
6. **Use hybrid architecture during training** (base model for sub-calls)
7. **Add self-imitation loss** (VAPO-style: extra cross_entropy on successful trajectories)
8. **Increase timeout** for long documents (120s + 90s per 100K chars)

Expected impact: The loss function fix alone should eliminate training instability. Combined with temperature and advantage fixes, we should see stable training for 20+ steps (vs current 10-step collapse). The prompt fix should directly address hard task regression.
