# RL Training Improvements Survey (2025-2026)

**Date:** 2026-03-12
**Goal:** Find techniques to break the zero-sum tradeoff (improving some tasks without hurting others).

## Executive Summary

Our core problem: training on task X improves X but regresses Y. The average stays flat (+0.5pp for V4-s5, -0.2pp for V9-s10). This survey identifies the most promising techniques to fix this, ranked by expected impact and implementation difficulty.

---

## TOP PRIORITY: Techniques Most Likely to Break the Zero-Sum Tradeoff

### 1. Turn-Level Credit Assignment (MT-GRPO / Turn-PPO / Bi-Level GAE)

**What it is:** Instead of assigning the same advantage to all tokens in a multi-turn trajectory, compute per-turn advantages. Each turn (code block + execution result) gets its own credit.

**Why it matters for us:** Our RLM trajectories are multi-turn by nature (3-8 turns of code + execution). Currently, if the final answer is wrong, ALL turns get negative advantage -- even the turns that correctly chunked the document. This teaches the model to avoid correct patterns just because a later turn failed.

**Papers:**
- Turn-PPO (arXiv:2512.17008) -- turn-level MDP, learnable turn critic
- MT-GRPO (NeurIPS 2025) -- extends GRPO with turn-level intermediate rewards
- Bi-Level GAE (VAGEN, 2025) -- hierarchical: turn-level advantage -> token-level propagation
- Segmental Advantage Estimation (arXiv:2601.07320, Jan 2026) -- partitions sequence into semantic segments, computes advantages at boundaries only

**Applicability:** HIGH. Our trajectories naturally segment into turns (each `[code]...[/code]` + execution is a turn). We can compute intermediate rewards per turn:
- Did the code execute without error? (+0.1)
- Did it call llm_query on a reasonable chunk size? (+0.05)
- Did it extract structured data? (+0.05)
- Did it call FINAL()? (+0.1)

**Implementation difficulty:** MEDIUM. Requires modifying advantage computation in `rl_tinker_v6.py`. The trajectory already has turn structure. Main work: define per-turn reward function, modify `trajectory_to_training_data_is()` to accept per-turn advantages.

**Expected improvement:** +3-5pp. This directly addresses format rigidity regression by not penalizing good chunking code when only the final extraction fails.

### 2. Experience Replay / Off-Policy Buffer (BAPO)

**What it is:** Maintain a replay buffer of successful trajectories. Mix on-policy rollouts with replayed successes. Re-evaluate hard prompts periodically.

**Why it matters for us:** Our training wastes data massively. When the model fails on cross_doc_compare (reward=0 for all K), we skip the group entirely. With a replay buffer, we'd have stored successful cross_doc trajectories from earlier steps and could learn from those.

**Papers:**
- Buffer Matters / BAPO (arXiv:2602.20722, Feb 2026) -- 12.5% improvement over GRPO
- RLEP (arXiv:2507.07451) -- two-phase: collect verified trajectories, then replay during training
- RRL (Retrospective Replay) -- dynamic replay with value-based state selection

**Applicability:** HIGH. We already have trajectory logs from all training runs. We could build a buffer of successful trajectories per task type.

**Implementation difficulty:** MEDIUM-LOW. Store successful trajectories (reward > 0.7) in a JSON buffer. Each step, mix 70% on-policy + 30% replayed. For replayed trajectories, use importance sampling correction.

**Expected improvement:** +3-5pp. Directly prevents regression by maintaining exposure to successful patterns from all task types.

### 3. Dr. GRPO Fixes (Already Partially Implemented)

**What it is:** Remove response-length normalization bias and std normalization from advantages.

**Why it matters:** We already use raw advantages (reward - mean) without std normalization. But we're NOT fixing the response-length bias. Our GRPO loss divides by sequence length, which means long incorrect trajectories (common in RLM -- the model generates lots of code but gets the wrong answer) get smaller penalties.

**Papers:**
- Dr. GRPO (arXiv:2503.20783, COLM 2025) -- remove std normalization + length normalization

**Applicability:** PARTIALLY DONE. We removed std normalization. Need to check if Tinker's `importance_sampling` or `ppo` loss internally normalizes by length.

**Implementation difficulty:** LOW. Check Tinker's loss computation, potentially switch to fixed-constant normalization.

**Expected improvement:** +1-2pp. Prevents the model from learning to generate long verbose failing code.

### 4. DAPO-Style Decoupled Clipping (Clip-Higher)

**What it is:** Use asymmetric clipping: [1-0.2, 1+0.28] instead of symmetric [1-0.2, 1+0.2]. The higher upper bound allows low-probability tokens (novel code patterns) to gain probability mass.

**Why it matters for us:** Our model converges to a single coding template (20K chunks, format-specific parsing). Clip-higher would let alternative patterns (smaller chunks, loose extraction) gain probability without being clipped out.

**Papers:**
- DAPO (arXiv:2503.14476) -- 50 pts on AIME 2024 with Qwen2.5-32B
- Clip-Low/Clip-High analysis (arXiv:2509.26114)

**Applicability:** HIGH. We already have `clip_high` and `clip_low` parameters in our code but use them for advantage scaling, not ratio clipping. We'd need to apply this to the IS ratio clipping in Tinker.

**Implementation difficulty:** LOW-MEDIUM. If using Tinker's `importance_sampling` loss, we need to check if we can set asymmetric clip bounds. If not, we can implement the clipping ourselves before passing to Tinker.

**Expected improvement:** +2-3pp. Directly combats mode collapse / format rigidity.

---

## MEDIUM PRIORITY: Solid Improvements, Moderate Effort

### 5. Critical Token KL (CT-KL)

**What it is:** Instead of uniform KL penalty across all tokens, reduce/remove KL for "critical tokens" (decision points like function calls, variable names, format strings) while keeping KL high for boilerplate.

**Why it matters for us:** Our KL penalty is broken (`|mean_logprob - ref|` -- not even real KL). Even with proper KL, uniform penalty prevents the model from learning new extraction patterns while allowing it to drift on irrelevant tokens.

**Paper:** "Ignore the KL Penalty!" (NAACL 2025 Findings, arXiv:2502.06533) -- up to 8.2% improvement on math reasoning.

**Applicability:** HIGH. In RLM, the critical tokens are: `llm_query(`, chunk boundaries, `FINAL(`, format parsing patterns. Non-critical: print statements, variable names, comments.

**Implementation difficulty:** MEDIUM-HIGH. Need to identify critical tokens (could use reward difference when masking tokens, or simpler heuristic: tokens in lines containing `llm_query`, `FINAL`, parsing logic). Then apply per-token KL weighting.

**Expected improvement:** +2-3pp. Would allow the model to explore better extraction patterns while staying close to base on general code.

### 6. Reinforce-Rej (Minimalist Alternative to GRPO)

**What it is:** Filter out groups where ALL trajectories succeed (nothing to learn) AND groups where ALL fail (no positive signal). Only train on groups with mixed outcomes.

**Why it matters for us:** We already skip all-same-reward groups. But RAFT (training only on positives) is competitive with GRPO, and the key insight is that GRPO's benefit comes from the implicit filtering, not the reward normalization.

**Paper:** "A Minimalist Approach to LLM Reasoning" (arXiv:2504.11343) -- RAFT matches GRPO; Reinforce-Rej improves KL efficiency.

**Applicability:** MEDIUM. We already do partial filtering. The insight is that we should also be more aggressive: even within mixed groups, weight the positive examples more heavily than negative ones.

**Implementation difficulty:** LOW. Modify the advantage computation to weight positive examples 2x vs negative.

**Expected improvement:** +1-2pp. Better KL efficiency means less drift, less regression.

### 7. Forward KL / JS Divergence Instead of Reverse KL

**What it is:** Replace reverse KL (mode-seeking, collapses diversity) with forward KL (mass-covering, preserves diversity) or a mixture (JS divergence).

**Why it matters for us:** Our model collapses to single coding templates. Reverse KL actively accelerates this by narrowing the policy. Forward KL would preserve the base model's diverse code generation patterns.

**Paper:** DPH-RL (2025) -- "The Choice of Divergence" (OpenReview) -- mass-covering f-divergences as rehearsal mechanism.

**Applicability:** MEDIUM. Our current "KL" is `|mean_logprob - ref|` which isn't even a proper divergence. Switching to proper token-level forward KL would be a significant improvement.

**Implementation difficulty:** MEDIUM. Need per-token reference logprobs (requires a forward pass through the reference model via Tinker). Then compute `sum(p_ref * log(p_ref/p_policy))` instead of `sum(p_policy * log(p_policy/p_ref))`.

**Expected improvement:** +2-3pp on regression prevention. Would directly address format rigidity by maintaining diverse coding patterns.

### 8. Self-Evolving Curriculum (SEC)

**What it is:** Automatically adjust task distribution based on model's current capabilities. Tasks where the model is improving fast get less weight; tasks where it's stagnating get more.

**Why it matters for us:** Our task distributions are manually set (v9, v10, v11 dists). When cross_doc declines mid-training, we don't notice until evaluation. SEC would automatically up-weight declining tasks.

**Paper:** Self-Evolving Curriculum (arXiv:2505.14970) -- dynamic curriculum with advantage-based weighting.

**Applicability:** HIGH. We already track per-task rewards in training logs. Could compute per-task learning gain (current vs. rolling average) and adjust sampling weights.

**Implementation difficulty:** MEDIUM. Add a `TaskScheduler` class that tracks per-task success rates and adjusts sampling probabilities each step. Simple version: inverse-success-rate weighting.

**Expected improvement:** +2-4pp. Directly prevents the "improve A, regress B" pattern by dynamically balancing training signal.

---

## LOWER PRIORITY: Good Ideas, Higher Effort or Less Certain Impact

### 9. Process Reward Model for Code (CodePRM / FunPRM)

**What it is:** Train a reward model that scores individual code steps (not just final answer correctness).

**Why it matters:** Would provide dense rewards for multi-turn training, not just sparse 0/1 at the end.

**Papers:**
- CodePRM (ACL 2025 Findings) -- execution feedback-enhanced PRM
- FunPRM (arXiv:2601.22249, Jan 2026) -- function-as-step PRM with meta-reward correction

**Applicability:** MEDIUM. Would be ideal but requires training a separate reward model, which is a significant effort.

**Implementation difficulty:** HIGH. Need to collect step-level labels, train a PRM, integrate into the training loop.

**Expected improvement:** +3-5pp if done well, but high effort.

### 10. Elastic Weight Consolidation (EWC) for LoRA

**What it is:** After training on task A, compute Fisher information matrix to identify important parameters. When training on task B, penalize changes to those parameters.

**Why it matters:** Would prevent the model from forgetting cross_doc patterns when training on niah patterns.

**Papers:** EWC + Gemma2 (arXiv:2505.05946, 2025) -- EWC for continual pre-training.

**Applicability:** LOW-MEDIUM. LoRA has few parameters, so Fisher computation is cheap. But: research shows LoRA already fails to prevent forgetting despite parameter isolation, so EWC on LoRA may have limited benefit.

**Implementation difficulty:** MEDIUM. Need to compute Fisher information after each task type, add regularization term to loss.

**Expected improvement:** +1-2pp, uncertain.

### 11. Training-Free GRPO (Test-Time Optimization)

**What it is:** No parameter updates. Instead, use GRPO-style scoring at inference time to select the best output from multiple candidates.

**Paper:** arXiv:2510.08191 -- 100 training samples, competitive with fine-tuning a 32B model.

**Applicability:** LOW for training (we want to train), but HIGH for inference. Could use at eval time as a free boost.

**Implementation difficulty:** LOW. Generate K outputs, score with GRPO-style ranking, return best. Already partially implemented via our strategy prompts.

**Expected improvement:** +2-3pp at eval time only (essentially best-of-K with learned scoring).

---

## Specific Fixes for Our Known Issues

### Fix 1: KL Penalty Is Broken

**Current code (line 1117-1122):**
```python
kl_approx = abs(traj["mean_logprob"] - ref_mean_logprob)
traj["reward"] -= kl_coeff * kl_approx
```

This is NOT KL divergence. It's the absolute deviation of mean logprob from a reference. It:
- Doesn't account for per-token distribution shift
- Uses mean logprob (average over all tokens) instead of per-token KL
- Uses absolute value instead of the actual KL formula

**Fix options (ranked by effort):**
1. **Drop KL entirely** (Dr. GRPO finding: KL is unnecessary for verifiable reward tasks). Just remove the penalty. Effort: 5 minutes.
2. **Use proper token-level KL**: Get reference model logprobs from Tinker `forward_backward`, compute `sum(exp(ref_lp) * (ref_lp - policy_lp))` per token. Effort: 2-3 hours.
3. **Use CT-KL**: Proper KL but only on non-critical tokens. Effort: 4-6 hours.

**Recommendation:** Option 1 (drop KL). Dr. GRPO and DAPO both found KL unnecessary. Our kl_coeff=0.01 is tiny anyway.

### Fix 2: Advantage Computation Ignores Turn Structure

**Current:** All tokens in a trajectory get the same advantage (trajectory-level reward - group mean).

**Fix:** Compute per-turn intermediate rewards, then per-turn advantages. Pseudo-code:
```python
turn_rewards = []
for turn in trajectory["turns"]:
    r = 0.0
    if turn["execution_success"]: r += 0.1
    if "llm_query" in turn["code"]: r += 0.05
    if "FINAL" in turn["code"]: r += outcome_reward
    turn_rewards.append(r)

# Discount future rewards back
for i in reversed(range(len(turn_rewards) - 1)):
    turn_rewards[i] += 0.95 * turn_rewards[i+1]

# Per-turn advantage = turn_reward - group mean for that turn position
```

### Fix 3: No Replay of Successful Patterns

**Current:** Each step generates fresh trajectories. If the model loses a skill (e.g., cross_doc two-pass), there's no mechanism to recover it.

**Fix:** Maintain a `replay_buffer.json` per task type:
- After each step, store trajectories with reward > 0.8
- Each step, 30% of training data comes from replay buffer
- Apply importance sampling correction for off-policy data
- Periodically evict old entries (keep most recent 100 per task type)

---

## Comparison with Prime Intellect / SkyRL Approaches

**Prime Intellect** uses their `verifiers` library with `RLMEnv` for training. Key differences:
- They use the verl framework for distributed training
- Their `RLMEnv` wraps the REPL environment for RL
- They've released INTELLECT-3 (100B+ MoE) trained with their RL stack

**SkyRL (Berkeley)** provides:
- Async dispatching for multi-turn environments (1.55x speedup)
- Integration with Tinker via `skyrl-tx`
- Multi-turn tool-use training pipeline
- Turn-level reward aggregation

**Key lesson:** Both groups emphasize turn-level credit assignment and async training for multi-turn tasks. We should adopt turn-level advantages.

---

## Recommended Implementation Order

1. **Drop broken KL penalty** (5 min, removes noise from training signal)
2. **Add experience replay buffer** (4-6 hours, prevents regression on trained tasks)
3. **Implement turn-level intermediate rewards** (6-8 hours, better credit assignment)
4. **Add self-evolving curriculum** (4-6 hours, dynamic task weighting)
5. **Implement DAPO-style clip-higher** (2-4 hours, prevents mode collapse)
6. **Proper forward KL if needed** (4-6 hours, only if regression persists)

Total estimated effort: 2-3 days of development for items 1-5, which should collectively provide +5-10pp improvement while breaking the zero-sum tradeoff.

---

## Sources

- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO: Correcting Token Aggregation Bias](https://arxiv.org/abs/2503.20783)
- [Turn-PPO: Turn-Level Advantage Estimation](https://arxiv.org/abs/2512.17008)
- [Segmental Advantage Estimation](https://arxiv.org/abs/2601.07320)
- [Reinforcing Multi-Turn Reasoning via Turn-Level Credit Assignment](https://arxiv.org/pdf/2505.11821)
- [A Practitioner's Guide to Multi-turn Agentic RL](https://openreview.net/pdf?id=yPWJG9wgll)
- [A Minimalist Approach: Rejection Sampling to Reinforce](https://arxiv.org/abs/2504.11343)
- [Ignore the KL Penalty! Critical Token KL](https://arxiv.org/abs/2502.06533)
- [Buffer Matters: Off-Policy RL for LLM Reasoning](https://arxiv.org/html/2602.20722v1)
- [RLEP: RL with Experience Replay for LLM Reasoning](https://arxiv.org/html/2507.07451v1)
- [Self-Evolving Curriculum for LLM Reasoning](https://arxiv.org/pdf/2505.14970)
- [Training-Free GRPO](https://arxiv.org/abs/2510.08191)
- [The Choice of Divergence: Mitigating Diversity Collapse](https://openreview.net/forum?id=xPEsxcO7F7)
- [CodePRM: Execution Feedback-enhanced PRM](https://aclanthology.org/2025.findings-acl.428/)
- [FunPRM: Function-as-Step PRM](https://arxiv.org/html/2601.22249v1)
- [ThinkPRM: Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [EWC for Continual Pre-Training of Gemma2](https://arxiv.org/html/2505.05946v1)
- [CURLoRA: Stable LLM Continual Fine-Tuning](https://arxiv.org/html/2408.14572v1)
- [Continual Learning with RL for LLMs](https://cameronrwolfe.substack.com/p/rl-continual-learning)
- [RLAR: Agentic Reward System for Multi-task RL](https://arxiv.org/html/2603.00724)
- [SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent](https://arxiv.org/abs/2511.16108)
- [Prime Intellect RLM Blog](https://www.primeintellect.ai/blog/rlm)
- [Prime Intellect Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers)
- [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL)
- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
