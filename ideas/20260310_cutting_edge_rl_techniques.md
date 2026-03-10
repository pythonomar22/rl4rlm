# Cutting-Edge RL Techniques for LLM Post-Training (March 2026 Survey)

Research survey for improving RLM training. Each technique analyzed for applicability to our
recursive language model that writes Python code in a REPL, trained via LoRA on Tinker API
with `importance_sampling` loss.

---

## 1. MaxRL — Maximum Likelihood Reinforcement Learning

**arXiv:** 2602.02710 (Feb 2026)
**Authors:** Tajwar, Zeng, Zhou, Song, Arora, Jiang, Schneider, Salakhutdinov, Feng, Zanette (CMU)
**Code:** github.com/tajwarfahim/maxrl (built on verl)

### Core Idea
Standard RL (REINFORCE/GRPO) optimizes pass@1 — the probability of a single correct sample.
MaxRL observes that the gradient of the maximum likelihood objective decomposes as:
∇J_ML = Σ_{k=1}^∞ (1/k) ∇pass@k — a harmonic mixture of ALL pass@k gradients.

Standard REINFORCE only captures the k=1 term. MaxRL truncates at T=N (number of rollouts),
automatically approximating ML by a **single-line change**: normalize the policy gradient by
the number of SUCCESSFUL samples K rather than total samples N.

- REINFORCE: gradient = (1/N) Σ r_i S_i
- MaxRL: gradient = (1/K) Σ r_i S_i  (K = number of successes)

### Key Results
- Pareto-dominates GRPO on Qwen3-1.7B and Qwen3-4B (POLARIS-53K)
- 7.9x-19.2x test-time scaling efficiency (same pass@k with fewer samples)
- Maintains diversity (no pass@k degradation) while improving pass@1
- Upweights harder problems (low pass-rate) — exactly what we need for hard tasks

### How to Apply to RLM
**This is the single most impactful change we should make.** Our current advantage computation:
```python
advantage = (reward - mean_r) / (std_r + 1e-8)
```
MaxRL says: instead of dividing by N (all trajectories), divide by K (successful ones).
For our `importance_sampling` loss, this means computing advantages differently:
- Group successes get upweighted relative to group failures
- Hard tasks where only 1/8 trajectories succeed get MUCH stronger gradient signal
- Easy tasks where 7/8 succeed get appropriately weaker signal

**Implementation:** Change advantage normalization in `rl_tinker_v6.py` lines 955-956.
Instead of group-relative (GRPO-style), use MaxRL-style: advantage = reward / K_success.

**Complexity:** EASY — literally a one-line change to advantage computation.

### Why This Matters for RLM Specifically
Our model struggles on hard tasks (hard_multi_hop: 10%, OOLONG: 15%, code_debug: 25%).
GRPO gives these tasks weak gradient because most trajectories fail (std is large, advantages
get divided by it). MaxRL would give these tasks STRONGER gradient — exactly right.

---

## 2. DAPO — Decoupled Clip and Dynamic Sampling Policy Optimization

**arXiv:** 2503.14476 (Mar 2025, ByteDance)
**Code:** Built on verl, fully open-source

### Core Idea
Four techniques that fix GRPO at scale:

1. **Clip-Higher (Asymmetric Clipping):** ε_low=0.2, ε_high=0.28. Standard symmetric
   clipping kills exploration by capping low-probability tokens. Asymmetric clipping
   lets rare/novel tokens grow faster while preventing catastrophic policy shifts.

2. **Dynamic Sampling:** Filter out groups where ALL trajectories succeed or ALL fail
   (zero advantage → zero gradient). Over-sample until batch has N gradient-producing groups.
   This is what we already do (skip groups with std < 1e-6) but DAPO formalizes it.

3. **Token-Level Policy Gradient Loss:** Instead of averaging loss per-sample then across
   samples, weight each TOKEN equally: Loss = (1/Σ|o_i|) × Σ_i Σ_t loss_i,t.
   This prevents long successful trajectories from being under-learned.

4. **Overlong Reward Shaping:** Linear penalty for sequences approaching max length:
   R_length = 0 if |y| ≤ L_max - L_cache, linearly to -1 at L_max.

### Key Results
- 50 points on AIME 2024 (Qwen2.5-32B), beating DeepSeek-R1-Zero (47) in 50% fewer steps

### How to Apply to RLM
- **Clip-Higher:** We use Tinker's `importance_sampling` which has built-in clipping.
  Check if Tinker supports asymmetric clip bounds. If not, implement via
  `forward_backward_custom` with custom IS loss. Or use `ppo` loss with custom clip_eps.
- **Dynamic Sampling:** Already doing this (skip same-reward groups). Could formalize
  with over-sampling to maintain batch size.
- **Token-Level Loss:** Tinker's `importance_sampling` is already token-level. But we
  should verify it doesn't per-sample normalize. Our trajectories vary wildly in length
  (short NIAH vs long multi-hop). This matters.
- **Overlong Reward Shaping:** Already have truncation handling, but could add the
  smooth linear penalty instead of hard cutoff.

**Complexity:** MEDIUM — most changes are in reward/advantage computation, not Tinker API.

---

## 3. VAPO — Value-Augmented Proximal Policy Optimization

**arXiv:** 2504.05118 (Apr 2025)

### Core Idea
First value-model-based RL to beat value-free methods on long-CoT tasks. Three innovations:

1. **Value-Pretraining:** Train critic on Monte Carlo returns from frozen SFT policy before
   RL starts. Prevents initial value model bias that causes instability.

2. **Decoupled GAE:** Separate λ for critic (λ_critic=1.0) and policy (λ_policy adaptive
   based on sequence length: λ = 1 - 1/(0.05 × length)). Long sequences need higher λ
   to prevent reward signal decay.

3. **Positive-Example Self-Imitation Loss:** Add NLL loss on correct trajectories:
   L = L_PPO + 0.1 × L_NLL(correct trajectories only). Prevents forgetting successful
   patterns during RL.

### Key Results
- 60.4 on AIME 2024 (beats DAPO by 10+ points)
- Stable: no crashes across 3 runs, converges in 5K steps

### How to Apply to RLM
- **Value model:** Not directly applicable with Tinker — would need a separate model.
  However, the SELF-IMITATION LOSS idea is gold. We could add an auxiliary SFT step
  on successful trajectories every N steps. Tinker supports `cross_entropy` alongside
  `importance_sampling` — we could alternate.
- **Decoupled GAE:** Our trajectories vary from 200 to 5000 tokens. The length-adaptive
  advantage idea could help. Longer trajectories' advantages should decay less.
- **Self-imitation:** After each step, take the top-K% successful trajectories and do
  a `cross_entropy` forward_backward on them. Simple to implement.

**Complexity:** MEDIUM (self-imitation easy, value model hard without second Tinker client)

---

## 4. ReTool — RL for Strategic Tool Use

**arXiv:** 2504.11536 (Apr 2025)

### Core Idea
Train LLMs to interleave reasoning with code execution via RL. Uses PPO with:
- Code execution inside XML tags during rollouts
- Interpreter feedback masked from loss (don't backprop through sandbox output)
- Cold-start SFT data generated by converting reasoning steps to code+execution pairs

### Key Results
- 67% on AIME2024 (Qwen2.5-32B) vs 40% text-only RL baseline
- Code invocation ratio increases to ~98% during training
- Emergent code self-correction behavior

### How to Apply to RLM
**This is essentially what we're already doing** — our RLM writes code in a REPL and gets
execution feedback. Key differences to adopt:

1. **Interpreter feedback masking:** We should verify that our `importance_sampling` loss
   only backprops through model-generated tokens, NOT through REPL output tokens that
   are fed back. Currently in `trajectory_to_training_data_is`, check that stdout/output
   tokens have zero advantage weight.

2. **Cold-start SFT quality:** ReTool shows that high-quality SFT data before RL is
   critical. Our V4-s5 checkpoint may benefit from additional SFT on curated trajectories.

3. **Code self-correction:** ReTool's model learns to fix its own code errors. We should
   reward this behavior explicitly (persistence_bonus does this somewhat).

**Complexity:** EASY — mostly verification that we're already doing the right thing.

---

## 5. REINFORCE++ (Global Advantage Normalization)

**arXiv:** 2501.03262 (Jan 2025)

### Core Idea
GRPO normalizes advantages per-prompt (local). REINFORCE++ normalizes across the entire
global batch. This is less biased (local normalization with small K is a biased estimator)
and slightly superior in compute efficiency and final performance.

### How to Apply to RLM
We currently do per-group normalization (GRPO-style). Switching to global normalization:
```python
# Current (GRPO): per-group
advantage = (reward - group_mean) / group_std

# REINFORCE++: global batch
advantage = (reward - batch_mean) / batch_std
```

This is especially relevant since our batch mixes different task types (NIAH, doc-classify,
multi-hop) with very different reward distributions. Per-group normalization may be better
for our case since task difficulties differ so much. But worth A/B testing.

**Complexity:** EASY — change normalization scope.

---

## 6. RAGEN/StarPO — Multi-Turn Agent RL

**arXiv:** 2504.20073 (Apr 2025)

### Core Idea
StarPO extends GRPO/PPO to multi-turn agent interactions. Key findings:

1. **Echo Trap:** Agents collapse to repetitive patterns. Detected by reward std drop →
   entropy drop → gradient spike → irreversible collapse. **This is exactly our mode
   collapse at step 10-14.**

2. **StarPO-S (Stabilized):** Three fixes:
   - Uncertainty-based trajectory filtering: only train on high-variance groups (top 25%)
   - Critic-based advantage estimation (PPO-style)
   - Asymmetric clipping (from DAPO)

3. **Key Findings:**
   - 4 responses × 8 prompts > 16 responses × 2 prompts (diversity matters)
   - 5-6 actions per turn optimal for multi-step planning
   - Fresh rollouts critical (online-1 > online-2)
   - Without fine-grained rewards, reasoning doesn't emerge

### How to Apply to RLM
**The Echo Trap is our #1 problem.** Our mode collapse at step 10-14 matches exactly.

1. **Uncertainty filtering (top 25%):** Instead of training on ALL groups, sort by reward
   std and only train on the top 25% most uncertain groups. This is a direct fix for
   mode collapse — we stop reinforcing repetitive patterns.

2. **Diversity over repetition:** Our K=8 per prompt may be too many. Try K=4 with 2x
   more unique prompts per step. More diverse tasks > more samples per task.

3. **Fresh rollouts:** We already do online RL (generate → train → refresh weights).
   Good. Never reuse old trajectories.

4. **Monitoring:** Add reward_std and gradient_norm tracking to detect collapse early.
   If reward_std drops below threshold, inject harder tasks or increase exploration.

**Complexity:** MEDIUM — uncertainty filtering is easy, monitoring needs new logging.

---

## 7. RLEF — RL from Execution Feedback

**arXiv:** 2410.02089 (ICML 2025 Spotlight)

### Core Idea
Frame iterative code generation as RL: generate code → execute → get feedback → retry.
The reward is based on whether final code passes held-out test cases. Key insight: RL
enables the model to actually IMPROVE with each retry attempt, while standard LLMs can't.

### Key Results
- 70B model: SOTA on code generation after RLEF
- 8B model outperforms MapCoder (GPT-3.5) with 3 samples
- Order-of-magnitude sample efficiency improvement

### How to Apply to RLM
Our RLM is inherently an RLEF system — the REPL provides execution feedback. But we could:
1. Add explicit retry rewards: if the model's first code fails but second succeeds, bonus
2. Track iteration-over-iteration improvement as a reward signal
3. Use execution output (success/error) as part of the reward, not just final answer

**Complexity:** EASY — reward function changes only.

---

## 8. Search-R1 — RL for Search-Augmented LLMs

**arXiv:** 2503.09516 (Mar 2025)

### Core Idea
Train LLMs to interleave reasoning with search engine calls. Critical technique:
**Retrieved token masking** — mask gradients on tokens returned by the tool (search results),
only backprop through the model's own reasoning and query generation tokens.

### How to Apply to RLM
This directly applies to our setup:
- `llm_query()` returns text that gets fed back to the model
- REPL stdout gets fed back to the model
- We should MASK all of these returned tokens from the RL loss

This is the same insight as ReTool's "interpreter feedback mask." If we're currently
backpropping through REPL output tokens, that's a source of noise.

**Complexity:** EASY — verify token masking in `trajectory_to_training_data_is`.

---

## 9. OAPL — Off-Policy RL for LLM Reasoning

**arXiv:** 2602.19362 (Feb 2026)

### Core Idea
Instead of fighting off-policyness with importance sampling correction, embrace it.
OAPL directly optimizes with off-policy data, tolerating policy lags of 400+ gradient
steps (100x more than prior work). Matches DeepCoder on LiveCodeBench with 3x fewer
generations.

### How to Apply to RLM
Our Tinker setup is inherently off-policy: we generate trajectories with the current
policy, then train (which shifts the policy), then generate again. The lag between
rollout policy and training policy grows within each step. OAPL suggests this isn't
as harmful as we think.

Practical implication: we might be able to reuse trajectories from 2-3 steps ago,
dramatically reducing sampling cost (our bottleneck is trajectory collection, not training).

**Complexity:** HARD — would need to implement OAPL's specific objective, but the insight
about trajectory reuse is immediately useful.

---

## 10. TIS/SORL — Turn-Level Importance Sampling for Long-Horizon Agents

**arXiv:** 2511.20718 (Nov 2025)

### Core Idea
Standard token-level importance sampling becomes high-variance for multi-turn interactions.
SORL proposes turn-level IS: compute importance ratios per-turn rather than per-token,
with clipping-triggered normalization to suppress unreliable updates.

SO-PPO and SO-GRPO variants prevent training instabilities on multi-turn search benchmarks.

### How to Apply to RLM
Our RLM has 2-6 turns per trajectory. Each turn has model generation + REPL execution.
Turn-level IS would:
1. Compute importance ratio per-turn (product of token ratios within the turn)
2. Clip at the turn level rather than token level
3. Better credit assignment: which TURN was responsible for success/failure?

This could help with our credit assignment problem: currently all tokens in all turns
get the same advantage. With turn-level IS, early turns that set up correct chunking
would get appropriately weighted.

**Complexity:** HARD — needs custom loss function via `forward_backward_custom`.

---

## 11. Process Reward Models for Code (FunPRM, CodePRM, PRLCoder)

### FunPRM (arXiv: 2601.22249, Jan 2026)
Treats functions as PRM steps. Meta-learning corrects noisy intermediate rewards using
clean final rewards (unit test results). SOTA on LiveCodeBench + BigCodeBench.

### CodePRM (ACL 2025)
Uses execution feedback to score individual reasoning steps. Generate-Verify-Refine
pipeline for dynamic error correction during code search.

### PRLCoder (arXiv: 2502.01715, Feb 2025)
Statement mutation + execution verification to auto-generate process supervision data.
Process-supervised RL significantly beats outcome-only RL on complex code tasks.

### How to Apply to RLM
We could build a simple process reward for RLM trajectories:
1. **Per-turn rewards:** Did this turn's code execute without error? Did it produce
   useful output? Did it call llm_query appropriately?
2. **Intermediate verification:** After each chunk is processed, check if extracted
   info is on-track (requires knowing the answer, only for training)
3. **Meta-correction:** Use final answer correctness to retroactively adjust per-turn
   rewards (FunPRM-style)

This would address our sparse reward problem: currently a 5-turn trajectory gets
ONE reward at the end. Process rewards give gradient signal at each turn.

**Complexity:** HARD — need to design per-turn reward function, may need custom loss.

---

## 12. Rejection Sampling Fine-Tuning (RAFT / Reinforce-Rej)

**arXiv:** 2504.11343 (Apr 2025)

### Core Idea
Surprising finding: simple rejection sampling (train only on correct trajectories with
SFT loss) is competitive with GRPO and PPO. The key insight: GRPO's effectiveness comes
from FILTERING (removing all-wrong groups), not from its reward normalization.

Reinforce-Rej: filter out both entirely wrong AND entirely correct groups. Only train on
groups with mixed outcomes. This improves KL efficiency and stability.

### How to Apply to RLM
We already do the filtering part (skip groups with std < 1e-6). But the RAFT insight
suggests a simpler training recipe might work just as well:

**Hybrid approach for V7:**
1. Collect K=8 trajectories per task
2. For groups where SOME succeed: use importance_sampling (GRPO-style) on all trajectories
3. For groups where ALL fail: skip (already doing this)
4. ADDITIONALLY: take all successful trajectories and do a `cross_entropy` SFT step
   (self-imitation from VAPO, matches RAFT's insight)

This combines RL's advantage of learning from negative examples with SFT's stability.

**Complexity:** EASY — add SFT step on successful trajectories.

---

## 13. Dr. GRPO — GRPO Without Length Normalization

### Core Idea
Removes response length normalization from GRPO's advantage computation and eliminates
per-question standard deviation. Results: more efficient training, fewer unnecessary
long answers, better token efficiency.

### How to Apply to RLM
Our current advantage: `(reward - mean) / std`. Dr. GRPO says: just use `(reward - mean)`,
no std normalization. This prevents the model from being encouraged to generate long
incorrect answers (which get large negative advantage when std is small).

For RLM: our incorrect trajectories tend to be LONG (multiple failed retries). Without
std normalization, their negative advantage signal would be proportionally larger.

**Complexity:** EASY — remove std division from advantage computation.

---

## 14. SkyRL-Agent / SkyRL-tx — Multi-Turn Agent RL on Tinker

**GitHub:** NovaSky-AI/SkyRL

### Core Idea
Berkeley's SkyRL implements Tinker-compatible RL for multi-agent, multi-turn tool-use.
SkyRL-tx is an open implementation of a Tinker API backend for local GPU clusters.
SkyRL-Agent handles long-horizon agent training and evaluation.

### How to Apply to RLM
SkyRL is built for exactly our use case: multi-turn tool-use RL on Tinker. We should:
1. Study their training loop patterns
2. Check if they handle turn-level credit assignment
3. Look at their reward design for tool-use agents
4. Consider adopting their evaluation framework

**Complexity:** EASY (study) to MEDIUM (adopt patterns).

---

## 15. ReflexiCoder — Self-Correction via RL-Zero

**arXiv:** 2603.05863 (Mar 2026)

### Core Idea
RL-zero training paradigm: no SFT, directly discover reflection-correction patterns.
Uses granular reward functions that incentivize both accurate error detection and
successful repair. Injects bounded sinusoidal perturbation to step penalties, making
them slightly non-stationary, which nudges policy away from repetitive local optima.

### How to Apply to RLM
The sinusoidal perturbation idea is novel and could help with mode collapse:
- Add small periodic noise to reward function (amplitude ~0.02, period ~10 steps)
- This prevents the optimizer from settling into a narrow basin
- Theoretically sound: non-stationary rewards promote continued exploration

**Complexity:** EASY — add noise term to reward computation.

---

## Priority Ranking for Implementation

### Tier 1: Implement Immediately (V7)
1. **MaxRL advantage normalization** — One-line change, biggest theoretical improvement
2. **Uncertainty-based trajectory filtering** (from RAGEN) — Direct fix for mode collapse
3. **Self-imitation SFT on successful trajectories** (from VAPO) — Prevents forgetting
4. **Interpreter feedback masking verification** (from ReTool/Search-R1)

### Tier 2: Implement in V7 if Tier 1 Shows Promise
5. **Clip-Higher (asymmetric clipping)** — Check Tinker PPO loss supports this
6. **Dr. GRPO (remove std normalization)** — May conflict with MaxRL; A/B test
7. **Overlong reward shaping** (from DAPO) — Smooth penalty instead of hard cutoff
8. **RAFT hybrid** — SFT step alongside RL step

### Tier 3: Research and Prototype
9. **Turn-level importance sampling** — Needs custom loss
10. **Process reward model** — Per-turn rewards for credit assignment
11. **Sinusoidal reward perturbation** — Novel, needs testing
12. **OAPL-style trajectory reuse** — Could cut sampling cost 3x

### Tier 4: Monitor and Evaluate
13. **VAPO full value model** — Requires second Tinker client
14. **FunPRM for RLM code** — Heavy engineering
15. **SkyRL-Agent patterns** — Study their codebase

---

## Concrete V7 Training Recipe

Based on this survey, here's the recommended V7 recipe:

```
V7 = V6 + {
    MaxRL advantage normalization,
    Uncertainty-based group filtering (top 50%),
    Self-imitation SFT on correct trajectories (μ=0.1),
    Asymmetric clipping (ε_low=0.2, ε_high=0.28),
    Verified interpreter token masking,
    Overlong reward shaping (linear penalty near max_tokens),
    Reward std monitoring for early collapse detection
}
```

Key principle: **More signal from hard tasks, less noise from easy tasks, and never
forget what works.**
