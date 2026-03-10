# Post-Training Techniques Survey V2: Cutting-Edge RL for Code/Tool-Use LLMs
**Date:** 2026-03-10
**Purpose:** Comprehensive survey of state-of-the-art post-training techniques for our RLM (recursive language model) public release

## Executive Summary

Our current approach (SC-GRPO = Strategy-Conditioned GRPO with importance_sampling loss on Tinker) is solid but leaves significant performance on the table. The field has moved dramatically since DeepSeek-R1's GRPO. The key insight from this survey: **the biggest gains come not from replacing GRPO, but from fixing its known biases, adding dense rewards, and managing entropy collapse properly.**

### Top 5 Actionable Improvements (Priority Order)

1. **Dr. GRPO bias correction** - Remove length normalization and std normalization from advantages (free, 1-2% gain)
2. **Clip-Higher from DAPO/VAPO** - Asymmetric clipping to prevent entropy collapse (directly implementable)
3. **Positive Example LM Loss from VAPO** - Add cross-entropy loss on correct trajectories alongside RL (6-point gain in VAPO)
4. **Process rewards via code execution** - Use REPL execution success as dense per-step reward
5. **Dynamic sampling from DAPO** - Filter out prompt groups where all K are correct or all wrong

---

## 1. VAPO: Value-based Augmented PPO
**Paper:** VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks
**ArXiv:** 2504.05118 (April 2025, ByteDance)
**Venue:** ICLR 2026

### Core Technique
VAPO is PPO with seven targeted fixes for reasoning tasks. It uses a value model (critic) but the key innovations are applicable without one:

1. **Clip-Higher** (eps_high=0.28, eps_low=0.2): Asymmetric clipping — allows probability increases more than decreases, preventing entropy collapse
2. **Positive Example LM Loss**: Adds standard cross-entropy loss on correct trajectories. This is effectively SFT-on-success mixed into the RL objective. Worth ~6 points.
3. **Length-Adaptive GAE**: Dynamically adjusts the GAE lambda based on sequence length, so short and long responses get balanced optimization
4. **Token-level loss** (not sequence-level): Worth ~7 points
5. **Value pretraining**: Pre-train the critic on outcome data before RL starts
6. **Decoupled GAE**: Prevents reward signal decay in long sequences
7. **Group sampling**: Multiple rollouts per prompt for variance reduction

### Key Results
- AIME24: 60.4 (vs GRPO's 47, DAPO's 50)
- Matches DAPO in 60% fewer steps
- No training crashes across multiple runs

### Applicability to RLM
**HIGH.** We can implement 3 of the 7 techniques without a value model:
- **Clip-Higher**: Tinker has `ppo` loss with `clip_eps`. We could use `forward_backward_custom` to implement asymmetric clipping, OR simply use the PPO loss with a larger clip range and add our own asymmetric logic.
- **Positive Example LM Loss**: After each GRPO step, take the best trajectory (highest reward) and also do a cross-entropy forward_backward on it. This is just an extra SFT step on wins. Trivially implementable on Tinker.
- **Token-level loss**: We already use importance_sampling which is token-level. Confirmed.

**Implementation on Tinker:**
```python
# After GRPO step, add positive example LM loss
best_traj = max(trajectories, key=lambda t: t['reward'])
if best_traj['reward'] > 0.5:  # only on genuine wins
    sft_datum = build_supervised_datum(best_traj)
    training_client.forward_backward([sft_datum], "cross_entropy")
    # Don't call optim_step yet — accumulate with RL gradient
```

---

## 2. Dr. GRPO: Removing Bias from GRPO
**Paper:** Understanding R1-Zero-Like Training: A Critical Perspective
**ArXiv:** 2503.20783 (March 2025)
**Venue:** COLM 2025

### Core Technique
Standard GRPO normalizes advantages by (1) group standard deviation and (2) response length. Both introduce bias:
- **Length normalization** causes the optimizer to favor longer wrong responses (more gradient mass)
- **Std normalization** distorts the gradient when variance differs across prompts

Dr. GRPO simply removes both normalizers: keep the group-mean baseline but drop variance and length divisors.

### Key Results
- Consistently outperforms vanilla GRPO in both training rewards and evaluation accuracy
- Prevents progressively longer incorrect responses
- Better token efficiency

### Applicability to RLM
**CRITICAL. This is a free lunch.** Our current advantage computation likely uses standard GRPO normalization. Fix:

```python
# BEFORE (standard GRPO):
advantages = (rewards - mean_reward) / (std_reward + 1e-8)
# Then divide by response length in the loss

# AFTER (Dr. GRPO):
advantages = rewards - mean_reward
# Do NOT divide by std or by response length
# The importance_sampling loss in Tinker handles token-level aggregation
```

**Implementation on Tinker:** Just change the advantage computation in our training loop. The Tinker `importance_sampling` loss takes per-token advantages — we set them all to the same value (the Dr. GRPO advantage). No API changes needed.

---

## 3. DAPO: Dynamic Sampling + Clip-Higher + Overlong Filtering
**Paper:** DAPO: An Open-Source LLM Reinforcement Learning System at Scale
**ArXiv:** 2503.14476 (March 2025, ByteDance)

### Core Technique
Four innovations:
1. **Clip-Higher**: Same as VAPO — asymmetric clipping (eps_high > eps_low)
2. **Dynamic Sampling**: Filter out prompts where accuracy=0 (all fail) or accuracy=1 (all succeed). Only train on prompts with mixed outcomes (some succeed, some fail). This eliminates zero-gradient and noise-only updates.
3. **Token-Level Policy Gradient Loss**: Critical for long-CoT/long-code scenarios
4. **Overlong Reward Shaping**: Mask loss for truncated samples + soft length penalty

### Key Results
- AIME24: 50 points (vs DeepSeek-R1-Zero's 47)
- 50% fewer training steps than R1-Zero

### Applicability to RLM
**HIGH.** Dynamic sampling is directly relevant:

```python
# After collecting K trajectories per prompt:
rewards = [t['reward'] for t in group]
if all(r > 0.8 for r in rewards):
    skip  # All correct — no gradient signal
if all(r < 0.1 for r in rewards):
    skip  # All wrong — noise only
# Only train on groups with variance
```

**Overlong filtering** matters for RLM because our trajectories can be very long (multiple REPL turns). If a trajectory was truncated (hit max_tokens), mask its loss contribution.

**Implementation on Tinker:** All in our training loop logic. No API changes.

---

## 4. REINFORCE++ with Global Baseline
**Paper:** REINFORCE++: Stabilizing Critic-Free Policy Optimization
**ArXiv:** 2501.03262 (January 2025)

### Core Technique
REINFORCE++ replaces GRPO's per-group normalization with **global batch normalization**:
- Normalize advantages across the entire batch, not per-prompt
- Achieves unbiased gradient estimates (bias vanishes as batch size grows)
- Equivalent to PPO without a critic, with gamma=1

### Key Results
- Outperforms both GRPO and full-critic PPO on average
- More stable training curves

### Applicability to RLM
**MEDIUM.** Our batch sizes are small (4 prompts x 8 rollouts = 32 trajectories). Global normalization might be noisy at this scale. But worth trying:

```python
# Instead of per-prompt normalization:
all_rewards = [r for group in batch for r in group]
global_mean = np.mean(all_rewards)
global_std = np.std(all_rewards) + 1e-8

# Per trajectory:
advantage = (reward - global_mean) / global_std  # or just (reward - global_mean) per Dr. GRPO
```

**Implementation on Tinker:** Trivial — just change advantage computation.

---

## 5. lambda-GRPO: Learnable Token Preferences
**Paper:** lambda-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences
**ArXiv:** 2510.06870 (October 2025)

### Core Technique
Unifies GRPO, DAPO, and Dr. GRPO under one framework with a learnable parameter lambda that controls token-level weighting. Instead of fixed heuristics (divide by length or not), lambda adapts dynamically during training.

### Key Results
- +1.9% accuracy on Qwen2.5-1.5B, +1.0% on 3B, +1.7% on 7B over standard GRPO
- No additional compute cost

### Applicability to RLM
**MEDIUM.** Requires implementing a learnable lambda parameter. With Tinker's `forward_backward_custom`, we could potentially implement this, but it adds complexity. Better to start with Dr. GRPO (the simplest fix) and only go to lambda-GRPO if we need more.

---

## 6. PRIME: Process Reinforcement through Implicit Rewards
**Paper:** Process Reinforcement through Implicit Rewards
**ArXiv:** 2502.01456 (February 2025)

### Core Technique
Uses an implicit PRM (Process Reward Model) derived from the policy model itself to provide dense per-token rewards. The PRM is trained with only outcome labels (no step-level annotations needed). Key: the PRM updates online alongside the policy, avoiding reward hacking from distribution shift.

### Key Results
- 2.5x sample efficiency vs outcome-only rewards
- 6.9% performance improvement

### Applicability to RLM
**HIGH POTENTIAL but complex.** For RLM, we have a natural source of dense rewards:
- **Code execution success/failure per REPL turn** — this IS a process reward
- **Sub-call responses** — whether llm_query returns useful results
- **Python variable state** — whether intermediate variables are well-formed

We don't need to train a separate PRM. We can compute process rewards directly from REPL execution:

```python
turn_rewards = []
for turn in trajectory['turns']:
    r = 0.0
    if turn['parsed_code']:
        r += 0.1  # Generated valid code
    if not turn['error']:
        r += 0.2  # Code executed without error
    if 'llm_query' in (turn['parsed_code'] or ''):
        r += 0.1  # Used recursive calls
    turn_rewards.append(r)
# Final turn gets the outcome reward
turn_rewards[-1] += outcome_reward
```

**Implementation on Tinker:** We'd need to set per-token advantages differently for each turn's tokens. This is already possible since `importance_sampling` takes per-token advantage tensors. Instead of uniform advantages across all tokens, weight by turn-level process rewards.

---

## 7. RLEF: RL from Execution Feedback
**Paper:** RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning
**ArXiv:** 2410.02089 (October 2024, updated Feb 2025)
**Venue:** ICML 2025

### Core Technique
End-to-end RL for multi-turn code generation with execution feedback. The model generates code, receives execution results, and refines iteratively. RL is applied to the full multi-turn trajectory.

### Key Results
- State-of-the-art on competitive programming with 8B and 70B models
- 10x reduction in required samples vs independent sampling

### Applicability to RLM
**DIRECTLY RELEVANT.** This is essentially what our RLM training does — the model writes code in a REPL and iterates. Key insight: the execution feedback (stdout, stderr) IS the environment observation, and RL should be applied to the full multi-turn trajectory, not just individual turns.

Our V6 training already does this. The key RLEF insight we should adopt: **treat execution errors as negative intermediate rewards**, not just noise. A trajectory that hits 3 errors before succeeding should have lower advantage for the error-producing turns.

---

## 8. ReflexiCoder: Self-Reflecting Code Generation via RL
**Paper:** ReflexiCoder: Teaching LLMs to Self-Reflect on Generated Code and Self-Correct
**ArXiv:** 2603.05863 (March 2026 -- very recent!)

### Core Technique
Trains the model to internalize self-debugging: generate code -> reflect on bugs -> self-correct, all within a single generation. Uses RL to learn this structured reasoning trajectory. No external oracle needed at inference time.

### Key Results
- Outperforms prior self-correction methods
- Fully autonomous debugging at inference time

### Applicability to RLM
**HIGH.** Our RLM already has multi-turn self-correction (the REPL loop). ReflexiCoder suggests we should explicitly reward:
1. Identifying errors in own code (reflection)
2. Generating corrected code that fixes the identified error
3. NOT repeating the same error

We could add a "reflection quality" reward:
```python
if turn_i_has_error and turn_i_plus_1_fixes_it:
    reflection_bonus += 0.1
if turn_i_repeats_same_error_as_turn_i_minus_1:
    repetition_penalty -= 0.1
```

---

## 9. Tool-R1: Sample-Efficient RL for Tool Use
**Paper:** Tool-R1: Sample-Efficient Reinforcement Learning for Agentic Tool Use
**ArXiv:** 2509.12867 (September 2025)

### Core Technique
RL framework for tool-using LLMs that generate executable Python code. Key innovation: **dynamic sample queue** that caches high-quality trajectories and reuses them, dramatically improving sample efficiency.

### Key Results
- Competitive on GAIA benchmark
- Significant sample efficiency gains via trajectory caching

### Applicability to RLM
**MEDIUM-HIGH.** The dynamic sample queue idea is excellent for our setting:
- We generate K=8 trajectories per prompt
- Best trajectories (reward > threshold) get cached
- In future steps, some "slots" in the batch use cached trajectories instead of fresh rollouts
- This reduces the cost of trajectory collection (our biggest bottleneck)

**Implementation on Tinker:**
```python
trajectory_cache = []  # Recent high-quality trajectories

for step in range(num_steps):
    # Generate some fresh trajectories
    fresh = collect_trajectories(prompts, K=6)
    # Supplement with cached trajectories
    cached = sample_from_cache(trajectory_cache, n=2)

    # Train on mixed batch
    train_on(fresh + cached)

    # Update cache with best fresh trajectories
    for t in fresh:
        if t['reward'] > 0.7:
            trajectory_cache.append(t)
```

---

## 10. Agent-R1 / RAGEN: Multi-Turn Agent RL
**Paper:** Agent-R1: Training Powerful LLM Agents with End-to-End RL
**ArXiv:** 2511.14460 (November 2025)
**Also:** RAGEN (arXiv:2504.20073, April 2025)

### Core Technique
Frameworks for multi-turn RL with tool use. Key insight: distinguish between agent-generated tokens (trained) and environment-response tokens (not trained). Use intermediate process rewards, not just final outcome.

RAGEN identifies the **"Echo Trap"** — a mode collapse where reward variance drops to zero and gradients spike. Their fix (StarPO-S): trajectory filtering + gradient stabilization.

### Key Results
- Agent-R1: GRPO delivers best performance for agent RL
- RAGEN: StarPO-S stabilizes training, prevents Echo Trap

### Applicability to RLM
**HIGH.** We already see mode collapse at step 10-14 (documented in our training_insights.md). The Echo Trap is likely what we're experiencing. RAGEN's fixes:

1. **Trajectory filtering**: Remove trajectories that are too similar (low diversity) — we already have code diversity bonus but could filter more aggressively
2. **Gradient clipping per-trajectory** (not just global): Prevents one outlier trajectory from dominating the update
3. **Process rewards on agent actions only**: Don't apply advantage to environment response tokens (REPL stdout). We should mask advantage to zero on REPL output tokens.

```python
# When building advantages tensor:
for token_idx in range(len(tokens)):
    if is_repl_output_token(token_idx):
        advantages[token_idx] = 0.0  # Don't train on environment output
    else:
        advantages[token_idx] = trajectory_advantage
```

---

## 11. SWE-RL: RL for Software Engineering Agents
**Paper:** SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution
**ArXiv:** 2502.18449 (February 2025, Meta)
**Venue:** NeurIPS 2025

### Core Technique
Applies GRPO to software engineering tasks using lightweight rule-based rewards from code execution. Key finding: training on code tasks with RL **transfers to out-of-domain reasoning** (math, language understanding).

### Key Results
- 41% solve rate on SWE-bench Verified (best for <100B models)
- Improves 5 out-of-domain tasks despite training only on code

### Applicability to RLM
**MEDIUM.** Validates that our approach (RL on code generation) should generalize. The transfer result is encouraging — our model may get better at general reasoning from RLM training. We should evaluate on out-of-domain benchmarks to check.

---

## 12. SPELL: Self-Play for Long-Context
**Paper:** SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models
**ArXiv:** 2509.23863 (September 2025)

### Core Technique
Three-role self-play: the same model acts as questioner (generates hard questions), responder (answers), and verifier (checks answers). Enables label-free optimization for long-context reasoning.

### Applicability to RLM
**LOW-MEDIUM for now.** Interesting for future work — could use the trained RLM to generate harder benchmark tasks for itself. But adds significant complexity. File for Phase 4.

---

## 13. Entropy Collapse Prevention: DAPO/VAPO/EDGE-GRPO/AEPO
Multiple papers, 2025

### Core Techniques
Our mode collapse at step 10-14 is entropy collapse. Multiple solutions exist:

1. **Clip-Higher** (DAPO/VAPO): Asymmetric clipping eps_high > eps_low
2. **EDGE-GRPO**: Entropy-driven advantages — weight advantages by token entropy
3. **AEPO**: Arbitrary Entropy Policy Optimization — stabilize entropy at target level
4. **CURE**: Identify critical high-entropy tokens and re-query
5. **STEER**: Apply stronger KL to low-entropy tokens specifically

### Recommended for RLM
Use **Clip-Higher** (simplest, proven effective) + **entropy monitoring with early stopping**:

```python
# Monitor per-step entropy
if step_entropy < entropy_floor:
    # Increase exploration: raise temperature, add entropy bonus to reward
    current_temperature = min(1.3, current_temperature + 0.05)
    entropy_bonus = max(0.0, (entropy_target - step_entropy) * 0.1)
```

---

## 14. Curriculum RL: Progressive Difficulty
**Papers:** E2H Reasoner (arXiv:2506.06632), VCRL (arXiv:2509.19803), TACLer (arXiv:2601.21711)

### Core Technique
Schedule training tasks from easy to hard. VCRL uses **reward variance** as difficulty signal — high variance means the model sometimes succeeds, sometimes fails (optimal training signal). Low variance means too easy or too hard.

### Applicability to RLM
**HIGH.** We already have adaptive difficulty in V6. VCRL's variance-based approach is more principled:

```python
# VCRL-style difficulty selection
for task_type in task_pool:
    # Estimate recent success variance
    recent_rewards = reward_history[task_type][-20:]
    variance = np.var(recent_rewards)
    # High variance = model is learning here = prioritize
    task_weights[task_type] = variance
# Sample tasks proportional to variance
```

---

## 15. Prime Intellect's RLMEnv
**Source:** https://www.primeintellect.ai/blog/rlm (January 2026)

### Core Architecture
Prime Intellect implemented RLM training in their `verifiers` library:
- Main RLM has Python REPL only
- Sub-LLMs get heavy tools (web search, file access)
- `llm_batch()` for parallel sub-queries
- `answer` variable for final solution

### Key Design Choices
They separate the main model's context (lean, code-focused) from sub-model contexts (tool-heavy). This is relevant for our design — our current approach puts everything in one context.

### Benchmarks Used
- DeepDive (web research)
- Math python (competition math)
- Oolong (long context classification)
- Verbatim copy (exact reproduction)

---

## 16. GRPO-CARE: Consistency-Aware RL
**Paper:** GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning
**ArXiv:** 2506.16141 (June 2025)

### Core Technique
Standard GRPO improves accuracy but reduces reasoning-answer consistency (only 57.9% consistent). GRPO-CARE adds a consistency bonus: the reasoning chain should logically lead to the answer.

**Replaces KL penalty** with an adaptive consistency bonus.

### Applicability to RLM
**MEDIUM-HIGH.** In our setting, "consistency" means the code should logically produce the answer. We could reward:
- Code that actually uses the result of llm_query calls (not ignoring sub-call results)
- FINAL() call with a value derived from computed variables (not hallucinated)
- Matching between last code output and FINAL() argument

---

## 17. VerlTool: Holistic Agentic RL with Tool Use
**Paper:** VerlTool: Towards Holistic Agentic RL with Tool Use
**ArXiv:** 2509.01055 (September 2025)
**Venue:** ICLR 2026 Workshop

### Core Technique
Unified framework for RL with diverse tools: code execution, search, SQL, vision. Key: **asynchronous rollout execution** achieving ~2x speedup.

### Applicability to RLM
**MEDIUM.** Architecture reference. Our Tinker-based training already handles async. But their standardized tool API design could inform our scaffold improvements.

---

## 18. CodePRM / FunPRM: Process Rewards for Code
**Papers:**
- CodePRM (ACL 2025): Execution feedback-enhanced PRM
- FunPRM (arXiv:2601.22249, January 2026): Function-as-step PRM

### Core Technique
**CodePRM**: Train PRM using code execution results to distinguish good vs bad reasoning steps. Generate-Verify-Refine pipeline.
**FunPRM**: Treat each function in modular code as a reasoning step. Score functions individually.

### Applicability to RLM
**HIGH.** Our RLM naturally produces modular code (each REPL turn is a "step"). We can use execution success as process reward per turn:

```python
# Per-turn process reward based on execution
for i, turn in enumerate(trajectory['turns']):
    if turn['error']:
        process_reward[i] = -0.1  # Execution error
    elif turn['stdout'] and len(turn['stdout']) > 10:
        process_reward[i] = 0.1   # Produced meaningful output
    if 'llm_query' in (turn['parsed_code'] or '') and not turn['error']:
        process_reward[i] += 0.15  # Successfully delegated to sub-model
```

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (implement in V7)
These require only changes to our training loop, no Tinker API changes:

1. **Dr. GRPO**: Remove length and std normalization from advantages
2. **Dynamic sampling (DAPO)**: Skip all-correct and all-wrong prompt groups
3. **Mask environment tokens**: Set advantage=0 on REPL stdout tokens
4. **Overlong filtering**: Mask loss for truncated trajectories

### Phase 2: Medium Effort (V7 or V8)
5. **Positive Example LM Loss (VAPO)**: Extra cross_entropy step on best trajectory per group
6. **Process rewards per turn**: Use execution success as dense per-turn reward, set per-token advantages accordingly
7. **Clip-Higher**: Use `forward_backward_custom` with asymmetric clipping, or use Tinker's `ppo` loss with appropriate clip range
8. **Variance-based curriculum (VCRL)**: Prioritize tasks with high reward variance

### Phase 3: Advanced (V8+)
9. **Trajectory caching (Tool-R1)**: Reuse high-quality trajectories
10. **Self-correction reward (ReflexiCoder)**: Bonus for fixing own errors across turns
11. **Global advantage normalization (REINFORCE++)**: When batch sizes are larger
12. **Entropy-driven advantages (EDGE-GRPO)**: Weight advantages by token entropy

### Phase 4: Research Contributions
13. **RLM-specific PRM**: Train implicit PRM on REPL execution traces
14. **Self-play curriculum (SPELL)**: Model generates its own harder tasks
15. **Multi-level recursion training**: Train RLM to call RLM to call RLM

---

## Compatibility with Tinker API

| Technique | Tinker Loss | Implementation |
|-----------|-------------|----------------|
| Dr. GRPO | importance_sampling | Change advantage computation only |
| DAPO dynamic sampling | importance_sampling | Filter prompts in training loop |
| Clip-Higher | ppo or forward_backward_custom | Use PPO loss with asymmetric clip |
| Positive Example LM | cross_entropy | Extra forward_backward on best traj |
| Process rewards | importance_sampling | Per-token advantage tensors |
| REINFORCE++ | importance_sampling | Global advantage normalization |
| Lambda-GRPO | forward_backward_custom | Custom loss function |
| Overlong filtering | importance_sampling | Set advantages=0 for truncated |
| Env token masking | importance_sampling | Set advantages=0 for env tokens |

**Key insight:** Almost everything is implementable with Tinker's existing `importance_sampling` loss by manipulating the per-token advantage tensor. The advantages tensor is our main control surface.

---

## Sources

- VAPO: https://arxiv.org/abs/2504.05118
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- REINFORCE++: https://arxiv.org/abs/2501.03262
- Lambda-GRPO: https://arxiv.org/abs/2510.06870
- PRIME: https://arxiv.org/abs/2502.01456
- RLEF: https://arxiv.org/abs/2410.02089
- ReflexiCoder: https://arxiv.org/abs/2603.05863
- Tool-R1: https://arxiv.org/abs/2509.12867
- Agent-R1: https://arxiv.org/abs/2511.14460
- RAGEN: https://arxiv.org/abs/2504.20073
- SWE-RL: https://arxiv.org/abs/2502.18449
- SPELL: https://arxiv.org/abs/2509.23863
- GRPO-CARE: https://arxiv.org/abs/2506.16141
- VerlTool: https://arxiv.org/abs/2509.01055
- CodePRM: https://aclanthology.org/2025.findings-acl.428/
- FunPRM: https://arxiv.org/abs/2601.22249
- E2H Reasoner: https://arxiv.org/abs/2506.06632
- VCRL: https://arxiv.org/abs/2509.19803
- EDGE-GRPO: https://arxiv.org/abs/2507.21848
- GTPO: https://arxiv.org/abs/2508.03772
- DiverseGRPO: https://arxiv.org/abs/2512.21514
- Prime Intellect RLM: https://www.primeintellect.ai/blog/rlm
- Prime Intellect verifiers: https://github.com/PrimeIntellect-ai/verifiers
- ASPO: https://arxiv.org/abs/2510.06062
- AReaL: https://arxiv.org/abs/2505.24298
- ReST-RL: https://arxiv.org/abs/2508.19576
- CodeRL+: https://arxiv.org/abs/2510.18471
