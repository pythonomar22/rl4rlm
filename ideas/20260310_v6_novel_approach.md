# V6: Novel GRPO Training Approach
**Date:** 2026-03-10

## Motivation

Deep analysis of V4b step 1 logs revealed critical issues:
1. **Zero gradient from easy tasks**: 10K multi-hop prompt had all 8 trajectories correct (reward std=0). The only learning came from 150K prompts.
2. **Mode collapse still happens on easy tasks**: When all K trajectories produce identical correct answers, advantage = 0 for all.
3. **kl_coeff was never used**: V1-V5 had zero KL regularization, accelerating mode collapse.
4. **Multiple optim_steps per training step**: Each mini-batch of 4 datums triggered a separate weight update, creating noisy SGD instead of clean gradient accumulation.
5. **Temperature bias in importance weights**: Logprobs from T=1.5 are systematically lower, inflating importance ratios.

## Key Innovations in V6

### 1. Gradient Accumulation (Critical Fix)
**Before (V1-V5):** For N datums, we did ceil(N/4) forward_backward + optim_step pairs. Each mini-batch updated weights, so later batches operated on already-updated weights. This made the optimizer behave like SGD with batch_size=4.

**V6:** Submit ALL forward_backward calls first, then ONE optim_step. This gives a clean full-batch gradient for the step. Expected effect: more stable training, fewer oscillations.

### 2. Adaptive Task Difficulty (Novel)
Track per-task-type skip rate (fraction of groups where all K trajectories get the same reward). When skip rate exceeds 60% for 3 consecutive steps, automatically increase difficulty:
- NIAH: 20K-100K → 50K-200K → 100K-500K
- Doc-Classify: more documents, harder categories
- Multi-Hop: longer documents

This keeps the learning signal alive throughout training. **No prior GRPO work has reported adaptive difficulty for code generation RL.**

### 3. Multi-Turn Persistence Bonus (Novel)
Extra reward for trajectories that:
- Use 2+ code turns AND get the correct answer
- Up to +0.15 bonus for 4+ turns

This directly incentivizes the iterative debugging behavior that makes RLMs powerful (see DataFrame QA 4-turn trajectory). Standard GRPO only rewards final correctness; we reward the *process*.

### 4. Code Diversity Bonus (Novel)
Within each group of K trajectories, compute pairwise 3-gram Jaccard distance of generated code. Trajectories with more unique code patterns get a small bonus (up to +0.03).

This creates a soft "anti-mode-collapse" pressure directly in the reward function. Even when two trajectories get the same final score, the one with more diverse code gets a slightly higher reward, breaking the std=0 deadlock.

### 5. KL Penalty via Reward Shaping
Since Tinker's kl_coeff isn't wired up in the V5 code, V6 approximates KL as `|mean_logprob - ref_mean_logprob|` and subtracts `kl_coeff * KL_approx` from each trajectory's reward. The reference is set from step 1.

### 6. Narrower Temperature Schedule
V5: [0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.5]
V6: [0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2]

Analysis showed temperatures > 1.2 mainly cause failures (hallucinated answers, timeouts) rather than creative correct solutions. The narrower range preserves diversity while reducing noise.

### 7. Linear Warmup + Cosine Decay
V5 started at full LR immediately. V6 uses 2-step linear warmup before cosine decay. This stabilizes early training when the model first encounters the reward landscape.

### 8. Minimum 20K Context Length
No training tasks under 20K characters. Short contexts (5K-10K) always fit in one chunk, so the model just calls `llm_query(f"... {context}")` — a trivially correct strategy that teaches nothing about RLM behavior.

## Task Mix: mixed_v6
| Task | Weight | Rationale |
|------|--------|-----------|
| hard_multi_hop | 25% | Primary learning target: decomposition |
| multi_hop_qa | 20% | Core multi-hop reasoning at moderate length |
| notebook_qa | 15% | Structured data extraction |
| dataframe_qa | 10% | Numerical analysis with iteration |
| code_debug | 10% | Code understanding |
| doc_classify | 10% | O(N) processing |
| niah | 5% | Prevent catastrophic forgetting only |
| multi_niah | 5% | Prevent catastrophic forgetting only |

## Expected Effects
1. **Longer training before mode collapse** — adaptive difficulty + diversity bonus + KL
2. **Better hard multi-hop performance** — persistence bonus + decomposition reward + higher weight
3. **No regression on easy tasks** — 10% allocation prevents forgetting
4. **Cleaner gradients** — gradient accumulation + narrower temperatures

## Launch Plan
Start V6 from best checkpoint (V4-s5 if confirmed best, or V4b-s5 when available).
Run for 30 steps with save-every=5. Monitor:
- Per-task skip rates (should decrease as adaptive difficulty kicks in)
- Hard multi-hop reward trend (target: >0.3 by step 15)
- Multi-turn trajectory frequency (target: >50% of trajectories use 2+ turns by step 10)
- Difficulty level upgrades (expected: NIAH upgrades by step 5-7)
