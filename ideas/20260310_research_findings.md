# Research Findings — Recent RLM & Code RL Literature (2026-03-10)

## Most Actionable (Implement Next)

### 1. ProGRPO's Advantage Re-weighting (ARM) — arXiv:2602.05281
- Re-weight advantages by prompt perplexity and answer confidence
- Penalize over-confident solutions, promote rare correct ones
- +5.7% Pass@1, +13.9% Pass@32 on Qwen2.5-7B
- **For V8:** When computing advantages, weight down high-confidence correct trajectories

### 2. NGRPO's Virtual Max-Reward — arXiv:2509.18851
- For all-wrong groups: add virtual optimal completion with reward=1.0
- Gives negative advantage to all actual completions → push away from failures
- **For V8:** Useful for event_counting and hard_multi_hop where all K=8 often fail

### 3. Dense Intermediate Rewards — arXiv:2603.02146 (LongRLVR)
- Reward correct chunking and sub-calls, not just final answer
- Proves that answer-only rewards cause vanishing gradients for context grounding
- **For V8:** Add reward for: correct chunk size, proper aggregation, multi-turn decomposition

### 4. Asymmetric Clipping — arXiv:2509.26114
- clip-high decreases entropy (causes collapse), clip-low increases entropy
- Use clip_low=0.3, clip_high=0.15 instead of symmetric 0.2
- **Simple hyperparameter change for V8**

### 5. Train Short → Test Long (LoongRL) — arXiv:2510.19363
- Training at 16K generalizes to 128K
- KeyChain synthesis: embed UUID chains as distractors
- **Saves significant Tinker compute** — train at 50K, test at 200K+

## Good Ideas (Lower Priority)

### 6. GAPO Frequency-Aware Reward — arXiv:2511.12596
- Penalize duplicate trajectories within each K-group
- **Not needed since SC-GRPO already solved mode collapse**

### 7. iGRPO Self-Feedback — arXiv:2602.09000
- Generate K drafts, select best, condition next GRPO round on best
- SOTA on AIME24 (85.62%)
- **Could apply to RLM: show model a good trajectory, ask it to do better**

### 8. llm_batch() Parallel Sub-calls — Prime Intellect
- Add batch inference for parallel chunk processing
- **Could speed up inference 3-5x for O(N) tasks**

### 9. REPL Output Cap (8192 chars) — Prime Intellect
- Forces programmatic data handling instead of dumping raw text
- **Good idea but may break debugging/observation during development**

### 10. Avoid Depth > 1 — arXiv:2603.02615
- Deeper recursion is counterproductive: accuracy degrades, cost explodes
- Depth=1 is optimal for most tasks
- **Already using depth=1, validated**

## Papers to Cite
- ProGRPO: arXiv:2602.05281
- NGRPO: arXiv:2509.18851
- GAPO: arXiv:2511.12596
- LoongRL: arXiv:2510.19363
- QwenLong-L1: arXiv:2505.17667
- LongRLVR: arXiv:2603.02146
- LongR: arXiv:2602.05758
- SkyRL-Agent: arXiv:2511.16108
- DAPO: arXiv:2503.14476
- iGRPO: arXiv:2602.09000
- Wang 2026 (Depth>1 analysis): arXiv:2603.02615
- RLM-JB: arXiv:2602.16520
