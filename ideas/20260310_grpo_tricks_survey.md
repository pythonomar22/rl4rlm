# GRPO++ Tricks for V9+ (from Cameron Wolfe's survey + DAPO + Dr. GRPO)

## Already Implemented in V8
1. **Asymmetric Advantage Scaling** — clip_high=0.5, clip_low=1.5 (our version of DAPO's "clip higher")
2. **Dynamic Sampling** — NGRPO virtual reward recovers signal from all-wrong groups (better than just filtering)
3. **MaxRL** — difficulty-aware normalization by K_success

## Should Implement in V9

### 1. Token-Level Loss Aggregation (HIGH PRIORITY)
- **Problem:** Per-sample normalization biases against long trajectories
- **Fix:** Normalize loss by total tokens in batch, not per-sample
- **Impact:** Eliminates length bias in RLM (where trajectory length varies 1-10 turns)
- **Our context:** RLM trajectories vary 1-5 turns, huge length variance. Token-level aggregation would equalize gradient contribution
- Source: DAPO paper

### 2. Dr. GRPO Simplified Advantages (MEDIUM PRIORITY)
- **Problem:** std_dev normalization creates extreme advantages for very easy/hard tasks
- **Fix:** A = reward - mean_reward (no std_dev division)
- **Impact:** More stable training, especially for mixed difficulty
- **Note:** Conflicts with MaxRL which multiplies by N/K_success. Could combine: A = (reward - mean_r) * N/K_success
- Source: Dr. GRPO paper

### 3. Decoupled Clipping Bounds (LOW PRIORITY for us)
- ε_low=0.2, ε_high=0.28 instead of symmetric ε=0.2
- **Our context:** We don't directly control the Tinker importance_sampling clip range, so this may not be configurable
- Source: DAPO paper

### 4. Overlong Reward Shaping (MEDIUM PRIORITY)
- Soft penalty in [L_max - L_cache, L_max] range instead of hard truncation
- **Our context:** We have timeout-based truncation. Could add soft penalty as context length increases
- Source: DAPO paper

### 5. Dr. GRPO Fixed Constant Loss Normalization
- Divide loss by MAX_TOKENS constant instead of actual sequence length
- Removes length-gradient correlation
- Source: Dr. GRPO paper

## Not Applicable to Us
- Template formatting — we use full system prompts, not raw templates
- Domain-specific pretraining — already doing task-specific RL
- TIS — already using importance_sampling on Tinker

## GRPO-LEAD (arXiv:2504.09696, EMNLP 2025)
- Difficulty-aware advantage reweighting — similar to our MaxRL but different approach
- Length-regularized rewards — encourages conciseness
- GitHub: github.com/aeroplanepaper/GRPO-LEAD
- Worth checking implementation details for V9
