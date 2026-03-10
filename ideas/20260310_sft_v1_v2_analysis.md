# SFT v1/v2 Analysis: Catastrophic Forgetting on Small Data
**Date:** 2026-03-10

## Experiment

### SFT v1 (LR=2e-3, 3 epochs)
- Session: a79e3136-cf44-50f6-8675-3701cf9161fe
- 155 samples, batch_size=4, 117 steps
- Loss not tracked (bug), but training completed
- **Result: Model lost ability to generate ```repl``` blocks entirely**
- Complete catastrophic forgetting

### SFT v2 (LR=1e-3, 2 epochs)
- Session: e52d5ee3-99eb-5ea1-86f8-8c766a4aba5b
- Same 155 samples, 78 steps
- Loss: 174 → 131 (24.6% reduction)
- **NIAH-20: 75.0%** (vs base 81.0%) — slight regression but reasonable
- **Multi-NIAH: ~0%** — complete failure, model generates text without ```repl``` blocks
- Root cause: 114/155 training samples are NIAH tasks with identical code pattern
  - Model converges to one template: chunk_size=5000, overlap=500, single search pattern
  - Multi-NIAH requires different code pattern (collecting multiple values)
  - SFT erased the model's ability to write diverse code

## Key Insights

1. **The base model is already very strong** (77.5% avg, 97.8% multi-NIAH). SFT on 155 samples can only hurt.

2. **Small-data SFT causes catastrophic forgetting.** Even with conservative LR (1e-3 after LoRA scaling), 2 epochs on biased data destroys capabilities.

3. **Data composition matters enormously.** 114/155 samples are NIAH → model collapses to NIAH pattern.

4. **The code pattern is too uniform.** All trajectories use nearly identical chunk-and-search code. The model memorizes this template instead of learning diverse strategies.

## Revised Strategy

### Option A: Very Conservative SFT + Diversity
- Use 1e-5 base LR (1e-4 after LoRA scaling)
- 1 epoch only
- Balance task types (equal NIAH, multi-NIAH, doc-classify)
- Add negative examples (incorrect trajectories with 0 weight)

### Option B: Skip SFT, Go Straight to RL (GRPO)
- Base model is strong enough for RL (77.5% avg)
- GRPO doesn't require pre-training on trajectories
- Use task rewards directly: correct answer → positive advantage
- Mix task types in each batch for diversity
- **This is the recommended approach**

### Option C: Rejection Sampling + Very Light SFT
- Collect K=4 trajectories per task
- Only SFT on the best-of-K (highest score)
- Use very small LR, 1 epoch
- Ensure task diversity

## Next Steps

Going with **Option B: Direct GRPO from base model** as primary strategy, with a very conservative SFT as a comparison point.

The base model already achieves:
- NIAH: 81.0%
- Multi-NIAH: 97.8%
- Doc-Classify: 53.6%

RL should improve Doc-Classify (biggest gap) without degrading Multi-NIAH.
