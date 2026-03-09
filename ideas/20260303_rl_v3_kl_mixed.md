# 20260303: RL-v3 with KL Regularization + Mixed Tasks

## Hypothesis
Adding KL regularization (anchor regularization: L2 penalty on LoRA weight drift from SFT reference) will stabilize GRPO training and prevent the reward oscillation observed in RL-v2 (0.730 → 0.213 → 0.761). Training on mixed tasks (NIAH + multi-NIAH) will produce a model that generalizes across benchmarks rather than overfitting to one.

## Method
- Start from SFT-v2 checkpoint (90% NIAH, 57.9% multi-NIAH)
- GRPO with K=4 trajectories per prompt
- KL coefficient: 0.02 (anchor regularization on LoRA weights)
- Mixed tasks: 50% NIAH (5K-50K), 50% multi-NIAH
- 40 steps, batch 2 prompts/step, LR 5e-6
- Recall reward for multi-NIAH, composite reward for NIAH

## Expected Outcome
- RL-v3 NIAH ≥ 92% (match or beat RL-v2)
- RL-v3 Multi-NIAH ≥ 58% (beat SFT, unlike RL-v2 which regressed to 54.6%)
- More stable training dynamics (less reward oscillation)
- KL loss should increase gradually, not spike

## What Would Disprove This
- RL-v3 Multi-NIAH < 54.6% (worse than RL-v2 despite KL)
- Reward oscillation same magnitude as RL-v2
- KL penalty too strong → no learning (reward stays flat)

## Results

### Training Dynamics
| Step | Reward | Correct | Loss | Updates |
|------|--------|---------|------|---------|
| 3    | 0.538  | 5/8     | 0.078 | 4 |
| 6    | 0.663  | 6/8     | -0.160 | 8 |
| 9    | 0.725  | 7/8     | 0.158 | 3 |
| 12   | 0.663  | 6/8     | 0.016 | 8 |
| 15   | 0.763  | 8/8     | -0.106 | 4 |
| 18   | 0.628  | 6/8     | 0.159 | 4 |
| 21   | 0.515  | 5/8     | 0.062 | 4 |
| 24   | 0.808  | 8/8     | 0.063 | 4 |
| 27   | 0.425  | 4/8     | 0.000 | **0** |
| 30   | 0.785  | 8/8     | 0.132 | 2 |
| 33   | 0.653  | 6/8     | -0.036 | 8 |
| 36   | 0.650  | 6/8     | -0.135 | 4 |
| 39   | 0.613  | 6/8     | 0.025 | 4 |

Total time: 3297s (0.92 GPU-hours). Final avg reward: 0.875.

### Key Finding: KL Coefficient Too Small
The KL regularization (L2 on LoRA weight drift) stayed at 0.000000 throughout training. The issue: normalizing by 17.4M parameters makes the per-parameter L2 negligible (order 1e-10). The coefficient of 0.02 times this is effectively zero.

### Mixed-Task Training Helps Somewhat
Despite ineffective KL, the oscillation amplitude is smaller than RL-v2:
- RL-v2: range [0.213, 0.761] (amplitude 0.548)
- RL-v3: range [0.425, 0.808] (amplitude 0.383)
Mixed-task training may provide implicit regularization by requiring the model to maintain performance on both NIAH and multi-NIAH.

### Step 27: Collapse Event
At step 27, all trajectories got the same reward (Loss=0.0, Updates=0). This is the same GRPO collapse as RL-v1, showing it can happen even with continuous rewards.

### Eval Results

| Model | NIAH-100 | M-NIAH-24 | DocCls-20 | Average |
|-------|----------|-----------|-----------|---------|
| Base | 72.0% | 38.3% | 80.3% | 63.5% |
| SFT-v2 | 90.0% | 57.9% | 82.4% | 76.8% |
| RL-v2 | 92.0% | 54.6% | 81.4% | 76.0% |
| **RL-v3** | **90.0%** | **41.4%** | **83.9%** | **71.8%** |
| STaR (SFT-v3) | 87.0% | 58.4% | 83.4% | 76.3% |

RL-v3 NIAH by length: 5K=80%, 10K=95%, 20K=85%, 50K=90%, 100K=100%
RL-v3 M-NIAH by needles: 3=50%, 5=57.5%, 8=26.6%, 10=30%

### Degenerate Outputs
RL-v3 produces severely degenerate outputs on multi-NIAH:
- Repeating values: "4H7-PURPLE" repeated >100 times
- Sequential numbers: "3,4,5,...,280..."
- Template leakage: "SECRET_CODE_[NAME]: [VALUE]" as literal output
- Worse than base model (41.4% vs 38.3%)

## Conclusion

**Hypothesis DISPROVED.** RL-v3 multi-NIAH = 41.4% — dramatically worse than SFT (57.9%) and even worse than the untrained base (38.3%). The KL regularization was ineffective due to normalization by 17.4M parameters. Mixed-task training provided some implicit regularization (reduced oscillation amplitude) but was insufficient to prevent catastrophic policy collapse on multi-NIAH.

**Key lessons:**
1. Weight-space L2 regularization is an inadequate KL proxy for LoRA models. Per-parameter penalties are negligible when d=17.4M. Need either: unnormalized L2, much higher beta (>>100), or proper token-level KL.
2. GRPO without effective regularization produces degenerate repetitive outputs that are worse than no training at all.
3. STaR (iterative SFT) on the same tasks achieves 58.4% multi-NIAH — proving the data is not the problem, the optimization algorithm is.
4. RL-v3 achieves 100% on 100K NIAH (best ever) but at the cost of catastrophic multi-NIAH regression. The optimization is not just failing to improve — it is actively destroying capabilities.
