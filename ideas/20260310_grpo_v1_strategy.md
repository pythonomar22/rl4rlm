# GRPO v1: Direct RL from Base Model
**Date:** 2026-03-10

## Motivation

SFT on 155 small-data trajectories caused catastrophic forgetting (see `20260310_sft_v1_v2_analysis.md`). The base Qwen3.5-35B-A3B is already strong (77.5% avg), so we pivot to GRPO RL directly from the base model.

## Strategy

**Online rejection sampling with advantage weighting** (approximate GRPO):
1. Sample K=8 trajectories per task using current policy
2. Score each trajectory
3. Compute group-relative advantages: `(reward - mean) / std`
4. Train on turns with positive advantages using cross_entropy + advantage weights
5. Refresh sampling client every 3 steps

This is equivalent to REINFORCE with advantage normalization, where we only train on above-average trajectories. True GRPO would use `importance_sampling` loss with old logprobs — we can add this later by capturing logprobs from TinkerModel.

## Hyperparameters

- **Model:** Qwen/Qwen3.5-35B-A3B (base, no SFT warmup)
- **LR:** 5e-6 base → ~5e-5 after LoRA scaling
- **LoRA rank:** 32
- **K:** 8 trajectories per task
- **Batch size:** 3 tasks per step (1 niah + 1 multi_niah + 1 doc_classify)
- **Steps:** 30 (exploratory)
- **Save every:** 10 steps
- **Temperature:** 0.8 (exploration)
- **Advantage threshold:** |advantage| > 0.1

## Task Mix Rationale

Equal mix (1/3 each) is critical to avoid the SFT failure mode:
- **NIAH (81% base):** Room to improve, especially on hard positions/lengths
- **Multi-NIAH (97.8% base):** Include as "anchor" — must not regress
- **Doc-Classify (53.6% base):** Biggest improvement target

## Expected Outcomes

- Doc-Classify should improve significantly (largest signal gap)
- NIAH should improve modestly
- Multi-NIAH should be maintained (near-ceiling already)
- If multi-NIAH drops, reduce its weight or use KL penalty

## Risks

1. **Slow trajectory collection:** 24 trajectories/step × ~15s each = ~6 min/step
2. **Cross_entropy ≠ true GRPO:** Only trains on positive advantages. May need importance_sampling later.
3. **Advantage variance:** With K=8, some groups will have all-same rewards → skipped
4. **LR too high/low:** Monitor loss and reward trends, adjust after 10 steps

## Next Steps After v1

1. If results are promising: extend to 100+ steps
2. Add logprob capture for true importance_sampling GRPO
3. Try DPO on collected trajectory pairs
4. Add harder tasks (longer documents, more needles)
5. External benchmarks (OOLONG)
