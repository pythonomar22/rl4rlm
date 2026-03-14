# V18 RS-SFT: Fixed doc_classify + multi_niah Data Gap (2026-03-13)

## Critical Bug Found

The trajectory mining script (`scripts/mine_eval_trajectories.py`) used `task_result.get("score", 0)` to identify correct trajectories. But two benchmarks use different score field names:
- **doc_classify**: uses `accuracy` field (not `score`)
- **multi_niah**: uses `recall` field (not `score`)

This means **ALL previous RS-SFT models (V10, V16, V17) were trained with ZERO doc_classify and ZERO multi_niah examples**. Despite this, V17 still scored 80.2% on doc_classify and 95.1% on multi_niah — these are entirely base model capabilities.

## Fix

Updated `mine_eval_trajectories.py` to check `score`, `accuracy`, and `recall` fields.

## Impact

- Before fix: 3,956 total correct trajectories (0 doc_classify, 0 multi_niah)
- After fix: 5,161 total correct trajectories (+858 doc_classify, +798 multi_niah)
- V18 training dataset: 4,660 balanced samples (was 3,644 in V17)
  - doc_classify: 400 samples (was 0)
  - multi_niah: 400 samples (was 0)

## V18 Training Config

- Same as V17: 3 epochs, lr 2e-5, LoRA rank 32, batch 4x4=16
- 873 steps (was 681 in V17)
- Session: d03d55e5-2f01-53e0-a154-92c3bd4809e9
- Estimated: ~3 hours

## Hypothesis

Adding 800 previously-missing trajectories for two benchmark types should:
1. Maintain or improve doc_classify (80.2% in V17, trained with 0 examples)
2. Maintain or improve multi_niah (95.1% in V17, trained with 0 examples)
3. Potentially improve other benchmarks via better generalization
4. The additional data diversity may help with V17's regressions on notebook_qa, verbatim_copy, hard_multi_hop
