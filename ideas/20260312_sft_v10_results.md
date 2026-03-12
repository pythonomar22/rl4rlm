# SFT V10 Quick Evaluation Results (2026-03-12)

## Training Config
- Model: Qwen/Qwen3.5-35B-A3B
- Data: data/sft/sft_v8_balanced.jsonl (592 samples)
- LR: 5e-5 (effective: 5e-4 after LoRA scaling)
- Epochs: 2
- Session: 67acfbca-af2a-5a0a-816c-c6ee60a66c5f

## Loss Curve (Stable, No Spikes)
- Epoch 1 avg: 187.1 (started ~261, ended ~232)
- Epoch 2 avg: 116.3 (started ~131, ended ~99)
- Total time: 1113s (18.5 min)
- Previous SFT V8 (LR=1e-4) exploded at epoch 3; V9 (LR=1e-4) diverged in epoch 1
- LR=5e-5 is the right choice for this model

## Full Quick Eval Results (4 benchmarks, seed-offset=10000)
| Checkpoint | NIAH | Multi-NIAH | Doc-Classify | DFQA |
|-----------|------|-----------|-------------|------|
| Base model | 60.0% | 91.5% | 81.6% | 54.0% |
| EP1 (state-0148) | 45.0% | 63.3% | **87.2%** | 65.5% |
| EP2 (state-0296) | 55.0% | 70.9% | 71.5% | **76.5%** |
| V11-s5 (GRPO) | **80.0%** | 87.8% | **99.2%** | 40.0% |

## Key Findings
1. **SFT degrades search dramatically**: NIAH drops 15pp at EP1, recovers only partially at EP2
2. **SFT improves tabular parsing**: DFQA 76.5% at EP2 is the BEST of any model (even base+strategy 75%)
3. **SFT and GRPO complement**: SFT teaches format/parsing, GRPO teaches search — neither does both
4. **Doc-Classify inversely correlated with epochs**: EP1 87.2% > EP2 71.5% — more SFT hurts classification

## SFT V9 (More Training) = Catastrophe
- SFT V9 state-0100 (same data, LR=1e-4): NIAH 0%, Multi-NIAH 0%
- Confirms higher LR and more training completely destroys search capability

## Failure Mode
- SFT model answers "Not found" instead of chunking and searching context
- Occurs primarily on larger contexts (20K+) where systematic search is needed
- The model has learned SFT formats but lost the "try harder" exploration behavior

## Next Steps
- GRPO V15 launched from EP2 (state-0296) with conservative LR=5e-7, 10 steps
- Hypothesis: light GRPO refinement will push NIAH above 60% by reinforcing search
- If V15 preserves DFQA 76.5% while recovering NIAH, it validates the SFT→GRPO pipeline
- Key target: NIAH ≥ 60% + DFQA ≥ 60% simultaneously
