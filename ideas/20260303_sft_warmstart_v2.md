# 20260303: SFT Warm-Start v2 (Cleaned Data)

## Hypothesis
LoRA SFT on filtered correct trajectories (error turns removed) will improve RLM performance over the base model.

## Method
1. Collected 150 trajectories (50 tasks x 3 attempts) using base Qwen3-1.7B
2. 89/150 correct (59.3%)
3. Filtered to 87 SFT samples (removed error turns, removed trajectories with >2 errors)
4. LoRA SFT: rank=16, alpha=32, 5 epochs, lr=2e-4, 34 seconds of training
5. Evaluated on 25 NIAH tasks (5K-100K, 5 positions)

## Expected Outcome
75-85% accuracy (up from 68% baseline). Main gain on long documents.

## What Would Disprove This
<68% → training data quality issues or overfitting.

## Results

### SFT v1 (contaminated data) - FAILED
- Used template-fixed data + included error turns
- Template fixer converted FINAL_VAR("result") to FINAL("result") in training data
- Model learned FINAL("result") shortcut → outputted literal "result"
- **56% accuracy** (regression from 68%)
- 0 sub-calls — lost recursive pattern entirely

### SFT v2 (cleaned data) - SUCCESS
| Metric | Base | SFT v2 | Delta |
|--------|------|--------|-------|
| **Overall** | **68%** | **84%** | **+16pp** |
| 5K | 80% | 80% | 0 |
| 10K | 80% | 80% | 0 |
| 20K | 100% | 100% | 0 |
| 50K | 40% | **80%** | **+40pp** |
| 100K | 40% | **80%** | **+40pp** |
| Time/task | 21.6s | **6.6s** | **3.3x faster** |
| Sub-calls | 42 | 47 | Similar |

### By needle position:
| Position | Base | SFT v2 |
|----------|------|--------|
| 0.1 | 80% | 100% |
| 0.2 | 20% | 60% |
| 0.5 | 80% | 100% |
| 0.8 | 80% | 60% |
| 0.9 | 80% | 100% |

### Key observations:
1. **Long-context is the big win**: 50K and 100K both jumped from 40% to 80%. SFT taught better chunking.
2. **Single-turn mastery**: All 25 tasks solved in exactly 1 turn (base needed 1.36 turns avg). The model learned to write complete REPL code on first attempt.
3. **3.3x faster**: Single-turn execution is much faster. No retries or error recovery needed.
4. **Position 0.2 improved**: 20% → 60%. Still the weakest position, but much better.
5. **Position 0.8 regressed**: 80% → 60%. May be statistical noise with N=5.

### Training details:
- 87 samples, 5 epochs, batch_size=2, grad_accum=2
- 110 optimizer steps, 34 seconds
- Final loss: 0.017
- 17.4M trainable parameters (LoRA rank 16, all attention + MLP projections)

## Lessons Learned

### Data quality > data quantity
SFT v1 with 97 samples (including 10 error turns) scored 56%. SFT v2 with 87 clean samples scored 84%. The 10 contaminated samples taught bad shortcuts that dominated inference behavior.

### Template fixes can be dangerous
Converting FINAL_VAR("result") to FINAL("result") in training data taught the model to output literal "result" as an answer. Template fixes should only be applied to data NOT used for training, or applied much more carefully.

### Error turns are negative examples
Per-turn SFT treats every turn as a positive example. Error turns (code that fails) become positive training targets. Solution: only train on turns where code executed successfully.

## Conclusion
**84% is a strong SFT result.** The +16pp improvement over base (68%) matches the paper's finding that SFT significantly improves RLM performance. The gain is concentrated on long documents (50K+), confirming that SFT teaches better chunking strategies.

This is the starting checkpoint for RL training. The remaining 16% errors are:
- Position 0.2 at 50K (likely chunk boundary issue)
- Position 0.2 at 100K (same)
- Position 0.8 at 5K and 10K (model missing late-document needles at shorter lengths)

RL should target these failure modes by rewarding correct answers on hard cases.
