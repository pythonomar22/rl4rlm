# Training Data Strategy — What We Need Next
**Date:** 2026-03-10

## Current State

### What's Working (keep training on these)
1. **Doc-Classify**: 53.6% → 95% — model learned to process all N documents
2. **NIAH**: 81% → 85% — slight improvement from better search patterns
3. **Multi-NIAH**: 97.8% → 95% — maintained, slight regression acceptable
4. **Multi-Hop QA**: 50% → 70% — compound queries, but NOT multi-step decomposition yet

### What's Broken (need targeted training)
1. **DataFrame QA**: 80% → 50% — text RL hurts numerical capabilities
2. **Code-Debug**: 25% → 25% — no improvement, needs code-specific signal
3. **Multi-Hop (real decomposition)**: Model still does single-pass chunking

## Strategy: Targeted Task Design

### Priority 1: Force Multi-Step Decomposition
The model keeps doing single-pass chunking even on multi-hop tasks because:
- Sometimes it works (facts in same chunk) → positive reward
- Single-pass is simpler → model's prior is to try it first

To force multi-step learning:
- **Increase doc lengths**: Use 100K+ docs where facts are ALWAYS in separate chunks
- **Wider fact separation**: Place facts at 0.1 and 0.9 (not 0.3 and 0.7)
- **More hops**: 3-hop and 4-hop questions where single-pass fundamentally can't work
- **Fact dependency**: Each subsequent fact references the previous fact's entity
  (you MUST find fact 1 before you can search for fact 2)

### Priority 2: DataFrame QA Recovery
The model's numerical/analytical abilities regressed because:
- RL training uses text-search reward signals
- The model's code changes optimize for text search, not data analysis

Fix: Add more DataFrame QA to training mix. In GRPO v4:
- 20% DataFrame QA (up from 10%)
- Include multi-step DFQA tasks (find best sector → find top stock in sector)
- Reward partial credit for correct methodology even if answer is slightly off

### Priority 3: Code-Debug Training
Current code-debug reward is binary (found bug or not). The model needs:
- **Intermediate rewards**: Finding the right function (+0.3), identifying bug type (+0.3), giving correct fix (+0.4)
- **Systematic code scanning reward**: Bonus for examining all functions, not just the first few
- **Larger training proportion**: 15% in mix (up from 10%)

### Priority 4: Notebook QA Training
Already showing transfer learning (+15% without training). Adding to mix should help further:
- Variable trace tasks specifically reward multi-step computation
- Cross-cell tasks reward cross-referencing (similar to multi-hop)
- 10% in GRPO v4 mix

## Proposed GRPO v4 Task Mix

| Task Type | Percentage | Rationale |
|-----------|------------|-----------|
| NIAH | 15% | Reduce — already strong, prevent overfit |
| Multi-NIAH | 10% | Maintain |
| Doc-Classify | 10% | Near-ceiling, reduce |
| Multi-Hop QA (hard) | 20% | KEY — force decomposition on 100K+ docs |
| DataFrame QA | 15% | Recovery — include multi-step |
| Code Debug | 10% | Include with better reward signal |
| Notebook QA | 10% | New — variable trace + cross-cell |
| Hard NIAH | 5% | Extreme lengths + distractors |
| Verbatim Copy | 5% | Precision task |

## Data Collection Ideas

### Teacher Distillation for Multi-Step
Use Qwen3.5-397B-A17B (or Claude/GPT) to generate gold multi-step trajectories:
1. Give it a multi-hop task with explicit instruction to decompose
2. Collect the decomposed solution (Step 1: find X, Step 2: use X to find Y)
3. Use these as SFT data or as positive examples in DPO

### Synthetic Curriculum
Start v4 with easy tasks (short docs, 2-hop) and progressively increase:
- Steps 1-5: 10K-20K docs, 2-hop only
- Steps 5-10: 20K-50K docs, 2-3 hop
- Steps 10-20: 50K-100K docs, 2-4 hop
- Steps 20-30: 100K+ docs, 3-4 hop

### Negative Mining for DPO
Collect pairs from training:
- Positive: Multi-step trajectories that succeed on long multi-hop tasks
- Negative: Single-pass trajectories that fail on the same tasks
- Use DPO to directly contrast good vs bad strategies

## Connection to Paper Story

The paper narrative is:
1. Standard LLMs fail on long-context multi-hop reasoning
2. RLM scaffold helps but model still uses suboptimal strategies
3. RL training teaches the model better strategies
4. **Key result**: Model learns multi-step decomposition through RL

For this story to work, we MUST show the model learning multi-step decomposition,
not just getting lucky with compound queries. V3/V4 training should produce this.
