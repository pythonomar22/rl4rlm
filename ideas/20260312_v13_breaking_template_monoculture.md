# V13: Breaking Template Monoculture

**Date:** 2026-03-12

## Problem

Trajectory analysis of V12 training (47 trajectories) revealed:
- **100% template monoculture**: Every single trajectory uses `while i < len(context): chunk = context[i:i+chunk_size]; answer = llm_query(...)`
- **91% single-turn**: Only 4/47 trajectories use 2+ turns (those that do are 100% correct)
- **6.4% gibberish**: MoE routing failure produces multilingual noise
- **No algorithmic diversity**: Zero use of regex, Python string ops, binary search, or two-pass verification
- **Deduplication is the key skill gap**: For counting tasks, reward 0.03 vs 0.72 depends on `set()` usage

Root cause: The system prompt had 5 examples, ALL using the same chunk+llm_query+Counter template. The model learned one algorithm.

## Changes

### 1. System Prompt V2 (most impactful)
- Added "Choosing Your Approach" decision guide
- Example 2: Python-direct parsing for structured/tabular data (NO llm_query)
- Example 3: Counting with `set()` deduplication
- Example 5: Cross-document comparison with TWO separate passes
- Key Tips section emphasizing deduplication, verification, error recovery

### 2. Per-Turn Credit Assignment
Function: `compute_per_turn_advantages(trajectory_dict, base_advantage)`

Successful trajectories:
- FINAL turns: 1.5x advantage (they produce the answer)
- llm_query turns: 1.0x (real work)
- Setup/print turns: 0.7x (scaffolding)
- Error turns: 0.3x (lucky recovery)

Failed trajectories:
- Timeout turns: 1.5x blame (biggest failure mode)
- Error turns: 1.2x
- FINAL turns: 1.3x (wrong answer is key failure)
- Other: 1.0x

Normalized so mean = 1.0 (preserves total gradient magnitude).

### 3. Gibberish Filter
Function: `is_gibberish_trajectory(trajectory_dict)`
- Checks each turn for >15% non-ASCII characters
- Removes before gradient computation (wasted compute otherwise)

### 4. Drop Broken KL Penalty
- Previous: `|mean_logprob - ref_mean_logprob|` (not real KL)
- Now: monitoring only, no reward penalty
- PPO clipping in Tinker's loss function already constrains policy drift

## V13 Config
- From BASE model (fresh LoRA)
- 15 steps (save every 1)
- LR 1e-6
- K=8, batch=4
- Distribution: niah 20%, event_counting 20%, hard_multi_hop 20%, doc_classify 15%, multi_hop_qa 15%, key_value_retrieval 10%
- All V13 features: credit assignment, gibberish filter, no KL, V2 system prompt

## Parallel: RS-SFT Collection
Also collecting 6×20×8 = 960 trajectories from base model for rejection sampling SFT.
If the best trajectories (score>0.9) are filtered and used for SFT, this avoids the negative gradient problem entirely.

## Hypotheses
1. V2 system prompt will produce trajectories with >1 distinct algorithmic approach (vs 0 currently)
2. Multi-turn trajectories will increase from 8.6% to >20% due to Example 4/5 and credit assignment
3. Event counting accuracy will improve due to explicit dedup teaching
4. Gibberish filter will save ~6% of wasted training compute
5. Per-turn credit will especially help hard_multi_hop (only benchmark that benefits from multi-turn)

## Success Criteria
V13-s5 should show:
- Average accuracy > 62% (vs V4-s5's 61.4% and base's 60.9%)
- No benchmark regresses more than 5pp from base
- NIAH ≥ 70%, Doc-Classify ≥ 95%, Event Counting ≥ 55%
