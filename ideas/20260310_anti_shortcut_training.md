# Anti-Shortcut Training: Making RL Reward Genuine RLM Behavior
**Date:** 2026-03-10

## The Problem: RL Teaches Shortcuts, Not RLM

Analysis of ALL saved trajectories (v1-v4, steps 5-30) reveals a devastating pattern:
**RL training teaches the model to AVOID recursive behavior.**

### Evidence
- v1 step 10: model used 2000-char chunks with 4 sub-calls on 5K doc (genuine multi-chunk)
- v1 step 30: model uses 5000-char chunks with 3 sub-calls on 10K doc (still multi-chunk)
- v2-v4: model sets chunk_size = context_length, making 1 sub-call (shortcut)

### Why This Happens
1. **Sub-call window (~30K chars) is larger than most training contexts**
2. A single `llm_query(f"... {context}")` gets reward 0.915 when correct
3. Multi-chunk processing also gets 0.915 when correct, but takes longer
4. RL gradient favors simpler solutions → single-pass dominates
5. The model is correctly maximizing reward — WE are wrong about the reward structure

### The Fix: Three Complementary Approaches

## 1. Minimum Context Length = 50K (Mandatory)

If the context always exceeds the sub-call window, single-pass is physically impossible.
The model MUST chunk to succeed.

- NIAH: 50K-500K (was 5K-100K)
- Multi-NIAH: 50K-200K
- Doc-Classify: 50K-200K
- Multi-Hop: 50K-150K
- Hard Multi-Hop: 100K-300K
- DataFrame QA: 50K-200K (larger CSV)
- Code Debug: keep as-is (code context is naturally short)
- Notebook QA: 50K-150K

Exception: code_debug can stay short because code snippets are naturally short.

## 2. Sub-Call Count Reward Bonus (Novel)

Add a reward bonus proportional to the number of meaningful sub-calls:
```python
n_subcalls = count_llm_query_calls(trajectory)
if context_length > 30000 and n_subcalls >= 2:
    subcall_bonus = min(0.10, 0.03 * n_subcalls)
else:
    subcall_bonus = 0.0
```

This directly rewards the model for using multiple sub-calls on long documents.
Combined with minimum context length, this creates a strong gradient toward chunking.

**Key insight:** We don't just want the RIGHT answer — we want the right answer
obtained through RECURSIVE PROCESSING. A model that somehow guesses correctly
from metadata is not a useful RLM.

## 3. Chunking Strategy Reward (Novel)

Analyze the code structure and reward proper chunking patterns:
```python
code = trajectory["turns"][0]["parsed_code"]
has_loop = "while" in code or "for " in code
has_chunk_var = "chunk" in code and "chunk_size" in code
has_overlap = "overlap" in code and int(overlap_value) > 0
has_aggregation = "results" in code and ("join" in code or "append" in code)

strategy_bonus = 0.0
if context_length > 30000:
    if has_loop: strategy_bonus += 0.02
    if has_overlap: strategy_bonus += 0.02
    if has_aggregation: strategy_bonus += 0.02
    if n_subcalls >= 3: strategy_bonus += 0.02
```

## 4. Multi-Pass Verification Reward (Novel)

Give a bonus when the model processes chunks TWICE:
- First pass: extract information
- Second pass: verify or refine

This rewards the most sophisticated RLM pattern without requiring decomposition.

## Expected Impact

With minimum 50K contexts:
- The model CANNOT avoid chunking on any task
- Every successful trajectory will demonstrate genuine RLM behavior
- RL will optimize the QUALITY of chunking, not whether to chunk
- Temperature diversity will produce different chunking strategies (varying chunk_size, overlap, aggregation)

## Risk: Slow Training

50K-500K contexts mean 3-10 sub-calls per trajectory × K=8 × batch_size=4 = 96-320 sub-calls per step.
At ~10s per sub-call, that's 16-53 minutes per step (vs current ~37 min).

Mitigation:
- Reduce batch_size to 3 for longer contexts
- Use K=6 instead of K=8 (tradeoff: less stable advantages)
- Implement `llm_batch()` for parallel sub-calls (from Prime Intellect's approach)

## For Paper

**This is a publishable insight:** Standard GRPO on RLMs optimizes away the very behavior
that makes RLMs valuable. The reward signal rewards correctness, but correctness on short
contexts doesn't require recursion. Only by forcing the model to operate beyond its context
window does RL training produce genuine recursive strategies.

"Training recursive models requires training contexts that mandate recursion."
