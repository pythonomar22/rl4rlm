# Paper Trajectory Examples
**Date:** 2026-03-10

## 1. DataFrame QA: Iterative Debugging (V4-s5, 4 turns, correct)

**Task:** "Which ticker in the Energy sector has the highest return over the period?"
**Expected:** PSX  **Got:** PSX ✓

**Turn 1:** Model tries `import pandas as pd` to parse CSV. Encounters parsing issue.
**Turn 2:** Re-parses context by finding CSV header line manually (`context.split('\n')`).
**Turn 3:** Debugs by printing DataFrame head. Finds data loaded correctly, columns exist.
**Turn 4:** Fixes merge issue (sector column lost during join), computes return per sector, finds PSX.

**Why this matters:** Base model does this in 1 turn and gets wrong answer (JNJ at v3-s5).
The persistent REPL allows iterative debugging that mirrors real data analysis workflows.

## 2. Multi-Hop QA: Compound Query Success (V4-s5, 1 turn, correct)

**Task:** "What technology is used by the project at the office managed by Patrick Hernandez?"
**Expected:** quantum computing  **Got:** quantum computing ✓

**Code:**
```python
chunk_size = 20000
overlap = 2000
results = []
i = 0
while i < len(context):
    chunk = context[i:i+chunk_size]
    answer = llm_query(f"Find the project based at the office managed by Patrick Hernandez and identify the technology it uses. Return ONLY the technology name. If not found, say 'NOT FOUND'.\n\n{chunk}")
    if "not found" not in answer.lower():
        results.append(answer.strip())
    i += chunk_size - overlap
result = results[0] if results else "Not found"
FINAL_VAR("result")
```

**Why this succeeds:** 50K doc → 3 chunks → both facts happen to land in same chunk.
**Why this fails on hard_multi_hop:** 150K doc → 8 chunks → facts always in different chunks.

## 3. Hard Multi-Hop: Distractor Trap (V3-s10, 1 turn, wrong)

**Task:** "What is the budget for the project led by X?"
**Expected:** $4.2 million  **Got:** $15.7 million ✗

The model scanned all chunks and found "$15.7 million" — the distractor answer.
The distractor entity (Y) is positioned closer to the question in the document,
so the model's scan finds it first and returns it.

**What decomposition would do:**
Turn 1: "Who leads the project?" → find X (from fact at position 0.15)
Turn 2: "What project does X lead?" → find Project Phoenix (from fact at position 0.75)
Turn 3: "What is Project Phoenix's budget?" → find $4.2M (correct)

## For Paper

These 3 trajectories illustrate the core thesis:
1. **RLMs enable iterative debugging** (DataFrame QA) → practical value
2. **Compound queries work on shorter docs** (Multi-Hop QA at 50K) → explains why base model succeeds
3. **Compound queries fail on longer docs** (Hard Multi-Hop at 150K) → decomposition gap
4. **RL teaches persistence but not decomposition** → key insight
