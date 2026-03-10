# Multi-Hop QA Failure Analysis — Why RLMs Need Multi-Step Decomposition

## The Problem

Base model + RLM scaffold achieves 50% on multi-hop QA (10 tasks, 2-3 hops, 10K-100K chars).
GRPO v2 step 5 achieves 50% — no improvement yet.

## Failure Pattern: Single-Pass Chunking

Every failed task (5/10) uses the same strategy:

```python
chunk_size = 20000
overlap = 2000
results = []
i = 0
while i < len(context):
    chunk = context[i:i+chunk_size]
    answer = llm_query(f"Find [FINAL ANSWER] in this text. If not found, say 'NOT FOUND'.\n\n{chunk}")
    if "not found" not in answer.lower():
        results.append(answer.strip())
    i += chunk_size - overlap

result = results[0] if results else "Not found"
FINAL_VAR("result")
```

The model asks each chunk for the **final answer** directly. For 2-hop questions like:
- "What department does the winner of the Innovation Award work in?"
- Fact 1 (pos 0.2): "Alice Johnson won the Innovation Award"
- Fact 2 (pos 0.7): "Alice Johnson works in R&D"

The sub-call gets ONE chunk and tries to answer the whole question. If Fact 1 and Fact 2 are in different chunks, no single chunk contains enough information.

## Why This Happens

The model learned to do single-pass chunking from NIAH training:
- NIAH tasks: single fact, one chunk contains the answer → single pass works
- Multi-NIAH tasks: multiple independent facts → single pass still works
- Doc-classify tasks: each chunk can be classified independently → single pass works

**None of the current training tasks require multi-step reasoning**, so the model never learns it.

## What Multi-Step Strategy Looks Like

A correct approach would be:

```python
# Step 1: Find who won the award
winner = None
chunk_size = 20000
for i in range(0, len(context), chunk_size):
    chunk = context[i:i+chunk_size]
    result = llm_query(f"Who won the Innovation Award? Return just the name or 'NOT FOUND'.\n\n{chunk}")
    if "not found" not in result.lower():
        winner = result.strip()
        break

# Step 2: Find their department
if winner:
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i+chunk_size]
        result = llm_query(f"What department does {winner} work in? Return just the department or 'NOT FOUND'.\n\n{chunk}")
        if "not found" not in result.lower():
            FINAL(result.strip())
```

Key difference: **decompose the question** into independent lookups, then chain results.

## Training Signal for Learning Multi-Step

The RL reward is clear:
- Single-pass chunking on multi-hop → score 0 → negative advantage
- Multi-step decomposition on multi-hop → score 1 → positive advantage

The model needs enough multi-hop examples in training for the RL signal to push it toward multi-step strategies. This is why GRPO v3 includes 15% Multi-Hop QA in the training mix.

## Difficulty Scaling

| Hops | Doc Length | Success Rate (base) | Why |
|------|-----------|--------------------|----|
| 2 | 10K | ~100% | Both facts in same chunk |
| 2 | 20K | ~80% | Usually in same chunk |
| 2 | 50K | ~40% | Often in different chunks |
| 2 | 100K | ~20% | Almost always in different chunks |
| 3 | Any | ~40% | 3 facts rarely in same chunk |

## Connection to Paper's RLM Value Proposition

This is arguably the strongest argument for RLMs:
1. **Standard LLMs** can't solve this because facts exceed context window
2. **RAG** can retrieve relevant chunks but doesn't chain reasoning across them
3. **RLMs** can iteratively search, extract intermediate results, and chain them

The multi-hop QA improvement (if achieved) would be the key paper result.

## GRPO v3 Predictions

With 15% Multi-Hop QA in training:
- The model should discover multi-step strategies within 5-10 steps
- Key indicator: multi-hop score improving from 50% while NIAH doesn't regress
- If successful, this validates the core RLM thesis: RL can teach recursive reasoning
