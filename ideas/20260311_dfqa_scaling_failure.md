# DFQA Sub-Call Scaling Failure

**Date:** 2026-03-11

## Finding

RL-trained models systematically fail on large DataFrame QA tasks because training teaches chunk+llm_query patterns that don't scale to large tabular datasets. This is a structural problem, not a training distribution issue.

## Evidence (Clean H2H, all seed-offset 10000)

| Data Size | Context | Base | V10-s5 | V11-s5 | Pattern |
|-----------|---------|------|--------|--------|---------|
| 5t/30d | 7K chars | 40% | 76% | 60% | llm_query works (fits in sub-call context) |
| 15t/60d | 54K chars | 40% | 20% | 0% | Sub-calls timeout on large chunks |
| 30t/120d | 217K chars | 80% | ~0% | ~0% | Context overflow (185K+ tokens per sub-call) |
| 50t/250d | 753K chars | 56% | ~0% | ~0% | Same context overflow |

**Reversal**: Training IMPROVES small tasks (+36pp on 5t) but REGRESSES large tasks (-80pp on 30t+).

### Task-level detail (V10-s5 vs Base, clean h2h)
| Task | Size | Type | Base | V10-s5 | Winner |
|------|------|------|------|--------|--------|
| 0-4 | 5t | mixed | 40% | 76% | V10-s5 (+36pp) |
| 5-9 | 15t | mixed | 40% | 20% | Base (+20pp) |
| 10-14 | 30t | mixed | 80% | 20% | Base (+60pp) |
| 15-19 | 50t | mixed | 56% | ~0% | Base (+56pp) |

Base model IMPROVES with larger contexts (40%→80%→56%) because Python parsing works better with more data.
Trained model DEGRADES (76%→20%→20%→~0%) because llm_query sub-calls overflow.

## Root Cause

### Base Model Pattern (works at any scale)
```python
# Parse CSV directly in Python
lines = context.strip().split("\n")
header = lines[0].split(",")
for line in lines[1:]:
    fields = line.split(",")
    # Direct computation in Python
```
- No sub-calls needed for data extraction
- Scales to any context size (Python string ops are fast)
- Only uses llm_query for natural language understanding

### Trained Model Pattern (breaks at 15t+)
```python
chunk_size = 30000
for i in range(0, len(context), chunk_size):
    chunk = context[i:i+chunk_size]
    answer = llm_query(f"Find the ticker with highest volume in this data:\n{chunk}")
    # Parse answer...
```
- Each sub-call gets 30K chars → 25K+ tokens
- Sub-call model processes 25K tokens to extract a number
- With 54K+ context → multiple sub-calls → each times out
- With 217K+ context → single chunk exceeds model context (65K tokens)

## Why RL Teaches This

1. **Training tasks are small**: Most DFQA training examples use 5t/30d (7K chars) or 15t/60d (54K chars)
2. **Reward doesn't penalize approach**: A correct answer from llm_query gets the same reward as Python-only
3. **llm_query pattern generalizes across tasks**: The model learns chunk+llm_query as a universal pattern from NIAH, doc_classify, etc.
4. **No negative signal for timeouts during training**: Timeouts produce 0 reward but the model doesn't learn WHY

## Potential Fixes

### 1. Explicit "no llm_query for tabular data" system prompt
Add to system prompt: "For CSV/tabular data, parse directly in Python using string operations. Do NOT use llm_query for data extraction from structured formats."
- Pros: Easy to implement, eval-time only
- Cons: Needs task detection; might hurt other tasks

### 2. Large DFQA tasks in training distribution
Include 30t/120d and 50t/250d tasks at 5% weight. The timeout penalty will teach the model to avoid llm_query for large data.
- Pros: Model learns the right pattern
- Cons: Expensive (each task takes 10+ min to generate)

### 3. Hybrid architecture for DFQA
Use base model for sub-calls → base model's Python parsing works naturally.
Already tested: V10-hybrid DFQA reward = 0.754 vs V10 standard = 0.135 (5.6x).
But hybrid at eval time requires knowing which tasks need it (oracle knowledge).

### 4. Reward bonus for Python-only solutions on tabular data
During training, if the task type is dataframe_qa and the code doesn't use llm_query, add +0.1 reward bonus. This directly incentivizes Python-only patterns for tabular tasks.

## Recommendation

Short term: Add "direct Python for structured data" instruction to system prompt for DFQA eval.
Medium term: Include large DFQA tasks in training with explicit timeout penalty.
Long term: The model should learn to detect structured vs unstructured data and choose the appropriate approach (Python vs llm_query). This is a meta-cognitive skill.
