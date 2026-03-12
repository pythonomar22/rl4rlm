# Format Rigidity: Root Cause of Training Regressions

**Date:** 2026-03-11

## Finding

Trajectory analysis of cross_doc_compare reveals a systematic pattern explaining why trained models regress on extraction tasks: **RL training teaches overly rigid formatting in sub-call code.**

## Evidence from Cross-Doc Trajectories

### Base Model Pattern (43% accuracy)
```python
# Loose extraction — tolerates noisy sub-call responses
answer = llm_query(f"Extract all employee names from this document...\n{chunk}")
for line in answer.strip().split("\n"):
    if line:
        names.add(line.strip())
```
- Accepts any reasonable format from sub-calls
- Filters in aggregation step, not parsing step
- Robust to sub-call output variability

### V4-s5 Trained Pattern (28.6% accuracy)
```python
# Strict format requirements
answer = llm_query(f"Extract ALL names. Format: 'ORG: Full Name'. Return ONLY names...\n{chunk}")
if "ORG: " in line:
    parts = line.split("ORG: ", 1)
    org, name = parts[0].strip(), parts[1].strip()
    if "Olympus" in org:
        olympus_employees.add(name)
```
- Requires specific format ("ORG: Name") from sub-calls
- Fails silently when format isn't matched
- Returns "No X found" instead of empty (false confidence)

## Why RL Training Causes This

1. **Reward optimization for clean execution**: Code that parses structured output and runs without errors gets higher reward than code with KeyError/ValueError fallbacks
2. **Template convergence**: SC-GRPO sees the same task type multiple times → model converges on a "winning" code template that worked for some instances
3. **No negative signal for format failures**: When strict parsing misses data, the model gets low reward but doesn't learn WHY — it just sees "different code = lower reward"
4. **Sub-call non-determinism**: Sub-calls are stochastic — the same prompt may return data in different formats. Training rewards averaged over K trajectories favor templates that work "most of the time" even if they fail on format edge cases

## Task Types Affected

| Task Type | Base Pattern | Trained Pattern | Why Trained Fails |
|-----------|-------------|-----------------|-------------------|
| cross_doc | Loose text → set operations | Strict "ORG: Name" → parse | Sub-call format varies |
| dataframe_qa | Direct CSV parsing in Python | chunk+llm_query for CSV rows | Sub-call adds noise to structured data |
| notebook_qa | Natural adaptive approach | Sequential cell tracking | Over-constrains processing order |
| multi_hop_qa | Chain-of-queries | Single-pass aggregation | Loses decomposition ability |

## Quantification

The format rigidity explains the regression pattern:
- Tasks improved by training (NIAH, doc_classify, event_counting): These are **search/classification** tasks where the answer is a label or count, NOT extracted text. Format doesn't matter.
- Tasks regressed by training (cross_doc, DFQA, notebook_qa, multi_hop): These require **exact extraction** of text/numbers from sub-call responses. Format rigidity kills accuracy.

## Potential Fixes

### 1. Reward shaping for extraction fidelity
Add reward component: "did the code correctly parse ALL relevant data from sub-call responses?" This requires a reference extraction to compare against.

### 2. Format-diverse training
During SC-GRPO trajectory collection, vary the sub-call model's temperature/system prompt so it returns data in different formats. This teaches the model to write format-robust parsing code.

### 3. Fallback-reward bonus
Reward code that includes fallback parsing (try strict format first, fall back to loose regex). Currently, simpler code (no fallback) gets rewarded because it runs faster.

### 4. Anti-rigidity KL
Increase KL penalty specifically for extraction-heavy tasks, preventing the model from drifting too far from the base model's loose parsing approach.

### 5. Hybrid architecture (partial fix)
Using base model for sub-calls (hybrid) helps because base sub-calls return data in more natural/varied formats. But if the root code still expects strict format, hybrid doesn't fully solve the problem.

## Recommendation

**Short term:** Use base model (no training) with task-specific strategies for extraction benchmarks (DFQA, notebook_qa). Training only helps search/classification tasks.

**Medium term:** Format-diverse training (fix #2) is the most promising — teach the model that sub-call output varies, so code must handle multiple formats.

**Long term:** The fundamental tension is that RL optimizes for expected reward under the training distribution, which biases toward rigid templates that work on average. A better objective would optimize for worst-case extraction fidelity.
