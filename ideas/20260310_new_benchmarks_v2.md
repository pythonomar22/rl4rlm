# New Benchmarks v2 — Expanding the Evaluation Suite
**Date:** 2026-03-10

## New Benchmarks Added

### 1. Notebook QA (`eval/benchmarks/notebook_qa.py`)
- **Motivation:** User specifically excited about RLMs in Jupyter-notebook style environments
- **Structure:** Simulates Jupyter notebooks with interleaved code cells, markdown, and outputs
- **Task types:**
  - `output_lookup` (40%): Find a specific cell output in a long notebook
  - `variable_trace` (30%): Trace a variable's value across multiple cells (multi-hop in code form)
  - `cross_cell` (30%): Cross-reference model comparison results across cells
- **Doc lengths:** 15K-60K chars, 15-256 cells
- **Why RLMs excel:** Must navigate structured documents, understand code execution flow, chain info across cells
- **Complexity:** O(K) for K relevant cells

### 2. Hard NIAH (`eval/benchmarks/hard_niah.py`)
- **Motivation:** Standard NIAH saturated at 80-100% — need harder variants
- **Difficulty types:**
  - `distractor` (40%): 3-7 adversarial similar-but-wrong values near the needle. E.g., "current access code" vs "previous access code" vs "test environment code"
  - `extreme_length` (30%): 200K, 500K, 1M character documents. Tests RLM at extreme scale.
  - `boundary` (30%): Needles at positions 0.01-0.03 or 0.97-0.99. Tests edge handling.
- **Why it matters:** Distractors test precision; extreme lengths test scalability; boundaries test chunking edge cases

### 3. Verbatim Copy (TODO — in progress)
- Faithful reproduction of specific paragraphs from context
- Tests precision of information retrieval (legal, medical use cases)
- Character-level similarity scoring

## Evaluation Plan

For each checkpoint, run all 8 benchmarks:
1. NIAH (standard)
2. Multi-NIAH
3. Doc-Classify
4. Multi-Hop QA
5. DataFrame QA
6. Code Debug
7. **Notebook QA** ← NEW
8. **Hard NIAH** ← NEW

## Training Integration

Both notebook_qa and hard_niah are registered in:
- `eval/run_eval.py` — for evaluation
- `training/rl_tinker.py` — score_trajectory can compute rewards (for future GRPO v4)

## Expected Results (Hypotheses)

### Notebook QA
- Base model: ~40-50% (can do output_lookup, struggles with variable_trace/cross_cell)
- GRPO v1 step 10: ~50-60% (improved search, still weak on multi-step)
- GRPO v3 (with multi-hop training): ~60-70% (if multi-step reasoning transfers)

### Hard NIAH
- Base model: distractor ~50-60%, extreme ~40-50%, boundary ~70-80%
- GRPO v1 step 10: distractor ~60-70%, extreme ~50-60%, boundary ~80-90%
- Key test: Can RLM handle 1M char documents? This would be a paper-worthy result.

## Paper Significance

The notebook QA benchmark is novel:
- No existing benchmark tests LLMs on Jupyter notebook comprehension at scale
- RLMs are uniquely suited: notebooks are structured, require navigation and code understanding
- Direct practical application: AI assistants that can analyze and answer questions about notebooks
- Could be contributed as a standalone benchmark

The hard NIAH with distractors tests a realistic failure mode:
- In real documents, the answer is often surrounded by similar-looking but wrong information
- Standard NIAH is too clean — one unique needle in uniform filler
- Distractor NIAH tests whether the model can distinguish current/valid from historical/invalid info
