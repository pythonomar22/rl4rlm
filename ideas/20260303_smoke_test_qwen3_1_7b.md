# 20260303: Smoke Test Qwen3-1.7B as RLM

## Hypothesis
Qwen3-1.7B can produce valid Python REPL code that uses `context`, `llm_query()`, `FINAL()`, and `FINAL_VAR()` when given an appropriate system prompt.

## Method
Ran 5 simple tasks (needle search, line counting, word extraction, entity extraction, length check) through the full RLM scaffold with the 2B system prompt.

## Expected Outcome (with numbers)
At minimum 40% valid Python production (needed for SFT to be viable). Hoped for 60%+.

## What Would Disprove This
< 20% valid Python → model fundamentally can't write code, need to scale up or use constrained decoding.

## Results
| Metric | Value |
|--------|-------|
| Valid Python | 100% |
| Uses `context` | 80% |
| Uses `llm_query()` | 40% |
| Uses `FINAL`/`FINAL_VAR` | 100% |
| Correct answers | 20% (but 40-60% "close") |
| Total time | ~125s for 5 tasks |

### Detailed per-task analysis:

1. **needle_in_haystack: CORRECT** — Single turn, perfect RLM pattern: chunk → llm_query per chunk → filter → FINAL. This is the gold standard.
2. **count_lines: Close (103 vs 100)** — Correct code (`context.split('\n')`), answer off because prompt format adds 3 lines.
3. **first_and_last: Close** — Got "QUESTION: LAST_WORD" because context starts with "QUESTION:" prefix.
4. **simple_extraction: Template mistake** — Called `FINAL_VAR("name_and_city")` but variable doesn't exist. Also used `"{context}"` instead of `f"{context}"`. Exactly the failure mode the paper describes.
5. **length_check: Close (5061 vs 5000)** — Actually correct — context includes prefix.

### Key failure modes observed:
- **FINAL_VAR with wrong variable name** (paper: 13% of errors)
- **Missing f-string** in llm_query prompts
- **Inconsistent code blocks** — sometimes outputs raw code without ```repl wrapper
- **Thinking mode** — Qwen3 outputs long `<think>` blocks that waste tokens (handled by stripping)
- **Literal string interpolation** — writes `"{context}"` instead of `f"{context}"`

## Conclusion
**The 1.7B model CAN write REPL code.** All failure modes are fixable via SFT on correct trajectories and RL with outcome-based reward. This is a strong starting point — better than the paper's report of Qwen3-8B base (which scored near zero without SFT).

Note: Qwen3.5-2B (hybrid linear attention, 262K context) would be ideal but isn't supported by transformers yet. Qwen3-1.7B has 40K context, which is sufficient for the benchmarks but limits the length advantage of RLM. Should revisit when transformers adds qwen3_5 support.

### Model choice: Qwen/Qwen3-1.7B
- 1.72B parameters, 3.44GB in bf16
- 40K max context (40960 tokens)
- Supports thinking mode (`<think>...</think>`)
- qwen3 architecture, fully supported by transformers 5.3.0+

### Next steps
1. Improve prompt to fix observed failure modes (f-string instructions, variable naming)
2. Build trajectory collection pipeline using teacher model (Qwen3.5-9B or self-bootstrapping)
3. SFT warm-start on filtered trajectories
4. RL training
