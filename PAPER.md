# PAPER.md — Recursive Language Models

## Reference

**Title:** Recursive Language Models
**Authors:** Alex L. Zhang, Tim Kraska, Omar Khattab (MIT OASYS Lab)
**arXiv:** https://arxiv.org/abs/2512.24601 (v1: Dec 31, 2025; v2: Jan 28, 2026)
**Official code:** https://github.com/alexzhang13/rlm
**Blog (Oct 2025):** https://alexzhang13.github.io/blog/2025/rlm/

```bibtex
@misc{zhang2026recursivelanguagemodels,
    title={Recursive Language Models},
    author={Alex L. Zhang and Tim Kraska and Omar Khattab},
    year={2025},
    eprint={2512.24601},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Core Idea

LLMs suffer from **context rot** — performance degrades as context grows, even within the physical context window. The effective context window is much shorter than the physical one, and shrinks further for harder tasks.

RLMs solve this by **externalizing the prompt**:
1. Full prompt stored as Python variable `context` in a persistent REPL
2. LLM sees only metadata (length, prefix, available functions) — never the full context
3. LLM writes code to examine, chunk, and process context programmatically
4. LLM can call `llm_query(text)` — a recursive sub-call on a chunk
5. LLM terminates with `FINAL(answer)` or `FINAL_VAR(variable_name)`

Analogy: **out-of-core algorithms** — the model iteratively fetches and processes pieces of data, rather than loading everything at once.

## Algorithm 1: The RLM Loop

```
Input: prompt P, model M, max_iterations T
1. E ← InitREPL(P)              # P stored as `context` in REPL
2. meta ← Metadata(E)           # |P|, prefix(P), functions, variables
3. messages ← [system_prompt, meta]
4. for t = 1..T:
5.   response ← M.generate(messages)
6.   code ← ParseCode(response)     # Extract ```repl block
7.   result ← E.execute(code)        # Persistent REPL execution
8.   if result.terminated: return result.answer
9.   messages += [assistant=response, user=StdoutMeta(result) + Metadata(E)]
10. return None  # max iterations
```

### What the LLM Sees (Metadata)

```
Context length: 50000 characters
Context prefix:
QUESTION: Find all secret codes in this document.
DOCUMENTS: The weather was pleasant...
... [49500 more characters]

Available functions: llm_query(prompt_str), FINAL(answer), FINAL_VAR(variable_name)
Available variable: context (the full input, 50000 chars)
User variables:
  results = ['7X9-BLUE', '3K2-RED']
Turn: 3
```

### Recursive Sub-Calls

`llm_query(text)` invokes a fresh LLM instance — clean context window, no accumulated state. Typical pattern:

```python
chunks = [context[i:i+20000] for i in range(0, len(context), 20000)]
results = []
for chunk in chunks:
    answer = llm_query(f"Find secret codes in:\n\n{chunk}")
    if "not found" not in answer.lower():
        results.append(answer)
```

The model can also grep, regex, slice, use Python's full standard library — the REPL is a real Python interpreter.

## Key Results

- RLMs handle inputs **up to 100× beyond model context windows** (tested to 10M+ tokens)
- Even for short prompts that fit in context, RLMs **dramatically outperform** vanilla LLMs
- Cost per query is **comparable or cheaper** than direct calls
- Far **less degradation** as length and complexity increase

### Benchmarks Used

| Benchmark | Complexity | What it Tests |
|-----------|-----------|---------------|
| S-NIAH | O(1) | Find one needle in haystack |
| OOLONG (trec_coarse) | O(N) | Classify all documents in long context |
| OOLONG-Pairs | O(N²) | Pairwise comparison across documents |
| DeepDive | Multi-hop | Deep research requiring search + reasoning |
| LoCoDiff | O(N) | Track long git diff histories |

GPT-5 drops below 10% on LoCoDiff at 75K+ tokens. RLM maintains strong performance.

### Natively Recursive Model

The paper post-trains **RLM-Qwen3-8B**:
- Base: Qwen3-8B
- Training: SFT on correct trajectories → RL (GRPO)
- **+28.3% average improvement** over base Qwen3-8B
- Approaches vanilla GPT-5 quality on three tasks

**This is what we're replicating at larger scale** — our target is Qwen3.5-35B-A3B.

## Task Complexity Classes

| Class | Description | Example | RLM Strategy |
|-------|-------------|---------|--------------|
| O(1) | Answer in one location | Single NIAH | Search/grep, one sub-call |
| O(K) | K scattered pieces | Multi-needle | Systematic sweep, K sub-calls |
| O(N) | Must process everything | Classification, summarization | Chunk and iterate all |
| O(N²) | Cross-reference pairs | OOLONG-Pairs | Nested iteration |

## Training Recipe

### Phase 1: Trajectory Collection (STaR)

Run base/instruct model through RLM scaffold → score → filter correct ones → this is your SFT data.

Key decisions:
- Use the model itself (self-bootstrap) or a larger teacher
- Filter criteria: correct answer + proper termination + reasonable turns (<8) + few errors (<2)
- Convert to **per-turn samples**: `(conversation_history_so_far, assistant_code_response)`
- Only non-error turns become positive examples; error turns stay in context for later turns

### Phase 2: SFT

- LoRA fine-tuning on filtered trajectories
- Per-turn: loss only on assistant completion tokens, prompt masked
- Creates starting checkpoint for RL

### Phase 3: RL (GRPO)

From DeepSeek-R1's Group Relative Policy Optimization:
1. For each prompt, sample K trajectories (K=8)
2. Score each with reward function
3. Advantage = (reward - group_mean) / group_std
4. Policy gradient on trajectory tokens, conditioned on conversation context
5. KL constraint vs frozen reference model

**Critical:** Log-probs must be **conditioned on full conversation** — `log P(code | system_prompt, metadata, prior_turns)`. Not `log P(code)` unconditionally.

## Common Model Failures

| Failure | Frequency | Fix |
|---------|-----------|-----|
| FINAL() with plan/explanation | 16% | Template fixing, SFT on correct patterns |
| FINAL_VAR() with literal text | 13% | Convert to FINAL(), SFT |
| Missing f-string in llm_query | Common at 2B | Explicit prompt instruction, SFT |
| Reading entire context at once | Common | Prompt instruction + training |
| No llm_query sub-calls | Common at 2B | Worked examples in prompt, training |
| Code outside repl blocks | Common at 2B | Explicit format instructions |

Programmatic fixes in `scripts/fix_templates.py`. Training (SFT+RL) should eliminate most of these.

## Design Decisions That Matter

1. **Metadata, not context:** The model never sees the full prompt. This is the core insight — prevents context rot.

2. **Persistent REPL:** Variables survive across turns. The model can build up results incrementally.

3. **Recursive sub-calls:** `llm_query()` gets a fresh context window. No state leakage between sub-calls.

4. **Stdout truncation:** REPL output is truncated in feedback to the model (default ~1000 chars). Forces the model to use variables to store and access full output.

5. **No special tokens or schema:** The model outputs standard Python in markdown code blocks. No tool-call JSON, no function-calling format.

## Related Work

- **Context compaction:** Summarize context to fit. Loses information. RLMs never summarize — programmatic access.
- **RAG:** Retrieve chunks. Requires index. Can't handle tasks needing full-context reasoning.
- **Code agents:** Write code to process data. RLMs add recursive self-invocation.
- **Prime Intellect's RLM:** https://www.primeintellect.ai/blog/rlm — different implementation, integrated with `verifiers` and `prime-rl`. Tests on DeepDive, OOLONG, LoCoDiff, verbatim-copy. Worth studying.
- **LoongRL** (arXiv): RL for long-context reasoning. Different approach but related goal.
- **QwenLong-L1.5:** Post-training for long-context memory management.

## Our Contributions (Planned)

What makes our work different from the original paper:

1. **Larger model:** Qwen3.5-35B-A3B vs Qwen3-8B. Shows RLM training scales to bigger models.
2. **Qwen3.5 family:** Latest generation, potentially better base capabilities.
3. **MoE architecture:** First RLM on a MoE model. Cost-efficient for deployment (3B active).
4. **Tinker-based training:** Demonstrates the recipe is reproducible without a private GPU cluster.
5. **Extended benchmarks:** More diverse evaluation beyond the paper's benchmarks.
6. **Public release:** Open weights, open training code, reproducible recipe.
7. **Training insights:** What works, what doesn't, hyperparameter sensitivity analysis.

## Open Questions

- Does MoE architecture help or hurt RLM training? (Sparse experts might specialize differently)
- How far can context length extrapolation go? (Train on 50K, test on 5M?)
- Can the model learn multi-level recursion? (RLM → sub-RLM → sub-sub-RLM)
- What's the optimal sub-call model size? (Same model? Smaller? Larger?)
- How does thinking (Qwen3's `<think>` blocks) interact with code generation in the REPL?
- Can we combine RLMs with RAG for hybrid retrieval-generation?