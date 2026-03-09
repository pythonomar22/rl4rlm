# PAPER_REFERENCE.md

Technical reference for the RLM paper (Zhang, Kraska, Khattab 2026).

Paper: arXiv:2512.24601v2 | Code: https://github.com/alexzhang13/rlm

## Algorithm 1: The RLM Loop

```python
def rlm(prompt: str, model, max_iterations: int = None) -> str:
    state = init_repl(prompt=prompt)              # P stored as REPL variable
    state = add_function(state, "llm_query", sub_rlm(model))
    hist = [metadata(state)]                      # Only metadata, never full prompt

    while True:
        code = model.generate(hist)               # Root generates ```repl ... ```
        state, stdout = repl_execute(state, code) # Execute in persistent REPL
        hist = hist + [code, metadata(stdout)]    # Only metadata of stdout appended
        if "Final" in state:
            return state["Final"]
```

**metadata(state):** context type, total char count, chunk lengths, available functions, short prefix.

**metadata(stdout):** first N chars + total length. Truncation forces use of REPL variables.

**Termination:** `FINAL(answer)` → literal string. `FINAL_VAR(variable_name)` → REPL variable value. Models confuse these constantly — key RL signal.

**Implementation note:** `model.generate()` here is a direct in-process call. `llm_query()` inside the REPL is also an in-process call to the same model (or a loaded teacher). No servers, no HTTP.

## Algorithm 2: The Anti-Pattern

```python
def bad_scaffold(prompt, model):
    hist = [metadata(actions), prompt]     # FLAW: prompt in context window
    while True:
        action, val = model.generate(hist)
        if action == "Finish": return val  # FLAW: output limited to window
        out = run(action, val)             # FLAW: sub-calls verbalized, not programmatic
        hist += [(action, val, out)]
        if tok(hist) > K: hist = compact(hist)
```

Three flaws: prompt in window, output bounded by window, sub-calls verbalized not programmatic.

## System Prompts

### GPT-5 RLM Prompt (Key Elements)
- References `context` variable and `llm_query` function
- Sub-LLM: ~500K chars capacity
- Three worked examples: chunking, batched chunking, markdown-header decomposition
- FINAL() and FINAL_VAR() termination
- "Execute immediately — don't just say 'I will do this'"

### Qwen3-Coder-480B Diff
Added batching warning: "aim for ~200k characters per call." Without this → 1000+ sub-calls.

### Qwen3-8B Diff
- Window: ~32k tokens. Sub-LLM: ~100k chars. Batching: ~10k-15k chars per call.
- 2-3 documents per sub-call. Example chunk: 1000 chars.

### For Qwen3.5-2B (We Must Write This)

The 2B has weaker instruction following and a smaller effective capacity. The prompt needs to be:
- **Shorter.** The model may get confused by long system prompts. Cut examples to 1-2.
- **More explicit.** Show exact code patterns. Don't assume it infers what `llm_query` returns.
- **More constrained.** Hard limit on sub-calls. Exact format for REPL code blocks.
- **Matched to its context window.** Check model card. Scale sub-call character limits.
- **Clear about output format.** "Write Python code inside \`\`\`repl blocks. Do NOT write explanations. Do NOT wrap in markdown. Just code."

**The 2B will have its own failure modes.** Validate the prompt on 5-10 tasks. Look at raw output. If the model produces broken Python or misuses FINAL — fix the prompt, not the scaffold.

## Paper's SFT Recipe

**Data:** Qwen3-Coder-480B teacher, 750 LongBenchPro tasks, 3 trajectories each → 2250 → filtered to ~1072 → per-turn SFT samples.

**Template fixing:** 16% incorrect FINAL (plan as answer), 13% incorrect FINAL_VAR (literal text). Programmatic fixing essential.

**Training:** prime-rl, batch 64, 300 steps, 48 H100-hours on Qwen3-8B.

**For our 2B:** ~4x smaller, ~12 GPU-hours for equivalent step count. Can afford more iteration.

## Results to Beat

### Qwen3-8B (reference — our 2B will be lower)

| Method | CodeQA | BrowseComp+ | OOLONG | OOLONG-Pairs |
|--------|--------|-------------|--------|--------------|
| Base | 4.0 | 0.0 | 0.0 | 0.1 |
| RLM prompted | 26.0 | 2.0 | 24.0 | 4.3 |
| RLM SFT | 32.0 | 14.0 | 32.0 | 5.2 |

### Frontier

| Method | CodeQA | BrowseComp+ | OOLONG | OOLONG-Pairs |
|--------|--------|-------------|--------|--------------|
| RLM(Qwen3-Coder-480B) | 56.0 | 44.7 | 48.0 | 23.1 |
| RLM(GPT-5) | 62.0 | 91.3 | 56.5 | 58.0 |

## Emergent Trajectory Patterns

1. **Chunking + recursive sub-calling.** Divide by newline/header/char, sub-call per chunk, aggregate.
2. **Regex filtering from priors.** Targeted regex before reading documents. Key to cost efficiency.
3. **Variable-based output construction.** Build answer in REPL variables, stitch sub-call outputs.
4. **Anti-pattern: redundant re-computation.** Build correct answer → discard → redo → return wrong answer. Primary RL target.

## RL Design Space

The paper did not do RL. This is our contribution.

### Reward

**Start: trajectory-level binary.** Correct answer = 1, wrong = 0. Benchmark's native metric.

**Later auxiliary signals:**
- Penalty for excessive sub-calls/turns
- Bonus for correct FINAL/FINAL_VAR usage
- Penalty for REPL errors (syntax, exceptions)
- Bonus for answers from REPL variables vs root-generated text

### Credit Assignment

1. **Trajectory-level** (start): same reward to all turns. GRPO at trajectory level.
2. **Turn-level** (later): per-turn credit.
3. **Hybrid:** trajectory reward + KL penalty against SFT checkpoint.

### Algorithm

**GRPO (start here):** Sample K trajectories per prompt, rank by reward, update. Proven on reasoning (DeepSeek-R1).

Later: PPO, STaR (generate → filter correct → SFT → repeat), DPO on trajectory pairs.

### Practical Concerns on 1-2× H200 with 2B Model

- **Memory is not the constraint.** 2B + optimizer ≈ 16GB. 143GB per GPU. The constraint is experiment turnaround.
- **Rollouts are the bottleneck.** RLM trajectories are multi-turn — each rollout needs multiple `model.generate()` calls plus REPL execution plus sub-calls. Batch prompts where possible.
- **Sub-calls = same model in-process.** `llm_query()` inside the REPL just calls `model.generate()` again. No separate server, no separate model. Same weights, same GPU.
- **GRPO needs K trajectories per prompt.** K=4-8 typical. With 2B being fast to generate, this is feasible.
- **Two-GPU option:** If both GPUs 6+7 are free, load the model on one and run rollout generation there while the other GPU does gradient updates. But sequential on 1 GPU is fine to start.
- **Generating training trajectories with a teacher:** Load a larger model (Qwen3.5-9B: ~18GB, fits on 1 GPU; or Qwen3.5-27B: ~56GB, fits on 1 GPU in bf16) temporarily. Generate trajectories. Unload. Then train the 2B. This is a one-time batch job, not a persistent service.

## Key Concern: 2B Coding Ability

The paper: "models without sufficient coding capabilities struggle as RLMs."

**The RLM API surface is tiny:** `context` (variable), `llm_query(prompt_str)` (function), `print()`, `FINAL()`/`FINAL_VAR()`. The model doesn't need general coding — just these 5 patterns.

**Mitigations:**
1. SFT on high-quality trajectories teaches the specific patterns.
2. RL reinforces correct code through outcome reward.
3. Very simple, explicit prompts. Show the model exactly what valid code looks like.
4. If the 2B can't produce valid Python at all: (a) scale to 4B or higher, or (b) constrained decoding.

**First thing to check:** Load Qwen3.5-2B, give it the RLM prompt, run 5 tasks. Read raw output. Can it write `context[:1000]` and `llm_query(...)`? This determines the entire approach.

## Benchmarks: Data Sources

| Benchmark | Reference |
|-----------|-----------|
| S-NIAH / RULER | arXiv:2404.06654 |
| BrowseComp+ | arXiv:2508.06600 |
| OOLONG / OOLONG-Pairs | arXiv:2511.02817 |
| LongBench-v2 CodeQA | arXiv:2412.15204 |
| LongBenchPro (training data) | arXiv:2601.02872 |

## Key References

- RLM paper + code: arXiv:2512.24601v2, github.com/alexzhang13/rlm
- prime-rl: github.com/PrimeIntellect-ai/prime-rl
- Qwen3.5 models: huggingface.co/collections/Qwen/qwen35
- DeepSeek-R1 / GRPO: arXiv:2501.12948
- STaR: arXiv:2203.14465