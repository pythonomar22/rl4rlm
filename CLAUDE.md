# CLAUDE.md

This is a research project to train natively recursive language models (RLMs) via reinforcement learning. The RLM paper (Zhang, Kraska, Khattab 2026) showed that SFT on filtered trajectories improved Qwen3-8B by 28.3% on long-context tasks. We believe RL can go much further. Read `PAPER_REFERENCE.md` for algorithms, prompts, and findings.

## ⚠️ Cluster Safety

**Read `CLUSTER_SAFETY.md` before running anything.** Shared 8×H200 server. GPUs 6 and 7 only. Check they are free before every use. Release immediately after. No external API calls. No inference servers. No interaction with processes on GPUs 0–5. No exceptions.

## ⚠️ Think Like a Researcher

You are not just writing code — you are running experiments on a system with many moving parts. The most dangerous bugs produce plausible-looking but wrong results.

**Read the actual outputs.** When something doesn't work, look at what the model actually generated. Print the raw text. If the model outputs `"Great! Here's the code:\n```python\ndef solve():\n```"` but you're trying to execute the raw response as Python, the fix is to change the prompt or parse the output — not the execution engine. The model's raw output is ground truth for debugging.

**Read logs line by line when stuck.** Not summaries. Not the last 5 lines. Scroll up. The root cause is almost always upstream of where symptoms appear.

**Maintain intuition about magnitudes.** Training loss of 0.001 on step 1 → something is wrong. A 2B model scoring 90% on a task GPT-5 scores 62% on → data leakage or eval bug. Generation taking 0.1s for 100K tokens → not actually processing context. Always ask: does this number make sense?

**Look at actual samples, not just metrics.** Before celebrating a reward increase, read 10 trajectories. Is the model solving the task or gaming the format?

**Iterate small with validation.** Build the REPL → test with a hardcoded trajectory. Build eval → test with a known-correct answer. Build the reward → score 5 trajectories by hand. Validate each piece before composing.

**Fix prompts before fixing code.** 90% of "the model can't do X" is "the prompt doesn't clearly ask for X." A 2B model needs very clear, simple instructions. If it's wrapping code in markdown, tell it not to. If it's writing plans instead of code, tell it to write code. Iterate the prompt on 3-5 examples before changing anything else.

**Diff against what worked.** When something breaks, `git diff` first. Know exactly what changed.

## Hardware & Model Decisions

**Training target: Qwen3.5-2B (instruct).** ~4GB in bf16. Full fine-tune + AdamW ≈ 16GB. On a 143GB H200, this leaves >125GB headroom.

**Why 2B:** Fast to train, fast to generate rollouts, fast to iterate. If RL can make a 2B model a competent RLM, that's a strong result. If it's too weak, scale up with the same pipeline.

**No servers. No APIs. Everything in-process.** Load models with `transformers` or `vllm.LLM()` directly. Generate with `model.generate()`. No HTTP, no ports, no sglang, no vLLM server mode. This is simpler, safer (no orphan processes), and sufficient for a 2B model that fits trivially in memory.

**Teacher model for trajectory collection:** Temporarily load a larger model (e.g., Qwen3.5-9B on one GPU, or Qwen3.5-27B across both GPUs 6+7) in-process. Collect trajectories. Unload it. Then load and train the 2B. This is a one-time data collection step — not a persistent service.

Alternatively, skip the teacher entirely and use STaR-style bootstrapping: generate rollouts with the 2B itself, filter correct ones, train on them, repeat. The 2B may be good enough as its own teacher after SFT warm-start.

**Sub-call model at eval/rollout time:** Qwen3.5-2B itself. Same model, same process, same weights. A sub-call is just another `model.generate()` call. No separate model needed.

**GPU allocation:**
- Training the 2B: 1 GPU (~16GB). Use the other for anything else or leave it free.
- If using 2 GPUs: one for training, one for rollout generation (batch generate → train on batch → repeat). But with a 2B model, sequential on 1 GPU is fast enough to start.
- Loading a teacher model (9B/27B) for trajectory collection: may need 1 or 2 GPUs depending on model size. Free them when done.

## What Is an RLM

An RLM treats the user prompt as part of the environment. Given a prompt P, the RLM loads it into a persistent Python REPL, then writes code to peek into, decompose, and recursively invoke itself over slices of P.

1. **Prompt in the environment.** P is a REPL variable. The root LLM sees only metadata (length, prefix, available functions).
2. **Output in the environment.** Final answer via `FINAL(answer)` or `FINAL_VAR(variable_name)`.
3. **Symbolic recursion.** `llm_query()` callable inside loops over slices of P.

## The Training Insight

Leaf sub-calls are just regular LLM requests. The hard part is the **root model's** ability to write correct REPL code and decide when to sub-call. Focus training here.

For a 2B model: it needs to produce valid Python that uses `context`, `llm_query()`, `print()`, and terminates with `FINAL()`/`FINAL_VAR()`. It does NOT need general coding ability — just this specific API.

## Project Goals

1. Build the RLM scaffold (REPL, eval harness, trajectory collection).
2. Collect trajectories (teacher model or STaR bootstrapping).
3. SFT warm-start Qwen3.5-2B (stepping stone, not deliverable).
4. **Train via RL on trajectory-level rewards from the SFT checkpoint.**
5. Evaluate and analyze where RL improves over SFT.

## Critical Invariants

1. **The REPL is the source of truth.** Full prompt P never enters the context window. If P is in messages, it's broken.
2. **Trajectory quality > quantity.** Filter and validate before training.
3. **Evaluate with the full scaffold.** RLM eval requires the REPL.
4. **Simple rewards first.** Binary trajectory-level. Add complexity only after understanding failure modes.
5. **Track costs.** GPU hours, tokens generated, wall-clock time. Report quartiles.

## Pipeline

```
Phase 0: Infrastructure
  REPL environment, eval harness, trajectory tooling.
  Validate each piece with hardcoded test cases.

Phase 1: Data Collection
  Option A: Load teacher model (9B/27B) on GPUs 6+7. Run as RLM on tasks.
            Collect trajectories. Unload teacher. Filter and clean.
  Option B: Skip teacher. Use Qwen3.5-2B directly with STaR bootstrapping.
  Either way: fix FINAL/FINAL_VAR template mistakes before training.

Phase 2: SFT Warm-Start
  LoRA fine-tune Qwen3.5-2B on filtered trajectories. GPU 6 or 7.
  Quick validation: run as RLM on a few tasks, confirm improvement.
  This is the RL starting checkpoint.

Phase 3: RL Training (The Main Event)
  Generate on-policy rollouts: load model, run RLM loop, score outcomes.
  Update with GRPO. Iterate.
  LOOK AT ACTUAL TRAJECTORIES EVERY FEW ITERATIONS.

Phase 4: Evaluation & Analysis
  All benchmarks, multiple context lengths.
  Compare: base → SFT → RL.
  Analyze trajectories qualitatively.
```

## Benchmarks

| Benchmark | Complexity | Metric |
|-----------|-----------|--------|
| S-NIAH (50 tasks) | O(1) | Accuracy |
| BrowseComp+ (150 tasks, 1K docs) | O(1) multi-hop | Accuracy |
| OOLONG trec_coarse (50 tasks) | O(n) | Custom score |
| OOLONG-Pairs (20 tasks) | O(n²) | F1 |
| LongBench-v2 CodeQA | Fixed | Accuracy |

**Reference numbers (Qwen3-8B from the paper — our 2B will be lower):**

| Method | CodeQA | BrowseComp+ | OOLONG | OOLONG-Pairs |
|-------|--------|-------------|--------|--------------|
| Base | 4.0 | 0.0 | 0.0 | 0.1 |
| RLM prompted | 26.0 | 2.0 | 24.0 | 4.3 |
| RLM SFT | 32.0 | 14.0 | 32.0 | 5.2 |

Research question: **does RL close the gap between prompted RLM and SFT, and between SFT and frontier, faster than scaling model size?**

## Lessons Learned from the Paper

### 1. Template Mistakes Dominate
16% of turns: FINAL with plan as answer. 13%: FINAL_VAR with literal text. Build `scripts/fix_templates.py` early.

### 2. Small Models Need Coding Ability
Qwen3-8B base couldn't write REPL code. **At 2B, this is our biggest risk.** Before any training: run Qwen3.5-2B on 5 tasks with the RLM prompt. Look at raw output. Can it write `context[:1000]` and `llm_query(...)`? If not, how close? This determines everything. Consider 

### 3. Prompts Must Match the Model
Each model needs its own prompt. The 2B needs simpler, shorter instructions than 8B. Validate on 5-10 tasks before committing.

### 4. Sub-Call Batching Matters
Scale batching guidance to the model's context window. For 2B, keep it conservative.

### 5. RLMs Are Worse on Short Inputs
Crossover ~16K tokens. Include short-context baselines.

### 6. Correct Answers Get Discarded
Models build correct REPL variables then return wrong root-generated text. RL target: reward FINAL_VAR(correct_variable).

## Directory Structure

```
rlm-training/
├── CLAUDE.md
├── CLUSTER_SAFETY.md
├── PAPER_REFERENCE.md
├── scaffold/
│   ├── repl.py                 # REPL environment
│   ├── rlm.py                  # Main RLM loop
│   ├── llm_query.py            # Sub-LLM (same model, in-process)
│   └── prompts/
├── data/                       # Gitignored except configs
│   ├── trajectories/
│   ├── filtered/
│   ├── sft/
│   └── collection_configs/
├── training/
│   ├── sft.py
│   ├── rl.py                   # GRPO
│   ├── rewards.py
│   └── configs/
├── eval/
│   ├── run_eval.py
│   ├── benchmarks/
│   ├── scoring.py
│   └── data/                   # Gitignored
├── analysis/
├── results/                    # Gitignored
│   └── {experiment}/{YYYYMMDD_HHMMSS}/
│       ├── eval_results.json
│       ├── trajectories/
│       ├── training_logs/
│       ├── cost_report.json
│       ├── config.json
│       └── stdout.log
├── ideas/                      # YYYYMMDD_short_name.md
└── scripts/
    ├── collect_trajectories.py
    ├── filter_trajectories.py
    └── fix_templates.py
```

## Required Outputs (Every Experiment)

1. **`eval_results.json`** — Scores per benchmark, task, context length.
2. **`cost_report.json`** — GPU hours, tokens generated, wall-clock time. Quartiles.
3. **`config.json`** — Full snapshot: model, hyperparams, prompt version, git hash.
4. **`stdout.log`** — Complete console output.
5. **`trajectories/`** — ≥10 sample trajectories as full REPL logs.

## Code Standards

- Simple, modular, readable.
- `uv` for package management. `tqdm` for progress. Type hints.
- Correctness first, performance second.
- **Print raw model outputs during debugging.** Not just metrics — the actual text.  Log everything. Every experiment that is useful for my eyes to look at: you will work autonomously, and I will come back and view these logs when I can, so log everything that I can see and that would be important for me to look at. 

## Git

- Commit early and often. Imperative mood, under 72 chars.
- Never commit weights, trajectories, or results. `.gitignore`: `data/`, `results/`, `*.pt`, `*.safetensors`.
- Branch for experiments: `feat/repl-scaffold`, `exp/sft-warmstart`, `exp/rl-grpo-v1`.
- Tag milestones.

## Experiment Tracking

Every experiment gets `ideas/YYYYMMDD_short_name.md`:

```markdown
# YYYYMMDD: Short Name
## Hypothesis
## Method
## Expected Outcome (with numbers)
## What Would Disprove This
## Results
## Conclusion
```

## When in Doubt

Stop and ask. And when something isn't working: read the logs, look at the model's actual output, check your prompts, reason about what the model is seeing, before changing code.