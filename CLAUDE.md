# CLAUDE.md — Training a Natively Recursive Language Model

## What We're Building

We are training **the first open-weight natively recursive language model based on Qwen3.5**. When released publicly, this model will be able to solve long-context tasks that defeat even frontier models — by writing Python code in a persistent REPL, chunking and searching its input programmatically, and recursively calling itself on sub-problems.

The original RLM paper (Zhang, Kraska & Khattab, arXiv:2512.24601) trained RLM-Qwen3-8B and showed a 28.3% average improvement over base Qwen3-8B, approaching GPT-5 on long-context tasks. **We are going bigger and better** — targeting Qwen3.5-35B-A3B (MoE) on Tinker's training API, which removes all GPU/VRAM constraints.

See `PAPER.md` for the full paper reference. See `TINKER.md` for the Tinker API reference.

## Target Model

**Primary: `Qwen3.5-35B-A3B`** (MoE, 35B total / 3B active parameters)

Why this model:
- MoE architecture is **cost-effective on Tinker** — you pay per active parameter, not total
- 35B total params means rich representations and strong code generation
- 3B active params means fast inference — practical for real RLM deployment where each task requires multiple LLM calls
- Significantly larger than the paper's Qwen3-8B — potential for a meaningful contribution
- Qwen3.5 is the latest family, better base capabilities than Qwen3

**Fallback: `Qwen3.5-27B`** (dense) if MoE causes issues with LoRA training on Tinker.

**For cheap iteration / debugging: `Qwen3.5-4B`** (dense, small, fast turnaround).

## Research Goals

1. **Train RLM-Qwen3.5-35B-A3B** that dramatically outperforms the base model on long-context tasks
2. **Beat the paper's results** — RLM-Qwen3-8B got +28.3% over base. We should do better with a stronger base model.
3. **Evaluate on diverse benchmarks** — not just S-NIAH. Include multi-needle, document classification, and ideally external benchmarks (OOLONG, LoCoDiff, BABILong, RULER, etc.)
4. **Public release** — weights on HuggingFace, paper (icmltemplate/ has the LaTeX), reproducible training recipe
5. **Novel contributions** where possible — better training recipes, new benchmarks, analysis of what makes RLMs work

## The RLM in 30 Seconds

The full prompt **never enters the LLM's context window**. Instead:
1. Prompt is stored as `context` variable in a Python REPL
2. LLM sees only metadata (length, prefix, available functions)
3. LLM writes code: chunk context, call `llm_query()` on chunks, aggregate results
4. `llm_query(text)` invokes a fresh LLM on a substring — the recursive call
5. LLM calls `FINAL(answer)` when done

This lets a model with a 32K context window process **millions of tokens** with no degradation.

## Project Structure

```
rlm/
├── CLAUDE.md              # THIS FILE — project guide for the agent
├── TINKER.md              # Tinker API reference (consult for any API questions)
├── PAPER.md               # RLM paper reference & key design decisions
├── paper.md               # Our paper draft (working notes)
├── icmltemplate/          # LaTeX template for the final paper
├── ideas/                 # Research ideas, experiment plans, observations
│   ├── YYYYMMDD_*.md      # One file per idea/experiment — keep adding here
├── results/               # All experiment results (organized by experiment name)
├── data/                  # Training data, trajectories, filtered samples
│   ├── trajectories/      # Raw collected trajectories
│   ├── filtered/          # Filtered SFT-ready samples
│   ├── sft/               # SFT checkpoints
│   └── rl/                # RL checkpoints
├── scaffold/              # Core RLM runtime (model-agnostic)
│   ├── repl.py            # Persistent Python REPL
│   ├── rlm.py             # Main RLM loop (Algorithm 1)
│   ├── llm_query.py       # Model wrappers (HFModel, MockModel, → add TinkerModel)
│   └── prompts/           # System prompts per model
├── eval/                  # Evaluation harness & benchmarks
│   ├── benchmarks/        # Synthetic benchmark generators (pure Python)
│   └── run_eval.py        # Eval runner
├── training/              # Training scripts (rewriting for Tinker)
├── scripts/               # Data pipeline & utilities
├── tests/                 # Unit tests
├── pyproject.toml         # Project config (use uv)
└── uv.lock
```

## Development Rules

### Tools & Workflow
- **Always use `uv`** for package management (`uv pip install`, `uv run python`)
- **Always use `git`** — commit early and often with meaningful messages
- **Write ideas to `ideas/`** — one markdown file per idea, dated (YYYYMMDD_description.md)
- **Log all experiment results** — save to `results/`, include configs, metrics, and observations
- **Update `paper.md`** as results come in — keep a running draft

### Git Discipline
```bash
git add -A && git commit -m "descriptive message"
```
Commit after: every successful experiment, every code change, every new idea, every bug fix. Never let work go uncommitted.

### Research Mindset
This is a **research project**, not a software project. The agent should:
- **Form hypotheses** before running experiments
- **Analyze results carefully** — look at per-task breakdowns, failure modes, sample trajectories
- **Read sample trajectories** — understand what the model is actually doing wrong or right
- **Iterate on the approach** — if something doesn't work, understand why before trying the next thing
- **Document everything** in `ideas/` and `paper.md`
- **Think about what makes a publishable contribution** — what's new, what's better, what's the insight

## Searching Online for Research Ideas

**This is cutting-edge work.** The agent should actively search the web for ideas to improve training, evaluation, and the model's recursive capabilities.

### Where to Look
- **arXiv:** Search "recursive language model", "long context RL", "RLM training", "context folding", "natively recursive"
- **Official RLM repo:** https://github.com/alexzhang13/rlm — check for updates, new benchmarks, new results
- **Prime Intellect's RLM work:** https://www.primeintellect.ai/blog/rlm — they have a different training pipeline with `verifiers` and `prime-rl`
- **Tinker cookbook:** https://github.com/thinking-machines-lab/tinker-cookbook — reference RL training patterns
- **Papers citing arXiv:2512.24601** — improvements, ablations, new benchmarks
- **Related approaches:** LoongRL, QwenLong-L1.5, RAPTOR, SkyRL (Berkeley, uses Tinker for multi-agent/tool-use RL)
- **Twitter/X:** Search #RLM, "recursive language model", discussions around Qwen3.5 fine-tuning
- **HuggingFace:** Look for existing RLM models, datasets, training scripts people have shared

### When to Search
- **Before starting a new training phase** — check if anyone has found better hyperparameters or recipes
- **When stuck on a failure mode** — search for how others handled similar issues
- **When designing new benchmarks** — look for established ones to include for credibility
- **When writing the paper** — check for concurrent work to cite and position against
- **Periodically (every few major iterations)** — the field moves fast, new work appears weekly
- **When considering novel ideas** — check if someone already tried it

### What to Do with Findings
- Write a brief note in `ideas/` capturing what you found and how it applies
- If it suggests a code change, implement it and test
- If it's a benchmark to add, add it to `eval/benchmarks/`
- If it's a training recipe improvement, try it and compare
- Always cite sources in `paper.md`

## Current State & History

### What Was Done (on the old 8×H200 cluster, Qwen3-1.7B)

The codebase contains a complete pipeline developed on a shared GPU cluster with Qwen3-1.7B:

1. **Smoke test** — verified Qwen3-1.7B can write REPL code (it barely can — biggest risk at 2B)
2. **Trajectory collection** — STaR with base model and SFT checkpoints
3. **Template fixing** — automated fixes for FINAL_VAR literals, missing f-strings, plan-as-answer
4. **SFT v1-v3** — LoRA on filtered trajectories, progressively harder tasks
5. **GRPO v1-v4** — RL with progressive bug fixes (unconditional logprobs → conditioned, L2 KL → token KL)
6. **DPO v1** — preference-based training on trajectory pairs
7. **Comprehensive evals** — NIAH, multi-NIAH, doc-classify at multiple checkpoints
8. **Results analysis** — comparison tables with CIs

All results are in `results/` and experiment notes in `ideas/`. These contain valuable lessons even though we're switching models.

### What Changes Now

The codebase was built for **local GPU execution**: `CUDA_VISIBLE_DEVICES` assertions, `AutoModelForCausalLM.from_pretrained()`, local `model.generate()`, local `loss.backward()`, `peft` LoRA. All training and inference happened in-process on GPU 6 or 7.

**Tinker changes the paradigm completely:**
- Training (forward/backward, optimizer) runs **remotely** via API calls
- Generation runs **remotely** via `sample()` API
- You write the training loop locally (CPU only), Tinker executes it on their GPU cluster
- LoRA is the only option (which is fine — we were already using LoRA)
- No VRAM limits — we can train models we could never fit locally

### Migration Checklist

**Must rewrite for Tinker:**
- `scaffold/llm_query.py` — add `TinkerModel` class wrapping `sampling_client.sample()` for both root generation and sub-calls
- `training/sft.py` — `forward_backward("cross_entropy")` + `optim_step()` instead of local training
- `training/rl.py` / `training/rl_v4.py` — Tinker RL primitives (`importance_sampling`/`ppo`)
- `training/dpo.py` — `forward_backward_custom` for DPO loss
- `eval/run_eval.py` — use TinkerModel
- `scripts/collect_trajectories.py`, `scripts/collect_star_v2.py` — use TinkerModel
- `scripts/smoke_test_2b.py` — rewrite for Qwen3.5-35B-A3B on Tinker
- Remove all `CUDA_VISIBLE_DEVICES` assertions

**Keep as-is (pure Python, no GPU):**
- `scaffold/repl.py` — the REPL runs locally, this is the core
- `scaffold/rlm.py` — model-agnostic loop, just needs the model wrapper to change
- `scaffold/prompts/` — keep old prompts, add new ones for Qwen3.5
- `eval/benchmarks/` — all benchmark generation and scoring
- `training/rewards.py` — reward computation
- `scripts/filter_trajectories.py`, `scripts/fix_templates.py`, `scripts/analyze_results.py`
- `tests/` — MockModel tests still work

**Can likely remove:**
- `training/logprobs.py` — Tinker returns logprobs natively from `forward_backward` and `sample()`
- `CLUSTER_SAFETY.md` — no longer relevant

## Research Plan

### Phase 0: Infrastructure (do this first)
- [ ] Add `TinkerModel` to `scaffold/llm_query.py`
- [ ] Rewrite training scripts for Tinker
- [ ] Test end-to-end on `Qwen3.5-4B` (cheap)
- [ ] Verify: can generate, can train, can evaluate

### Phase 1: Baselines (critical — establishes the floor)
- [ ] **Base Qwen3.5-35B-A3B direct** (no RLM scaffold) on all benchmarks — the floor
- [ ] **Base Qwen3.5-35B-A3B + RLM scaffold** (no fine-tuning) — what scaffolding alone gives you
- [ ] **Base Qwen3.5-4B + RLM scaffold** — shows model size matters
- [ ] Document in `results/` and `paper.md`

### Phase 2: Data Collection & SFT
- [ ] Collect trajectories from base model (STaR round 1)
- [ ] If success rate < 30%, use Qwen3.5-397B-A17B as teacher to generate gold trajectories
- [ ] Filter, fix templates, convert to SFT samples
- [ ] SFT warm-start with LoRA on Tinker
- [ ] Evaluate — should see meaningful improvement

### Phase 3: RL
- [ ] GRPO on SFT checkpoint
- [ ] Experiment: reward functions, KL coefficients, K values
- [ ] Try DPO as alternative/complement
- [ ] Multiple STaR rounds (collect harder trajectories from best model → retrain)
- [ ] Evaluate after each round

### Phase 4: Hardening & New Benchmarks
- [ ] Add external benchmarks (OOLONG, LoCoDiff, BABILong, RULER)
- [ ] Create harder synthetic benchmarks
- [ ] Stress test at extreme context lengths (1M+ tokens)
- [ ] Ablation studies

### Phase 5: Release
- [ ] Final comprehensive eval
- [ ] Upload weights to HuggingFace
- [ ] Write paper (icmltemplate/)
- [ ] Clean repo, write public README

## Benchmarks

### Current (in eval/benchmarks/)
- **S-NIAH** (niah.py): O(1) — single needle in 5K–100K chars
- **Multi-NIAH** (multi_niah.py): O(K) — K=3-10 needles in 10K–100K chars
- **Doc-Classify** (doc_classify.py): O(N) — classify N=5-20 articles into 6 categories

### Should Add
- **OOLONG** (arXiv:2511.02817): Document classification from long context. The paper's main benchmark.
- **OOLONG-Pairs**: Cross-document comparison. O(N²) complexity.
- **LoCoDiff**: Track long git diff histories. GPT-5 < 10% at 75K+ tokens.
- **BABILong**: Extended bAbI tasks at long context.
- **RULER**: Synthetic long-context with multiple task types.
- **Verbatim Copy**: Faithfully reproduce context segments. Tests precision.
- **Multi-hop QA at scale**: Questions requiring chaining facts from different parts of a huge context.

### Evaluation Strategy
1. Always compare: **base (direct)** vs **base (RLM scaffold)** vs **fine-tuned (RLM scaffold)**
2. Report per-task breakdowns — aggregates hide patterns
3. Track trajectory quality — code quality, turn count, sub-call patterns
4. Save and read sample trajectories for qualitative analysis
5. Report cost — total tokens, time per task

## System Prompt Design

Current prompt (`scaffold/prompts/qwen2b.py`) was tuned for 1.7B. For Qwen3.5-35B-A3B:
- Much stronger instruction following — can give richer prompts
- 2-3 worked examples covering different task types
- Consider separate prompts for different complexity classes
- Look online for strategies used by other RLM implementations
- Create new file: `scaffold/prompts/qwen35_35b.py`

## Key Lessons (Don't Repeat These Mistakes)

1. **GRPO v1-v3 trained on unconditional code** — taught patterns in general, not conditioned on the task. Always use conditioned log-probs.
2. **Weight-space L2 ≈ 0** as KL — meaningless. Use token-level KL with frozen reference.
3. **K=4 too small** for stable advantages. Use K=8+.
4. **2B models barely write Python** — 35B MoE will be dramatically better. Still smoke test first.
5. **16% FINAL() with plans, 13% FINAL_VAR() with literals** — fix_templates.py helps, but training should eliminate these.
6. **f-string bug** — `llm_query("{context}")` instead of `f"{context}"`. Watch for it even at 35B.

## Ideas to Explore (flesh out in ideas/)

- **Teacher distillation:** Qwen3.5-397B-A17B generates gold trajectories → SFT the 35B model
- **Curriculum learning:** Start easy (short docs, single needle), progressively increase difficulty
- **Multi-task mixing:** NIAH + multi-NIAH + doc-classify + external benchmarks in each batch
- **Sub-call model:** Use smaller model for llm_query() sub-calls during inference (cost savings)
- **Multi-level recursion:** Can we train RLM to call RLM to call RLM?
- **Code quality rewards:** Reward clean, efficient code patterns, not just correctness
- **Context length extrapolation:** Train on 50K, test on 500K — does it generalize?
- **Comparison with Prime Intellect** — what can we learn from their approach?
- **Thinking tokens:** Qwen3.5 supports `<think>` blocks — should the RLM think before coding?
- **Tool augmentation:** Give the REPL additional tools beyond llm_query (regex search, embeddings?)

## Quick Reference

```bash
# Run tests
uv run python tests/test_repl.py
uv run python tests/test_rlm.py

# Commit
git add -A && git commit -m "description"

# Environment
export TINKER_API_KEY=<key>

# Write an idea
# ideas/YYYYMMDD_short_description.md

# Check Tinker API
# See TINKER.md in this repo
```