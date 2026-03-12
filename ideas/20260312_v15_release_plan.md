# V15 Release Plan: Best Publicly Accessible RLM

**Date:** 2026-03-12

## Strategy: Two-Phase Training (SFT → Light GRPO)

Based on all experiments to date, the optimal training pipeline is:

### Phase 1: SFT Warmstart (V8)
- **Data:** 592 balanced samples across all 14 task types
- **Source:** Correct trajectories from all eval runs (base, V4, V11, hybrid)
- **Config:** 3 epochs, LR=1e-4, batch=4, LoRA rank 32
- **Session:** 5c203420-1891-5999-8cef-23cbb0be8402
- **Expected:** 90%+ NIAH, 95%+ multi-NIAH, 95%+ doc-classify (based on SFT V1 ep5 results)

### Phase 2: GRPO Refinement (V15)
- **From:** Best SFT V8 checkpoint
- **Config:** 5-10 steps, LR=5e-7 (very conservative), all bug fixes applied
- **Distribution:** v12 (conservative — only proven-improvement tasks)
- **Key fixes in V15:**
  - Hybrid model weight update bug FIXED
  - Multi-turn turn_idx alignment FIXED
  - Temperature capped at 1.0 (no gibberish)
  - Per-turn credit assignment
  - New strategies for all weak benchmarks

### Phase 3: Strategy Prompt Optimization
- Create per-benchmark optimal strategies
- Package strategies with the model for release
- Oracle model selection: use trained for search, base for extraction

## Release Checklist
- [ ] Final checkpoint with best average across all 14 benchmarks
- [ ] Strategy prompts for each benchmark type
- [ ] System prompt V2 (scaffold/prompts/qwen35_35b.py)
- [ ] Clean evaluation results with confidence intervals
- [ ] Upload weights to HuggingFace
- [ ] Write paper (icmltemplate/)
- [ ] Clean repo, public README
- [ ] Security: sandbox the REPL exec() before public release

## Key Metrics to Beat
- Base model average: 60.9%
- Best trained (V4-s5): 61.4%
- Best with strategies (V9-s10+strat): 66.5%
- Oracle best-of-all: 68.2%
- **Target: >70% average with single model + strategies**

## Bug Fixes Applied (2026-03-12)
1. FinalAnswer → BaseException (can't be caught by model code)
2. REPLTimeoutError (no more shadowing)
3. Hybrid model weight update (critical for multi-step hybrid training)
4. turn_idx increment on all paths (critical for multi-turn IS data)
5. score_trajectory handles all 14 task types
6. sample_tasks_v6 generates all 14 task types
7. Temperature schedule capped (no MoE gibberish)
8. notebook_qa variable_trace proper f-string escaping
9. key_value_retrieval single format per document
10. NGRPO virtual reward uses scores not rewards for all-correct check
