# Comprehensive Audit for Public Release

**Date:** 2026-03-12

## Scoring Bugs (FIXED)

### 1. cross_doc_compare recall > 1.0 (FIXED)
- **File:** `eval/benchmarks/cross_doc_compare.py:955`
- **Bug:** Multiple predicted items could substring-match the same expected item, producing `true_positives > len(expected)` → recall > 1.0 → F1 > 1.0
- **Fix:** Track matched expected items in a set; each expected item can only be matched once
- **Impact:** Training rewards for cross_doc could have exceeded 1.0, creating noisy gradient signal

### 2. code_debug scoring gameable (FIXED)
- **File:** `eval/benchmarks/code_debug.py:485`
- **Bug:** Only checked if function name appeared in answer. Listing all function names = 100% score.
- **Fix:** Now requires function name + evidence of understanding (description keywords or bug-related words)
- **Impact:** RL could have learned to list function names without finding bugs

## Scoring Issues (NOT YET FIXED — document for paper)

### 3. doc_classify substring matching
- **File:** `eval/benchmarks/doc_classify.py:407`
- **Issue:** `expected.lower() in pred.lower()` — listing all 6 categories would match everything
- **Risk:** Low in practice (categories don't substring-overlap, output is parsed from "N: Category" format)
- **Fix:** Use exact match on parsed category name

### 4. multi_niah precision = recall
- **File:** `eval/benchmarks/multi_niah.py:245`
- **Issue:** `precision = recall` is hardcoded. F1 is meaningless.
- **Risk:** None (eval uses recall as primary metric, not F1)
- **Fix:** Actually compute precision or don't report it

## Training Loop Issues

### 5. KL divergence is not real KL (known, low impact)
- **File:** `training/rl_tinker_v6.py:1104`
- **Issue:** Uses `|mean_logprob - ref_mean_logprob|` instead of token-level KL with frozen reference
- **Impact:** Low — `kl_coeff=0.01` is very small
- **Fix for paper:** Either implement proper KL or remove/relabel as "log-likelihood drift penalty"

### 6. MaxRL k_success threshold is task-agnostic (0.3)
- **File:** `training/rl_tinker_v6.py:1186`
- **Issue:** 0.3 doesn't correspond to actual task success
- **Impact:** Low-medium — for binary tasks (NIAH), 0.3 separates success/failure well. For partial credit tasks, it's more arbitrary.
- **Fix:** Consider per-task-type thresholds or use the correctness component only

### 7. Advantage clamping at ±3 truncates MaxRL amplification
- **File:** `training/rl_tinker_v6.py:409,491`
- **Issue:** MaxRL multiplier can reach 8x, but clamp at 3 truncates it
- **Impact:** Reduces MaxRL effectiveness for very hard tasks
- **Fix:** Increase clamp to ±8 or compute from K value

### 8. Asymmetric advantage "clipping" is actually scaling
- Not PPO/DAPO-style IS ratio clipping
- Works in practice but naming is misleading for the paper

## Evaluation Methodology Issues

### 9. --seed-offset only works for NIAH (CRITICAL for paper)
- **File:** `eval/run_eval.py:1180`
- **Issue:** Only NIAH receives the CLI `--seed-offset`. All other benchmarks use function-level defaults.
- **Impact:** Config.json recording is misleading. No actual train/eval leakage with rl_tinker_v6.py.
- **Fix:** Pass `seed_offset` to all benchmark runners

### 10. cross_doc_compare and key_value_retrieval seeds hardcoded
- Not configurable at all — can't do bootstrap resampling
- **Fix:** Add seed_offset parameter to both

### 11. Temperature 0.7 for eval creates variance
- With N=10-20 tasks, run-to-run variance is significant
- OOLONG (N=10) has 95% CI of ±31pp for a 50% score
- **For paper:** Must report confidence intervals or run multiple seeds
- **Note:** Changing temperature now would invalidate all results

### 12. Inconsistent max_iterations per benchmark
- Some benchmarks use 8, others 10 or 12
- Not recorded per-benchmark in config.json
- **Fix:** Record actual per-benchmark config

### 13. Statistical significance concerns
- N=10 benchmarks (OOLONG, hard_multi_hop, verbatim_copy): too few for meaningful comparisons
- The +0.5pp average improvement (60.9% → 61.4%) is not statistically significant
- **For paper:** Run larger N or report this as a limitation

## Security / Public Release

### 14. API key must not be in published repo
- .env is in .gitignore but verify git history is clean
- Rotate key before release

### 15. Unsandboxed REPL exec()
- Full system access during training and eval
- **For release:** Add import whitelist, disable filesystem/network access
- **Minimum:** Document the security model and recommend sandboxed execution

### 16. CUDA assertions in legacy files
- `training/sft.py`, `training/rl.py`, etc. have `CUDA_VISIBLE_DEVICES` assertions
- **Fix:** Remove or make configurable

## Verified Correct

- [x] GRPO advantage computation (Dr. GRPO, no std normalization)
- [x] NGRPO virtual reward (threshold 0.9 correctly skips near-perfect groups)
- [x] Conditioned logprobs (full prompt context from Tinker sampling)
- [x] Importance sampling delegation to Tinker PPO
- [x] LoRA config (rank 32, all layers, Tinker defaults)
- [x] Learning rate schedule (cosine with warmup)
- [x] Training and eval use same scoring functions (no mismatch)
- [x] Gradient accumulation + optim_step timing
- [x] REPL timeout scaling with context length
- [x] No train/eval seed overlap with rl_tinker_v6.py (offset 100000+)

## Priority for V12+ Training

1. **Scoring fixes applied** — cross_doc and code_debug now score correctly
2. **V12 is clean** — starts from base, only 6 proven-improvement tasks, correct scoring
3. **All running evals (V10-s5, V11-s5, V11-s10) use OLD scoring** — results are still valid but cross_doc scores may be slightly inflated
4. **Future training** should use the fixed scoring (V12 will naturally pick it up since it imports the benchmark modules)
