# Complete Bug Audit & Fixes

Date: 2026-03-11

## Bugs Found and Fixed

### BUG 1: HybridTinkerModel.sub_query() uses wrong temperature [CRITICAL, FIXED]
**File**: `scaffold/llm_query.py:394`
**Problem**: `HybridTinkerModel.sub_query()` used `self.temperature` (0.7 in eval, up to 1.3 in training) instead of fixed 0.3 for factual extraction. Regular `TinkerModel.sub_query()` correctly used 0.3 (line 213).
**Impact**: ALL previous hybrid evaluation results had noisier sub-calls than intended. Hybrid results may IMPROVE with fix.
**Fix**: Changed to `sub_temp = getattr(self, 'sub_temperature', 0.3)`

### BUG 2: Event counting uses global random [MEDIUM, FIXED]
**File**: `eval/benchmarks/event_counting.py:90,98`
**Problem**: `_generate_event_line()` and `_generate_filler_block()` used `random.choice()` (global random) instead of the seeded `rng` parameter.
**Impact**: Event counting task filler text was NOT reproducible across runs. Task instances varied even with same seed_offset.
**Fix**: Added `rng` parameter to both functions, passed through from callers.

### BUG 3: Config doesn't record seed_offset, model_path, hybrid [MEDIUM, FIXED]
**File**: `eval/run_eval.py:1260`
**Problem**: Saved config.json didn't include `seed_offset`, `model_path`, or `hybrid` flag.
**Impact**: Cannot verify which eval runs used which seeds. Multiple runs per benchmark with different scores (e.g., hard_multi_hop: 40%, 10%, 30% for V4-s5) and no way to determine which is canonical.
**Fix**: Added all three fields to config dict.

### BUG 4: hard_multi_hop and event_counting eval don't accept seed_offset [MEDIUM, FIXED]
**File**: `eval/run_eval.py:639,707`
**Problem**: `run_hard_multi_hop_eval()` and `run_event_counting_eval()` had no `seed_offset` parameter. They always used default (seed=42 and seed=0 respectively).
**Impact**: These benchmarks' seeds couldn't be controlled from the command line. They always generated the same tasks.
**Fix**: Added `seed_offset` parameter with unique defaults (96000 and 94000).

### BUG 5: NGRPO virtual reward threshold too strict [MEDIUM, FIXED]
**File**: `training/rl_tinker_v6.py:988`
**Problem**: NGRPO only activated when `mean_r < 0.5`. Hard tasks with partial credit (rewards ~0.3-0.5) missed this threshold.
**Impact**: Groups where all K=8 trajectories scored 0.4 got skipped instead of receiving negative feedback.
**Fix**: Changed threshold to `mean_r < 0.9` (almost all uniform groups get NGRPO).

### BUG 6: Advantage clamping too tight with MaxRL [LOW, FIXED]
**File**: `training/rl_tinker_v6.py:348`
**Problem**: Advantage clamped at ±2, but MaxRL can produce advantages up to ±8. Clamping destroys MaxRL's signal amplification.
**Fix**: Increased clamp to ±3. Still prevents gradient explosions while preserving MaxRL benefit.

### BUG 7: Mixed datum formats in training batch [LOW, FIXED]
**File**: `training/rl_tinker_v6.py:1081`
**Problem**: Training data could mix PPO datums (with "advantages" field) and cross_entropy datums (with "weights" field). Loss function dispatch checked only `training_data[0]`. If first datum was wrong format, all datums got wrong loss.
**Impact**: Unlikely in practice (Tinker always returns logprobs), but a latent bug.
**Fix**: Separate datums by format, use PPO if any PPO datums exist, log warning if mixed.

## Bugs Found but NOT Fixed (Deliberate Design)

### ISSUE 1: binary_reward uses substring match
**File**: `training/rewards.py:23`
**Problem**: `expected.lower() in predicted.lower()` means "3" matches "13", "300", etc.
**Impact**: NOT used in training (score_trajectory dispatches to per-benchmark scorers). Only affects legacy code.

### ISSUE 2: OOLONG always 0% for trained models
**Root cause**: OOD — model never trains on unstructured D&D transcript aggregation.
**Impact**: Cannot fix with code changes alone. Needs training data.

### ISSUE 3: Notebook QA degrades in hybrid mode
**Root cause**: Base model sub-calls can't parse Jupyter structure as well as trained sub-calls.
**Impact**: Design tradeoff, not a bug.

## Multiple Evaluation Runs Issue

Previous results had MULTIPLE runs per benchmark per config with DIFFERENT scores. Examples:
- Baseline code_debug: 18.9%, 28.9%, 22.2%
- V4-s5 hard_multi_hop: 10.0%, 0.0%, 30.0%

Root cause: Configs didn't record seed_offset, so we can't verify which runs used consistent settings. Early runs may have used different seeds or n_tasks.

**Resolution**: Running clean head-to-head evaluation with ALL fixes applied, consistent seed_offset=10000, consistent n_tasks=20.

## Clean Evaluation Status

Running now (2026-03-11 09:00):
- Base model (PID 1124157): clean_headtohead_base
- V4-s5 non-hybrid (PID 1124261): clean_headtohead_v4s5
- V4-s5 hybrid (PID 1124184): clean_headtohead_v4s5_hybrid
