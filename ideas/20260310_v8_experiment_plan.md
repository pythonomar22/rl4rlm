# V8 Experiment Plan — SC-GRPO + NGRPO + Cross-Doc

## Background
V7 (SC-GRPO) eliminated mode collapse. V8 adds research-backed improvements.

## New Features in V8

### 1. NGRPO Virtual Max-Reward (arXiv:2509.18851)
- **Problem:** When all K=8 trajectories get the same reward (all fail or all succeed),
  the group is SKIPPED. This wastes compute and gives no learning signal on hard tasks.
- **Solution:** For all-wrong groups (mean_r < 0.5), add a virtual optimal completion
  with reward=1.0. This gives all actual completions a negative advantage, pushing the
  model away from failure patterns.
- **Impact:** V7 step 1 had 1/4 groups skipped, step 2 had 0/4. On hard tasks like
  event_counting and hard_multi_hop, many groups fail entirely. NGRPO recovers signal.
- **Flag:** `--ngrpo-virtual-reward`

### 2. Asymmetric Advantage Scaling (arXiv:2509.26114)
- **Problem:** Standard GRPO clips advantages symmetrically. But positive advantages
  reinforce existing patterns → entropy collapse. Negative advantages push away from
  failures → preserves diversity.
- **Solution:** Scale positive advantages by clip_high (e.g., 0.5 = halve them) and
  negative advantages by clip_low (e.g., 1.5 = amplify them).
- **Impact:** Combined with SC-GRPO, should further prevent mode collapse even at
  high step counts (>10).
- **Flags:** `--clip-high 0.5 --clip-low 1.5`

### 3. Cross-Document Comparison Benchmark
- **New benchmark:** `cross_doc_compare` — tests comparing information across 2 documents
- **4 task types:** overlap_entities, budget_diff, timeline_conflict, metric_comparison
- **Added to training mix:** 10% of mixed_v6 tasks (taken from hard_multi_hop: 20%→15%)
- **Why:** Tests genuine multi-step reasoning (find A, find B, compare in Python)
- **Strategy weights:** map_reduce=0.3, two_pass=0.3, extract_compute=0.2, standard=0.2

### 4. MaxRL Advantage Normalization (arXiv:2602.02710)
- **Problem:** Standard GRPO normalizes advantages by total samples N. On hard tasks
  where only 1/8 trajectories succeed, the gradient signal is weak.
- **Solution:** Normalize by K_success (number of successful trajectories). Hard tasks
  with few successes get STRONGER gradient signal, not weaker.
- **Impact:** 7.9x-19.2x test-time scaling efficiency on Qwen3-1.7B and 4B.
- **Flag:** `--maxrl`

## V8 Launch Command
```bash
mkdir -p data/rl/grpo_35b_v8 && uv run python training/rl_tinker_v6.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --model-path "tinker://SESSION_ID:train:0/weights/STATE" \
    --steps 30 --K 8 --batch-size 4 --lr 2e-6 \
    --kl-coeff 0.01 \
    --task-type mixed_v6 --save-every 5 \
    --warmup-steps 2 --grad-accum-batch 4 \
    --strategy-conditioning \
    --ngrpo-virtual-reward \
    --clip-high 0.5 --clip-low 1.5 \
    --maxrl \
    --experiment-name grpo_35b_v8
```

## When to Launch
- After V7 reaches step 5 checkpoint and is evaluated
- Use V7 step-5 checkpoint as starting point (or V4-s5 if V7 disappoints)
- OR: use SFT checkpoint from teacher distillation (Stage 1 of multi-stage pipeline)

## Expected Improvements
1. **NGRPO:** +5-10% on hard tasks (event_counting, hard_multi_hop) where all-wrong groups are common
2. **Asymmetric advantages:** Slower entropy decay, better diversity maintenance at step 10+
3. **Cross-doc training:** New capability dimension, improves multi-step reasoning generally
4. **Combined with SC-GRPO:** All four innovations working together

## Ablation Plan
If V8 shows improvement, run ablations:
- V8a: SC-GRPO only (V7 baseline)
- V8b: SC-GRPO + NGRPO only
- V8c: SC-GRPO + asymmetric only
- V8d: SC-GRPO + NGRPO + asymmetric (V8 full)
This shows contribution of each component.
