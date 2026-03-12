# The Specialization-Generalization Tradeoff in Code-Generation RL

**Date:** 2026-03-11

## The Core Finding

RL training for code-generation RLMs (using GRPO/SC-GRPO) produces a **zero-sum tradeoff** between task types. Without strategy prompts at eval time, V9-s10 scores -0.3pp below base average across 14 benchmarks.

The model improves on 5 benchmarks (search/classification tasks) and regresses on 5 (extraction/reasoning tasks), with 4 ties.

## Why This Happens

### What RL Optimizes
RL training optimizes the code generation policy for reward. The reward is: "did the answer match the expected answer?" This creates implicit biases:

1. **Speed bias**: Faster completion = more reward per unit compute. The model learns larger chunks (20K→30K) to reduce API calls. This helps NIAH/doc_classify (fewer calls needed) but hurts key_value (misses boundary entries).

2. **Simplification bias**: Single-pass solutions are faster than multi-pass. The model collapses multi-step reasoning (extract A, extract B, compare) into single-pass patterns. This helps when one pass is sufficient but fails when genuine decomposition is needed (multi_hop, cross_doc).

3. **Template convergence**: SC-GRPO assigns strategies, but the model converges toward a few dominant patterns across ALL task types. The "chunk+scan+aggregate" pattern works for search but fails for extraction.

4. **Format precision loss**: RL rewards exact match or partial credit. The model learns that approximate answers often score > 0 (e.g., "0.87" instead of "87%"). This erodes format fidelity over training.

## The Strategy Prompt Paradox

Strategy prompts at eval time add +5.9pp on average. But this isn't "the model learned to use strategies" — it's "the model needs external guidance to compensate for what training damaged."

Key evidence:
- Multi-Hop QA: training causes -15pp regression, strategy adds +25pp, net +10pp
- Cross-Doc: training causes -7.8pp, cross_doc_separate strategy adds -11pp MORE, net -18.8pp
- Same model, same weights — strategy is the ONLY difference between 70% and 95% on multi_hop

This means strategy prompts are a **separate intervention** from training, not a consequence of it. The base model could also benefit from strategy prompts (untested).

## Implications

### 1. Training Alone is Insufficient
No amount of standard GRPO/SC-GRPO training will beat base on ALL tasks simultaneously. The policy gradient pushes toward task-specific optima that are incompatible.

### 2. Strategy Prompts are the Primary Improvement
The +5.9pp from strategy prompts dwarfs the -0.3pp from training. If we could find the right strategy for every benchmark, we might get +10pp without ANY training.

### 3. The Right Baseline
Papers comparing "trained model + strategies" vs "base model without strategies" are inflated. Fair comparison: base + strategies vs trained + strategies.

## Potential Solutions

### A. Minimal Training + Strategy Engineering
- Train for 2-3 steps only (V4-s5 was +1.1pp without strategy)
- Focus training budget on strategy prompt optimization
- Develop per-task strategy library
- This is the pragmatic approach

### B. Task-Conditional Training
- Train separate LoRA adapters per task type (search adapter, extraction adapter, comparison adapter)
- Use a router to select adapter at inference
- Each adapter doesn't interfere with others
- Significantly more complex

### C. Conservative Training with Strong KL
- Increase KL penalty (current 0.005 → 0.05 or higher)
- Train only on tasks where base is weak (doc_classify, event_counting)
- Explicit "do no harm" constraint: reject updates that decrease base performance on held-out tasks
- Would need online eval during training

### D. Reward Shaping for Format Preservation
- Add format fidelity bonus to reward (exact format match vs approximate)
- Add "decomposition" bonus for multi-step solutions
- Add "chunk overlap" bonus to penalize boundary-missing patterns
- Requires careful reward engineering

### E. Progressive Evaluation During Training
- After each step, evaluate on 5 key benchmarks
- Early-stop any task where performance drops below base
- Automatically reduce weight of regressed tasks
- Computationally expensive but would prevent the tradeoff

### F. Accept the Tradeoff (for the paper)
- Present the specialization-generalization tradeoff as a finding
- Show that strategy prompts are the real mechanism for improvement
- Propose the tradeoff as a fundamental limitation of single-adapter RL
- Position as motivation for future work on task-conditional training

## Recommendation

For the paper: **Option F** — present honestly. The tradeoff is itself a contribution.

For practical deployment: **Option A** — minimal training + strategy prompts. This gives the best cost/benefit ratio.

For future work: **Option C or E** — conservative training with evaluation guardrails.
