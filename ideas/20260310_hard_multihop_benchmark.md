# Hard Multi-Hop QA Benchmark — Forcing True Decomposition
**Date:** 2026-03-10

## Motivation

The original multi_hop_qa benchmark has a critical flaw: the model can solve many tasks with single-pass compound queries. For example, asking each chunk "find the budget of the project completed by R&D" works when both facts are in the same chunk (which happens often at 10K-20K doc lengths).

**Hard Multi-Hop** fixes this by:
1. Using 100K-200K char documents (facts are ALWAYS in different chunks)
2. Adding distractor entity chains that match partial queries
3. Requiring entity discovery (you must find entity B from fact 1 before searching for fact 2)

## Design

### 2-Hop Tasks (100K-150K docs)
- Question asks about entity A
- Fact 1 links A → B (bridge entity)
- Fact 2 provides answer about B
- Distractor: A' → B' with different answer
- Fact positions: 0.08-0.20 and 0.70-0.88 (maximally separated)
- Distractor positions: 0.30-0.45 and 0.50-0.65 (between target facts)

### 3-Hop Tasks (150K-200K docs)
- Chain: A → B → C → answer
- Distractor chain: A' → B' → C' → different answer
- Three facts at positions 0.05-0.15, 0.40-0.50, 0.75-0.90
- Three distractors at positions 0.20-0.30, 0.55-0.65, 0.92-0.97

## Baseline Results

**Base model: 0% (0/10)** — as expected!

The model consistently:
- Gets distractor answers (finds wrong entity's budget/date/city)
- Fails to link the correct chain of entities
- Uses single-pass compound queries that don't work on 150K docs

### Specific Failures
- Expected $1.9M → Got $1.25M (random distractor)
- Expected Jan 10, 2027 → Got March 15, 2026 (distractor date)
- Expected $4.2M → Got "Not found" (couldn't even find it)
- Expected $7.8M → Got $1.9M (found distractor chain's budget)

## Why This Is the Perfect RLM Benchmark

A model with true multi-step decomposition would:
1. Step 1: Search for "who is VP of Engineering?" → find "Rachel Thomas"
2. Step 2: Search for "what project does Rachel Thomas lead?" → find "Project Phoenix"
3. Step 3: Search for "what is the budget of Project Phoenix?" → find "$4.2M"

Each step uses the output of the previous step as input — this is exactly what RLMs are designed to do. A single-pass approach fundamentally cannot solve this because:
- No single chunk contains all three facts
- Distractor chains match partial compound queries
- The bridge entity must be discovered first

## Training Implications

This benchmark should be added to GRPO v4 training mix:
- **15-20% of tasks** should be hard multi-hop
- Start with 2-hop at 100K, progress to 3-hop at 200K
- Reward function should give partial credit for finding intermediate entities
- This will force the model to learn multi-step decomposition strategies

## Paper Significance

If the RL-trained model improves from 0% to even 30-40% on this benchmark, it would be a compelling demonstration that RL can teach models genuine multi-step reasoning — not just better single-pass search patterns. This is the key contribution of the paper.
