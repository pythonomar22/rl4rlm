# Why RL Doesn't Teach Multi-Step Decomposition (Yet)
**Date:** 2026-03-10

## The Core Finding

Despite training on multi-hop QA (15% of v3 task mix), the model does NOT learn
true multi-step decomposition. It consistently uses single-pass compound queries:

```python
# What the model does (compound query):
answer = llm_query(f"Find the budget of the project completed by R&D. {chunk}")

# What it SHOULD do (decomposition):
# Step 1: Find which project R&D completed
project = llm_query(f"What project did R&D complete? {chunk}")
# Step 2: Find the budget of that project
budget = llm_query(f"What is the budget of {project}? {chunk}")
```

## Evidence from Trajectories

### V3-s5 Hard Multi-Hop Eval (30% accuracy)
- All 10 trajectories use single-pass compound queries
- Task 0 (4 turns): Same query repeated 4x with different chunk sizes — NOT decomposition
- Task 7 (4 turns): Same query repeated 4x — same pattern
- Task 1 (1 turn, correct): Got lucky — both facts in same 20K chunk
- Task 4 (2 turns, correct): Found partial match, second turn just formatted answer

### V3 Training Trajectories
- Multi-hop reward peaked at 1.000 (step 7) — but this is on REGULAR multi-hop
- Regular multi-hop has shorter docs (10K-50K) where compound queries often work
- The model gets high reward WITHOUT learning decomposition
- **This is a form of reward hacking:** high reward, wrong skill

## Why This Happens

1. **No decomposition pressure:** Regular multi-hop tasks (10K-50K chars) can be solved
   with compound queries because both facts often land in the same chunk
2. **Sparse reward:** Only final answer is scored. No intermediate credit for finding
   bridge entities. The model gets 0 or 1, with no signal about HOW to solve it.
3. **Path of least resistance:** Compound queries are simpler (1 turn vs 3+ turns),
   so RL reinforces the simpler strategy that sometimes works
4. **No negative examples:** The model never sees what decomposition LOOKS like

## What Would Fix This

### 1. Hard Multi-Hop in Training (v4, launched)
- 20% of v4 mix is hard multi-hop (100K-200K chars)
- Facts are ALWAYS in different chunks → compound queries fail more often
- RL pressure should push toward multi-turn strategies
- BUT: still sparse reward, no decomposition signal

### 2. Intermediate Rewards (not implemented)
```python
# Proposed: reward for finding bridge entities
def score_decomposition(trajectory, task):
    score = 0
    # Check if bridge entities appear in llm_query calls
    for entity in task.bridge_entities:
        if entity in trajectory.intermediate_outputs:
            score += 0.2  # partial credit per bridge entity
    # Final answer
    if is_correct(trajectory.answer, task.expected):
        score += 0.6
    return score
```

### 3. Teacher Demonstrations (not implemented)
- Use Qwen3.5-397B-A17B to generate gold decomposition trajectories
- SFT on these demonstrations to show the model what decomposition looks like
- Then RL to refine the strategy

### 4. Process Reward Model (PRM)
- Train a separate model to score intermediate reasoning steps
- Use PRM as reward signal instead of just final answer
- Much more signal per trajectory → faster learning

## Implications for Paper

This analysis is a **key contribution** for the paper:
- We demonstrate that final-answer RL teaches better search patterns, not reasoning
- The taxonomy of skills (scan > search > decomposition) maps to the information
  the reward signal provides
- Decomposition requires richer reward signals (intermediate, process-based)
- This explains why the original RLM paper showed improvements on S-NIAH and
  OOLONG-style tasks but likely wouldn't improve on multi-hop reasoning either

## V4 Predictions

With 20% hard multi-hop training:
- **Optimistic:** Model discovers decomposition pattern, hard multi-hop → 50%+
- **Realistic:** Model learns better chunk sizes for 150K docs, hard multi-hop → 35-40%
- **Pessimistic:** Sparse reward insufficient, no improvement on hard multi-hop
- **Key metric:** Does the model ever issue a 2-step query (find entity, then use it)?
