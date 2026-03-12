# Per-Turn Credit Assignment for RLM RL Training

**Date:** 2026-03-12
**Status:** Design document with implementation code

## Problem

In `training/rl_tinker_v6.py`, the functions `trajectory_to_training_data` and
`trajectory_to_training_data_is` assign the **same scalar advantage** to every
turn in a trajectory. A 3-turn trajectory where T0 sets up chunking, T1 queries
sub-calls, and T2 calls FINAL with the right answer all receive identical
advantage weight.

This is wasteful and potentially harmful:
- **Successful trajectories**: The FINAL turn produced the answer. Setup turns
  are necessary but less directly causal. Giving them equal credit dilutes the
  signal.
- **Failed trajectories**: A timeout on T1 should get negative credit, but a
  reasonable setup on T0 shouldn't be penalized as harshly.
- **Mixed trajectories**: Some turns produce errors, some succeed. Uniform
  advantage blurs this distinction.

## Tinker API Constraints

From TINKER.md, the key constraint:

> All losses are **token-level**. Tensors have shape `(N,)` where N = `model_input.length`.

The `advantages` tensor in `importance_sampling` / `ppo` is per-token. The
current code already sets per-token advantages (zero for prompt tokens, uniform
for response tokens). **Per-turn credit assignment is fully supported** --- each
turn is a separate `tinker.Datum` with its own `advantages` array. We simply
scale the advantage scalar differently per turn.

Both `trajectory_to_training_data` and `trajectory_to_training_data_is` iterate
over assistant messages (turns) and create one `Datum` per turn. The `advantage`
parameter is currently the same float for all turns. The change is: compute a
**per-turn advantage** before passing it to the Datum construction.

## Current Code Flow

```
trajectory_to_training_data_is(trajectory_dict, advantage, renderer, tokenizer)
  for each assistant message (turn):
    build Datum with advantages_aligned = [advantage] * n_response_tokens
```

The fix is a new function that takes `trajectory_dict` and the **trajectory-level
advantage** and returns a `list[float]` of per-turn advantages.

---

## Approach 1: Outcome-Weighted Turns

### Idea
- Successful trajectory: FINAL turn gets amplified advantage; setup turns get
  attenuated advantage.
- Failed trajectory: All turns get uniform negative advantage (we can't
  distinguish which turn caused the failure).

### Rationale
The FINAL turn is the most directly responsible for the outcome. In a 3-turn
trajectory [setup, query, FINAL], the model's decision in the FINAL turn to call
`FINAL("Amsterdam")` vs `FINAL("Berlin")` is the proximate cause of success.
Setup turns established context but are less decisive.

### Weighting Scheme
For a trajectory with N turns and trajectory-level advantage A:
- If A > 0 (success):
  - FINAL turn: `A * alpha` where alpha = 1.5
  - Other turns: `A * (1 - (alpha - 1) / (N - 1))` to keep mean = A
  - Simplification: FINAL turn gets 1.5x, others get ~0.75x (for 3 turns)
- If A <= 0 (failure):
  - All turns get A (uniform --- we don't know which turn caused failure)

### Implementation

```python
def compute_per_turn_advantages_outcome(
    trajectory_dict: dict,
    advantage: float,
    alpha: float = 1.5,
) -> list[float]:
    """Outcome-weighted per-turn advantages.

    FINAL turn gets alpha * advantage. Other turns get scaled down
    so the mean advantage across turns equals the original advantage.
    """
    turns = trajectory_dict.get("turns", [])
    messages = trajectory_dict.get("messages", [])

    # Count assistant turns (these are the ones that become training data)
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")

    if n_assistant <= 1 or advantage <= 0:
        # Single turn or failure: uniform advantage
        return [advantage] * n_assistant

    # Find which assistant turn is the FINAL turn
    final_turn_idx = None
    assistant_idx = 0
    for turn in turns:
        if turn.get("terminated"):
            final_turn_idx = assistant_idx
        if turn.get("parsed_code") is not None or turn.get("raw_response"):
            assistant_idx += 1

    if final_turn_idx is None:
        # No FINAL found (shouldn't happen for successful trajectories)
        return [advantage] * n_assistant

    # Compute weights: FINAL gets alpha, others share the remainder
    # Constraint: mean of weights = 1.0 (so mean advantage = original advantage)
    # FINAL weight = alpha, other weights = (N - alpha) / (N - 1)
    other_weight = (n_assistant - alpha) / max(n_assistant - 1, 1)
    other_weight = max(other_weight, 0.1)  # Floor: don't zero out setup turns

    per_turn = []
    for i in range(n_assistant):
        if i == final_turn_idx:
            per_turn.append(advantage * alpha)
        else:
            per_turn.append(advantage * other_weight)

    return per_turn
```

### Pros
- Simple, fast, no additional information needed
- Mean advantage preserved --- same total gradient magnitude
- Conservative: only amplifies the clearest signal (FINAL turn)

### Cons
- Assumes FINAL turn is most important, which isn't always true (the setup
  turn that correctly chunks the document is equally critical)
- Doesn't penalize bad turns in otherwise successful trajectories

---

## Approach 2: Sub-Call Credit

### Idea
Turns are scored based on observable signals:
- llm_query sub-calls made (indicates useful work)
- Errors/timeouts (indicates wasted computation)
- Variable assignments vs FINAL calls (setup vs output)

### Weighting Scheme
Each turn gets a "usefulness score" u_i, then advantages are:
`advantage_i = advantage * (u_i / mean(u))`

Scoring:
- Base score: 1.0
- +0.5 per llm_query call in the code (max +1.5)
- +0.3 if turn produces stdout (generated useful output)
- +0.5 if turn calls FINAL (produced the answer)
- -0.5 if turn has an error
- -1.0 if turn timed out
- -0.3 if turn has no parsed code (unparseable response)

### Implementation

```python
def compute_per_turn_advantages_subcall(
    trajectory_dict: dict,
    advantage: float,
) -> list[float]:
    """Sub-call weighted per-turn advantages.

    Turns with more useful work (sub-calls, FINAL) get more credit.
    Turns with errors/timeouts get less or negative credit.
    """
    turns = trajectory_dict.get("turns", [])
    messages = trajectory_dict.get("messages", [])

    n_assistant = sum(1 for m in messages if m["role"] == "assistant")

    if n_assistant <= 1:
        return [advantage] * n_assistant

    # Score each turn
    usefulness = []
    for turn in turns:
        score = 1.0  # base

        code = turn.get("parsed_code") or ""
        stdout = turn.get("stdout") or ""
        error = turn.get("error")
        terminated = turn.get("terminated", False)

        # Sub-call credit
        n_subcalls = code.count("llm_query")
        score += min(1.5, 0.5 * n_subcalls)

        # Output credit
        if stdout and len(stdout) > 10:
            score += 0.3

        # FINAL credit
        if terminated:
            score += 0.5

        # Error penalty
        if error:
            if "Timeout" in str(error):
                score -= 1.0
            else:
                score -= 0.5

        # No-code penalty
        if not code:
            score -= 0.3

        usefulness.append(max(score, 0.1))  # Floor at 0.1

    # Pad if turns < assistant messages (shouldn't happen, but safety)
    while len(usefulness) < n_assistant:
        usefulness.append(1.0)
    usefulness = usefulness[:n_assistant]

    # Normalize: multiply advantage by (u_i / mean_u)
    mean_u = sum(usefulness) / len(usefulness)
    if mean_u < 1e-6:
        return [advantage] * n_assistant

    per_turn = [advantage * (u / mean_u) for u in usefulness]
    return per_turn
```

### Pros
- Uses observable signals, not just position
- Penalizes timeout turns even in successful trajectories
- Rewards actual sub-call usage (the core RLM behavior we want)

### Cons
- Heuristic scoring --- the weights (0.5, 0.3, etc.) are arbitrary
- `code.count("llm_query")` counts appearances in code, not actual executions
  (a loop with llm_query might appear once but execute 5 times)
- Doesn't consider the *quality* of sub-call results

---

## Approach 3: Progressive Credit

### Idea
Later turns built on information from earlier turns. Turns later in the
trajectory are "more informed" and thus more deserving of credit.

### Weighting Scheme

Exponential decay from first to last:
`w_i = decay^(N - 1 - i)` where decay in (0, 1)

With decay=0.7 and N=3: w = [0.49, 0.70, 1.00] (normalized to mean 1.0: [0.67, 0.96, 1.37])

### Implementation

```python
def compute_per_turn_advantages_progressive(
    trajectory_dict: dict,
    advantage: float,
    decay: float = 0.7,
) -> list[float]:
    """Progressive per-turn advantages: later turns get more credit.

    Rationale: later turns are more informed, having seen outputs of
    earlier turns. The FINAL turn (always last in successful trajectories)
    naturally gets the highest weight.
    """
    messages = trajectory_dict.get("messages", [])
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")

    if n_assistant <= 1:
        return [advantage] * n_assistant

    # Exponential weights: last turn = 1.0, earlier turns decay
    raw_weights = [decay ** (n_assistant - 1 - i) for i in range(n_assistant)]

    # Normalize so mean = 1.0
    mean_w = sum(raw_weights) / len(raw_weights)
    normalized = [w / mean_w for w in raw_weights]

    return [advantage * w for w in normalized]
```

### Variant: Information Gain

Instead of pure position, weight by whether the turn changed the model's
"working state" (new variables, new sub-call results):

```python
def compute_per_turn_advantages_info_gain(
    trajectory_dict: dict,
    advantage: float,
) -> list[float]:
    """Information-gain weighted advantages.

    Turns that produce new information (variables, sub-call results)
    get more credit than turns that just reorganize existing data.
    """
    turns = trajectory_dict.get("turns", [])
    messages = trajectory_dict.get("messages", [])
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")

    if n_assistant <= 1:
        return [advantage] * n_assistant

    info_scores = []
    for turn in turns:
        code = turn.get("parsed_code") or ""
        stdout = turn.get("stdout") or ""

        score = 0.0
        # New variable assignments = new information
        import re
        assignments = re.findall(r'^(\w+)\s*=', code, re.MULTILINE)
        score += min(1.0, 0.2 * len(assignments))

        # Sub-call results = new information from the document
        n_subcalls = code.count("llm_query")
        score += 0.5 * n_subcalls

        # Long stdout = rich output
        if len(stdout) > 100:
            score += 0.3

        # FINAL = answer synthesis
        if turn.get("terminated"):
            score += 0.5

        info_scores.append(max(score, 0.1))

    while len(info_scores) < n_assistant:
        info_scores.append(0.5)
    info_scores = info_scores[:n_assistant]

    mean_s = sum(info_scores) / len(info_scores)
    if mean_s < 1e-6:
        return [advantage] * n_assistant

    return [advantage * (s / mean_s) for s in info_scores]
```

### Pros
- Position-based version is dead simple
- Information-gain variant captures actual contribution
- Naturally aligns with FINAL turn (always last in successful trajectories)

### Cons
- Position-based is crude --- turn 2 isn't inherently more valuable than turn 1
- Information-gain uses heuristics that may not correlate with actual contribution
- Neither accounts for turns that produce correct intermediate results vs wrong ones

---

## Approach 4: Hindsight Credit Assignment

### Idea
After seeing the final answer, trace back which turns actually contributed to
finding it. If the final answer is "Amsterdam", check which sub-call results
contained "Amsterdam" or information leading to it.

### Weighting Scheme

For successful trajectories:
1. Get the final answer string
2. For each turn, check if any of these conditions hold:
   - Sub-call stdout contains the answer (or part of it)
   - Turn's code directly produces the answer (FINAL turn)
   - Turn's variables are used by later turns that produce the answer
3. Turns with answer-relevant content get amplified credit

### Implementation

```python
def compute_per_turn_advantages_hindsight(
    trajectory_dict: dict,
    advantage: float,
    amplify: float = 2.0,
) -> list[float]:
    """Hindsight credit assignment: trace the answer back through turns.

    Turns whose outputs contain information that appears in the final
    answer get amplified credit. This is a lightweight version of
    hindsight experience replay adapted for multi-turn code generation.
    """
    turns = trajectory_dict.get("turns", [])
    messages = trajectory_dict.get("messages", [])
    answer = trajectory_dict.get("answer") or ""

    n_assistant = sum(1 for m in messages if m["role"] == "assistant")

    if n_assistant <= 1 or not answer or advantage <= 0:
        return [advantage] * n_assistant

    answer_lower = answer.lower().strip()
    # Extract key tokens from the answer (words > 2 chars, not stopwords)
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                 "can", "had", "her", "was", "one", "our", "out", "has",
                 "its", "with", "this", "that", "from", "they", "been"}
    answer_tokens = set(
        w for w in answer_lower.split()
        if len(w) > 2 and w not in stopwords
    )

    if not answer_tokens:
        # Answer is too short/generic to trace
        return [advantage] * n_assistant

    # Score each turn by answer relevance
    relevance = []
    for turn in turns:
        stdout = (turn.get("stdout") or "").lower()
        code = (turn.get("parsed_code") or "").lower()

        # Check if this turn's output contains answer-relevant tokens
        turn_text = stdout + " " + code
        matched = sum(1 for t in answer_tokens if t in turn_text)
        frac = matched / len(answer_tokens)

        if turn.get("terminated"):
            # FINAL turn always gets high relevance
            score = 1.0
        elif frac > 0.5:
            # More than half the answer tokens found in this turn
            score = amplify
        elif frac > 0:
            # Some answer tokens found
            score = 1.0 + (amplify - 1.0) * frac
        else:
            # No answer tokens --- pure setup
            score = 0.5

        relevance.append(score)

    while len(relevance) < n_assistant:
        relevance.append(0.5)
    relevance = relevance[:n_assistant]

    # Normalize so mean = 1.0
    mean_r = sum(relevance) / len(relevance)
    if mean_r < 1e-6:
        return [advantage] * n_assistant

    return [advantage * (r / mean_r) for r in relevance]
```

### Pros
- Most principled approach --- directly identifies which turns contributed
- Handles multi-hop tasks well: T0 finds entity A, T1 finds entity B,
  T2 combines --- all get credit proportional to their contribution
- Works with the actual answer content, not heuristics about code structure

### Cons
- Only works for successful trajectories (need an answer to trace back)
- Token matching is crude --- "Amsterdam" appearing in stdout doesn't mean
  the turn *used* it correctly
- Computationally cheap but logically fragile (what if the answer is "yes"?)
- Can't handle numeric answers well ("42" appears in many contexts)

---

## Recommended Implementation: Hybrid (Approaches 1 + 2)

The best practical approach combines the simplicity of outcome weighting with
the observable signals of sub-call credit. This avoids the fragility of
hindsight matching while capturing more signal than pure position weighting.

### Combined Implementation

```python
def compute_per_turn_advantages(
    trajectory_dict: dict,
    advantage: float,
    method: str = "hybrid",  # "uniform", "outcome", "subcall", "progressive", "hindsight", "hybrid"
    alpha: float = 1.5,      # FINAL turn amplification (outcome/hybrid)
    decay: float = 0.7,      # Progressive decay
) -> list[float]:
    """Compute per-turn advantage weights for credit assignment.

    Args:
        trajectory_dict: Full trajectory with turns and messages
        advantage: Trajectory-level advantage (from GRPO)
        method: Credit assignment method
        alpha: Amplification factor for FINAL turn
        decay: Decay factor for progressive method

    Returns:
        List of per-turn advantages (one per assistant message)
    """
    if method == "uniform":
        messages = trajectory_dict.get("messages", [])
        n = sum(1 for m in messages if m["role"] == "assistant")
        return [advantage] * max(n, 1)
    elif method == "outcome":
        return compute_per_turn_advantages_outcome(trajectory_dict, advantage, alpha)
    elif method == "subcall":
        return compute_per_turn_advantages_subcall(trajectory_dict, advantage)
    elif method == "progressive":
        return compute_per_turn_advantages_progressive(trajectory_dict, advantage, decay)
    elif method == "hindsight":
        return compute_per_turn_advantages_hindsight(trajectory_dict, advantage)
    elif method == "hybrid":
        return _compute_hybrid_advantages(trajectory_dict, advantage, alpha)
    else:
        raise ValueError(f"Unknown credit assignment method: {method}")


def _compute_hybrid_advantages(
    trajectory_dict: dict,
    advantage: float,
    alpha: float = 1.5,
) -> list[float]:
    """Hybrid: outcome weighting + sub-call signals + timeout penalties.

    For successful trajectories (advantage > 0):
      - FINAL turn: alpha * base
      - Turns with sub-calls: 1.0 * base
      - Pure setup turns: 0.7 * base
      - Timeout turns: 0.3 * base

    For failed trajectories (advantage <= 0):
      - Timeout turns: 1.5 * advantage (more negative = more penalty)
      - Error turns: 1.2 * advantage
      - Other turns: 1.0 * advantage (uniform)

    All weights are normalized so mean = 1.0 (preserving total gradient).
    """
    turns = trajectory_dict.get("turns", [])
    messages = trajectory_dict.get("messages", [])
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")

    if n_assistant <= 1:
        return [advantage] * max(n_assistant, 1)

    raw_weights = []
    for turn in turns:
        code = turn.get("parsed_code") or ""
        error = turn.get("error")
        terminated = turn.get("terminated", False)
        has_subcall = "llm_query" in code
        is_timeout = error and "Timeout" in str(error)
        is_error = error and not is_timeout

        if advantage > 0:
            # Successful trajectory: amplify useful turns
            if terminated:
                w = alpha
            elif is_timeout:
                w = 0.3  # Timeout even in success = bad turn
            elif is_error:
                w = 0.5  # Error but trajectory still succeeded
            elif has_subcall:
                w = 1.0  # Core work: querying sub-calls
            else:
                w = 0.7  # Pure setup (variable assignment, imports)
        else:
            # Failed trajectory: penalize problematic turns more
            if is_timeout:
                w = 1.5  # Timeouts are likely failure cause
            elif is_error:
                w = 1.2  # Errors are likely failure cause
            elif not code:
                w = 1.3  # No parseable code = bad
            else:
                w = 1.0  # Uniform for other turns

        raw_weights.append(w)

    # Pad/truncate to match assistant count
    while len(raw_weights) < n_assistant:
        raw_weights.append(1.0)
    raw_weights = raw_weights[:n_assistant]

    # Normalize so mean = 1.0
    mean_w = sum(raw_weights) / len(raw_weights)
    if mean_w < 1e-6:
        return [advantage] * n_assistant

    return [advantage * (w / mean_w) for w in raw_weights]
```

---

## Integration into rl_tinker_v6.py

The changes required are minimal. The advantage computation happens at lines
1134-1218 in the training loop, where `trajectory_to_training_data_is` and
`trajectory_to_training_data` are called with a scalar `advantage`. The fix:

### Change 1: Add `--credit-assignment` CLI flag

In the argparse section (~line 1390):

```python
parser.add_argument("--credit-assignment", type=str, default="uniform",
                    choices=["uniform", "outcome", "subcall", "progressive", "hindsight", "hybrid"],
                    help="Per-turn credit assignment method (default: uniform = current behavior)")
```

### Change 2: Modify the advantage-to-training-data path

Replace the current pattern (lines 1160-1167, 1210-1217):

```python
# BEFORE (current code, ~line 1160):
if turn_has_logprobs:
    data = trajectory_to_training_data_is(
        traj_dict, advantage, renderer, tokenizer
    )
else:
    data = trajectory_to_training_data(
        traj_dict, advantage, renderer, tokenizer
    )
```

With:

```python
# AFTER (with per-turn credit assignment):
per_turn_advs = compute_per_turn_advantages(
    traj_dict, advantage, method=credit_assignment
)
turn_has_logprobs = any(
    t.get("logprobs") is not None for t in traj_dict.get("turns", [])
)
if turn_has_logprobs:
    data = trajectory_to_training_data_is_per_turn(
        traj_dict, per_turn_advs, renderer, tokenizer
    )
else:
    data = trajectory_to_training_data_per_turn(
        traj_dict, per_turn_advs, renderer, tokenizer
    )
```

### Change 3: New per-turn-aware training data functions

These are minimal modifications of the existing functions. The only difference
is that `advantage` changes from a single float to a list indexed by assistant
turn position.

```python
def trajectory_to_training_data_is_per_turn(
    trajectory_dict: dict,
    per_turn_advantages: list[float],
    renderer,
    tokenizer,
) -> list[tinker.Datum]:
    """Like trajectory_to_training_data_is but with per-turn advantages."""
    data = []
    all_messages = trajectory_dict.get("messages", [])
    turns = trajectory_dict.get("turns", [])

    if not all_messages:
        return data

    turn_idx = 0
    assistant_idx = 0
    for i, msg in enumerate(all_messages):
        if msg["role"] != "assistant":
            continue

        if turn_idx >= len(turns):
            break
        turn = turns[turn_idx]
        turn_idx += 1

        # Per-turn advantage (falls back to last if list is short)
        if assistant_idx < len(per_turn_advantages):
            advantage = per_turn_advantages[assistant_idx]
        else:
            advantage = per_turn_advantages[-1] if per_turn_advantages else 0.0
        assistant_idx += 1

        turn_logprobs = turn.get("logprobs")
        turn_tokens = turn.get("tokens")
        if turn_logprobs is None or turn_tokens is None:
            continue

        try:
            full_msgs = all_messages[:i] + [msg]
            model_input, weights = renderer.build_supervised_example(full_msgs)

            from tinker_cookbook.supervised.common import (
                create_rightshifted_model_input_and_leftshifted_targets,
            )
            input_model_input, target_tokens = (
                create_rightshifted_model_input_and_leftshifted_targets(
                    model_input.chunks
                )
            )

            if hasattr(weights, "tolist"):
                weights_list = weights.tolist()
            else:
                weights_list = list(weights)
            weights_shifted = weights_list[1 : len(target_tokens) + 1]

            n_targets = len(target_tokens)
            logprobs_aligned = [0.0] * n_targets
            advantages_aligned = [0.0] * n_targets

            resp_start = None
            for j, w in enumerate(weights_shifted):
                if float(w) > 0:
                    resp_start = j
                    break

            if resp_start is not None:
                n_resp = min(len(turn_logprobs), n_targets - resp_start)
                clamped_adv = float(max(min(advantage, 3.0), -3.0))
                for j in range(n_resp):
                    logprobs_aligned[resp_start + j] = float(turn_logprobs[j])
                    advantages_aligned[resp_start + j] = clamped_adv

            datum = tinker.Datum(
                model_input=input_model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[len(target_tokens)],
                    ),
                    "logprobs": tinker.TensorData(
                        data=logprobs_aligned,
                        dtype="float32",
                        shape=[len(logprobs_aligned)],
                    ),
                    "advantages": tinker.TensorData(
                        data=advantages_aligned,
                        dtype="float32",
                        shape=[len(advantages_aligned)],
                    ),
                },
            )
            data.append(datum)
        except Exception as e:
            logger.warning(f"Failed to convert turn for IS (per-turn): {e}")

    return data


def trajectory_to_training_data_per_turn(
    trajectory_dict: dict,
    per_turn_advantages: list[float],
    renderer,
    tokenizer,
) -> list[tinker.Datum]:
    """Like trajectory_to_training_data but with per-turn advantages."""
    data = []
    all_messages = trajectory_dict.get("messages", [])

    if not all_messages:
        return data

    assistant_idx = 0
    for i, msg in enumerate(all_messages):
        if msg["role"] != "assistant":
            continue

        if assistant_idx < len(per_turn_advantages):
            advantage = per_turn_advantages[assistant_idx]
        else:
            advantage = per_turn_advantages[-1] if per_turn_advantages else 0.0
        assistant_idx += 1

        try:
            full_msgs = all_messages[:i] + [msg]
            model_input, weights = renderer.build_supervised_example(full_msgs)

            from tinker_cookbook.supervised.common import (
                create_rightshifted_model_input_and_leftshifted_targets,
            )
            input_model_input, target_tokens = (
                create_rightshifted_model_input_and_leftshifted_targets(
                    model_input.chunks
                )
            )

            if hasattr(weights, "tolist"):
                weights_list = weights.tolist()
            else:
                weights_list = list(weights)
            weights_shifted = weights_list[1 : len(target_tokens) + 1]

            advantage_weight = max(min(advantage, 3.0), -3.0)
            weighted = [float(w) * advantage_weight for w in weights_shifted]

            datum = tinker.Datum(
                model_input=input_model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[len(target_tokens)],
                    ),
                    "weights": tinker.TensorData(
                        data=weighted,
                        dtype="float32",
                        shape=[len(weighted)],
                    ),
                },
            )
            data.append(datum)
        except Exception as e:
            logger.warning(f"Failed to convert turn {i} (per-turn): {e}")

    return data
```

---

## Approach Comparison

| Approach | Complexity | Info Required | Risk | Best For |
|----------|-----------|---------------|------|----------|
| Outcome-weighted | Low | Turn position, terminated flag | Low | Quick improvement, safe default |
| Sub-call credit | Medium | Code content, error/timeout flags | Medium | Rewarding RLM-specific behavior |
| Progressive | Low | Turn count only | Low | When unsure about turn quality |
| Hindsight | Medium | Final answer, stdout content | High | Multi-hop tasks with traceable answers |
| Hybrid (1+2) | Medium | All turn metadata | Low | General-purpose, recommended |

## Recommendation

**Start with "hybrid" as default.** It:
1. Amplifies the FINAL turn (clear signal for success)
2. Penalizes timeouts even in successful trajectories (prevents learning slow patterns)
3. Differentiates sub-call turns from setup turns (rewards core RLM behavior)
4. Uses uniform weighting for failures (conservative --- don't over-blame)
5. Preserves total gradient magnitude (normalization to mean=1.0)

**Run an A/B experiment:**
- V12-uniform: current behavior (baseline)
- V12-hybrid: hybrid credit assignment
- Same everything else (LR, task distribution, K, steps)
- Compare after 10 steps on the 14-benchmark suite

**Hypothesis:** Hybrid credit assignment will improve training stability and
reduce the specialization-generalization tradeoff by giving stronger signal to
the turns that actually matter, rather than diluting it across setup turns.

## Logging Changes

Add per-turn advantage stats to the training log:

```python
# In the training loop, after computing per-turn advantages:
if credit_assignment != "uniform":
    turn_adv_ratios = [a / advantage if abs(advantage) > 1e-6 else 1.0
                       for a in per_turn_advs]
    logger.info(f"    Per-turn advantage ratios: {[f'{r:.2f}' for r in turn_adv_ratios]}")
```

This lets us verify the credit assignment is working as intended and correlate
turn-level signals with training outcomes.
