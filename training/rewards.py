"""
Reward functions for RLM training.

Start simple: trajectory-level binary reward.
The reward is 1.0 if the final answer matches the expected answer, 0.0 otherwise.

Later: add auxiliary signals for format quality, code correctness, etc.
"""

from __future__ import annotations

import re
from typing import Any


def binary_reward(predicted: str | None, expected: str) -> float:
    """
    Binary trajectory-level reward.
    1.0 if expected answer is found in predicted, 0.0 otherwise.
    """
    if predicted is None:
        return 0.0
    return 1.0 if expected.lower() in predicted.lower() else 0.0


def format_reward(trajectory: dict) -> float:
    """
    Auxiliary reward for correct formatting.

    +0.1 for using ```repl blocks
    +0.1 for using FINAL/FINAL_VAR correctly
    +0.1 for no syntax errors
    -0.1 for each error turn
    """
    score = 0.0
    turns = trajectory.get("turns", [])

    if not turns:
        return 0.0

    for turn in turns:
        code = turn.get("parsed_code", "")
        raw = turn.get("raw_response", "")
        error = turn.get("error")

        # Reward for using repl blocks
        if "```repl" in raw:
            score += 0.1 / len(turns)

        # Penalty for errors
        if error:
            score -= 0.1

    # Reward for termination
    if trajectory.get("terminated"):
        score += 0.1

    return max(-1.0, min(1.0, score))


def composite_reward(
    predicted: str | None,
    expected: str,
    trajectory: dict,
    correctness_weight: float = 0.8,
    format_weight: float = 0.2,
) -> float:
    """
    Composite reward combining correctness and format quality.

    Default: 80% correctness, 20% format.
    """
    correct = binary_reward(predicted, expected)
    fmt = format_reward(trajectory)
    return correctness_weight * correct + format_weight * fmt
