"""
OOLONG benchmark integration for RLMs.

OOLONG (arXiv:2511.02817) is a challenging aggregation benchmark
for long-context models. Tasks require analyzing D&D episode transcripts
to answer questions about rolls, spells, and character actions.

Uses the HuggingFace dataset: oolongbench/oolong-real (toy_dnd config).

Context lengths range from 50K to 900K+ characters — ideal for testing RLMs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class OolongTask:
    """A single OOLONG task."""
    task_id: str
    prompt: str
    expected_answer: str
    question_type: str  # singledoc_rolls, multidoc_rolls, singledoc_spells, multidoc_spells
    context_length: int
    question: str


def load_oolong_tasks(
    n_tasks: int = 20,
    max_context_chars: int = 200000,
    question_types: list[str] | None = None,
    seed: int = 42,
) -> list[OolongTask]:
    """Load OOLONG tasks from HuggingFace.

    Args:
        n_tasks: Number of tasks to load
        max_context_chars: Maximum context length in characters
        question_types: Filter by question type (None = all)
        seed: Random seed for sampling
    """
    from datasets import load_dataset

    ds = load_dataset('oolongbench/oolong-real', 'toy_dnd', split='validation')

    # Filter by context length and question type
    candidates = []
    for ex in ds:
        ctx_len = len(ex['context_window_text'])
        if ctx_len > max_context_chars:
            continue
        if question_types and ex['question_type'] not in question_types:
            continue
        candidates.append(ex)

    # Sample tasks
    rng = random.Random(seed)
    if len(candidates) > n_tasks:
        candidates = rng.sample(candidates, n_tasks)

    tasks = []
    for i, ex in enumerate(candidates):
        prompt = f"QUESTION: {ex['question']}\n\nCONTEXT:\n{ex['context_window_text']}"
        tasks.append(OolongTask(
            task_id=f"oolong_{i:03d}_{ex['question_type']}",
            prompt=prompt,
            expected_answer=str(ex['answer']),
            question_type=ex['question_type'],
            context_length=len(prompt),
            question=ex['question'],
        ))

    return tasks


def score_oolong(predicted: str | None, expected: str) -> dict:
    """Score an OOLONG prediction.

    OOLONG uses exact match scoring, but we add some flexibility:
    - Numeric answers: tolerance within 5%
    - Text answers: case-insensitive containment
    """
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    pred = predicted.strip()
    exp = expected.strip()

    # Exact match
    if pred.lower() == exp.lower():
        return {"score": 1.0, "match_type": "exact"}

    # Containment (prediction contains expected)
    if exp.lower() in pred.lower():
        return {"score": 1.0, "match_type": "contains"}

    # Numeric tolerance
    try:
        pred_num = float(pred.replace(",", ""))
        exp_num = float(exp.replace(",", ""))
        if exp_num != 0:
            rel_error = abs(pred_num - exp_num) / abs(exp_num)
            if rel_error < 0.05:
                return {"score": 0.5, "match_type": "numeric_close"}
    except ValueError:
        pass

    # Comma-separated list comparison (for multidoc tasks)
    if "," in exp:
        exp_items = set(item.strip().lower() for item in exp.split(","))
        pred_items = set(item.strip().lower() for item in pred.split(","))
        if exp_items == pred_items:
            return {"score": 1.0, "match_type": "set_match"}
        overlap = exp_items & pred_items
        if overlap and len(overlap) >= len(exp_items) * 0.5:
            return {"score": 0.5, "match_type": "partial_set"}

    return {"score": 0.0, "match_type": "none"}
