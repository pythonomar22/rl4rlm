"""
LongBench-v2 CodeQA benchmark for RLMs.

From Bai et al. 2025 (arXiv:2412.15204), "LongBench v2: Towards Deeper
Understanding and Reasoning on Realistic Long-context Multitasks."

The Code Repository Understanding split contains 50 multiple-choice questions
about code repositories. Context lengths range from ~100K to ~16M characters.

This is the SAME benchmark used in the original RLM paper (Zhang, Kraska, Khattab,
arXiv:2502.14155), where RLM(GPT-5) achieved 62.0% vs base GPT-5 at 24.0%.

Uses HuggingFace dataset: THUDM/LongBench-v2 (train split, domain='Code Repository Understanding').
Scoring: exact match on A/B/C/D answer.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass
class LongBenchCodeTask:
    """A single LongBench-v2 CodeQA task."""
    task_id: str
    prompt: str
    expected_answer: str  # A, B, C, or D
    question: str
    choices: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    difficulty: str   # "easy" or "hard"
    context_length: int  # len of context in chars


def load_longbench_codeqa_tasks(
    n_tasks: int = 50,
    max_context_chars: int | None = None,
    seed: int = 42,
) -> list[LongBenchCodeTask]:
    """Load LongBench-v2 Code Repository Understanding tasks.

    Args:
        n_tasks: Number of tasks (max 50)
        max_context_chars: Optional cap on context length (None = no limit)
        seed: Random seed for sampling
    """
    from datasets import load_dataset

    ds = load_dataset('THUDM/LongBench-v2', split='train')

    # Filter to Code Repository Understanding
    candidates = []
    for ex in ds:
        if ex['domain'] != 'Code Repository Understanding':
            continue
        if max_context_chars and len(ex['context']) > max_context_chars:
            continue
        candidates.append(ex)

    if not candidates:
        raise ValueError("No Code Repository Understanding tasks found")

    # Sample if needed
    rng = random.Random(seed)
    if len(candidates) > n_tasks:
        candidates = rng.sample(candidates, n_tasks)

    tasks = []
    for i, ex in enumerate(candidates):
        choices = {
            "A": ex['choice_A'],
            "B": ex['choice_B'],
            "C": ex['choice_C'],
            "D": ex['choice_D'],
        }

        # Format the question with choices
        choices_text = "\n".join(f"  {k}. {v}" for k, v in choices.items())
        question_text = (
            f"{ex['question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Answer with ONLY the letter (A, B, C, or D)."
        )

        prompt = f"QUESTION: {question_text}\n\nCODE REPOSITORY:\n{ex['context']}"

        tasks.append(LongBenchCodeTask(
            task_id=f"longbench_code_{i:03d}",
            prompt=prompt,
            expected_answer=ex['answer'],
            question=ex['question'],
            choices=choices,
            difficulty=ex['difficulty'],
            context_length=len(ex['context']),
        ))

    return tasks


def score_longbench_codeqa(predicted: str | None, expected: str) -> dict:
    """Score a LongBench-v2 CodeQA prediction.

    Scoring: exact match on the letter A/B/C/D.
    """
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    pred = predicted.strip()

    # Extract the letter answer from the prediction
    # The model might say "A", "A.", "(A)", "The answer is A", etc.
    pred_letter = _extract_letter(pred)

    if pred_letter and pred_letter == expected.strip().upper():
        return {"score": 1.0, "match_type": "exact"}

    # Direct comparison (case-insensitive)
    if pred.upper() == expected.strip().upper():
        return {"score": 1.0, "match_type": "exact"}

    return {"score": 0.0, "match_type": "wrong"}


def _extract_letter(text: str) -> str | None:
    """Extract a single letter answer (A/B/C/D) from model output."""
    text = text.strip()

    # If the answer is just a single letter
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()

    # Common patterns: "A.", "(A)", "Answer: A", "The answer is A"
    patterns = [
        r'^([A-D])\.',           # "A."
        r'^\(([A-D])\)',          # "(A)"
        r'^([A-D])\)',            # "A)"
        r'answer\s*(?:is\s*)?([A-D])\b',  # "answer is A", "answer: A"
        r'^([A-D])\b',           # "A" at start
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Last resort: find any standalone A/B/C/D
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()

    return None
