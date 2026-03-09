"""
Synthetic Needle-In-A-Haystack (S-NIAH) benchmark.

Generates tasks where a specific piece of information is hidden in a long
document of filler text. Tests the model's ability to search and extract.

O(1) complexity: the answer is always in one location.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass
class NIAHTask:
    """A single NIAH task."""
    task_id: str
    prompt: str          # Full text (question + document)
    needle: str          # The hidden information
    expected_answer: str # What the model should return
    needle_position: float  # 0.0-1.0 relative position in doc
    doc_length: int      # Total document length in chars
    question: str        # The question asked


# Filler text patterns
FILLER_SENTENCES = [
    "The weather was pleasant with clear skies and moderate temperatures.",
    "Recent studies have shown interesting patterns in data analysis.",
    "The committee met to discuss various administrative procedures.",
    "Market trends indicate steady growth across multiple sectors.",
    "The research team published their findings in the quarterly report.",
    "Infrastructure improvements were planned for the upcoming fiscal year.",
    "The conference attracted participants from numerous countries.",
    "Environmental monitoring systems recorded normal activity levels.",
    "Quality assurance protocols were updated according to new standards.",
    "The development team completed the scheduled maintenance tasks.",
    "Budget allocations were reviewed during the annual planning session.",
    "Training programs were expanded to include additional departments.",
    "Communication systems operated within expected parameters.",
    "The logistics department streamlined several delivery processes.",
    "Safety inspections confirmed compliance with all regulations.",
]

# Needle templates
NEEDLE_TEMPLATES = [
    ("The secret code is {value}.", "What is the secret code?", "{value}"),
    ("The password for the vault is {value}.", "What is the vault password?", "{value}"),
    ("The meeting is scheduled for {value}.", "When is the meeting scheduled?", "{value}"),
    ("The project codename is {value}.", "What is the project codename?", "{value}"),
    ("The key contact person is {value}.", "Who is the key contact person?", "{value}"),
    ("The budget total is {value}.", "What is the budget total?", "{value}"),
    ("The critical deadline is {value}.", "What is the critical deadline?", "{value}"),
    ("The assigned location is {value}.", "What is the assigned location?", "{value}"),
]

# Values to insert
NEEDLE_VALUES = [
    "ALPHA-7", "BRAVO-42", "CHARLIE-19", "DELTA-88", "ECHO-365",
    "FOXTROT-11", "GOLF-77", "HOTEL-256", "INDIA-33", "JULIET-99",
    "March 15 2025", "September 3 2024", "January 22 2026",
    "Dr. Elena Rodriguez", "Professor James Chen", "Agent Sarah Miller",
    "$4.2 million", "$850,000", "$12.7 billion",
    "Building 7 Room 304", "Sector 9 North Wing", "Lab Complex Alpha",
]


def _generate_filler(length: int, seed: int) -> str:
    """Generate filler text of approximately `length` characters."""
    rng = random.Random(seed)
    parts = []
    current_len = 0
    while current_len < length:
        sentence = rng.choice(FILLER_SENTENCES)
        parts.append(sentence)
        current_len += len(sentence) + 1  # +1 for newline
    text = "\n".join(parts)
    return text[:length]


def generate_niah_task(
    task_idx: int,
    doc_length: int = 10000,
    needle_position: float = 0.5,
    seed: int | None = None,
) -> NIAHTask:
    """
    Generate a single NIAH task.

    Args:
        task_idx: Task index (for deterministic generation)
        doc_length: Total document length in characters
        needle_position: Where to place the needle (0.0=start, 1.0=end)
        seed: Random seed (defaults to task_idx)
    """
    seed = seed if seed is not None else task_idx
    rng = random.Random(seed)

    # Pick needle template and value
    template, question, answer_template = rng.choice(NEEDLE_TEMPLATES)
    value = rng.choice(NEEDLE_VALUES)
    needle = template.format(value=value)
    expected_answer = answer_template.format(value=value)

    # Generate filler before and after needle
    needle_char_pos = int(doc_length * needle_position)
    before_len = max(0, needle_char_pos - len(needle) // 2)
    after_len = max(0, doc_length - before_len - len(needle))

    before_text = _generate_filler(before_len, seed * 1000 + 1)
    after_text = _generate_filler(after_len, seed * 1000 + 2)

    document = before_text + "\n" + needle + "\n" + after_text

    # Build full prompt
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    task_id = f"niah_{task_idx:03d}_{doc_length}_{needle_position:.1f}"

    return NIAHTask(
        task_id=task_id,
        prompt=prompt,
        needle=needle,
        expected_answer=expected_answer,
        needle_position=needle_position,
        doc_length=len(document),
        question=question,
    )


def generate_niah_suite(
    n_tasks: int = 50,
    doc_lengths: list[int] | None = None,
    positions: list[float] | None = None,
    seed_offset: int = 0,
) -> list[NIAHTask]:
    """
    Generate a full NIAH benchmark suite.

    Default: 50 tasks across varying document lengths and needle positions.

    Args:
        seed_offset: Offset for task seeds to avoid overlap with training data.
                     Use 0 for training, 10000+ for eval.
    """
    if doc_lengths is None:
        doc_lengths = [5000, 10000, 20000, 50000, 100000]
    if positions is None:
        positions = [0.1, 0.25, 0.5, 0.75, 0.9]

    tasks = []
    idx = 0
    for doc_len in doc_lengths:
        for pos in positions:
            for _ in range(n_tasks // (len(doc_lengths) * len(positions)) or 1):
                tasks.append(generate_niah_task(
                    idx + seed_offset, doc_len, pos, seed=idx + seed_offset,
                ))
                idx += 1
                if len(tasks) >= n_tasks:
                    return tasks

    # Fill remaining with random combinations
    rng = random.Random(42 + seed_offset)
    while len(tasks) < n_tasks:
        tasks.append(generate_niah_task(
            idx + seed_offset,
            rng.choice(doc_lengths),
            rng.choice(positions),
            seed=idx + seed_offset,
        ))
        idx += 1

    return tasks


def score_niah(predicted: str | None, expected: str) -> float:
    """Score a NIAH prediction. Returns 1.0 if expected is found in predicted."""
    if predicted is None:
        return 0.0
    return 1.0 if expected.lower() in predicted.lower() else 0.0
