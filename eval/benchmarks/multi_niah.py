"""
Multi-Needle-In-A-Haystack (MN-NIAH) benchmark.

Hides K needles in a long document. Model must find ALL of them.
Tests O(K) complexity: requires systematic search through the entire document.

More challenging than single-needle NIAH because:
1. Model can't get lucky with a single chunk
2. Must iterate and aggregate results
3. Tests the model's ability to build a complete answer from multiple sub-calls

Difficulty scales with:
- Number of needles (K)
- Document length (L)
- K/L ratio (needle density)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class MultiNIAHTask:
    """A single multi-needle NIAH task."""
    task_id: str
    prompt: str            # Full text (question + document)
    needles: list[dict]    # [{code_name, value, position}]
    expected_answers: list[str]  # All values that should be found
    doc_length: int        # Total document length in chars
    question: str
    n_needles: int


# Code names for needles
CODE_NAMES = [
    "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO",
    "FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIET",
    "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR",
    "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO",
]

# Values that are distinctive and easy to score
CODE_VALUES = [
    "7X9-BLUE", "3K2-RED", "8M4-GREEN", "5P1-GOLD", "2W6-SILVER",
    "9J3-ORANGE", "4H7-PURPLE", "1N8-BLACK", "6T5-WHITE", "0R9-CYAN",
    "QZ4-RUBY", "VL2-JADE", "FS7-ONYX", "BH1-PEARL", "DK6-AMBER",
    "GT3-CORAL", "WN8-IVORY", "XP5-SLATE", "YM2-BRONZE", "UC7-CHROME",
]

# Filler sentences (reuse pattern from NIAH but different seed = different order)
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
    "The annual review highlighted several areas for improvement.",
    "International partnerships were strengthened through diplomatic efforts.",
    "New policies were implemented to address emerging challenges.",
    "The technical team resolved several outstanding infrastructure issues.",
    "Regional offices submitted their quarterly performance summaries.",
]


def _generate_filler(length: int, seed: int) -> str:
    """Generate filler text of approximately `length` characters."""
    rng = random.Random(seed)
    parts = []
    current_len = 0
    while current_len < length:
        sentence = rng.choice(FILLER_SENTENCES)
        parts.append(sentence)
        current_len += len(sentence) + 1
    text = "\n".join(parts)
    return text[:length]


def generate_multi_niah_task(
    task_idx: int,
    doc_length: int = 20000,
    n_needles: int = 5,
    seed: int | None = None,
) -> MultiNIAHTask:
    """
    Generate a single multi-needle NIAH task.

    Each needle is formatted as: SECRET_CODE_[NAME]: [VALUE]
    Needles are distributed throughout the document.
    """
    seed = seed if seed is not None else task_idx + 50000
    rng = random.Random(seed)

    # Pick needle identifiers and values
    names = rng.sample(CODE_NAMES, n_needles)
    values = rng.sample(CODE_VALUES, n_needles)

    # Distribute needles across document with some randomness
    # Evenly spaced with jitter to avoid predictable positions
    base_positions = [(i + 0.5) / n_needles for i in range(n_needles)]
    positions = []
    for bp in base_positions:
        jitter = rng.uniform(-0.3 / n_needles, 0.3 / n_needles)
        positions.append(max(0.05, min(0.95, bp + jitter)))
    positions.sort()

    needles = []
    for name, value, pos in zip(names, values, positions):
        needles.append({
            "code_name": name,
            "value": value,
            "position": pos,
        })

    # Build document with needles inserted
    # First, generate all filler text
    total_filler = doc_length - sum(
        len(f"SECRET_CODE_{n['code_name']}: {n['value']}") + 2  # +2 for newlines
        for n in needles
    )
    total_filler = max(total_filler, 1000)

    # Split filler into segments between needles
    n_segments = n_needles + 1
    segment_lengths = []
    remaining = total_filler
    for i in range(n_segments):
        if i < n_segments - 1:
            # Proportional to gap between positions
            if i == 0:
                frac = positions[0]
            else:
                frac = positions[i] - positions[i - 1]
            seg_len = int(total_filler * frac)
        else:
            seg_len = remaining
        segment_lengths.append(max(100, seg_len))
        remaining -= segment_lengths[-1]

    # Generate document
    parts = []
    for i in range(n_needles):
        parts.append(_generate_filler(segment_lengths[i], seed * 100 + i))
        parts.append(f"\nSECRET_CODE_{needles[i]['code_name']}: {needles[i]['value']}\n")
    parts.append(_generate_filler(segment_lengths[-1], seed * 100 + n_needles))

    document = "".join(parts)

    # Build prompt
    question = (
        f"This document contains {n_needles} hidden secret codes. "
        f"Each code is formatted as 'SECRET_CODE_[NAME]: [VALUE]'. "
        f"Find ALL {n_needles} secret codes and return ONLY their values "
        f"as a comma-separated list (e.g., 'value1, value2, value3')."
    )

    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    task_id = f"mniah_{task_idx:03d}_{doc_length}_{n_needles}needles"

    return MultiNIAHTask(
        task_id=task_id,
        prompt=prompt,
        needles=needles,
        expected_answers=values,
        doc_length=len(document),
        question=question,
        n_needles=n_needles,
    )


def generate_multi_niah_suite(
    n_tasks: int = 30,
    seed_offset: int = 50000,
) -> list[MultiNIAHTask]:
    """
    Generate multi-needle NIAH benchmark suite.

    Configurations test scaling across document length and needle count:
    - Short docs with few needles (easy): 10K, 3 needles
    - Medium docs with moderate needles: 20K, 5 needles
    - Long docs with many needles (hard): 50K, 8 needles
    - Very long with many needles (very hard): 100K, 10 needles
    """
    configs = [
        # (doc_length, n_needles, count)
        (10000, 3, 4),     # Easy: short doc, few needles
        (20000, 5, 4),     # Medium: medium doc, moderate needles
        (50000, 5, 4),     # Medium-hard: long doc, moderate needles
        (50000, 8, 4),     # Hard: long doc, many needles
        (100000, 8, 4),    # Very hard: very long doc, many needles
        (100000, 10, 4),   # Extreme: very long doc, lots of needles
    ]

    tasks = []
    idx = 0
    for doc_len, n_needles, count in configs:
        for _ in range(count):
            if len(tasks) >= n_tasks:
                return tasks
            tasks.append(generate_multi_niah_task(
                task_idx=idx,
                doc_length=doc_len,
                n_needles=n_needles,
                seed=idx + seed_offset,
            ))
            idx += 1

    return tasks[:n_tasks]


def score_multi_niah(predicted: str | None, expected_values: list[str]) -> dict:
    """
    Score a multi-needle NIAH prediction.

    Returns:
        dict with 'recall' (fraction of needles found), 'precision',
        'f1', and 'found' (count of needles found).
    """
    if predicted is None:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "found": 0, "total": len(expected_values)}

    predicted_lower = predicted.lower()

    # Count how many expected values appear in the prediction
    found = sum(1 for v in expected_values if v.lower() in predicted_lower)
    total = len(expected_values)

    recall = found / total if total > 0 else 0.0

    # Estimate precision: count distinct code values mentioned
    # (We can't perfectly compute precision without parsing, so use recall as proxy
    # for tasks where we know the total)
    precision = recall  # Simplification: assume model doesn't hallucinate extra codes

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "found": found,
        "total": total,
    }
