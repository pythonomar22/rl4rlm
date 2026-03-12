"""
OOLONG benchmark (trec_coarse split) for RLMs.

OOLONG (Bertsch et al., arXiv:2511.02817) is a long-context reasoning benchmark
that requires aggregating information across the full input. The trec_coarse split
contains TREC questions labeled with 6 coarse categories; tasks ask about label
frequencies, comparisons, and counts.

Uses HuggingFace dataset: oolongbench/oolong-synth (validation split, dataset='trec_coarse').

This is the SAME benchmark used in the original RLM paper (Zhang, Kraska, Khattab,
arXiv:2502.14155), which evaluates on 50 tasks at a given context length.

Scoring follows the paper: 0.75^|y - y_hat| for numeric answers, exact match otherwise.
"""

from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import dataclass


@dataclass
class OolongTask:
    """A single OOLONG task."""
    task_id: str
    prompt: str
    expected_answer: str
    answer_type: str     # ANSWER_TYPE.LABEL, NUMERIC, USER, COMPARISON
    task_type: str       # TASK_TYPE.MOST_FREQ, LEAST_FREQ, etc.
    context_length: int  # context_len token bucket from the dataset
    question: str


def load_oolong_tasks(
    n_tasks: int = 50,
    context_len: int = 131072,
    seed: int = 42,
) -> list[OolongTask]:
    """Load OOLONG trec_coarse tasks from HuggingFace.

    Args:
        n_tasks: Number of tasks to load (max 50 per context_len)
        context_len: Token-count bucket to filter on.
            Available: 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576, 2097152, 4194304.
            Default 131072 (128K tokens) is a good RLM test point.
        seed: Random seed for sampling when n_tasks < available
    """
    from datasets import load_dataset

    ds = load_dataset('oolongbench/oolong-synth', split='validation')

    # Filter to trec_coarse at the requested context length
    candidates = []
    for ex in ds:
        if ex['dataset'] != 'trec_coarse':
            continue
        if ex['context_len'] != context_len:
            continue
        candidates.append(ex)

    if not candidates:
        available_lens = sorted(set(
            ex['context_len'] for ex in ds if ex['dataset'] == 'trec_coarse'
        ))
        raise ValueError(
            f"No trec_coarse tasks at context_len={context_len}. "
            f"Available: {available_lens}"
        )

    # Sample if needed
    rng = random.Random(seed)
    if len(candidates) > n_tasks:
        candidates = rng.sample(candidates, n_tasks)

    tasks = []
    for i, ex in enumerate(candidates):
        # The dataset provides both raw text and labeled text
        # Use raw text (context_window_text) as the context
        context_text = ex['context_window_text']

        # Parse the answer from the dataset's string-list format
        # Answers look like "['abbreviation']" or "['42']" or "['more common than']"
        raw_answer = ex['answer']
        try:
            parsed = ast.literal_eval(raw_answer)
            if isinstance(parsed, list) and len(parsed) == 1:
                answer = str(parsed[0])
            elif isinstance(parsed, list):
                # Multi-value answer (e.g. USER tasks with multiple labels)
                answer = ", ".join(str(x) for x in parsed)
            else:
                answer = str(parsed)
        except (ValueError, SyntaxError):
            answer = str(raw_answer)

        # The dataset uses 'task' field (e.g. TASK_TYPE.MOST_FREQ), not 'task_type'
        task_type_str = ex.get('task', ex.get('task_type', 'unknown'))

        prompt = f"QUESTION: {ex['question']}\n\nDOCUMENT:\n{context_text}"

        tasks.append(OolongTask(
            task_id=f"oolong_{i:03d}_{task_type_str}",
            prompt=prompt,
            expected_answer=answer,
            answer_type=ex.get('answer_type', 'unknown'),
            task_type=task_type_str,
            context_length=ex['context_len'],
            question=ex['question'],
        ))

    return tasks


def score_oolong(predicted: str | None, expected: str, answer_type: str = "") -> dict:
    """Score an OOLONG prediction using the paper's scoring method.

    Scoring (from Bertsch et al. 2025):
    - Numeric answers: score = 0.75^|y - y_hat|
    - All other answers: exact match (case-insensitive)
    """
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    pred = predicted.strip().strip("'\"[]")
    exp = expected.strip().strip("'\"[]")

    # Exact match (case-insensitive)
    if pred.lower() == exp.lower():
        return {"score": 1.0, "match_type": "exact"}

    # Check if this is a numeric answer
    is_numeric = "NUMERIC" in answer_type.upper() if answer_type else False

    # Try numeric scoring: 0.75^|y - y_hat|
    pred_num = _extract_number(pred)
    exp_num = _extract_number(exp)

    if pred_num is not None and exp_num is not None:
        diff = abs(pred_num - exp_num)
        score = math.pow(0.75, diff)
        match_type = "numeric_exact" if diff == 0 else f"numeric_partial_diff{diff:.1f}"
        return {"score": score, "match_type": match_type}

    # For non-numeric: check if the expected label appears in the prediction
    # This handles cases where the model wraps the answer in extra text
    if exp.lower() in pred.lower():
        # Verify it's not a substring match (e.g., "entity" in "no entity found")
        # Check word boundaries
        pattern = r'\b' + re.escape(exp.lower()) + r'\b'
        if re.search(pattern, pred.lower()):
            return {"score": 1.0, "match_type": "contains"}

    return {"score": 0.0, "match_type": "none"}


def _extract_number(s: str) -> float | None:
    """Try to extract a number from a string."""
    s = s.strip()
    # Direct parse
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    # Find a number in the string
    m = re.search(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', s)
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            pass
    return None
