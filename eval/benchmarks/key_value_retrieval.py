"""
Key-Value Retrieval benchmark — tests exact lookup of structured entries.

Generates long documents (50K-200K chars) containing a registry/database of
structured entries. Each entry has a unique REGISTRY_ID, a numeric VALUE,
a STATUS, and a CATEGORY. Target entries are scattered among thousands of
similar-looking distractor entries. The model must navigate the long context,
find specific entries by key, and return exact field values.

This benchmark is challenging because:
- Thousands of visually similar distractor entries
- Target entries are randomly placed among noise
- Exact values must be returned (partial credit for close matches)
- Aggregate tasks require finding multiple entries and computing over them

Task types:
1. single_lookup: Find one entry by REGISTRY_ID, return a specific field
2. multi_lookup: Find 3-5 entries by REGISTRY_ID, return a field for each
3. aggregate: Find all entries matching a CATEGORY and sum their VALUEs
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field


@dataclass
class KeyValueTask:
    """A single key-value retrieval task."""
    task_id: str
    prompt: str
    expected_answer: str
    task_type: str  # single_lookup, multi_lookup, aggregate
    doc_length: int
    n_target_entries: int
    n_distractor_entries: int
    question: str
    target_entries: list[dict] = field(default_factory=list)


# Status values
STATUSES = ["active", "inactive", "pending", "suspended", "archived", "verified"]

# Category values
CATEGORIES = [
    "infrastructure", "personnel", "logistics", "research",
    "finance", "security", "operations", "maintenance",
]

# Department prefixes for registry IDs
DEPT_PREFIXES = [
    "INF", "PER", "LOG", "RES", "FIN", "SEC", "OPS", "MNT",
    "ADM", "ENG", "QAS", "DEV", "SRV", "NET", "DAT", "CLD",
]

# Filler narrative text between entry blocks
FILLER_SENTENCES = [
    "The registry was last audited on schedule per standard operating procedure.",
    "All entries in this section have been cross-referenced with the central database.",
    "Periodic reviews ensure that the records remain accurate and up to date.",
    "The following entries were migrated from the legacy tracking system.",
    "Compliance checks confirmed that all records met regulatory requirements.",
    "This section of the registry covers allocations from the current fiscal period.",
    "Data integrity verification was performed by the automated validation suite.",
    "Records in this block have been approved by the designated review authority.",
    "The classification schema follows the organizational taxonomy version 4.2.",
    "Entries are indexed by their unique registry identifier for efficient retrieval.",
    "The next scheduled review for this section is pending administrative approval.",
    "Historical records from the previous system have been archived separately.",
    "Redundant entries were identified and merged during the last consolidation.",
    "Access controls for this registry section follow the standard tier-2 policy.",
    "The automated reconciliation process found no discrepancies in this batch.",
    "Metadata tags were updated to reflect the revised classification guidelines.",
    "This portion of the registry was populated during the initial data migration.",
    "Quality assurance sampling confirmed a 99.7% accuracy rate for this section.",
    "The registration authority verified all entries against the master reference list.",
    "Backup copies of this registry section are maintained in three geographic zones.",
]


def _generate_registry_id(rng: random.Random) -> str:
    """Generate a unique-looking registry ID like 'INF-83721'."""
    prefix = rng.choice(DEPT_PREFIXES)
    number = rng.randint(10000, 99999)
    return f"{prefix}-{number}"


def _generate_entry(
    registry_id: str,
    value: float,
    status: str,
    category: str,
    rng: random.Random,
) -> str:
    """Generate a single registry entry line in one of several formats."""
    templates = [
        (
            f"REGISTRY_ID: {registry_id} | VALUE: {value:.2f} "
            f"| STATUS: {status} | CATEGORY: {category}"
        ),
        (
            f"[{registry_id}]  value={value:.2f}  status={status}  "
            f"category={category}"
        ),
        (
            f"  {registry_id}    {value:.2f}    {status:<12s}  {category}"
        ),
        (
            f"Entry {registry_id}: val {value:.2f}, "
            f"sts {status}, cat {category}"
        ),
    ]
    # Use the same format index per document to keep the registry consistent
    fmt_idx = rng.getrandbits(2)  # 0-3, chosen once per document via doc-level rng
    return templates[fmt_idx]


def _generate_filler_block(target_chars: int, rng: random.Random) -> str:
    """Generate filler narrative text of approximately target_chars length."""
    lines: list[str] = []
    total = 0
    while total < target_chars:
        line = rng.choice(FILLER_SENTENCES)
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _generate_document(
    target_entries: list[dict],
    n_distractors: int,
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build a registry document with target entries hidden among distractors.

    Returns the full document text. The format index is chosen once so that
    all entries in a document share the same visual format, making it harder
    to distinguish targets from distractors.
    """
    # Choose a single entry format for this document
    fmt_rng = random.Random(rng.randint(0, 2**31))

    # Collect all registry IDs to avoid collisions
    used_ids = {e["registry_id"] for e in target_entries}

    # Generate distractor entries
    distractors: list[dict] = []
    for _ in range(n_distractors):
        rid = _generate_registry_id(rng)
        while rid in used_ids:
            rid = _generate_registry_id(rng)
        used_ids.add(rid)
        distractors.append({
            "registry_id": rid,
            "value": round(rng.uniform(0.01, 9999.99), 2),
            "status": rng.choice(STATUSES),
            "category": rng.choice(CATEGORIES),
        })

    # Merge targets and distractors, then assign random positions
    all_entries = list(target_entries) + distractors
    rng.shuffle(all_entries)

    # Build document in blocks: filler -> entries -> filler -> entries -> ...
    # Group entries into blocks of 5-15 to simulate registry sections
    block_size_range = (5, 15)
    entry_blocks: list[list[dict]] = []
    i = 0
    while i < len(all_entries):
        bs = rng.randint(*block_size_range)
        entry_blocks.append(all_entries[i:i + bs])
        i += bs

    # Target filler chars between blocks
    n_blocks = len(entry_blocks)
    total_entry_chars = sum(
        len(_generate_entry(e["registry_id"], e["value"], e["status"], e["category"], fmt_rng))
        for block in entry_blocks for e in block
    )
    remaining_chars = max(0, doc_length - total_entry_chars)
    filler_per_gap = remaining_chars // (n_blocks + 1) if n_blocks > 0 else remaining_chars

    parts: list[str] = []
    # Opening header
    parts.append("=" * 72)
    parts.append("  CONSOLIDATED REGISTRY — INTERNAL USE ONLY")
    parts.append("=" * 72)
    parts.append("")

    for block_idx, block in enumerate(entry_blocks):
        # Filler before block
        filler_target = max(80, filler_per_gap + rng.randint(-200, 200))
        parts.append(_generate_filler_block(filler_target, rng))
        parts.append("")
        parts.append(f"--- Section {block_idx + 1:04d} ---")
        for entry in block:
            parts.append(
                _generate_entry(
                    entry["registry_id"], entry["value"],
                    entry["status"], entry["category"], fmt_rng,
                )
            )
        parts.append("")

    # Final filler
    current_len = sum(len(p) + 1 for p in parts)
    if current_len < doc_length:
        parts.append(_generate_filler_block(doc_length - current_len, rng))

    parts.append("")
    parts.append("=" * 72)
    parts.append("  END OF REGISTRY")
    parts.append("=" * 72)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def generate_single_lookup_task(
    task_id: str, doc_length: int, n_distractors: int, seed: int,
) -> KeyValueTask:
    """Find one entry by REGISTRY_ID and return a specific field."""
    rng = random.Random(seed)

    # Create the target entry
    target = {
        "registry_id": _generate_registry_id(rng),
        "value": round(rng.uniform(100.0, 9999.99), 2),
        "status": rng.choice(STATUSES),
        "category": rng.choice(CATEGORIES),
    }

    document = _generate_document([target], n_distractors, doc_length, rng)

    # Ask for a specific field
    field_choice = rng.choice(["value", "status", "category"])
    if field_choice == "value":
        question = (
            f"Find the registry entry with ID '{target['registry_id']}' "
            f"and return its VALUE. Return ONLY the numeric value."
        )
        expected = f"{target['value']:.2f}"
    elif field_choice == "status":
        question = (
            f"Find the registry entry with ID '{target['registry_id']}' "
            f"and return its STATUS. Return ONLY the status word."
        )
        expected = target["status"]
    else:
        question = (
            f"Find the registry entry with ID '{target['registry_id']}' "
            f"and return its CATEGORY. Return ONLY the category word."
        )
        expected = target["category"]

    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return KeyValueTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=expected,
        task_type="single_lookup",
        doc_length=len(document),
        n_target_entries=1,
        n_distractor_entries=n_distractors,
        question=question,
        target_entries=[target],
    )


def generate_multi_lookup_task(
    task_id: str, doc_length: int, n_distractors: int, seed: int,
) -> KeyValueTask:
    """Find 3-5 entries by REGISTRY_ID and return a field for each."""
    rng = random.Random(seed)

    n_targets = rng.randint(3, 5)
    used_ids: set[str] = set()
    targets: list[dict] = []
    for _ in range(n_targets):
        rid = _generate_registry_id(rng)
        while rid in used_ids:
            rid = _generate_registry_id(rng)
        used_ids.add(rid)
        targets.append({
            "registry_id": rid,
            "value": round(rng.uniform(100.0, 9999.99), 2),
            "status": rng.choice(STATUSES),
            "category": rng.choice(CATEGORIES),
        })

    document = _generate_document(targets, n_distractors, doc_length, rng)

    # Ask for VALUE of each target
    id_list = ", ".join(t["registry_id"] for t in targets)
    question = (
        f"Find the registry entries with the following IDs: {id_list}. "
        f"For each entry, return its VALUE. "
        f"Format your answer as 'ID: value' pairs separated by commas, "
        f"in the same order as listed above."
    )

    # Expected: "INF-12345: 456.78, SEC-99881: 123.45, ..."
    expected_parts = [
        f"{t['registry_id']}: {t['value']:.2f}" for t in targets
    ]
    expected = ", ".join(expected_parts)

    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return KeyValueTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=expected,
        task_type="multi_lookup",
        doc_length=len(document),
        n_target_entries=n_targets,
        n_distractor_entries=n_distractors,
        question=question,
        target_entries=targets,
    )


def generate_aggregate_task(
    task_id: str, doc_length: int, n_distractors: int, seed: int,
) -> KeyValueTask:
    """Find all entries matching a category and sum their values."""
    rng = random.Random(seed)

    # Pick a target category
    target_category = rng.choice(CATEGORIES)

    # Create 5-10 target entries with this category
    n_targets = rng.randint(5, 10)
    used_ids: set[str] = set()
    targets: list[dict] = []
    for _ in range(n_targets):
        rid = _generate_registry_id(rng)
        while rid in used_ids:
            rid = _generate_registry_id(rng)
        used_ids.add(rid)
        targets.append({
            "registry_id": rid,
            "value": round(rng.uniform(100.0, 9999.99), 2),
            "status": rng.choice(STATUSES),
            "category": target_category,
        })

    # Ensure distractors do NOT have the target category so the sum is
    # unambiguous. We monkey-patch the category choices for distractors.
    other_categories = [c for c in CATEGORIES if c != target_category]

    # Build distractors manually so we control their categories
    distractor_rng = random.Random(rng.randint(0, 2**31))
    distractors: list[dict] = []
    for _ in range(n_distractors):
        rid = _generate_registry_id(distractor_rng)
        while rid in used_ids:
            rid = _generate_registry_id(distractor_rng)
        used_ids.add(rid)
        distractors.append({
            "registry_id": rid,
            "value": round(distractor_rng.uniform(0.01, 9999.99), 2),
            "status": distractor_rng.choice(STATUSES),
            "category": distractor_rng.choice(other_categories),
        })

    # Build doc with custom distractor list
    document = _generate_document_from_all(targets, distractors, doc_length, rng)

    total_value = round(sum(t["value"] for t in targets), 2)
    question = (
        f"Find ALL registry entries with CATEGORY '{target_category}' and "
        f"compute the sum of their VALUEs. Return ONLY the numeric sum, "
        f"rounded to two decimal places."
    )
    expected = f"{total_value:.2f}"

    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return KeyValueTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=expected,
        task_type="aggregate",
        doc_length=len(document),
        n_target_entries=n_targets,
        n_distractor_entries=n_distractors,
        question=question,
        target_entries=targets,
    )


def _generate_document_from_all(
    targets: list[dict],
    distractors: list[dict],
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build registry document from pre-built target and distractor lists."""
    fmt_rng = random.Random(rng.randint(0, 2**31))

    all_entries = list(targets) + list(distractors)
    rng.shuffle(all_entries)

    block_size_range = (5, 15)
    entry_blocks: list[list[dict]] = []
    i = 0
    while i < len(all_entries):
        bs = rng.randint(*block_size_range)
        entry_blocks.append(all_entries[i:i + bs])
        i += bs

    n_blocks = len(entry_blocks)
    total_entry_chars = sum(
        len(_generate_entry(e["registry_id"], e["value"], e["status"], e["category"], fmt_rng))
        for block in entry_blocks for e in block
    )
    remaining_chars = max(0, doc_length - total_entry_chars)
    filler_per_gap = remaining_chars // (n_blocks + 1) if n_blocks > 0 else remaining_chars

    parts: list[str] = []
    parts.append("=" * 72)
    parts.append("  CONSOLIDATED REGISTRY — INTERNAL USE ONLY")
    parts.append("=" * 72)
    parts.append("")

    for block_idx, block in enumerate(entry_blocks):
        filler_target = max(80, filler_per_gap + rng.randint(-200, 200))
        parts.append(_generate_filler_block(filler_target, rng))
        parts.append("")
        parts.append(f"--- Section {block_idx + 1:04d} ---")
        for entry in block:
            parts.append(
                _generate_entry(
                    entry["registry_id"], entry["value"],
                    entry["status"], entry["category"], fmt_rng,
                )
            )
        parts.append("")

    current_len = sum(len(p) + 1 for p in parts)
    if current_len < doc_length:
        parts.append(_generate_filler_block(doc_length - current_len, rng))

    parts.append("")
    parts.append("=" * 72)
    parts.append("  END OF REGISTRY")
    parts.append("=" * 72)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Suite generation
# ---------------------------------------------------------------------------

def generate_key_value_suite(
    n_tasks: int = 10,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[KeyValueTask]:
    """Generate a suite of key-value retrieval tasks.

    Default: 50K-200K char documents with hundreds to thousands of
    distractor entries (roughly 1 distractor per 50 chars of target
    doc length).

    Args:
        n_tasks: Number of tasks to generate.
        doc_lengths: List of document lengths (chars) to sample from.
        seed_offset: Offset for task seeds (use 0 for training, 10000+ for eval).
    """
    if doc_lengths is None:
        doc_lengths = [50000, 100000, 150000, 200000]

    generators = [
        generate_single_lookup_task,
        generate_multi_lookup_task,
        generate_aggregate_task,
    ]

    tasks: list[KeyValueTask] = []
    for i in range(n_tasks):
        seed = seed_offset + i * 13
        rng = random.Random(seed)
        doc_length = rng.choice(doc_lengths)
        # Scale distractors with document size: ~1 entry per 60 chars of
        # structured content, leaving room for filler. Aim for hundreds
        # to low thousands of entries.
        n_distractors = max(200, doc_length // 80)
        gen = generators[i % len(generators)]
        task = gen(
            task_id=f"kv_retrieval_{i:03d}",
            doc_length=doc_length,
            n_distractors=n_distractors,
            seed=seed,
        )
        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_key_value(answer: str | None, task: KeyValueTask) -> dict:
    """Score a key-value retrieval answer.

    Returns a dict with 'score' (0.0-1.0) and 'match_type'.

    Scoring rules:
    - single_lookup: exact match of the field value (partial credit for
      close numeric matches).
    - multi_lookup: fraction of correctly retrieved values. Each entry
      scores 1.0 (exact), 0.5 (value present but not matched to correct
      ID), or 0.0.
    - aggregate: exact numeric match of the sum (partial credit for
      close values).
    """
    if answer is None:
        return {"score": 0.0, "match_type": "none"}

    answer = answer.strip()
    expected = task.expected_answer.strip()

    if task.task_type == "single_lookup":
        return _score_single(answer, expected)
    elif task.task_type == "multi_lookup":
        return _score_multi(answer, expected, task.target_entries)
    elif task.task_type == "aggregate":
        return _score_aggregate(answer, expected)
    else:
        return {"score": 0.0, "match_type": "unknown_task_type"}


def _score_single(answer: str, expected: str) -> dict:
    """Score a single-lookup answer."""
    # Exact match (case-insensitive)
    if answer.lower() == expected.lower():
        return {"score": 1.0, "match_type": "exact"}

    # Check if expected is contained in answer (model may include extra text)
    if expected.lower() in answer.lower():
        return {"score": 1.0, "match_type": "contains"}

    # Numeric close match
    try:
        ans_num = float(answer.replace(",", ""))
        exp_num = float(expected.replace(",", ""))
        if abs(ans_num - exp_num) < 0.01:
            return {"score": 1.0, "match_type": "numeric_exact"}
        if exp_num != 0:
            ratio = abs(ans_num - exp_num) / abs(exp_num)
            if ratio <= 0.01:
                return {"score": 0.9, "match_type": "numeric_close"}
            if ratio <= 0.05:
                return {"score": 0.5, "match_type": "numeric_approx"}
        return {"score": 0.0, "match_type": "wrong_value"}
    except ValueError:
        pass

    return {"score": 0.0, "match_type": "wrong"}


def _score_multi(answer: str, expected: str, targets: list[dict]) -> dict:
    """Score a multi-lookup answer.

    Expected format: 'ID1: val1, ID2: val2, ...'
    """
    # Parse expected pairs
    expected_pairs: dict[str, str] = {}
    for pair in expected.split(","):
        if ":" in pair:
            key, val = pair.split(":", 1)
            expected_pairs[key.strip()] = val.strip()

    if not expected_pairs:
        return {"score": 0.0, "match_type": "parse_error"}

    # Parse answer pairs (be lenient with formatting)
    answer_pairs: dict[str, str] = {}
    # Try comma-separated first
    for pair in answer.replace("\n", ",").split(","):
        if ":" in pair:
            key, val = pair.split(":", 1)
            answer_pairs[key.strip()] = val.strip()

    n_expected = len(expected_pairs)
    if not answer_pairs:
        # Check if any expected values appear anywhere in the answer
        found = sum(1 for v in expected_pairs.values() if v in answer)
        if found > 0:
            return {
                "score": found / n_expected * 0.5,  # Half credit — no ID match
                "match_type": "values_only",
            }
        return {"score": 0.0, "match_type": "no_pairs_found"}

    # Score each expected entry
    entry_scores: list[float] = []
    for eid, eval_ in expected_pairs.items():
        if eid in answer_pairs:
            # Exact value match
            if answer_pairs[eid] == eval_:
                entry_scores.append(1.0)
            else:
                # Try numeric comparison
                try:
                    a_num = float(answer_pairs[eid].replace(",", ""))
                    e_num = float(eval_.replace(",", ""))
                    if abs(a_num - e_num) < 0.01:
                        entry_scores.append(1.0)
                    elif e_num != 0 and abs(a_num - e_num) / abs(e_num) <= 0.01:
                        entry_scores.append(0.8)
                    else:
                        entry_scores.append(0.0)
                except ValueError:
                    entry_scores.append(0.0)
        else:
            # Check if the value appears anywhere (partial credit)
            if eval_ in answer:
                entry_scores.append(0.3)
            else:
                entry_scores.append(0.0)

    avg_score = sum(entry_scores) / n_expected
    match_type = "exact" if avg_score == 1.0 else "partial"
    return {"score": avg_score, "match_type": match_type}


def _score_aggregate(answer: str, expected: str) -> dict:
    """Score an aggregate (sum) answer."""
    # Try to parse both as floats
    try:
        exp_num = float(expected.replace(",", ""))
    except ValueError:
        return {"score": 0.0, "match_type": "expected_parse_error"}

    # Extract a number from the answer (model might include extra text)
    ans_num = _extract_number(answer)
    if ans_num is None:
        return {"score": 0.0, "match_type": "no_number_found"}

    # Exact match (within floating point tolerance)
    if abs(ans_num - exp_num) < 0.01:
        return {"score": 1.0, "match_type": "exact"}

    # Proportional closeness
    if exp_num != 0:
        ratio = abs(ans_num - exp_num) / abs(exp_num)
        if ratio <= 0.005:
            return {"score": 1.0, "match_type": "numeric_exact"}
        if ratio <= 0.02:
            return {"score": 0.8, "match_type": "very_close"}
        if ratio <= 0.05:
            return {"score": 0.5, "match_type": "close"}
        if ratio <= 0.10:
            return {"score": 0.3, "match_type": "approximate"}
        if ratio <= 0.20:
            return {"score": 0.1, "match_type": "rough"}

    return {"score": 0.0, "match_type": "wrong"}


def _extract_number(text: str) -> float | None:
    """Extract the first plausible numeric value from text."""
    import re
    # Match numbers like 12345.67 or 12,345.67
    matches = re.findall(r"-?[\d,]+\.?\d*", text.replace(" ", ""))
    for m in matches:
        try:
            return float(m.replace(",", ""))
        except ValueError:
            continue
    return None
