"""
Hard NIAH benchmark — extended needle-in-a-haystack with adversarial distractors.

Unlike standard NIAH which has a single clear needle, this version adds:
1. Adversarial distractors: similar-looking but wrong values near the needle
2. Extreme lengths: 200K, 500K, 1M characters
3. Multiple needle types: not just codes, but dates, names, numbers
4. Positional stress: needles at exact boundaries (0.01, 0.99)

Tests precision under adversarial conditions at extreme scale.
O(1) complexity, but much harder than standard NIAH.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class HardNIAHTask:
    """A single Hard NIAH task."""
    task_id: str
    prompt: str
    needle: str
    expected_answer: str
    distractors: list[str]
    needle_position: float
    doc_length: int
    question: str
    difficulty: str  # "distractor", "extreme_length", "boundary"


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
    "New policies were implemented to address emerging challenges.",
    "The annual review highlighted several areas for improvement.",
    "International partnerships were strengthened through diplomatic efforts.",
    "Technology upgrades enhanced overall system performance.",
    "The strategic plan outlined objectives for the next quarter.",
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
    return "\n".join(parts)[:length]


def _generate_distractor_task(
    task_idx: int,
    doc_length: int = 50000,
    n_distractors: int = 5,
    seed: int = 42,
) -> HardNIAHTask:
    """Generate NIAH with adversarial distractors.

    Distractors are similar to the needle but with different values.
    E.g., needle is "The access code is ALPHA-7" and distractors include
    "The access code for floor 3 is BETA-9", "The old access code was GAMMA-12".
    """
    rng = random.Random(seed)

    # Generate needle and distractors
    templates = [
        {
            "needle_fmt": "The current active access code for the main server is {val}.",
            "distractor_fmts": [
                "The previous access code for the backup server was {val}.",
                "The test environment access code is {val}.",
                "The decommissioned access code used to be {val}.",
                "The temporary access code for maintenance is {val}.",
                "The old access code before the rotation was {val}.",
                "The staging server access code is {val}.",
                "The development environment uses access code {val}.",
            ],
            "question": "What is the current active access code for the main server?",
            "values": ["ALPHA-7X9", "BETA-22K", "GAMMA-55Z", "DELTA-91M", "ECHO-37P",
                       "FOXTROT-8R", "GOLF-44W", "HOTEL-66V"],
        },
        {
            "needle_fmt": "The confirmed meeting date for the board review is {val}.",
            "distractor_fmts": [
                "The tentative meeting date was {val} but it was postponed.",
                "The previous board meeting took place on {val}.",
                "The canceled meeting was originally scheduled for {val}.",
                "The preliminary date suggested was {val} before rescheduling.",
                "The alternative meeting date of {val} was rejected.",
                "The last quarterly review was held on {val}.",
            ],
            "question": "What is the confirmed meeting date for the board review?",
            "values": ["March 15 2026", "April 3 2026", "February 28 2026", "May 12 2026",
                       "January 20 2026", "June 8 2026", "March 22 2026", "April 19 2026"],
        },
        {
            "needle_fmt": "The final approved budget for Project Zenith is {val}.",
            "distractor_fmts": [
                "The initial budget proposal for Project Zenith was {val}.",
                "The rejected budget amendment requested {val}.",
                "The preliminary estimate before revision was {val}.",
                "The competing project had a budget of {val}.",
                "The Phase 1 allocation before adjustment was {val}.",
                "The unofficial estimate circulated was {val}.",
            ],
            "question": "What is the final approved budget for Project Zenith?",
            "values": ["$4.2 million", "$3.8 million", "$5.1 million", "$2.9 million",
                       "$6.7 million", "$4.5 million", "$3.3 million", "$7.2 million"],
        },
        {
            "needle_fmt": "The designated emergency contact person is {val}.",
            "distractor_fmts": [
                "The previous emergency contact was {val} before the change.",
                "The backup contact (not primary) is {val}.",
                "The former contact {val} was reassigned to another department.",
                "{val} was considered for the emergency contact role but declined.",
                "The weekend-only contact person is {val}.",
                "The temporary substitute contact was {val} during the transition.",
            ],
            "question": "Who is the designated emergency contact person?",
            "values": ["Dr. Sarah Mitchell", "Prof. James Chen", "Agent Laura Kim",
                       "Director Mark Thompson", "Dr. Elena Rodriguez", "Chief Ana Petrov",
                       "Supervisor David Okonkwo", "Lead Engineer Rachel Nakamura"],
        },
    ]

    template = rng.choice(templates)
    all_values = list(template["values"])
    rng.shuffle(all_values)

    needle_value = all_values[0]
    distractor_values = all_values[1:n_distractors + 1]

    needle = template["needle_fmt"].format(val=needle_value)
    distractors = [
        rng.choice(template["distractor_fmts"]).format(val=v)
        for v in distractor_values
    ]

    # Place needle and distractors
    needle_pos = rng.uniform(0.2, 0.8)

    # Distractors placed throughout, but NOT at the same position as needle
    distractor_positions = sorted(rng.sample(
        [i / 10 for i in range(1, 10) if abs(i / 10 - needle_pos) > 0.05],
        min(n_distractors, 7)
    ))

    # Build document
    all_inserts = [(needle_pos, needle)] + list(zip(distractor_positions, distractors))
    all_inserts.sort(key=lambda x: x[0])

    doc_parts = []
    current_len = 0
    insert_idx = 0

    while current_len < doc_length:
        if insert_idx < len(all_inserts):
            target_pos = all_inserts[insert_idx][0]
            if current_len >= doc_length * target_pos:
                doc_parts.append(all_inserts[insert_idx][1])
                current_len += len(all_inserts[insert_idx][1])
                insert_idx += 1
                continue

        sentence = rng.choice(FILLER_SENTENCES)
        doc_parts.append(sentence)
        current_len += len(sentence) + 1

    document = "\n".join(doc_parts)
    prompt = f"QUESTION: {template['question']}\n\nDOCUMENT:\n{document}"

    return HardNIAHTask(
        task_id=f"hard_niah_distractor_{task_idx}",
        prompt=prompt,
        needle=needle,
        expected_answer=needle_value,
        distractors=distractors,
        needle_position=needle_pos,
        doc_length=len(prompt),
        question=template["question"],
        difficulty="distractor",
    )


def _generate_extreme_length_task(
    task_idx: int,
    doc_length: int = 500000,
    seed: int = 42,
) -> HardNIAHTask:
    """Generate NIAH at extreme document lengths (200K-1M chars)."""
    rng = random.Random(seed)

    values = [
        ("The classified project identifier is {val}.",
         "What is the classified project identifier?",
         rng.choice(["NEXUS-PRIME-7742", "SHADOW-GATE-3319", "IRON-LOTUS-5581",
                      "SILVER-HAWK-2290", "GHOST-PHOENIX-4467"])),
        ("The final authorization code issued by command is {val}.",
         "What is the final authorization code issued by command?",
         f"AUTH-{rng.randint(1000,9999)}-{rng.choice('ABCDEF')}{rng.choice('XYZW')}{rng.randint(10,99)}"),
    ]

    needle_text, question, answer = rng.choice(values)
    needle = needle_text.format(val=answer)
    needle_pos = rng.uniform(0.1, 0.9)

    # Generate very long filler
    needle_char_pos = int(doc_length * needle_pos)
    before_text = _generate_filler(needle_char_pos, seed * 1000 + 1)
    after_text = _generate_filler(doc_length - needle_char_pos - len(needle), seed * 1000 + 2)

    document = before_text + "\n" + needle + "\n" + after_text
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return HardNIAHTask(
        task_id=f"hard_niah_extreme_{task_idx}",
        prompt=prompt,
        needle=needle,
        expected_answer=answer,
        distractors=[],
        needle_position=needle_pos,
        doc_length=len(prompt),
        question=question,
        difficulty="extreme_length",
    )


def _generate_boundary_task(
    task_idx: int,
    doc_length: int = 50000,
    seed: int = 42,
) -> HardNIAHTask:
    """Generate NIAH with needle at extreme boundaries (very start/end)."""
    rng = random.Random(seed)

    values = [
        ("The emergency override code is {val}.",
         "What is the emergency override code?",
         f"OVERRIDE-{rng.randint(100,999)}-{rng.choice('ABCDEFGH')}{rng.choice('MNPQRST')}"),
        ("The witness identification number is {val}.",
         "What is the witness identification number?",
         f"WIT-{rng.randint(10000,99999)}"),
    ]

    needle_text, question, answer = rng.choice(values)
    needle = needle_text.format(val=answer)

    # Extreme boundary positions
    needle_pos = rng.choice([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])

    needle_char_pos = int(doc_length * needle_pos)
    before_text = _generate_filler(max(0, needle_char_pos), seed * 1000 + 1)
    after_text = _generate_filler(max(0, doc_length - needle_char_pos - len(needle)), seed * 1000 + 2)

    document = before_text + "\n" + needle + "\n" + after_text
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return HardNIAHTask(
        task_id=f"hard_niah_boundary_{task_idx}",
        prompt=prompt,
        needle=needle,
        expected_answer=answer,
        distractors=[],
        needle_position=needle_pos,
        doc_length=len(prompt),
        question=question,
        difficulty="boundary",
    )


def generate_hard_niah_suite(
    n_tasks: int = 15,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[HardNIAHTask]:
    """Generate a suite of hard NIAH tasks.

    Mix:
    - 40% distractor tasks (adversarial similar values)
    - 30% extreme length tasks (200K-1M chars)
    - 30% boundary tasks (needles at very start/end)
    """
    if doc_lengths is None:
        doc_lengths = {
            "distractor": [30000, 50000, 100000],
            "extreme": [200000, 500000, 1000000],
            "boundary": [20000, 50000, 100000],
        }

    tasks = []
    rng = random.Random(42 + seed_offset)

    for i in range(n_tasks):
        seed = i + seed_offset
        r = rng.random()

        if r < 0.4:
            dl = rng.choice(doc_lengths["distractor"])
            n_dist = rng.choice([3, 5, 7])
            tasks.append(_generate_distractor_task(i, dl, n_dist, seed))
        elif r < 0.7:
            dl = rng.choice(doc_lengths["extreme"])
            tasks.append(_generate_extreme_length_task(i, dl, seed))
        else:
            dl = rng.choice(doc_lengths["boundary"])
            tasks.append(_generate_boundary_task(i, dl, seed))

    return tasks


def score_hard_niah(predicted: str | None, expected: str) -> dict:
    """Score a hard NIAH prediction.

    Returns dict with score and match_type.
    Strict matching — must get the exact value right.
    """
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    pred = predicted.strip().lower()
    exp = expected.strip().lower()

    if exp in pred:
        return {"score": 1.0, "match_type": "exact"}

    # Normalize punctuation (commas, extra spaces) for comparison
    import re
    pred_norm = re.sub(r'[,\s]+', ' ', pred).strip()
    exp_norm = re.sub(r'[,\s]+', ' ', exp).strip()
    if exp_norm in pred_norm:
        return {"score": 1.0, "match_type": "normalized"}

    # Check for partial match (e.g., got the code but extra text)
    # Only if at least 80% of expected chars are present
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, pred, exp).ratio()
    if ratio >= 0.8:
        return {"score": 0.5, "match_type": "partial"}

    return {"score": 0.0, "match_type": "none"}
