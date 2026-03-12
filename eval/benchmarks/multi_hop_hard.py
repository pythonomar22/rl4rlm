"""
Hard Multi-Hop QA benchmark for RLMs — forces true multi-step decomposition.

Key differences from multi_hop_qa.py:
1. Documents are always 100K+ chars (facts CANNOT be in same chunk)
2. Questions require sequential reasoning (can't ask compound query)
3. Entity discovery: question mentions entity A, fact 1 links A→B, fact 2 uses B
   (you MUST find B from fact 1 before you can search for fact 2)
4. Distractor entities: similar-looking entities to prevent lucky guesses

Example hard task:
- Question: "What is the delivery date for the project led by the VP of Engineering?"
- Fact 1 (position 0.15): "Rachel Thomas serves as VP of Engineering"
- Fact 2 (position 0.75): "Rachel Thomas leads Project Phoenix"
- Fact 3 (position 0.55): "Project Phoenix has a delivery date of March 15, 2026"
- Distractors:
  - (0.30): "David Garcia serves as VP of Marketing"
  - (0.45): "David Garcia leads Project Nexus"
  - (0.85): "Project Nexus has a delivery date of June 22, 2026"

The model CANNOT just ask "what's the delivery date for the VP of Engineering's project?"
because no single chunk contains all three facts. It must:
1. Find who is VP of Engineering → Rachel Thomas
2. Find what project Rachel Thomas leads → Project Phoenix
3. Find the delivery date of Project Phoenix → March 15, 2026
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class HardMultiHopTask:
    """A hard multi-hop QA task requiring true decomposition."""
    task_id: str
    prompt: str
    expected_answer: str
    n_hops: int
    facts: list[dict]  # [{text, position, role}]
    distractors: list[dict]  # [{text, position}]
    doc_length: int
    question: str
    decomposition: list[str]  # Expected reasoning steps


# Person names (larger set to avoid collisions with distractors)
FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
    "Henry", "Iris", "Jack", "Karen", "Leo", "Maria", "Nathan",
    "Olivia", "Patrick", "Quinn", "Rachel", "Samuel", "Tina",
    "Victor", "Wendy", "Xavier", "Yara", "Zachary",
]

LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Garcia", "Martinez", "Anderson",
    "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson",
    "Thompson", "White", "Lopez", "Lee", "Clark", "Robinson",
    "Walker", "Young", "Hall", "Allen", "King", "Wright", "Scott",
]

DEPARTMENTS = [
    "Research and Development", "Marketing", "Finance", "Operations",
    "Human Resources", "Engineering", "Quality Assurance", "Sales",
    "Product Management", "Data Science", "Customer Success", "Legal",
]

TITLES = [
    "VP of", "Director of", "Head of", "Chief of",
    "Senior Manager of", "Lead of", "Principal of",
]

PROJECTS = [
    "Project Phoenix", "Project Horizon", "Project Catalyst",
    "Project Nexus", "Project Meridian", "Project Vertex",
    "Project Prism", "Project Forge", "Project Atlas", "Project Helix",
    "Project Orion", "Project Zenith", "Project Vanguard", "Project Summit",
]

CITIES = [
    "San Francisco", "New York", "London", "Tokyo", "Berlin",
    "Singapore", "Toronto", "Sydney", "Amsterdam", "Seoul",
    "Dublin", "Zurich", "Boston", "Chicago", "Austin",
]

BUDGETS = [
    "$2.5 million", "$4.2 million", "$7.8 million", "$12.3 million",
    "$1.9 million", "$6.4 million", "$15.7 million", "$3.1 million",
]

DATES = [
    "March 15, 2026", "June 22, 2026", "September 8, 2026",
    "January 10, 2027", "April 5, 2026", "November 30, 2026",
    "August 14, 2026", "December 1, 2026", "February 28, 2027",
]

TECHNOLOGIES = [
    "quantum computing", "machine learning", "blockchain",
    "edge computing", "natural language processing",
    "computer vision", "robotics", "IoT sensors",
]

# Filler sentences (same style as other benchmarks)
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


def _build_document(
    facts: list[tuple[str, float]],
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build a document with facts placed at specified positions."""
    doc_parts = []
    current_len = 0
    inserted = [False] * len(facts)

    # Sort facts by position for correct insertion order
    indexed_facts = sorted(enumerate(facts), key=lambda x: x[1][1])

    while current_len < doc_length:
        for orig_idx, (text, pos) in indexed_facts:
            if not inserted[orig_idx] and current_len >= doc_length * pos:
                doc_parts.append(text)
                current_len += len(text)
                inserted[orig_idx] = True
                break
        else:
            sentence = rng.choice(FILLER_SENTENCES)
            doc_parts.append(sentence)
            current_len += len(sentence) + 1

    return "\n".join(doc_parts)


def generate_hard_2hop_task(
    task_idx: int,
    doc_length: int = 100000,
    seed: int = 42,
) -> HardMultiHopTask:
    """Generate a hard 2-hop task with distractors.

    The question asks about entity A.
    Fact 1 links A → B (the bridge entity).
    Fact 2 provides the answer about B.
    Distractor: A different entity A' → B', with B' having a different answer.
    """
    rng = random.Random(seed)

    # Choose target person and distractor person
    names = rng.sample(list(zip(FIRST_NAMES, LAST_NAMES)), 4)
    person1 = f"{names[0][0]} {names[0][1]}"
    person2 = f"{names[1][0]} {names[1][1]}"  # distractor

    dept1 = rng.choice(DEPARTMENTS)
    dept2 = rng.choice([d for d in DEPARTMENTS if d != dept1])
    title = rng.choice(TITLES)

    project1 = rng.choice(PROJECTS)
    project2 = rng.choice([p for p in PROJECTS if p != project1])

    city1 = rng.choice(CITIES)
    city2 = rng.choice([c for c in CITIES if c != city1])

    budget1 = rng.choice(BUDGETS)
    budget2 = rng.choice([b for b in BUDGETS if b != budget1])

    templates = [
        {
            "fact1": f"{person1} serves as the {title} {dept1}.",
            "fact2": f"{person1} is based in the {city1} office and works remotely three days a week.",
            "dist1": f"{person2} serves as the {title} {dept2}.",
            "dist2": f"{person2} is based in the {city2} office and works remotely two days a week.",
            "question": f"In which city is the {title} {dept1} based?",
            "answer": city1,
            "decomposition": [
                f"Find who is the {title} {dept1} → {person1}",
                f"Find where {person1} is based → {city1}",
            ],
        },
        {
            "fact1": f"{person1} has been appointed to lead {project1} starting this quarter.",
            "fact2": f"{project1} has been allocated a budget of {budget1} for the fiscal year.",
            "dist1": f"{person2} has been appointed to lead {project2} starting this quarter.",
            "dist2": f"{project2} has been allocated a budget of {budget2} for the fiscal year.",
            "question": f"What is the budget for the project led by {person1}?",
            "answer": budget1,
            "decomposition": [
                f"Find what project {person1} leads → {project1}",
                f"Find the budget of {project1} → {budget1}",
            ],
        },
        {
            "fact1": f"The {dept1} department is responsible for {project1}.",
            "fact2": f"{project1} uses {rng.choice(TECHNOLOGIES)} as its core technology platform.",
            "dist1": f"The {dept2} department is responsible for {project2}.",
            "dist2": f"{project2} uses {rng.choice(TECHNOLOGIES)} as its core technology platform.",
            "question": f"What technology does the project managed by {dept1} use?",
            "answer": None,  # Will be set from fact2
            "decomposition": [
                f"Find what project {dept1} manages → {project1}",
                f"Find what technology {project1} uses",
            ],
        },
    ]

    template = rng.choice(templates[:2])  # Skip template 3 (answer set dynamically)

    # Place facts far apart with distractors in between
    fact_positions = [
        (template["fact1"], rng.uniform(0.08, 0.20)),
        (template["fact2"], rng.uniform(0.70, 0.88)),
    ]
    distractor_positions = [
        (template["dist1"], rng.uniform(0.30, 0.45)),
        (template["dist2"], rng.uniform(0.50, 0.65)),
    ]

    all_facts = fact_positions + distractor_positions
    document = _build_document(all_facts, doc_length, rng)
    prompt = f"QUESTION: {template['question']}\n\nDOCUMENT:\n{document}"

    return HardMultiHopTask(
        task_id=f"hard_multihop_2hop_{task_idx}",
        prompt=prompt,
        expected_answer=template["answer"],
        n_hops=2,
        facts=[{"text": t, "position": p, "role": "target"} for t, p in fact_positions],
        distractors=[{"text": t, "position": p} for t, p in distractor_positions],
        doc_length=len(prompt),
        question=template["question"],
        decomposition=template["decomposition"],
    )


def generate_hard_3hop_task(
    task_idx: int,
    doc_length: int = 150000,
    seed: int = 42,
) -> HardMultiHopTask:
    """Generate a hard 3-hop task with distractors.

    Chain: Question mentions A → Fact1 links A→B → Fact2 links B→C → Fact3 gives answer about C
    Plus distractor chain: A'→B'→C' with different answer.
    """
    rng = random.Random(seed)

    names = rng.sample(list(zip(FIRST_NAMES, LAST_NAMES)), 4)
    person1 = f"{names[0][0]} {names[0][1]}"
    person2 = f"{names[1][0]} {names[1][1]}"

    dept1, dept2 = rng.sample(DEPARTMENTS, 2)
    project1, project2 = rng.sample(PROJECTS, 2)
    city1, city2 = rng.sample(CITIES, 2)
    budget1, budget2 = rng.sample(BUDGETS, 2)
    date1, date2 = rng.sample(DATES, 2)
    title = rng.choice(TITLES)

    templates = [
        {
            # Question → Person → Project → Budget
            "fact1": f"{person1} serves as the {title} {dept1} and oversees all major initiatives.",
            "fact2": f"The {title} {dept1} is the executive sponsor of {project1}.",
            "fact3": f"{project1} received final budget approval of {budget1} from the board.",
            "dist1": f"{person2} serves as the {title} {dept2} and oversees all major initiatives.",
            "dist2": f"The {title} {dept2} is the executive sponsor of {project2}.",
            "dist3": f"{project2} received final budget approval of {budget2} from the board.",
            "question": f"What is the approved budget for the project sponsored by the {title} {dept1}?",
            "answer": budget1,
            "decomposition": [
                f"Find who is {title} {dept1} → {person1}",
                f"Find what project the {title} {dept1} sponsors → {project1}",
                f"Find the budget of {project1} → {budget1}",
            ],
        },
        {
            # Question → Department → Project → Date
            "fact1": f"The {dept1} department recently celebrated the successful completion of a major milestone.",
            "fact2": f"This milestone was the on-time delivery of {project1} after 18 months of development.",
            "fact3": f"{project1} was delivered on {date1} to overwhelming positive feedback.",
            "dist1": f"The {dept2} department recently celebrated the successful completion of a major milestone.",
            "dist2": f"This milestone was the on-time delivery of {project2} after 24 months of development.",
            "dist3": f"{project2} was delivered on {date2} to positive stakeholder feedback.",
            "question": f"On what date was the project completed by the {dept1} department delivered?",
            "answer": date1,
            "decomposition": [
                f"Find what milestone {dept1} completed → delivery of {project1}",
                f"Find what project was delivered → {project1}",
                f"Find the delivery date of {project1} → {date1}",
            ],
        },
        {
            # Question → Person → Project → City
            "fact1": f"{person1} was nominated as the lead architect for a critical company initiative.",
            "fact2": f"The initiative led by {person1} is officially called {project1}.",
            "fact3": f"{project1} operations are headquartered in {city1} with satellite offices worldwide.",
            "dist1": f"{person2} was nominated as the lead architect for a critical company initiative.",
            "dist2": f"The initiative led by {person2} is officially called {project2}.",
            "dist3": f"{project2} operations are headquartered in {city2} with satellite offices worldwide.",
            "question": f"Where is the project led by {person1} headquartered?",
            "answer": city1,
            "decomposition": [
                f"Find what initiative {person1} leads → {project1}",
                f"Find where {project1} is headquartered → {city1}",
            ],
        },
    ]

    template = rng.choice(templates)

    # Place target facts and distractor facts with maximum separation
    fact_positions = [
        (template["fact1"], rng.uniform(0.05, 0.15)),
        (template["fact2"], rng.uniform(0.40, 0.50)),
        (template["fact3"], rng.uniform(0.75, 0.90)),
    ]
    distractor_positions = [
        (template["dist1"], rng.uniform(0.20, 0.30)),
        (template["dist2"], rng.uniform(0.55, 0.65)),
        (template["dist3"], rng.uniform(0.92, 0.97)),
    ]

    all_facts = fact_positions + distractor_positions
    document = _build_document(all_facts, doc_length, rng)
    prompt = f"QUESTION: {template['question']}\n\nDOCUMENT:\n{document}"

    return HardMultiHopTask(
        task_id=f"hard_multihop_3hop_{task_idx}",
        prompt=prompt,
        expected_answer=template["answer"],
        n_hops=3,
        facts=[{"text": t, "position": p, "role": "target"} for t, p in fact_positions],
        distractors=[{"text": t, "position": p} for t, p in distractor_positions],
        doc_length=len(prompt),
        question=template["question"],
        decomposition=template["decomposition"],
    )


def generate_hard_multi_hop_suite(
    n_tasks: int = 10,
    doc_lengths_2hop: list[int] | None = None,
    doc_lengths_3hop: list[int] | None = None,
    seed_offset: int = 0,
) -> list[HardMultiHopTask]:
    """Generate a suite of hard multi-hop tasks.

    Default: 50% 2-hop (100K chars), 50% 3-hop (150K chars).
    All tasks have distractor chains to prevent lucky guesses.
    """
    if doc_lengths_2hop is None:
        doc_lengths_2hop = [100000, 150000]
    if doc_lengths_3hop is None:
        doc_lengths_3hop = [150000, 200000]

    tasks = []
    rng = random.Random(42 + seed_offset)

    for i in range(n_tasks):
        seed = i * 7 + seed_offset

        if rng.random() < 0.5:
            doc_len = rng.choice(doc_lengths_2hop)
            tasks.append(generate_hard_2hop_task(i, doc_len, seed))
        else:
            doc_len = rng.choice(doc_lengths_3hop)
            tasks.append(generate_hard_3hop_task(i, doc_len, seed))

    return tasks


def score_hard_multi_hop(predicted: str | None, expected: str) -> dict:
    """Score a hard multi-hop QA prediction."""
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    # Exact match
    if exp_lower in pred_lower:
        return {"score": 1.0, "match_type": "contains"}

    # Reverse containment (e.g., "$4.2 million" for "$4.2")
    if pred_lower in exp_lower and len(pred_lower) >= 3:
        return {"score": 0.5, "match_type": "partial"}

    # Numeric comparison for budgets
    import re
    pred_nums = re.findall(r'[\d,.]+', predicted)
    exp_nums = re.findall(r'[\d,.]+', expected)
    if pred_nums and exp_nums:
        try:
            pred_val = float(pred_nums[0].replace(',', ''))
            exp_val = float(exp_nums[0].replace(',', ''))
            if exp_val != 0 and abs(pred_val - exp_val) / exp_val < 0.01:
                return {"score": 1.0, "match_type": "numeric"}
        except ValueError:
            pass

    # Date comparison (normalize format)
    pred_clean = re.sub(r'[,\s]+', ' ', predicted).strip().lower()
    exp_clean = re.sub(r'[,\s]+', ' ', expected).strip().lower()
    if exp_clean in pred_clean:
        return {"score": 1.0, "match_type": "normalized"}

    # ISO date format comparison (e.g., "2026-09-08" vs "September 8, 2026")
    from datetime import datetime
    date_formats = [
        "%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
        "%m/%d/%Y", "%d %B %Y", "%d %b %Y",
    ]
    def parse_date(s):
        s = re.sub(r'[,\s]+', ' ', s).strip()
        for fmt in date_formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    pred_date = parse_date(predicted)
    exp_date = parse_date(expected)
    if pred_date and exp_date and pred_date == exp_date:
        return {"score": 1.0, "match_type": "date_normalized"}

    return {"score": 0.0, "match_type": "none"}
