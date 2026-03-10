"""
Multi-Hop QA benchmark for RLMs.

Generates tasks where answering requires chaining 2-3 facts from different
parts of a long document. Tests recursive reasoning, not just search.

Example: "What department does the person who won the Innovation Award work in?"
- Fact 1 (at position 0.2): "Alice Johnson won the Innovation Award"
- Fact 2 (at position 0.7): "Alice Johnson works in the R&D department"
- Answer: "R&D department"

O(K) complexity: must find K related facts scattered across the document.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MultiHopTask:
    """A single Multi-Hop QA task."""
    task_id: str
    prompt: str
    expected_answer: str
    n_hops: int
    facts: list[dict]  # List of {text, position, role}
    doc_length: int
    question: str


# Person names
FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
    "Henry", "Iris", "Jack", "Karen", "Leo", "Maria", "Nathan",
    "Olivia", "Patrick", "Quinn", "Rachel", "Samuel", "Tina",
]

LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Garcia", "Martinez", "Anderson",
    "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson",
    "Thompson", "White", "Lopez", "Lee", "Clark", "Robinson",
    "Walker", "Young",
]

# Departments
DEPARTMENTS = [
    "Research and Development", "Marketing", "Finance", "Operations",
    "Human Resources", "Engineering", "Quality Assurance", "Sales",
    "Product Management", "Data Science", "Customer Success", "Legal",
]

# Awards/achievements
AWARDS = [
    "Innovation Award", "Excellence in Leadership Award",
    "Outstanding Contributor Award", "Rising Star Award",
    "Best Team Player Award", "Technical Achievement Award",
    "Customer Impact Award", "Sustainability Champion Award",
]

# Projects
PROJECTS = [
    "Project Phoenix", "Project Horizon", "Project Catalyst",
    "Project Nexus", "Project Meridian", "Project Vertex",
    "Project Prism", "Project Forge", "Project Atlas", "Project Helix",
]

# Cities
CITIES = [
    "San Francisco", "New York", "London", "Tokyo", "Berlin",
    "Singapore", "Toronto", "Sydney", "Amsterdam", "Seoul",
]

# Filler sentences (same style as NIAH)
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


def generate_2hop_task(
    task_idx: int,
    doc_length: int = 20000,
    seed: int = 42,
) -> MultiHopTask:
    """Generate a 2-hop QA task.

    Pattern: Person → Attribute → Question about linked attribute
    Example: "Who won X award?" + "What department does [winner] work in?"
    Combined: "What department does the winner of X award work in?"
    """
    rng = random.Random(seed)

    person = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    department = rng.choice(DEPARTMENTS)
    award = rng.choice(AWARDS)
    project = rng.choice(PROJECTS)
    city = rng.choice(CITIES)

    # Choose a question template
    templates = [
        {
            "fact1": f"{person} received the {award} at the annual ceremony.",
            "fact2": f"{person} has been leading the team in the {department} department.",
            "question": f"What department does the recipient of the {award} work in?",
            "answer": department,
        },
        {
            "fact1": f"The {project} initiative was spearheaded by {person}.",
            "fact2": f"{person} is based in the {city} office.",
            "question": f"Which city is the leader of {project} based in?",
            "answer": city,
        },
        {
            "fact1": f"{person} was appointed as the lead researcher for {project}.",
            "fact2": f"{project} focuses on developing next-generation sustainability solutions.",
            "question": f"What does the project led by {person} focus on?",
            "answer": "next-generation sustainability solutions",
        },
        {
            "fact1": f"The {department} department is headed by {person}.",
            "fact2": f"{person} previously worked at the {city} branch for 5 years.",
            "question": f"Where did the head of {department} previously work?",
            "answer": f"{city} branch",
        },
        {
            "fact1": f"{person} won the {award} for their work on {project}.",
            "fact2": f"{project} was completed under budget and ahead of schedule.",
            "question": f"Was the project that earned {person} the {award} completed on time?",
            "answer": "ahead of schedule",
        },
    ]

    template = rng.choice(templates)

    # Place facts at different positions in the document
    pos1 = rng.uniform(0.1, 0.4)
    pos2 = rng.uniform(0.6, 0.9)

    # Build document with filler
    doc_parts = []
    current_len = 0
    fact1_inserted = False
    fact2_inserted = False

    while current_len < doc_length:
        # Check if we should insert fact 1
        if not fact1_inserted and current_len >= doc_length * pos1:
            doc_parts.append(template["fact1"])
            current_len += len(template["fact1"])
            fact1_inserted = True
            continue

        # Check if we should insert fact 2
        if not fact2_inserted and current_len >= doc_length * pos2:
            doc_parts.append(template["fact2"])
            current_len += len(template["fact2"])
            fact2_inserted = True
            continue

        # Add filler
        sentence = rng.choice(FILLER_SENTENCES)
        doc_parts.append(sentence)
        current_len += len(sentence) + 1  # +1 for newline

    document = "\n".join(doc_parts)

    prompt = f"QUESTION: {template['question']}\n\nDOCUMENT:\n{document}"

    return MultiHopTask(
        task_id=f"multihop_2hop_{task_idx}",
        prompt=prompt,
        expected_answer=template["answer"],
        n_hops=2,
        facts=[
            {"text": template["fact1"], "position": pos1, "role": "bridge_entity"},
            {"text": template["fact2"], "position": pos2, "role": "target_attribute"},
        ],
        doc_length=len(prompt),
        question=template["question"],
    )


def generate_3hop_task(
    task_idx: int,
    doc_length: int = 50000,
    seed: int = 42,
) -> MultiHopTask:
    """Generate a 3-hop QA task.

    Pattern: A → B → C → Answer
    Example: "Person won award" → "Award given for project" → "Project in city"
    Question: "In what city was the project that earned someone the award?"
    """
    rng = random.Random(seed)

    person = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    department = rng.choice(DEPARTMENTS)
    award = rng.choice(AWARDS)
    project = rng.choice(PROJECTS)
    city = rng.choice(CITIES)

    templates = [
        {
            "fact1": f"{person} was honored with the {award} this year.",
            "fact2": f"The {award} was given specifically for outstanding contributions to {project}.",
            "fact3": f"{project} is headquartered in {city} with a team of 50 engineers.",
            "question": f"In which city is the project for which {person} received an award headquartered?",
            "answer": city,
        },
        {
            "fact1": f"The {department} department recently completed a major milestone.",
            "fact2": f"This milestone was the successful delivery of {project}.",
            "fact3": f"{project} was funded with a budget of $4.2 million.",
            "question": f"What was the budget of the project completed by the {department} department?",
            "answer": "$4.2 million",
        },
        {
            "fact1": f"{person} manages the {city} regional office.",
            "fact2": f"The {city} office is the primary hub for {project}.",
            "fact3": f"{project} uses cutting-edge quantum computing technology.",
            "question": f"What technology does the project based at the office managed by {person} use?",
            "answer": "quantum computing",
        },
    ]

    template = rng.choice(templates)

    # Place facts at three different positions
    pos1 = rng.uniform(0.05, 0.25)
    pos2 = rng.uniform(0.35, 0.55)
    pos3 = rng.uniform(0.65, 0.90)

    # Build document with filler
    doc_parts = []
    current_len = 0
    facts_inserted = [False, False, False]
    fact_texts = [template["fact1"], template["fact2"], template["fact3"]]
    positions = [pos1, pos2, pos3]

    while current_len < doc_length:
        for i in range(3):
            if not facts_inserted[i] and current_len >= doc_length * positions[i]:
                doc_parts.append(fact_texts[i])
                current_len += len(fact_texts[i])
                facts_inserted[i] = True
                break
        else:
            sentence = rng.choice(FILLER_SENTENCES)
            doc_parts.append(sentence)
            current_len += len(sentence) + 1

    document = "\n".join(doc_parts)
    prompt = f"QUESTION: {template['question']}\n\nDOCUMENT:\n{document}"

    return MultiHopTask(
        task_id=f"multihop_3hop_{task_idx}",
        prompt=prompt,
        expected_answer=template["answer"],
        n_hops=3,
        facts=[
            {"text": fact_texts[i], "position": positions[i]} for i in range(3)
        ],
        doc_length=len(prompt),
        question=template["question"],
    )


def generate_multi_hop_suite(
    n_tasks: int = 20,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[MultiHopTask]:
    """Generate a suite of multi-hop QA tasks.

    Default: mix of 2-hop and 3-hop tasks at varying document lengths.
    """
    if doc_lengths is None:
        doc_lengths = [10000, 20000, 50000, 100000]

    tasks = []
    rng = random.Random(42 + seed_offset)

    for i in range(n_tasks):
        doc_len = rng.choice(doc_lengths)
        seed = i + seed_offset

        # 60% 2-hop, 40% 3-hop
        if rng.random() < 0.6:
            tasks.append(generate_2hop_task(i, doc_len, seed))
        else:
            tasks.append(generate_3hop_task(i, doc_len, seed))

    return tasks


def score_multi_hop(predicted: str | None, expected: str) -> dict:
    """Score a multi-hop QA prediction.

    Returns dict with:
    - score: 1.0 if expected answer found in prediction, 0.0 otherwise
    - partial: True if key terms overlap
    """
    if predicted is None:
        return {"score": 0.0, "partial": False}

    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()

    # Exact or contains match (prediction contains expected)
    if exp_lower in pred_lower:
        return {"score": 1.0, "partial": False}

    # Reverse containment (expected contains prediction — e.g. "Berlin" for "Berlin branch")
    if pred_lower in exp_lower and len(pred_lower) >= 3:
        return {"score": 0.5, "partial": True}

    # Check for key term overlap (partial credit)
    exp_words = set(exp_lower.split())
    pred_words = set(pred_lower.split())
    overlap = exp_words & pred_words
    if len(overlap) >= len(exp_words) * 0.7:
        return {"score": 0.5, "partial": True}

    return {"score": 0.0, "partial": False}
