"""
Event Counting benchmark — teaches extract-then-count-in-Python strategy.

Addresses the core OOLONG failure: models delegate counting to sub-models
which hallucinate counts. The correct strategy is:
1. Use sub-model to EXTRACT matching items (return verbatim lines)
2. Count/aggregate in Python code

This benchmark generates long documents (50K-200K) with embedded structured
events in free-text narrative, then asks counting/aggregation questions.

Task types:
1. count_value: "How many events with value X?" (filter + count)
2. count_entity: "How many events by entity Y?" (entity filter + count)
3. first_last: "What was the first/last event of type Z?" (ordered retrieval)
4. per_entity: "How many events per entity?" (group-by aggregation)
5. ratio: "What percentage of events were type X?" (two counts + arithmetic)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class EventCountingTask:
    """A single event counting task."""
    task_id: str
    prompt: str
    expected_answer: str
    task_type: str  # count_value, count_entity, first_last, per_entity, ratio
    doc_length: int
    n_events: int
    question: str
    events: list[dict] = field(default_factory=list)  # For debugging


# Entity names
ENTITY_NAMES = [
    "Alice Chen", "Bob Martinez", "Carol Williams", "David Johnson",
    "Emma Davis", "Frank Miller", "Grace Lee", "Henry Wilson",
    "Iris Thompson", "Jack Anderson", "Karen Brown", "Leo Garcia",
    "Maria Rodriguez", "Nathan Taylor", "Olivia White", "Patrick Harris",
]

# Event types
EVENT_TYPES = [
    "login", "logout", "upload", "download", "purchase",
    "refund", "comment", "share", "bookmark", "delete",
    "create", "update", "approve", "reject", "submit",
]

# Values (for count_value tasks)
EVENT_VALUES = list(range(1, 21))  # Values 1-20

# Filler text for narrative padding
FILLER_SENTENCES = [
    "The system continued to operate within normal parameters during this period.",
    "Routine maintenance was scheduled but postponed due to higher priorities.",
    "Several team members discussed the quarterly objectives in an informal meeting.",
    "The database indexes were rebuilt overnight to improve query performance.",
    "New monitoring alerts were configured for the production environment.",
    "The security team completed their monthly review of access policies.",
    "Documentation updates were requested for the latest API changes.",
    "A brief outage occurred in the staging environment but was quickly resolved.",
    "The deployment pipeline was updated to include additional validation steps.",
    "Performance metrics showed steady improvement across all service endpoints.",
    "The analytics dashboard was refreshed with the latest data transformation logic.",
    "Cross-functional teams aligned on the product roadmap for the next quarter.",
    "Load testing results confirmed the system can handle peak traffic volumes.",
    "The incident response procedures were reviewed and several improvements noted.",
    "A new feature flag was introduced to control the rollout of the beta version.",
    "The backup and recovery procedures were tested successfully on schedule.",
    "Customer feedback was aggregated from multiple channels for the weekly review.",
    "The CI/CD pipeline execution time was reduced by optimizing the build cache.",
    "Infrastructure costs were reviewed and several optimization opportunities found.",
    "The engineering team held a knowledge sharing session on distributed systems.",
]


def _generate_event_line(entity: str, event_type: str, value: int, event_id: int,
                         rng: random.Random | None = None) -> str:
    """Generate a single event line embedded in narrative text."""
    templates = [
        f"[EVENT #{event_id}] User: {entity} | Action: {event_type} | Value: {value}",
        f"At this point, {entity} performed a {event_type} action (event #{event_id}, value={value}).",
        f"The log shows event #{event_id}: {entity} executed {event_type} with value {value}.",
        f"Event #{event_id} recorded: {entity} — {event_type} — value {value}.",
    ]
    if rng is not None:
        return rng.choice(templates)
    return random.choice(templates)


def _generate_filler_block(target_chars: int, rng: random.Random | None = None) -> str:
    """Generate filler text of approximately target_chars length."""
    lines = []
    total = 0
    while total < target_chars:
        if rng is not None:
            line = rng.choice(FILLER_SENTENCES)
        else:
            line = random.choice(FILLER_SENTENCES)
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def generate_event_document(
    n_events: int,
    doc_length: int,
    rng: random.Random,
    entities: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """Generate a document with embedded events.

    Returns (document_text, list_of_events).
    Each event dict has: entity, event_type, value, position, event_id
    """
    if entities is None:
        n_entities = rng.randint(4, 8)
        entities = rng.sample(ENTITY_NAMES, n_entities)

    # Generate events with random attributes
    events = []
    for i in range(n_events):
        event = {
            "entity": rng.choice(entities),
            "event_type": rng.choice(EVENT_TYPES),
            "value": rng.choice(EVENT_VALUES),
            "event_id": i + 1,
            "position": rng.random(),  # 0-1 relative position in doc
        }
        events.append(event)

    # Sort by position for deterministic placement
    events.sort(key=lambda e: e["position"])

    # Build document: filler text with events embedded at positions
    chars_per_event = doc_length // (n_events + 1)
    parts = []
    for i, event in enumerate(events):
        # Filler before this event
        filler_chars = max(100, int(chars_per_event * (0.8 + 0.4 * rng.random())))
        parts.append(_generate_filler_block(filler_chars, rng=rng))
        # The event line
        event_line = _generate_event_line(
            event["entity"], event["event_type"], event["value"], event["event_id"],
            rng=rng,
        )
        parts.append(event_line)

    # Final filler
    remaining = doc_length - sum(len(p) for p in parts)
    if remaining > 100:
        parts.append(_generate_filler_block(remaining, rng=rng))

    document = "\n".join(parts)
    return document, events


def generate_count_value_task(
    task_id: str, doc_length: int, n_events: int, seed: int
) -> EventCountingTask:
    """Count events with a specific value."""
    rng = random.Random(seed)
    document, events = generate_event_document(n_events, doc_length, rng)

    # Pick a target value that appears at least once
    value_counts = {}
    for e in events:
        value_counts[e["value"]] = value_counts.get(e["value"], 0) + 1

    # Pick a value (prefer ones with low count for harder task)
    target_value = rng.choice(list(value_counts.keys()))
    expected_count = value_counts[target_value]

    question = f"How many events in this log have value {target_value}? Return ONLY the count as a number."
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return EventCountingTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=str(expected_count),
        task_type="count_value",
        doc_length=len(document),
        n_events=n_events,
        question=question,
        events=events,
    )


def generate_count_entity_task(
    task_id: str, doc_length: int, n_events: int, seed: int
) -> EventCountingTask:
    """Count events by a specific entity."""
    rng = random.Random(seed)
    entities = rng.sample(ENTITY_NAMES, rng.randint(4, 8))
    document, events = generate_event_document(n_events, doc_length, rng, entities)

    entity_counts = {}
    for e in events:
        entity_counts[e["entity"]] = entity_counts.get(e["entity"], 0) + 1

    target_entity = rng.choice(list(entity_counts.keys()))
    expected_count = entity_counts[target_entity]

    question = f"How many events were performed by {target_entity}? Return ONLY the count as a number."
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return EventCountingTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=str(expected_count),
        task_type="count_entity",
        doc_length=len(document),
        n_events=n_events,
        question=question,
        events=events,
    )


def generate_first_last_task(
    task_id: str, doc_length: int, n_events: int, seed: int
) -> EventCountingTask:
    """Find the first or last event of a specific type."""
    rng = random.Random(seed)
    document, events = generate_event_document(n_events, doc_length, rng)

    # Group by event type
    type_events = {}
    for e in events:
        if e["event_type"] not in type_events:
            type_events[e["event_type"]] = []
        type_events[e["event_type"]].append(e)

    target_type = rng.choice(list(type_events.keys()))
    is_first = rng.random() < 0.5
    target_events = type_events[target_type]

    if is_first:
        target_event = target_events[0]  # Already sorted by position
        question = f"Who performed the first '{target_type}' event in this log? Return ONLY the person's name."
    else:
        target_event = target_events[-1]
        question = f"Who performed the last '{target_type}' event in this log? Return ONLY the person's name."

    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return EventCountingTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=target_event["entity"],
        task_type="first_last",
        doc_length=len(document),
        n_events=n_events,
        question=question,
        events=events,
    )


def generate_per_entity_task(
    task_id: str, doc_length: int, n_events: int, seed: int
) -> EventCountingTask:
    """Count events per entity (group-by aggregation)."""
    rng = random.Random(seed)
    entities = rng.sample(ENTITY_NAMES, rng.randint(3, 5))  # Fewer entities for tractable answer
    document, events = generate_event_document(n_events, doc_length, rng, entities)

    entity_counts = {}
    for e in events:
        entity_counts[e["entity"]] = entity_counts.get(e["entity"], 0) + 1

    # Format: "Alice: 5, Bob: 3, Carol: 7"
    expected = ", ".join(
        f"{name}: {count}" for name, count in sorted(entity_counts.items())
    )

    question = "How many events did each person perform? Return as 'Name: count' pairs separated by commas, sorted alphabetically by name."
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return EventCountingTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=expected,
        task_type="per_entity",
        doc_length=len(document),
        n_events=n_events,
        question=question,
        events=events,
    )


def generate_ratio_task(
    task_id: str, doc_length: int, n_events: int, seed: int
) -> EventCountingTask:
    """Compute percentage of events of a specific type."""
    rng = random.Random(seed)
    document, events = generate_event_document(n_events, doc_length, rng)

    type_counts = {}
    for e in events:
        type_counts[e["event_type"]] = type_counts.get(e["event_type"], 0) + 1

    target_type = rng.choice(list(type_counts.keys()))
    percentage = round(100 * type_counts[target_type] / len(events))

    question = f"What percentage of events are '{target_type}' events? Round to the nearest integer. Return ONLY the number (no % sign)."
    prompt = f"QUESTION: {question}\n\nDOCUMENT:\n{document}"

    return EventCountingTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=str(percentage),
        task_type="ratio",
        doc_length=len(document),
        n_events=n_events,
        question=question,
        events=events,
    )


def generate_event_counting_suite(
    n_tasks: int = 10,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[EventCountingTask]:
    """Generate a suite of event counting tasks.

    Default: 50K-200K char documents with 30-100 events.
    """
    if doc_lengths is None:
        doc_lengths = [50000, 100000, 150000, 200000]

    generators = [
        generate_count_value_task,
        generate_count_entity_task,
        generate_first_last_task,
        generate_per_entity_task,
        generate_ratio_task,
    ]

    tasks = []
    for i in range(n_tasks):
        seed = seed_offset + i * 7
        rng = random.Random(seed)
        doc_length = rng.choice(doc_lengths)
        n_events = max(20, doc_length // 3000)  # ~1 event per 3K chars
        gen = generators[i % len(generators)]
        task = gen(
            task_id=f"event_count_{i:03d}",
            doc_length=doc_length,
            n_events=n_events,
            seed=seed,
        )
        tasks.append(task)

    return tasks


def score_event_counting(answer: str | None, expected: str) -> dict:
    """Score an event counting answer.

    For count tasks: exact match of the number.
    For entity tasks: exact match of the name.
    For per_entity: fraction of correct entity counts.
    """
    if answer is None:
        return {"score": 0.0, "match_type": "none"}

    answer = answer.strip()
    expected = expected.strip()

    # Try exact match first
    if answer.lower() == expected.lower():
        return {"score": 1.0, "match_type": "exact"}

    # For numeric answers: try parsing
    try:
        answer_num = int(answer)
        expected_num = int(expected)
        if answer_num == expected_num:
            return {"score": 1.0, "match_type": "numeric"}
        # Granular partial credit based on closeness
        if expected_num > 0:
            ratio = abs(answer_num - expected_num) / expected_num
            if ratio <= 0.1:
                return {"score": 0.7, "match_type": "close"}
            elif ratio <= 0.2:
                return {"score": 0.5, "match_type": "close"}
            elif ratio <= 0.5:
                return {"score": 0.2, "match_type": "approximate"}
        return {"score": 0.0, "match_type": "wrong_number"}
    except ValueError:
        pass

    # For entity name answers: substring match
    if expected.lower() in answer.lower():
        return {"score": 1.0, "match_type": "contains"}

    # For per_entity answers: parse and compare counts
    if ":" in expected and ":" in answer:
        expected_pairs = {}
        for pair in expected.split(","):
            if ":" in pair:
                name, count = pair.rsplit(":", 1)
                expected_pairs[name.strip().lower()] = int(count.strip())

        answer_pairs = {}
        for pair in answer.split(","):
            if ":" in pair:
                name, count = pair.rsplit(":", 1)
                try:
                    answer_pairs[name.strip().lower()] = int(count.strip())
                except ValueError:
                    pass

        if expected_pairs and answer_pairs:
            n_entities = len(expected_pairs)
            # Entity identification: did the model find the right entities?
            matched_entities = sum(
                1 for name in expected_pairs if name in answer_pairs
            )
            entity_score = matched_entities / n_entities  # 0-1

            # Count accuracy: for matched entities, how close are the counts?
            count_scores = []
            for name, expected_count in expected_pairs.items():
                if name in answer_pairs:
                    answer_count = answer_pairs[name]
                    if answer_count == expected_count:
                        count_scores.append(1.0)
                    elif expected_count > 0:
                        ratio = abs(answer_count - expected_count) / expected_count
                        if ratio <= 0.1:
                            count_scores.append(0.8)
                        elif ratio <= 0.2:
                            count_scores.append(0.5)
                        elif ratio <= 0.5:
                            count_scores.append(0.2)
                        else:
                            count_scores.append(0.0)
                    else:
                        count_scores.append(0.0 if answer_count != 0 else 1.0)
                else:
                    count_scores.append(0.0)

            avg_count_score = sum(count_scores) / n_entities if count_scores else 0
            # 40% entity identification + 60% count accuracy
            final_score = 0.4 * entity_score + 0.6 * avg_count_score
            return {"score": final_score, "match_type": "partial_entity"}

    return {"score": 0.0, "match_type": "none"}
