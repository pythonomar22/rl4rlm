"""
Verbatim Copy benchmark for RLMs.

Tests whether an RLM can faithfully reproduce a specific paragraph from a long
document. This is critical for practical applications (legal, medical, compliance)
where exact reproduction matters — not just gist or summary.

O(1) complexity: one target paragraph must be located and copied exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class VerbatimCopyTask:
    """A single Verbatim Copy task."""
    task_id: str
    prompt: str                    # Full text (instruction + document)
    expected_answer: str           # The target paragraph to reproduce
    doc_length: int                # Total document length in chars
    target_paragraph_position: float  # 0.0-1.0 relative position
    target_paragraph_length: int   # Length of the target paragraph in chars


# Filler sentences (same style as NIAH / multi_hop_qa)
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

# --- Target paragraph templates ---
# Each is a callable that takes an RNG and returns (paragraph_text, distinctive_phrase).
# The distinctive_phrase is what the prompt will reference so the model knows which
# paragraph to copy.

PERSON_FIRST = [
    "Eleanor", "Marcus", "Priya", "Tobias", "Ingrid", "Rafael",
    "Yuki", "Desmond", "Fiona", "Henrik", "Amara", "Lucien",
]

PERSON_LAST = [
    "Whitfield", "Okonkwo", "Chandra", "Sorensen", "Mbeki", "Petrov",
    "Nakamura", "Abernathy", "Kowalski", "Delgado", "Lindqvist", "Ashworth",
]


def _memo_paragraph(rng: random.Random) -> tuple[str, str]:
    """Generate a business memorandum paragraph."""
    sender = f"{rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    recipient = f"{rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    date = f"{rng.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} {rng.randint(1, 28)}, {rng.choice([2024, 2025, 2026])}"
    ref_number = f"MEM-{rng.randint(1000, 9999)}-{rng.choice(['A', 'B', 'C', 'D', 'X', 'Z'])}"
    amount = f"${rng.randint(50, 999)},{rng.randint(100, 999)}.{rng.randint(10, 99)}"
    division = rng.choice([
        "Western Regional Division", "Eastern Operations Group",
        "Northern Analytics Unit", "Southern Field Office",
        "Central Logistics Hub", "Pacific Compliance Bureau",
    ])

    text = (
        f"MEMORANDUM — Reference {ref_number}. From: {sender}. To: {recipient}. "
        f"Date: {date}. Subject: Budget Reallocation for {division}. "
        f"Following the review conducted on {date}, it has been determined that "
        f"an amount of {amount} shall be transferred from the general operating "
        f"fund to the {division} effective immediately. All department heads are "
        f"required to acknowledge receipt of this memorandum and confirm compliance "
        f"with the updated allocation schedule within five business days. Failure "
        f"to comply will result in a mandatory audit of departmental expenditures."
    )
    return text, ref_number


def _report_paragraph(rng: random.Random) -> tuple[str, str]:
    """Generate a quarterly report paragraph."""
    quarter = rng.choice(["Q1", "Q2", "Q3", "Q4"])
    year = rng.choice([2024, 2025, 2026])
    report_id = f"QR-{year}-{quarter}-{rng.randint(100, 999)}"
    revenue = f"${rng.randint(1, 99)}.{rng.randint(1, 9)} million"
    growth = f"{rng.randint(1, 35)}.{rng.randint(0, 9)}%"
    employees = rng.randint(120, 4500)
    director = f"{rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    region = rng.choice([
        "Asia-Pacific", "North America", "Europe", "Latin America",
        "Middle East and Africa", "Southeast Asia",
    ])

    text = (
        f"QUARTERLY REPORT {report_id}. The {region} region reported total "
        f"revenue of {revenue} for {quarter} {year}, representing year-over-year "
        f"growth of {growth}. The workforce expanded to {employees} full-time "
        f"employees during the reporting period. Regional Director {director} "
        f"noted that the growth was primarily driven by strong performance in "
        f"enterprise subscriptions and a 12% increase in contract renewals. "
        f"Operating margins improved by 2.3 percentage points compared to the "
        f"previous quarter, attributed to cost optimization measures introduced "
        f"in the prior fiscal year."
    )
    return text, report_id


def _spec_paragraph(rng: random.Random) -> tuple[str, str]:
    """Generate a technical specification paragraph."""
    spec_id = f"SPEC-{rng.randint(10000, 99999)}"
    version = f"{rng.randint(1, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 99)}"
    material = rng.choice([
        "titanium-reinforced carbon fiber composite",
        "aerospace-grade aluminum alloy 7075-T6",
        "high-density polyethylene with glass fiber reinforcement",
        "surgical-grade stainless steel 316L",
        "borosilicate glass with anti-reflective coating",
    ])
    temp_min = rng.randint(-60, -10)
    temp_max = rng.randint(150, 400)
    tolerance = f"{rng.choice([0.001, 0.002, 0.005, 0.01, 0.05])} mm"
    weight = f"{rng.randint(1, 999)}.{rng.randint(10, 99)} grams"
    pressure = f"{rng.randint(50, 800)} kPa"

    text = (
        f"TECHNICAL SPECIFICATION {spec_id} (Revision {version}). Material: "
        f"{material}. Operating temperature range: {temp_min} C to {temp_max} C. "
        f"Dimensional tolerance: plus or minus {tolerance}. Unit weight: {weight}. "
        f"Maximum operating pressure: {pressure}. The component shall undergo "
        f"thermal cycling testing for a minimum of 500 cycles before acceptance. "
        f"All units must be individually serialized and traceable to the original "
        f"material batch. Non-conforming units shall be quarantined and reported "
        f"to the quality engineering team within 24 hours of detection."
    )
    return text, spec_id


def _letter_paragraph(rng: random.Random) -> tuple[str, str]:
    """Generate a formal letter paragraph."""
    sender = f"Dr. {rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    recipient = f"Prof. {rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    ref_code = f"LTR-{rng.randint(2000, 9999)}"
    institution = rng.choice([
        "Meridian Institute of Technology",
        "Harwick University College of Sciences",
        "Thornfield Research Consortium",
        "Lakeview Center for Applied Mathematics",
        "Vanguard Institute for Computational Biology",
    ])
    grant = f"${rng.randint(100, 999)},{rng.randint(100, 999)}"
    duration = f"{rng.randint(2, 5)}-year"
    field = rng.choice([
        "quantum error correction", "protein folding dynamics",
        "renewable energy storage", "neural interface design",
        "large-scale climate modeling",
    ])

    text = (
        f"LETTER — Reference {ref_code}. Dear {recipient}, I am writing on "
        f"behalf of {institution} to formally invite your participation in our "
        f"upcoming {duration} collaborative research program on {field}. The "
        f"program is funded by a {grant} grant and will commence in the autumn "
        f"semester. Your expertise, as demonstrated in your recent publications, "
        f"would be invaluable to the consortium. Please respond by the end of "
        f"the month to confirm your interest. Yours sincerely, {sender}, "
        f"Director of Research Partnerships, {institution}."
    )
    return text, ref_code


def _legal_paragraph(rng: random.Random) -> tuple[str, str]:
    """Generate a legal notice paragraph."""
    case_number = f"CASE-{rng.randint(2020, 2026)}-{rng.randint(10000, 99999)}"
    plaintiff = f"{rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    defendant_org = rng.choice([
        "Broadfield Holdings LLC", "Centurion Logistics Corp",
        "Apex Manufacturing Group", "Pinnacle Financial Services Inc",
        "Redstone Ventures Partners",
    ])
    judge = f"Hon. {rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    filing_date = f"{rng.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])} {rng.randint(1, 28)}, {rng.choice([2024, 2025, 2026])}"
    damages = f"${rng.randint(1, 50)},{rng.randint(100, 999)},{rng.randint(100, 999)}.00"

    text = (
        f"LEGAL NOTICE — {case_number}. In the matter of {plaintiff} v. "
        f"{defendant_org}, filed on {filing_date} before {judge}. The plaintiff "
        f"alleges breach of contractual obligations under Section 4(b) of the "
        f"original agreement dated twelve months prior to filing. Damages sought: "
        f"{damages}. The defendant is hereby notified that a preliminary hearing "
        f"has been scheduled and all relevant documentation must be submitted to "
        f"the court clerk no later than thirty calendar days from the date of "
        f"this notice. Counsel for both parties are required to attend."
    )
    return text, case_number


def _research_paragraph(rng: random.Random) -> tuple[str, str]:
    """Generate a research summary paragraph."""
    study_id = f"STUDY-{rng.choice(['BIO', 'PHY', 'CHM', 'ENG', 'MED'])}-{rng.randint(1000, 9999)}"
    lead_author = f"Dr. {rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"
    sample_size = rng.randint(200, 15000)
    p_value = f"0.{rng.randint(1, 49):03d}" if rng.random() < 0.7 else f"0.{rng.randint(50, 99):03d}"
    duration_weeks = rng.randint(4, 104)
    effect = f"{rng.randint(5, 45)}.{rng.randint(0, 9)}%"
    metric = rng.choice([
        "cognitive recall accuracy", "cardiovascular endurance",
        "thermal conductivity", "signal-to-noise ratio",
        "cellular regeneration rate", "error reduction rate",
    ])

    text = (
        f"RESEARCH SUMMARY — {study_id}. Principal investigator: {lead_author}. "
        f"A randomized controlled trial with N={sample_size} participants conducted "
        f"over {duration_weeks} weeks demonstrated a statistically significant "
        f"improvement in {metric} of {effect} (p={p_value}). The intervention "
        f"group showed consistent gains across all measured time points. Secondary "
        f"endpoints including participant adherence and adverse event frequency "
        f"were within acceptable thresholds. The authors recommend a follow-up "
        f"study with an expanded cohort to confirm generalizability of the findings."
    )
    return text, study_id


# All paragraph generators, each yielding (text, distinctive_phrase)
PARAGRAPH_GENERATORS = [
    _memo_paragraph,
    _report_paragraph,
    _spec_paragraph,
    _letter_paragraph,
    _legal_paragraph,
    _research_paragraph,
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


def generate_verbatim_copy_task(
    task_idx: int,
    doc_length: int = 20000,
    target_length: int = 200,
    seed: int = 42,
) -> VerbatimCopyTask:
    """
    Generate a single Verbatim Copy task.

    A unique target paragraph is embedded in a long document of filler text.
    The model must find and reproduce the paragraph exactly.

    Args:
        task_idx: Task index (for deterministic generation).
        doc_length: Total document length in characters.
        target_length: Ignored — paragraph length is determined by the template.
                       Kept for API consistency.
        seed: Random seed.
    """
    rng = random.Random(seed)

    # Pick a paragraph generator
    gen_fn = rng.choice(PARAGRAPH_GENERATORS)
    target_paragraph, distinctive_phrase = gen_fn(rng)

    # Choose a random position for the target paragraph (0.1 to 0.9)
    target_position = rng.uniform(0.1, 0.9)

    # Generate filler before and after
    target_char_pos = int(doc_length * target_position)
    before_len = max(0, target_char_pos - len(target_paragraph) // 2)
    after_len = max(0, doc_length - before_len - len(target_paragraph))

    before_text = _generate_filler(before_len, seed * 1000 + 1)
    after_text = _generate_filler(after_len, seed * 1000 + 2)

    document = before_text + "\n" + target_paragraph + "\n" + after_text

    # Build prompt — instruct the model to reproduce exactly
    prompt = (
        f"INSTRUCTION: Find and reproduce EXACTLY the paragraph that contains "
        f"\"{distinctive_phrase}\". Copy the entire paragraph verbatim — do not "
        f"summarize, paraphrase, or omit any part of it. Your answer should be "
        f"the paragraph text and nothing else.\n\n"
        f"DOCUMENT:\n{document}"
    )

    task_id = f"verbatim_{task_idx:03d}_{doc_length}"

    return VerbatimCopyTask(
        task_id=task_id,
        prompt=prompt,
        expected_answer=target_paragraph,
        doc_length=len(document),
        target_paragraph_position=target_position,
        target_paragraph_length=len(target_paragraph),
    )


def generate_verbatim_copy_suite(
    n_tasks: int = 10,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[VerbatimCopyTask]:
    """
    Generate a full Verbatim Copy benchmark suite.

    Default: tasks spread across doc lengths [10000, 20000, 50000, 100000].
    Each task uses a different paragraph type for variety.

    Args:
        n_tasks: Total number of tasks to generate.
        doc_lengths: List of document lengths to cycle through.
        seed_offset: Offset for seeds to avoid overlap with training data.
                     Use 0 for training, 10000+ for eval.
    """
    if doc_lengths is None:
        doc_lengths = [10000, 20000, 50000, 100000]

    tasks = []
    rng = random.Random(42 + seed_offset)

    for i in range(n_tasks):
        doc_len = doc_lengths[i % len(doc_lengths)]
        seed = i + seed_offset

        tasks.append(generate_verbatim_copy_task(
            task_idx=i,
            doc_length=doc_len,
            seed=seed,
        ))

    return tasks


def score_verbatim_copy(predicted: str | None, expected: str) -> float:
    """
    Score a Verbatim Copy prediction using character-level similarity.

    Uses difflib.SequenceMatcher for character-level comparison (no external deps).

    Scoring thresholds:
        >= 95% similarity  -> 1.0   (near-perfect reproduction)
        >= 80% similarity  -> 0.75  (minor errors — typos, small omissions)
        >= 60% similarity  -> 0.5   (got the gist but significant drift)
        <  60% similarity  -> 0.0   (failed)

    Args:
        predicted: The model's output (may be None if no answer produced).
        expected: The target paragraph.

    Returns:
        Score between 0.0 and 1.0.
    """
    if predicted is None:
        return 0.0

    # Strip whitespace for fair comparison — leading/trailing whitespace
    # differences should not penalize the model.
    predicted = predicted.strip()
    expected = expected.strip()

    if not predicted:
        return 0.0

    # Character-level similarity via SequenceMatcher
    similarity = SequenceMatcher(None, predicted, expected).ratio()

    if similarity >= 0.95:
        return 1.0
    elif similarity >= 0.80:
        return 0.75
    elif similarity >= 0.60:
        return 0.5
    else:
        return 0.0
