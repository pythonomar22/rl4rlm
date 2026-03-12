"""
Cross-Document Comparison benchmark for RLMs.

Tests cross-document comparison: comparing information between 2 documents
embedded in a long context. This is an O(N) task that requires:
1. Finding and extracting structured info from Document A
2. Finding and extracting structured info from Document B
3. Comparing them in Python

Task types:
1. overlap_entities: Two org charts (departments/roles). Find employees in BOTH orgs.
2. budget_diff: Two project budgets. Find the project with the largest budget difference.
3. timeline_conflict: Two timelines of events. Find events with conflicting dates.
4. metric_comparison: Two performance reports. Which report shows better overall metrics?

Each task has 2 documents (30K-100K chars each) separated by clear markers,
with filler/noise text between and around documents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class CrossDocTask:
    """A single cross-document comparison task."""
    task_id: str
    prompt: str
    expected_answer: str
    task_type: str  # overlap_entities, budget_diff, timeline_conflict, metric_comparison
    doc_length: int  # Total context length
    question: str
    details: dict = field(default_factory=dict)  # For debugging


# ---------------------------------------------------------------------------
# Name and entity pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
    "Henry", "Iris", "Jack", "Karen", "Leo", "Maria", "Nathan",
    "Olivia", "Patrick", "Quinn", "Rachel", "Samuel", "Tina",
    "Victor", "Wendy", "Xavier", "Yara", "Zachary", "Brenda",
    "Charles", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kyle", "Laura", "Marcus", "Nina",
    "Oscar", "Paula", "Raymond", "Susan", "Thomas", "Ursula",
]

LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Garcia", "Martinez", "Anderson",
    "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson",
    "Thompson", "White", "Lopez", "Lee", "Clark", "Robinson",
    "Walker", "Young", "Hall", "Allen", "King", "Wright", "Scott",
    "Adams", "Baker", "Campbell", "Davis", "Edwards", "Foster",
    "Green", "Hill", "Irwin", "James", "Kelly", "Long",
    "Mitchell", "Nelson", "Owens", "Price", "Reed", "Stone",
]

COMPANY_NAMES = [
    "Apex Global", "BlueWave Industries", "Crestline Corp", "Delphi Systems",
    "Evergreen Partners", "Falcon Technologies", "Granite Holdings",
    "Horizon Dynamics", "Ionic Solutions", "Juniper Networks Inc",
    "Keystone Ventures", "Lumina Capital", "Meridian Group",
    "Northstar Enterprises", "Olympus Digital", "Pinnacle Resources",
]

DEPARTMENTS = [
    "Engineering", "Marketing", "Finance", "Operations",
    "Human Resources", "Sales", "Research & Development",
    "Quality Assurance", "Customer Success", "Data Science",
    "Product Management", "Legal", "IT Infrastructure",
    "Business Development", "Supply Chain", "Compliance",
]

ROLES = [
    "Director", "Senior Manager", "Vice President", "Manager",
    "Team Lead", "Principal Engineer", "Senior Analyst",
    "Department Head", "Associate Director", "Senior Consultant",
]

PROJECT_NAMES = [
    "Project Alpha", "Project Beta", "Project Gamma", "Project Delta",
    "Project Epsilon", "Project Zeta", "Project Eta", "Project Theta",
    "Project Iota", "Project Kappa", "Project Lambda", "Project Mu",
    "Project Nu", "Project Xi", "Project Omicron", "Project Pi",
    "Project Rho", "Project Sigma", "Project Tau", "Project Upsilon",
]

EVENT_NAMES = [
    "Product Launch", "Board Meeting", "Annual Conference",
    "Quarterly Review", "System Migration", "Office Relocation",
    "Audit Completion", "Partnership Signing", "Funding Round",
    "Regulatory Filing", "Training Program", "Hackathon",
    "Strategy Retreat", "Client Summit", "Infrastructure Upgrade",
    "Budget Approval", "Merger Announcement", "Compliance Audit",
    "Technology Demo", "Stakeholder Meeting",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

METRIC_NAMES = [
    "Revenue Growth", "Customer Satisfaction", "Employee Retention",
    "Operational Efficiency", "Market Share", "Product Quality Score",
    "Net Promoter Score", "Profit Margin", "Cost Reduction",
    "Innovation Index", "Customer Acquisition Rate", "Delivery Speed",
]

# ---------------------------------------------------------------------------
# Filler text
# ---------------------------------------------------------------------------

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
    "Team performance reviews were conducted on a rolling quarterly basis.",
    "The internal audit found no significant discrepancies in the records.",
    "Vendor contracts were renegotiated to achieve cost savings.",
    "Cross-functional collaboration improved over the course of the year.",
    "Customer feedback was incorporated into the product development cycle.",
    "Sustainability targets were reviewed and updated for the coming period.",
    "The organization expanded its presence in two additional regions.",
    "Employee engagement scores showed a marked improvement year-over-year.",
    "Compliance training was completed by all personnel before the deadline.",
    "Procurement processes were centralized for greater efficiency.",
]


def _generate_filler_block(target_chars: int, rng: random.Random) -> str:
    """Generate filler text of approximately target_chars length."""
    lines: list[str] = []
    total = 0
    while total < target_chars:
        line = rng.choice(FILLER_SENTENCES)
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _make_full_name(rng: random.Random, used: set[str] | None = None) -> str:
    """Generate a unique full name not in *used*."""
    for _ in range(200):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if used is None or name not in used:
            if used is not None:
                used.add(name)
            return name
    raise RuntimeError("Could not generate a unique name")


# ===================================================================
# Task 1: overlap_entities
# ===================================================================

def _build_org_chart_doc(
    company: str,
    entries: list[dict],  # [{name, department, role}]
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build a realistic-looking org chart document for a company.

    Each entry is embedded in a paragraph of narrative text so the model
    must actually parse the document rather than just scanning a table.
    """
    parts: list[str] = []
    parts.append(
        f"ORGANIZATIONAL DIRECTORY — {company}\n"
        f"{'=' * 50}\n"
        f"This document contains the current organizational structure and "
        f"personnel assignments for {company}. All information is accurate "
        f"as of the latest reporting period.\n"
    )

    chars_per_entry = max(400, (doc_length - 500) // (len(entries) + 1))

    for entry in entries:
        # Narrative paragraph embedding the structured info
        templates = [
            (
                f"In the {entry['department']} department, {entry['name']} currently "
                f"holds the position of {entry['role']}. {entry['name']} has been "
                f"instrumental in driving key initiatives within {entry['department']} "
                f"and reports directly to the divisional leadership. The team under "
                f"{entry['name']} has consistently met or exceeded quarterly targets."
            ),
            (
                f"{entry['name']} serves as {entry['role']} of {entry['department']} "
                f"at {company}. Since joining the organization, {entry['name']} has "
                f"led several cross-functional projects and contributed to strategic "
                f"planning efforts. The {entry['department']} function under their "
                f"leadership has seen measurable improvements in output quality."
            ),
            (
                f"The {entry['department']} division lists {entry['name']} as its "
                f"{entry['role']}. {entry['name']} was promoted to this role after "
                f"demonstrating exceptional leadership during the most recent "
                f"reorganization. Peers describe {entry['name']} as a collaborative "
                f"and results-oriented leader within {company}."
            ),
        ]
        para = rng.choice(templates)
        parts.append(para)

        # Filler between entries
        filler_len = max(100, chars_per_entry - len(para))
        parts.append(_generate_filler_block(filler_len, rng))

    # Pad to reach target doc_length
    current_len = sum(len(p) for p in parts)
    if current_len < doc_length:
        parts.append(_generate_filler_block(doc_length - current_len, rng))

    return "\n\n".join(parts)


def generate_overlap_entities_task(
    task_id: str,
    doc_length: int,
    seed: int,
) -> CrossDocTask:
    """Two org charts. Find employees who appear in BOTH organizations.

    Design:
    - Company A has 15-25 employees.
    - Company B has 15-25 employees.
    - 2-5 employees appear in both (different roles/depts — they moved).
    - The rest are unique to each company.
    """
    rng = random.Random(seed)

    company_a, company_b = rng.sample(COMPANY_NAMES, 2)
    used_names: set[str] = set()

    # Decide how many shared employees
    n_shared = rng.randint(2, 5)
    n_only_a = rng.randint(12, 20)
    n_only_b = rng.randint(12, 20)

    # Generate shared employees (appear in both, different dept/role)
    shared_names: list[str] = []
    shared_entries_a: list[dict] = []
    shared_entries_b: list[dict] = []
    depts_pool = list(DEPARTMENTS)
    for _ in range(n_shared):
        name = _make_full_name(rng, used_names)
        shared_names.append(name)
        dept_a, dept_b = rng.sample(depts_pool, 2)
        role_a, role_b = rng.sample(ROLES, 2)
        shared_entries_a.append({"name": name, "department": dept_a, "role": role_a})
        shared_entries_b.append({"name": name, "department": dept_b, "role": role_b})

    # Generate unique-to-A employees
    entries_a = list(shared_entries_a)
    for _ in range(n_only_a):
        name = _make_full_name(rng, used_names)
        entries_a.append({
            "name": name,
            "department": rng.choice(DEPARTMENTS),
            "role": rng.choice(ROLES),
        })
    rng.shuffle(entries_a)

    # Generate unique-to-B employees
    entries_b = list(shared_entries_b)
    for _ in range(n_only_b):
        name = _make_full_name(rng, used_names)
        entries_b.append({
            "name": name,
            "department": rng.choice(DEPARTMENTS),
            "role": rng.choice(ROLES),
        })
    rng.shuffle(entries_b)

    half_len = doc_length // 2
    doc_a = _build_org_chart_doc(company_a, entries_a, half_len, rng)
    doc_b = _build_org_chart_doc(company_b, entries_b, half_len, rng)

    # Expected answer: sorted list of shared names
    expected = ", ".join(sorted(shared_names))

    question = (
        f"Below are two organizational directories — one for {company_a} and "
        f"one for {company_b}. Identify all employees (by full name) who appear "
        f"in BOTH organizations. Return ONLY the names separated by commas, "
        f"sorted alphabetically."
    )

    filler_between = _generate_filler_block(rng.randint(2000, 5000), rng)
    filler_before = _generate_filler_block(rng.randint(1000, 3000), rng)
    filler_after = _generate_filler_block(rng.randint(1000, 3000), rng)

    full_prompt = (
        f"QUESTION: {question}\n\n"
        f"{filler_before}\n\n"
        f"DOCUMENT_A:\n{doc_a}\n\n"
        f"{filler_between}\n\n"
        f"DOCUMENT_B:\n{doc_b}\n\n"
        f"{filler_after}"
    )

    return CrossDocTask(
        task_id=task_id,
        prompt=full_prompt,
        expected_answer=expected,
        task_type="overlap_entities",
        doc_length=len(full_prompt),
        question=question,
        details={
            "company_a": company_a,
            "company_b": company_b,
            "shared_names": sorted(shared_names),
            "n_only_a": n_only_a,
            "n_only_b": n_only_b,
        },
    )


# ===================================================================
# Task 2: budget_diff
# ===================================================================

def _build_budget_doc(
    company: str,
    projects: list[dict],  # [{name, budget}]
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build a realistic budget report document."""
    parts: list[str] = []
    parts.append(
        f"BUDGET REPORT — {company}\n"
        f"{'=' * 50}\n"
        f"This report details the approved budget allocations for all active "
        f"projects at {company}. Figures are in USD and reflect the current "
        f"fiscal year planning cycle.\n"
    )

    chars_per_project = max(400, (doc_length - 500) // (len(projects) + 1))

    for proj in projects:
        budget_str = f"${proj['budget']:,.0f}"
        templates = [
            (
                f"{proj['name']} has been allocated a total budget of {budget_str} "
                f"for the current fiscal year. This figure includes personnel costs, "
                f"infrastructure investments, and a contingency reserve. The project "
                f"sponsor has confirmed that spending is on track and within the "
                f"approved envelope. Monthly burn rate reviews are scheduled to ensure "
                f"continued alignment with organizational priorities."
            ),
            (
                f"The finance committee approved a budget of {budget_str} for "
                f"{proj['name']}. This allocation covers all phases of the project "
                f"lifecycle from planning through deployment. Key cost drivers include "
                f"vendor contracts, talent acquisition, and technology licensing. "
                f"Quarterly variance reports will be submitted to the CFO office."
            ),
            (
                f"Under the current plan, {proj['name']} is budgeted at {budget_str}. "
                f"The project leadership team worked with finance to develop this "
                f"estimate based on historical spend patterns and projected resource "
                f"needs. Approximately 40% of the budget is earmarked for the first "
                f"two quarters, with the remainder allocated to execution and closing."
            ),
        ]
        para = rng.choice(templates)
        parts.append(para)

        filler_len = max(100, chars_per_project - len(para))
        parts.append(_generate_filler_block(filler_len, rng))

    current_len = sum(len(p) for p in parts)
    if current_len < doc_length:
        parts.append(_generate_filler_block(doc_length - current_len, rng))

    return "\n\n".join(parts)


def generate_budget_diff_task(
    task_id: str,
    doc_length: int,
    seed: int,
) -> CrossDocTask:
    """Two budget reports for the same set of projects. Find the project with
    the largest absolute budget difference between the two reports.

    Design:
    - 8-15 projects appear in both documents.
    - One project has a dramatically larger difference than the rest.
    - Budget values are realistic (100K - 50M).
    """
    rng = random.Random(seed)

    company_a, company_b = rng.sample(COMPANY_NAMES, 2)
    n_projects = rng.randint(8, 15)
    projects = rng.sample(PROJECT_NAMES, n_projects)

    # Generate budgets for company A
    budgets_a: dict[str, float] = {}
    budgets_b: dict[str, float] = {}

    for proj in projects:
        base = rng.randint(100_000, 50_000_000)
        budgets_a[proj] = float(base)
        # Most projects have small differences (0-15%)
        variation = rng.uniform(0.0, 0.15)
        direction = rng.choice([-1, 1])
        budgets_b[proj] = float(base * (1 + direction * variation))

    # Pick one project to have the biggest difference (30-80% change)
    target_project = rng.choice(projects)
    big_variation = rng.uniform(0.30, 0.80)
    direction = rng.choice([-1, 1])
    budgets_b[target_project] = float(
        budgets_a[target_project] * (1 + direction * big_variation)
    )

    entries_a = [{"name": p, "budget": budgets_a[p]} for p in projects]
    entries_b = [{"name": p, "budget": budgets_b[p]} for p in projects]
    rng.shuffle(entries_a)
    rng.shuffle(entries_b)

    half_len = doc_length // 2
    doc_a = _build_budget_doc(company_a, entries_a, half_len, rng)
    doc_b = _build_budget_doc(company_b, entries_b, half_len, rng)

    # Verify target_project truly has the largest diff
    diffs = {p: abs(budgets_a[p] - budgets_b[p]) for p in projects}
    actual_max_project = max(diffs, key=lambda p: diffs[p])
    # If by some chance another project ended up with a larger diff,
    # adjust: use the actual max
    expected = actual_max_project

    question = (
        f"Below are two budget reports — one from {company_a} and one from "
        f"{company_b} — covering the same set of projects. Identify the project "
        f"that has the LARGEST absolute budget difference between the two reports. "
        f"Return ONLY the project name (e.g. 'Project Alpha')."
    )

    filler_between = _generate_filler_block(rng.randint(2000, 5000), rng)
    filler_before = _generate_filler_block(rng.randint(1000, 3000), rng)
    filler_after = _generate_filler_block(rng.randint(1000, 3000), rng)

    full_prompt = (
        f"QUESTION: {question}\n\n"
        f"{filler_before}\n\n"
        f"DOCUMENT_A:\n{doc_a}\n\n"
        f"{filler_between}\n\n"
        f"DOCUMENT_B:\n{doc_b}\n\n"
        f"{filler_after}"
    )

    return CrossDocTask(
        task_id=task_id,
        prompt=full_prompt,
        expected_answer=expected,
        task_type="budget_diff",
        doc_length=len(full_prompt),
        question=question,
        details={
            "company_a": company_a,
            "company_b": company_b,
            "budgets_a": budgets_a,
            "budgets_b": budgets_b,
            "diffs": diffs,
            "target_project": expected,
            "target_diff": diffs[expected],
        },
    )


# ===================================================================
# Task 3: timeline_conflict
# ===================================================================

def _build_timeline_doc(
    source: str,
    events: list[dict],  # [{name, month, day, year}]
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build a realistic timeline/schedule document."""
    parts: list[str] = []
    parts.append(
        f"EVENT TIMELINE — {source}\n"
        f"{'=' * 50}\n"
        f"This document records the official schedule and dates for all major "
        f"events as tracked by {source}. Dates reflect the most recent "
        f"confirmed information.\n"
    )

    chars_per_event = max(400, (doc_length - 500) // (len(events) + 1))

    for evt in events:
        date_str = f"{evt['month']} {evt['day']}, {evt['year']}"
        templates = [
            (
                f"According to the latest schedule, the {evt['name']} is set for "
                f"{date_str}. All relevant stakeholders have been notified and "
                f"preparations are underway. The organizing committee confirmed "
                f"that venue and logistics arrangements are finalized. Attendance "
                f"projections suggest strong participation from key departments."
            ),
            (
                f"The {evt['name']} has been officially scheduled for {date_str}. "
                f"This date was confirmed after consultation with all participating "
                f"parties. The agenda includes presentations, working sessions, and "
                f"a networking reception. Travel arrangements should be completed "
                f"at least two weeks prior to the event date."
            ),
            (
                f"Records indicate that the {evt['name']} will take place on "
                f"{date_str}. The event coordinator has distributed preliminary "
                f"materials to all invitees. Key deliverables are expected to be "
                f"reviewed during the session. Follow-up actions will be tracked "
                f"through the project management system."
            ),
        ]
        para = rng.choice(templates)
        parts.append(para)

        filler_len = max(100, chars_per_event - len(para))
        parts.append(_generate_filler_block(filler_len, rng))

    current_len = sum(len(p) for p in parts)
    if current_len < doc_length:
        parts.append(_generate_filler_block(doc_length - current_len, rng))

    return "\n\n".join(parts)


def generate_timeline_conflict_task(
    task_id: str,
    doc_length: int,
    seed: int,
) -> CrossDocTask:
    """Two timelines for the same set of events. Some events have the same
    dates in both documents; a small number have CONFLICTING dates.
    The model must identify all events with conflicting dates.

    Design:
    - 8-15 events appear in both timelines.
    - 2-4 events have different dates (the conflicts).
    - The rest have identical dates.
    """
    rng = random.Random(seed)

    source_a = rng.choice(["Internal Planning Office", "Operations Division",
                           "Corporate Calendar", "Executive Scheduling"])
    source_b = rng.choice(["Regional Coordination Team", "External Affairs Office",
                           "Partner Relations", "Project Management Office"])
    while source_b == source_a:
        source_b = rng.choice(["Regional Coordination Team", "External Affairs Office",
                               "Partner Relations", "Project Management Office"])

    n_events = rng.randint(8, 15)
    n_conflicts = rng.randint(2, 4)

    event_names = rng.sample(EVENT_NAMES, n_events)
    year = rng.choice([2025, 2026, 2027])

    # Generate dates for document A
    events_a: list[dict] = []
    events_b: list[dict] = []

    conflict_indices = set(rng.sample(range(n_events), n_conflicts))
    conflict_events: list[str] = []

    for i, evt_name in enumerate(event_names):
        month_a = rng.choice(MONTHS)
        day_a = rng.randint(1, 28)

        evt_a = {"name": evt_name, "month": month_a, "day": day_a, "year": year}
        events_a.append(evt_a)

        if i in conflict_indices:
            # Generate a DIFFERENT date for document B
            month_b = rng.choice(MONTHS)
            day_b = rng.randint(1, 28)
            # Make sure the date is actually different
            while month_b == month_a and day_b == day_a:
                month_b = rng.choice(MONTHS)
                day_b = rng.randint(1, 28)
            evt_b = {"name": evt_name, "month": month_b, "day": day_b, "year": year}
            conflict_events.append(evt_name)
        else:
            # Same date in both documents
            evt_b = {"name": evt_name, "month": month_a, "day": day_a, "year": year}

        events_b.append(evt_b)

    # Shuffle event order within each document
    rng.shuffle(events_a)
    rng.shuffle(events_b)

    half_len = doc_length // 2
    doc_a = _build_timeline_doc(source_a, events_a, half_len, rng)
    doc_b = _build_timeline_doc(source_b, events_b, half_len, rng)

    expected = ", ".join(sorted(conflict_events))

    question = (
        f"Below are two event timelines — one from the {source_a} and one from "
        f"the {source_b}. Both cover the same set of events, but some events "
        f"have CONFLICTING dates between the two documents. Identify all events "
        f"whose dates differ between the two timelines. Return ONLY the event "
        f"names separated by commas, sorted alphabetically."
    )

    filler_between = _generate_filler_block(rng.randint(2000, 5000), rng)
    filler_before = _generate_filler_block(rng.randint(1000, 3000), rng)
    filler_after = _generate_filler_block(rng.randint(1000, 3000), rng)

    full_prompt = (
        f"QUESTION: {question}\n\n"
        f"{filler_before}\n\n"
        f"DOCUMENT_A:\n{doc_a}\n\n"
        f"{filler_between}\n\n"
        f"DOCUMENT_B:\n{doc_b}\n\n"
        f"{filler_after}"
    )

    return CrossDocTask(
        task_id=task_id,
        prompt=full_prompt,
        expected_answer=expected,
        task_type="timeline_conflict",
        doc_length=len(full_prompt),
        question=question,
        details={
            "source_a": source_a,
            "source_b": source_b,
            "events_a": events_a,
            "events_b": events_b,
            "conflict_events": sorted(conflict_events),
            "n_conflicts": n_conflicts,
        },
    )


# ===================================================================
# Task 4: metric_comparison
# ===================================================================

def _build_performance_doc(
    division: str,
    metrics: list[dict],  # [{name, value, unit}]
    doc_length: int,
    rng: random.Random,
) -> str:
    """Build a realistic performance report document."""
    parts: list[str] = []
    parts.append(
        f"PERFORMANCE REPORT — {division}\n"
        f"{'=' * 50}\n"
        f"This report presents the key performance indicators for {division} "
        f"for the most recent evaluation period. All metrics have been verified "
        f"by the internal audit team and represent finalized figures.\n"
    )

    chars_per_metric = max(400, (doc_length - 500) // (len(metrics) + 1))

    for m in metrics:
        val_str = f"{m['value']}{m['unit']}"
        templates = [
            (
                f"The {m['name']} for {division} was recorded at {val_str} during "
                f"the reporting period. This figure reflects sustained effort across "
                f"the team and is consistent with the targets established at the "
                f"start of the cycle. Management has noted that continued focus on "
                f"process improvement will be critical to maintaining this level "
                f"of performance in subsequent quarters."
            ),
            (
                f"For the period under review, {division} achieved a {m['name']} "
                f"of {val_str}. This result was driven by a combination of strategic "
                f"investments and operational discipline. The leadership team has "
                f"committed to building on this outcome through targeted capability "
                f"development and resource optimization initiatives."
            ),
            (
                f"{division} reported a {m['name']} of {val_str}. Compared to "
                f"industry benchmarks, this positions the division competitively "
                f"within its peer group. The analytics team attributed the result "
                f"to improvements in workflow efficiency and stakeholder engagement "
                f"practices implemented earlier in the year."
            ),
        ]
        para = rng.choice(templates)
        parts.append(para)

        filler_len = max(100, chars_per_metric - len(para))
        parts.append(_generate_filler_block(filler_len, rng))

    current_len = sum(len(p) for p in parts)
    if current_len < doc_length:
        parts.append(_generate_filler_block(doc_length - current_len, rng))

    return "\n\n".join(parts)


def generate_metric_comparison_task(
    task_id: str,
    doc_length: int,
    seed: int,
) -> CrossDocTask:
    """Two performance reports for different divisions. Determine which
    division has better overall metrics.

    Design:
    - Both divisions report the same 6-10 metrics.
    - All metrics use percentage units so they are comparable.
    - One division has a higher average across metrics.
    - The question asks which division has better OVERALL performance.
    """
    rng = random.Random(seed)

    division_a = rng.choice([
        "North America Division", "Eastern Region", "Global Operations Unit",
        "Consumer Products Group", "Enterprise Solutions Division",
    ])
    division_b = rng.choice([
        "Asia-Pacific Division", "Western Region", "Strategic Initiatives Unit",
        "Commercial Services Group", "Digital Transformation Division",
    ])
    while division_b == division_a:
        division_b = rng.choice([
            "Asia-Pacific Division", "Western Region", "Strategic Initiatives Unit",
            "Commercial Services Group", "Digital Transformation Division",
        ])

    n_metrics = rng.randint(6, 10)
    metric_names = rng.sample(METRIC_NAMES, n_metrics)

    # Decide which division wins (A or B)
    winner = rng.choice(["A", "B"])

    metrics_a: list[dict] = []
    metrics_b: list[dict] = []

    for mname in metric_names:
        # Generate realistic percentage metrics
        base_val = rng.uniform(40.0, 90.0)

        if winner == "A":
            # A tends to be higher
            val_a = round(base_val + rng.uniform(1.0, 15.0), 1)
            val_b = round(base_val - rng.uniform(0.0, 10.0), 1)
        else:
            # B tends to be higher
            val_a = round(base_val - rng.uniform(0.0, 10.0), 1)
            val_b = round(base_val + rng.uniform(1.0, 15.0), 1)

        metrics_a.append({"name": mname, "value": val_a, "unit": "%"})
        metrics_b.append({"name": mname, "value": val_b, "unit": "%"})

    # Verify which actually has the higher average
    avg_a = sum(m["value"] for m in metrics_a) / n_metrics
    avg_b = sum(m["value"] for m in metrics_b) / n_metrics

    if avg_a > avg_b:
        expected_division = division_a
    else:
        expected_division = division_b

    # Shuffle metric order within each document
    rng.shuffle(metrics_a)
    rng.shuffle(metrics_b)

    half_len = doc_length // 2
    doc_a = _build_performance_doc(division_a, metrics_a, half_len, rng)
    doc_b = _build_performance_doc(division_b, metrics_b, half_len, rng)

    question = (
        f"Below are two performance reports — one for {division_a} and one for "
        f"{division_b}. Both report the same set of metrics (all in percentages). "
        f"Which division has BETTER overall performance (i.e. a higher average "
        f"across all metrics)? Return ONLY the division name."
    )

    filler_between = _generate_filler_block(rng.randint(2000, 5000), rng)
    filler_before = _generate_filler_block(rng.randint(1000, 3000), rng)
    filler_after = _generate_filler_block(rng.randint(1000, 3000), rng)

    full_prompt = (
        f"QUESTION: {question}\n\n"
        f"{filler_before}\n\n"
        f"DOCUMENT_A:\n{doc_a}\n\n"
        f"{filler_between}\n\n"
        f"DOCUMENT_B:\n{doc_b}\n\n"
        f"{filler_after}"
    )

    return CrossDocTask(
        task_id=task_id,
        prompt=full_prompt,
        expected_answer=expected_division,
        task_type="metric_comparison",
        doc_length=len(full_prompt),
        question=question,
        details={
            "division_a": division_a,
            "division_b": division_b,
            "metrics_a": {m["name"]: m["value"] for m in metrics_a},
            "metrics_b": {m["name"]: m["value"] for m in metrics_b},
            "avg_a": round(avg_a, 2),
            "avg_b": round(avg_b, 2),
            "winner": expected_division,
        },
    )


# ===================================================================
# Suite generation
# ===================================================================

def generate_cross_doc_suite(
    n_tasks: int = 12,
    doc_lengths: list[int] | None = None,
    seed_offset: int = 0,
) -> list[CrossDocTask]:
    """Generate a suite of cross-document comparison tasks.

    Default: 60K-200K total chars (30K-100K per document).
    Tasks are evenly split across the 4 types.
    """
    if doc_lengths is None:
        doc_lengths = [60000, 100000, 150000, 200000]

    generators = [
        generate_overlap_entities_task,
        generate_budget_diff_task,
        generate_timeline_conflict_task,
        generate_metric_comparison_task,
    ]

    tasks: list[CrossDocTask] = []
    for i in range(n_tasks):
        seed = seed_offset + i * 13
        rng = random.Random(seed)
        doc_length = rng.choice(doc_lengths)
        gen = generators[i % len(generators)]
        task = gen(
            task_id=f"cross_doc_{i:03d}",
            doc_length=doc_length,
            seed=seed,
        )
        tasks.append(task)

    return tasks


# ===================================================================
# Scoring
# ===================================================================

def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_name_list(text: str) -> list[str]:
    """Parse a comma-separated list of names from model output.

    Handles various formats:
    - "Alice Johnson, Bob Smith"
    - "Alice Johnson; Bob Smith"
    - "- Alice Johnson\\n- Bob Smith"
    - "1. Alice Johnson\\n2. Bob Smith"
    """
    import re

    text = text.strip()

    # Remove markdown list markers, numbers with dots/parens, bullet points
    text = re.sub(r"^[\d]+[.)]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s*", "", text, flags=re.MULTILINE)

    # Split on common delimiters
    if "," in text:
        items = text.split(",")
    elif ";" in text:
        items = text.split(";")
    elif "\n" in text:
        items = text.split("\n")
    else:
        items = [text]

    # Clean each item
    cleaned: list[str] = []
    for item in items:
        item = item.strip().strip("'\"")
        if item:
            cleaned.append(item)

    return cleaned


def score_cross_doc(predicted: str | None, task: CrossDocTask) -> dict:
    """Score a cross-document comparison answer.

    Scoring varies by task type:
    - overlap_entities: F1 over the set of shared names
    - budget_diff: exact match of the project name
    - timeline_conflict: F1 over the set of conflicting event names
    - metric_comparison: exact match of the division name
    """
    if predicted is None:
        return {"score": 0.0, "match_type": "none"}

    predicted = predicted.strip()
    expected = task.expected_answer

    if task.task_type == "overlap_entities":
        return _score_name_set(predicted, expected)

    elif task.task_type == "budget_diff":
        return _score_exact_entity(predicted, expected)

    elif task.task_type == "timeline_conflict":
        return _score_name_set(predicted, expected)

    elif task.task_type == "metric_comparison":
        return _score_exact_entity(predicted, expected)

    return {"score": 0.0, "match_type": "unknown_type"}


def _score_name_set(predicted: str, expected: str) -> dict:
    """Score a set-based answer (overlap_entities, timeline_conflict).

    Uses F1 score over the predicted vs expected item sets.
    """
    pred_items = _parse_name_list(predicted)
    exp_items = _parse_name_list(expected)

    # Normalize for matching
    pred_normalized = {_normalize(item) for item in pred_items}
    exp_normalized = {_normalize(item) for item in exp_items}

    if not exp_normalized:
        return {"score": 0.0, "match_type": "empty_expected"}

    # Compute true positives, precision, recall, F1
    # Track matched expected items to prevent recall > 1.0
    matched_expected = set()
    matched_predicted = 0
    for pred_item in pred_normalized:
        for exp_item in exp_normalized:
            if exp_item in matched_expected:
                continue
            # Use substring matching for robustness
            if pred_item == exp_item or pred_item in exp_item or exp_item in pred_item:
                matched_predicted += 1
                matched_expected.add(exp_item)
                break

    if matched_predicted == 0:
        return {"score": 0.0, "match_type": "no_overlap"}

    precision = matched_predicted / len(pred_normalized) if pred_normalized else 0.0
    recall = matched_predicted / len(exp_normalized)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if f1 >= 1.0:
        match_type = "exact"
    elif f1 >= 0.5:
        match_type = "partial"
    else:
        match_type = "low_overlap"

    return {"score": round(f1, 4), "match_type": match_type}


def _score_exact_entity(predicted: str, expected: str) -> dict:
    """Score an exact entity match (budget_diff, metric_comparison).

    Accepts the answer if the expected string appears within the prediction
    (case-insensitive), to handle things like 'The answer is Project Alpha'.
    """
    pred_lower = _normalize(predicted)
    exp_lower = _normalize(expected)

    if exp_lower == pred_lower:
        return {"score": 1.0, "match_type": "exact"}

    if exp_lower in pred_lower:
        return {"score": 1.0, "match_type": "contains"}

    # Try partial match for project names: e.g., "Alpha" in "Project Alpha"
    # Split expected into words and check if all significant words appear
    exp_words = [w for w in exp_lower.split() if len(w) > 2 and w not in {"the", "and", "for"}]
    if exp_words and all(w in pred_lower for w in exp_words):
        return {"score": 0.8, "match_type": "partial_words"}

    return {"score": 0.0, "match_type": "none"}
