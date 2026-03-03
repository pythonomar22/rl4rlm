"""
Document Classification benchmark (OOLONG-style).

Given a collection of N short documents separated by markers, classify each
document into one of K categories.

Tests O(n) complexity: model must read and classify each document individually.
This is the core RLM use case — tasks that require iterating over all parts
of a long context rather than finding a single piece of information.

Inspired by OOLONG trec_coarse (arXiv:2511.02817).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class DocClassifyTask:
    """A single document classification task."""
    task_id: str
    prompt: str              # Full text (question + all documents)
    documents: list[dict]    # [{text, category, doc_idx}]
    expected_labels: list[str]  # Category for each document
    n_docs: int
    doc_length: int          # Total context length
    question: str
    categories: list[str]    # Available categories


# Categories (simplified TREC-style)
CATEGORIES = ["Science", "Sports", "Business", "World", "Health", "Technology"]

# Document templates per category — each produces ~300-800 char articles
# Templates use {placeholders} for variation
ARTICLE_TEMPLATES = {
    "Science": [
        "Researchers at the {university} have made a breakthrough in {field}. The team led by Dr. {name} discovered that {finding}. This could have major implications for {application}. The study, published in {journal}, involved {method}. \"We were surprised by the results,\" said Dr. {name}. \"This changes our understanding of {topic}.\" The research was funded by the {funder} and took {duration} to complete. Follow-up studies are already being planned to investigate {next_step}.",
        "A new species of {organism} has been identified in {location}. The discovery was made by a team from {university} during a {duration} expedition. The {organism}, named {species_name}, exhibits unique characteristics including {trait}. Scientists believe it evolved to adapt to {environment}. The finding was published in {journal} and has drawn attention from researchers studying {field}.",
        "The {observatory} has detected {phenomenon} in {cosmic_location}. This observation confirms theoretical predictions made by {name} in {year}. The data suggests that {implication}. \"This is the first direct evidence of {discovery},\" explained lead researcher Dr. {name}. The telescope used advanced {instrument} technology to make the measurement over {duration} of observation.",
    ],
    "Sports": [
        "The {team} secured a dramatic {score} victory over {opponent} in yesterday's {event}. {player} scored the winning {play} in the final minutes, sending fans into celebration. Coach {coach} praised the team's resilience after trailing for most of the game. \"We never gave up,\" said {player}. \"The fans deserved this performance.\" The win moves {team} to {position} in the standings with {record}.",
        "{player} announced their retirement from professional {sport} after a {duration} career spanning {achievements}. The {age}-year-old athlete holds the record for {record}. \"It's been an incredible journey,\" {player} said at a press conference. Tributes poured in from across the {sport} world, with {rival} calling them \"the greatest of their generation.\" A farewell {event} is planned for next month.",
        "The annual {tournament} kicked off in {city} with {number} athletes competing across {categories}. Early favorites include {player} and {player2}, both coming off strong seasons. Organizers expect record attendance of {attendance}. New rules this year include {rule_change}, which has been controversial among players. The tournament runs through {end_date} with the final scheduled at {venue}.",
    ],
    "Business": [
        "{company} reported quarterly earnings of {amount}, beating analyst expectations by {percent}. Revenue grew {growth}% year-over-year, driven by strong performance in the {division} segment. CEO {name} attributed the results to {strategy}. The company's stock rose {stock_change}% in after-hours trading. {company} also announced plans to {plan}, investing {investment} over the next {timeline}.",
        "A proposed merger between {company} and {company2} valued at {amount} is expected to reshape the {industry} industry. The deal, announced {day}, would create the world's largest {description}. Regulators in {region} are expected to review the transaction. Analysts predict the merger could lead to {impact}. Both companies' boards have unanimously approved the deal, pending shareholder vote.",
        "The {industry} sector saw {trend} this quarter as {cause}. Market analysts at {firm} noted that {observation}. Small businesses reported {impact} while larger corporations {response}. Industry group {org} called for {action} to address the challenges. Economists forecast {forecast} for the coming year, though {caveat} remains a concern.",
    ],
    "World": [
        "Leaders from {count} nations gathered in {city} for the {summit} to address {issue}. The meeting, hosted by {leader}, aims to reach agreement on {goal}. Negotiations have been tense, with {country} and {country2} at odds over {dispute}. Protesters outside the venue called for {demand}. A final communique is expected by {deadline}, though diplomats caution that significant differences remain.",
        "Humanitarian organizations are responding to {disaster} in {country} that has affected {number} people. The {org} has deployed emergency teams to provide {aid}. {leader} declared a state of emergency in {regions}. International aid pledges have reached {amount}, though agencies say {gap} more is needed. The crisis began {timeline} when {cause}.",
        "Elections in {country} resulted in a decisive victory for {party}, ending {duration} of {situation}. {leader} won with {percent}% of the vote, defeating {opponent}. The result is expected to shift {country}'s position on {policy}. International observers from {org} declared the election \"largely free and fair\" despite {issue}. Voter turnout reached {turnout}%, the highest in {duration2}.",
    ],
    "Health": [
        "A new study published in {journal} found that {finding}. Researchers at {institution} tracked {number} participants over {duration} and observed that {result}. Dr. {name}, the lead author, emphasized that {implication}. The findings could impact treatment protocols for {condition}. Health authorities recommend {recommendation} based on the emerging evidence.",
        "The {org} approved a new {treatment} for {condition}, marking a significant advance in {field}. Clinical trials showed that {result}, with {percent}% of patients experiencing improvement. The {treatment}, developed by {company}, works by {mechanism}. Side effects were {side_effects}. Experts estimate that {number} patients could benefit from the new therapy.",
        "Public health officials in {region} reported a {trend} in {condition} cases over the past {duration}. The {org} attributed the change to {cause}. Dr. {name} recommended that {demographic} should {action}. Prevention strategies include {prevention}. Hospitals in the area have {response} to handle the changing situation.",
    ],
    "Technology": [
        "{company} unveiled its latest {product} at the {event} conference in {city}. The device features {feature}, a first in the {industry} industry. CEO {name} demonstrated how {capability}. The {product} will be available starting {date} at a price of {price}. Analysts at {firm} predict it could capture {percent}% of the market within {timeline}.",
        "A team of engineers at {institution} developed a new {technology} that could revolutionize {field}. The system uses {method} to achieve {result}, surpassing previous approaches by {improvement}. \"This opens up entirely new possibilities for {application},\" said researcher {name}. The work was presented at {conference} and has attracted interest from {companies}.",
        "Cybersecurity experts warned about a new {threat} affecting {target}. The vulnerability, discovered by {company}, could allow attackers to {impact}. {org} issued an advisory recommending {action}. An estimated {number} systems worldwide may be affected. A patch is expected within {timeline}, and users are urged to {mitigation} in the meantime.",
    ],
}

# Placeholder values for template filling
NAMES = ["Smith", "Chen", "Rodriguez", "Patel", "Kim", "Johnson", "Williams", "Garcia", "Brown", "Davis"]
UNIVERSITIES = ["MIT", "Stanford", "Oxford", "Cambridge", "ETH Zurich", "Tokyo University", "Tsinghua"]
COMPANIES = ["TechCorp", "GlobalVentures", "Nextera", "Pinnacle Systems", "Atlas Industries", "Meridian Corp"]
CITIES = ["Geneva", "Singapore", "Berlin", "Tokyo", "New York", "London", "Sydney", "Dubai", "Boston", "Seoul"]
JOURNALS = ["Nature", "Science", "The Lancet", "IEEE Transactions", "PNAS", "Cell"]
TEAMS = ["the Lakers", "Manchester United", "the Eagles", "Bayern Munich", "the Warriors", "Real Madrid"]
PLAYERS = ["James Martinez", "Alex Thompson", "Maria Santos", "Tyler Brooks", "Sofia Lee", "David Park"]
SPORTS = ["basketball", "football", "soccer", "tennis", "baseball", "hockey"]
COUNTRIES = ["France", "Japan", "Brazil", "India", "Australia", "Canada", "Germany", "South Korea"]
ORGS = ["the WHO", "the UN", "Doctors Without Borders", "the Red Cross", "UNICEF"]
FIELDS = ["genetics", "quantum computing", "neuroscience", "climate science", "materials science"]
ORGANISMS = ["butterfly", "deep-sea fish", "plant", "fungus", "beetle"]
TREATMENTS = ["gene therapy", "monoclonal antibody", "vaccine", "drug", "surgical technique"]
CONDITIONS = ["diabetes", "Alzheimer's disease", "heart disease", "chronic pain", "rare blood disorder"]
PRODUCTS = ["smartphone", "AI chip", "wearable device", "electric vehicle", "robot assistant"]
THREATS = ["ransomware strain", "zero-day exploit", "phishing campaign", "supply chain attack"]
INDUSTRIES = ["technology", "healthcare", "finance", "energy", "automotive", "retail"]

# Map placeholder names to specific value pools
PLACEHOLDER_MAP = {
    "team": TEAMS, "opponent": TEAMS,
    "player": PLAYERS, "player2": PLAYERS, "rival": PLAYERS, "coach": PLAYERS,
    "sport": SPORTS, "event": ["championship game", "tournament final", "league match", "playoffs"],
    "tournament": ["Grand Slam", "World Cup qualifier", "National Championship", "Olympic trials"],
    "play": ["goal", "touchdown", "three-pointer", "home run"],
    "categories": ["individual and team events", "3 weight classes", "men's and women's divisions"],
    "venue": ["Madison Square Garden", "Wembley Stadium", "Tokyo Dome", "Olympic Park"],
    "attendance": ["45,000", "72,000", "28,000", "95,000"],
    "rule_change": ["a new video review system", "expanded rosters", "modified scoring"],
    "end_date": ["next Sunday", "March 20", "the end of the month"],
    "achievements": ["12 championship titles", "3 MVP awards", "multiple record-breaking seasons"],
    "country": COUNTRIES, "country2": COUNTRIES,
    "leader": ["President Nakamura", "Prime Minister Clarke", "Chancellor Weber"],
    "party": ["the National Alliance", "the Democratic Front", "the Reform Party"],
    "summit": ["G20 Summit", "Climate Conference", "Peace Summit", "Economic Forum"],
    "disaster": ["severe flooding", "a major earthquake", "widespread drought"],
    "aid": ["food, water, and medical supplies", "emergency shelter and medicine"],
    "issue": ["climate change", "trade policy", "regional security", "migration"],
    "goal": ["binding emissions targets", "a ceasefire agreement", "trade reform"],
    "dispute": ["carbon reduction timelines", "border security", "resource allocation"],
    "demand": ["stronger action on climate", "an end to the conflict", "democratic reforms"],
    "policy": ["foreign trade", "environmental regulation", "defense spending"],
    "situation": ["single-party rule", "coalition government", "political deadlock"],
    "regions": ["the northern provinces", "the coastal areas", "three major cities"],
    "gap": ["$200 million", "$50 million", "$1.5 billion"],
    "cause": ["prolonged heavy rainfall", "a shift in global markets", "policy changes"],
    "timeline": ["two weeks ago", "last month", "earlier this year"],
    "org": ORGS,
    "field": FIELDS,
    "finding": ["a novel protein structure", "unexpected correlations in patient data",
                 "a new mechanism for cellular repair"],
    "result": ["a 40% reduction in symptoms", "significant improvement in key metrics",
               "outcomes that exceeded expectations"],
    "implication": ["this could change standard treatment protocols",
                    "further research is urgently needed",
                    "patients may benefit within a few years"],
    "condition": CONDITIONS,
    "recommendation": ["regular screening for at-risk groups",
                       "updated vaccination schedules",
                       "increased physical activity"],
    "treatment": TREATMENTS,
    "mechanism": ["blocking a specific receptor", "targeting faulty genes",
                  "stimulating the immune response"],
    "side_effects": ["generally mild and temporary", "comparable to existing treatments"],
    "demographic": ["adults over 50", "children under 12", "healthcare workers"],
    "action": ["get regular check-ups", "follow updated guidelines", "maintain preventive care"],
    "prevention": ["regular exercise and balanced diet", "vaccination and hygiene practices"],
    "response": ["expanded capacity", "activated emergency protocols"],
    "trend": ["a significant decline", "a notable increase", "stable patterns"],
    "product": PRODUCTS,
    "feature": ["a neural processing unit", "holographic display", "quantum encryption"],
    "capability": ["the device processes natural language in real time",
                   "it adapts to user behavior automatically"],
    "technology": ["machine learning algorithm", "energy storage system", "quantum sensor"],
    "method": ["deep reinforcement learning", "advanced spectroscopy", "CRISPR gene editing"],
    "improvement": ["a factor of ten", "35%", "three orders of magnitude"],
    "application": ["medical diagnostics", "autonomous vehicles", "renewable energy"],
    "conference": ["NeurIPS 2025", "CES", "the ACM Conference"],
    "companies": ["major tech firms including Google and Apple", "several Fortune 500 companies"],
    "threat": THREATS,
    "target": ["enterprise networks", "mobile devices", "financial institutions"],
    "impact": ["unauthorized access to sensitive data", "widespread service disruption"],
    "mitigation": ["update software immediately", "enable two-factor authentication"],
    "industry": INDUSTRIES,
    "division": ["cloud services", "consumer electronics", "enterprise software"],
    "strategy": ["aggressive expansion into new markets", "cost optimization and AI integration"],
    "plan": ["expand operations to 15 new countries", "launch a new product line"],
    "investment": ["$2.5 billion", "$800 million", "$5 billion"],
    "company2": COMPANIES,
    "description": ["technology conglomerate", "healthcare provider", "financial services firm"],
    "observation": ["consumer confidence remains cautious", "innovation spending is accelerating"],
    "forecast": ["moderate growth of 3-5%", "continued uncertainty", "recovery by Q3"],
    "caveat": ["geopolitical instability", "supply chain disruption", "inflation"],
    "cosmic_location": ["the Andromeda galaxy", "a distant nebula", "the galactic center"],
    "phenomenon": ["gravitational waves", "a gamma-ray burst", "unusual radio signals"],
    "discovery": ["dark matter interaction", "a new type of stellar object"],
    "instrument": ["infrared imaging", "radio interferometry", "X-ray spectroscopy"],
    "observatory": ["the James Webb Space Telescope", "the European Southern Observatory",
                     "the Atacama Large Millimeter Array"],
    "topic": ["stellar evolution", "neurodegenerative disease", "ecosystem dynamics"],
    "funder": ["National Science Foundation", "European Research Council"],
    "next_step": ["clinical trials", "larger-scale experiments", "field deployments"],
    "species_name": ["Chromis Stellaris", "Dendroicus Novus", "Mycena Luminosa"],
    "trait": ["bioluminescence", "extreme heat tolerance", "a symbiotic relationship with algae"],
    "environment": ["deep ocean thermal vents", "high-altitude environments"],
    "organism": ORGANISMS,
    "location": CITIES,
}


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template with random placeholder values."""
    result = template
    used_values: dict[str, list] = {}  # Track used values per placeholder type

    while "{" in result:
        start = result.index("{")
        end = result.index("}", start)
        placeholder = result[start + 1:end]

        # Look up in placeholder map
        val = None
        if placeholder in PLACEHOLDER_MAP:
            pool = PLACEHOLDER_MAP[placeholder]
            val = rng.choice(pool)
        elif "name" in placeholder.lower():
            val = rng.choice(NAMES)
        elif "university" in placeholder or "institution" in placeholder:
            val = rng.choice(UNIVERSITIES)
        elif "company" in placeholder or "firm" in placeholder:
            val = rng.choice(COMPANIES)
        elif "city" in placeholder:
            val = rng.choice(CITIES)
        elif "journal" in placeholder:
            val = rng.choice(JOURNALS)
        elif "number" in placeholder or "count" in placeholder:
            val = str(rng.randint(50, 5000))
        elif "percent" in placeholder:
            val = str(rng.randint(5, 95))
        elif "amount" in placeholder:
            val = f"${rng.randint(1, 999)} {'million' if rng.random() > 0.5 else 'billion'}"
        elif "duration" in placeholder:
            val = rng.choice(["6 months", "2 years", "18 months", "5 years", "a decade"])
        elif "year" in placeholder:
            val = str(rng.randint(2010, 2025))
        elif "age" in placeholder:
            val = str(rng.randint(28, 42))
        elif "score" in placeholder:
            val = f"{rng.randint(1, 5)}-{rng.randint(0, 4)}"
        elif "date" in placeholder or "deadline" in placeholder:
            val = rng.choice(["March 15", "June 30", "September 1", "December 10", "next quarter"])
        elif "price" in placeholder:
            val = f"${rng.randint(299, 1999)}"
        elif "growth" in placeholder or "stock_change" in placeholder:
            val = str(rng.randint(3, 35))
        elif "record" in placeholder:
            val = f"{rng.randint(10, 50)} career records"
        elif "position" in placeholder:
            val = rng.choice(["first", "second", "third", "fourth"])
        elif "turnout" in placeholder:
            val = str(rng.randint(55, 92))
        else:
            val = rng.choice(["the initiative", "several key factors", "ongoing developments"])

        result = result[:start] + val + result[end + 1:]

    return result


def _generate_article(category: str, seed: int) -> str:
    """Generate a synthetic article of a given category."""
    rng = random.Random(seed)
    template = rng.choice(ARTICLE_TEMPLATES[category])
    return _fill_template(template, rng)


def generate_doc_classify_task(
    task_idx: int,
    n_docs: int = 10,
    seed: int | None = None,
) -> DocClassifyTask:
    """
    Generate a document classification task.

    Creates n_docs synthetic articles across categories, concatenates them
    with document markers, and asks the model to classify each.
    """
    seed = seed if seed is not None else task_idx + 70000
    rng = random.Random(seed)

    # Assign categories to documents (ensure at least 2 categories represented)
    categories_used = rng.sample(CATEGORIES, min(n_docs, len(CATEGORIES)))
    doc_categories = []
    for i in range(n_docs):
        doc_categories.append(categories_used[i % len(categories_used)])
    rng.shuffle(doc_categories)

    # Generate articles
    documents = []
    doc_parts = []
    for i, cat in enumerate(doc_categories):
        article = _generate_article(cat, seed * 100 + i)
        documents.append({
            "text": article,
            "category": cat,
            "doc_idx": i + 1,
        })
        doc_parts.append(f"=== DOCUMENT {i + 1} ===\n{article}")

    full_doc = "\n\n".join(doc_parts)

    # Build prompt
    cat_list = ", ".join(CATEGORIES)
    question = (
        f"Below are {n_docs} documents. Classify each document into exactly one "
        f"of these categories: {cat_list}.\n\n"
        f"Return your answer as a numbered list in this exact format:\n"
        f"1: Category\n2: Category\n...\n"
        f"Return ONLY the numbered classifications, nothing else."
    )

    prompt = f"QUESTION: {question}\n\nDOCUMENTS:\n{full_doc}"

    task_id = f"classify_{task_idx:03d}_{n_docs}docs"

    return DocClassifyTask(
        task_id=task_id,
        prompt=prompt,
        documents=documents,
        expected_labels=doc_categories,
        n_docs=n_docs,
        doc_length=len(full_doc),
        question=question,
        categories=CATEGORIES,
    )


def generate_doc_classify_suite(
    n_tasks: int = 20,
    seed_offset: int = 70000,
) -> list[DocClassifyTask]:
    """
    Generate document classification benchmark suite.

    Configurations:
    - 5 docs (~3K chars total) — easy
    - 10 docs (~6K chars total) — medium
    - 15 docs (~10K chars total) — hard
    - 20 docs (~14K chars total) — very hard
    """
    configs = [
        # (n_docs, count)
        (5, 5),    # Easy
        (10, 5),   # Medium
        (15, 5),   # Hard
        (20, 5),   # Very hard
    ]

    tasks = []
    idx = 0
    for n_docs, count in configs:
        for _ in range(count):
            if len(tasks) >= n_tasks:
                return tasks
            tasks.append(generate_doc_classify_task(
                task_idx=idx,
                n_docs=n_docs,
                seed=idx + seed_offset,
            ))
            idx += 1

    return tasks[:n_tasks]


def score_doc_classify(predicted: str | None, expected_labels: list[str]) -> dict:
    """
    Score a document classification prediction.

    Parses "1: Category\n2: Category\n..." format and compares to expected.

    Returns:
        dict with 'accuracy' (fraction correct), 'correct' (count),
        'total', and 'per_doc' (list of bool).
    """
    n_docs = len(expected_labels)

    if predicted is None:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": n_docs,
            "per_doc": [False] * n_docs,
        }

    # Parse predictions
    predicted_labels = {}
    for line in predicted.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Try parsing "N: Category" or "N. Category" or "Document N: Category"
        for sep in [":", ".", "-", ")"]:
            if sep in line:
                parts = line.split(sep, 1)
                try:
                    # Extract document number
                    num_str = parts[0].strip()
                    # Handle "Document N" format
                    num_str = num_str.lower().replace("document", "").replace("doc", "").strip()
                    doc_num = int(num_str)
                    category = parts[1].strip()
                    predicted_labels[doc_num] = category
                    break
                except (ValueError, IndexError):
                    continue

    # Score each document
    correct = 0
    per_doc = []
    for i, expected in enumerate(expected_labels):
        doc_num = i + 1
        pred = predicted_labels.get(doc_num, "")
        # Fuzzy match: check if expected category is substring of prediction
        match = expected.lower() in pred.lower()
        per_doc.append(match)
        if match:
            correct += 1

    accuracy = correct / n_docs if n_docs > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": n_docs,
        "per_doc": per_doc,
    }
