# Harder Benchmarks for Stronger Models
**Date:** 2026-03-10

## Current Benchmarks (Too Easy for GRPO model)
- Multi-NIAH: 100% at step 10 → ceiling
- Doc-Classify: 92% at step 10 → near ceiling for 5-doc tasks

## Needed: Harder Variants

### Extended NIAH
- Longer documents: 200K, 500K, 1M characters
- Multiple needle types (not just secret codes)
- Adversarial distractors (similar but wrong codes)
- Positional stress test: needles at exact boundaries

### Extended Multi-NIAH
- More needles: K=15, 20, 50
- Needles requiring cross-referencing (find code X, then find what X refers to)
- Mixed needle types (codes + names + dates)

### Hard Doc-Classify
- N=50, N=100 documents
- Ambiguous categories (Science vs Technology overlap)
- Very short documents (harder to classify)
- Novel categories beyond the 6 standard ones

### New Benchmarks to Implement

#### Multi-Hop QA
- Question requires chaining 2-3 facts from different parts of context
- E.g., "What project did the person who won the award work on?"
- Facts scattered across 50K+ character context
- Tests recursive reasoning, not just search

#### Verbatim Copy
- Reproduce specific paragraphs from context faithfully
- Tests precise information retrieval
- Important for practical applications (legal, medical)

#### OOLONG (External)
- Document classification benchmark from arXiv:2511.02817
- Real documents, harder categories
- Used by both original RLM paper and Prime Intellect
- Would add credibility to results

## Priority
1. Extended NIAH (longer docs) — easy to implement
2. Hard Doc-Classify (more docs) — easy to implement
3. Multi-Hop QA — medium difficulty, high value for paper
4. OOLONG — requires integration, high credibility value
