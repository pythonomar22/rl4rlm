# Hybrid RLM Architecture: Mechanism Analysis

Date: 2026-03-11

## The Discovery

Standard GRPO (V4-s5) + hybrid architecture (base model sub-calls) achieves our best results:
- Multi-Hop QA: 55% → 90% (+35pp over base, +25pp over non-hybrid trained)
- Hard Multi-Hop: 40% → 70% (+30pp over base, +60pp over non-hybrid trained)
- Multi-NIAH: 83.9% → 100.0%
- Overall: 62.4% → 67.6% (11 benchmarks), 69.6% → 81.1% (9 in-dist)

## Root Cause Analysis

### Why RL Degrades Sub-Call Quality

1. **Shared LoRA weights**: The same adapter handles both root code generation and llm_query() responses
2. **Training signal only rewards final answers**: The model gets reward for FINAL() being correct, not for sub-call accuracy
3. **Distribution shift**: Sub-calls receive arbitrary text chunks from context, not the structured prompts the model sees during generation
4. **Shortcut learning**: RL teaches the model to "return first plausible entity" for sub-calls, which works for NIAH but fails for multi-hop where precision matters

### Why Hybrid Fixes This

The hybrid architecture uses the **trained model for root code generation** and the **base (untrained) model for llm_query() sub-calls**.

1. **Division of labor**: Root model → planning, decomposition, aggregation. Sub-model → precise fact extraction.
2. **No RL bias in sub-calls**: Base model answers questions literally without learned shortcuts
3. **Composability**: Each sub-call is an independent, atomic lookup — base model excels at these

### Evidence from Trajectories

**Non-hybrid V4-s5 (Multi-Hop, 65%)**:
- Model asks compound questions: "Find the budget of the project completed by the HR department"
- Sub-call tries to resolve the entire chain in one pass → fails
- Returns "Not found" or picks up a distractor entity

**Hybrid V4-s5 (Multi-Hop, 90%)**:
- Root code decomposes: (1) "Who manages HR?" → entity, (2) "What project does {entity} lead?" → project, (3) "What budget is allocated to {project}?" → answer
- Each atomic query succeeds independently
- Python code chains the results

### The Tradeoff

Hybrid mode creates a **clear tradeoff**:

| Task Type | Non-hybrid | Hybrid | Delta | Why |
|-----------|-----------|--------|-------|-----|
| Multi-Hop QA | 65.0% | 90.0% | +25pp | Atomic decomposition works |
| Hard Multi-Hop | 10.0% | 70.0% | +60pp | Multi-step chains succeed |
| Multi-NIAH | 99.4% | 100.0% | +0.6pp | Near-ceiling already |
| NIAH | 85.0% | 80.0% | -5pp | Trained sub-calls slightly better for simple extraction |
| Event Counting | 61.2% | 55.1% | -6.1pp | Trained sub-calls better at structured extraction |
| Notebook QA | 73.3% | 66.7% | -6.6pp | Trained sub-calls understand Jupyter structure |
| Doc-Classify | 96.8% | 91.4% | -5.4pp | Trained sub-calls better at classification |

**Net effect**: Strongly positive because reasoning gains (+25pp, +60pp) >> extraction losses (-5 to -6pp)

## Design Principle for RLMs

> When training recursive models, separate the **planning module** (code generation, decomposition, orchestration) from the **perception module** (sub-call fact extraction). Train only the planning module; keep perception general-purpose.

This is analogous to the separation of concerns in cognitive architectures — System 2 (deliberate reasoning) should be trained, while System 1 (pattern matching/extraction) should remain general.

## Implications for V9 and Beyond

1. **V9 should be evaluated in hybrid mode**: If PPO + raw advantages produce even better root code, V9-hybrid could surpass V4-s5-hybrid
2. **Training should reward decomposition explicitly**: Add intermediate rewards for correct bridge entities
3. **Consider adapter gating**: Instead of binary hybrid/non-hybrid, learn when to use trained vs base for sub-calls
4. **OOLONG fix**: The real problem is OOD — neither trained nor base handles D&D transcripts. Need training data with unstructured aggregation.

## Paper Positioning

This is our **most novel finding** and distinguishes us from the original RLM paper (Zhang et al., 2025):

- Original RLM: Single model for both root and sub-calls (no LoRA separation issue since they use full fine-tuning)
- Our work: Discovers that LoRA fine-tuning causes root/sub-call interference, and proposes hybrid architecture as the solution
- This is a **new failure mode specific to parameter-efficient fine-tuning of recursive models** — not previously documented

The hybrid architecture is also practically useful: it's cheaper (sub-calls use the lighter base model) and more reliable.
