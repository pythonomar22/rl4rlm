#!/usr/bin/env python3
"""Create SFT V6: weak-task-only dataset.

ONLY train on tasks where base model accuracy < 70%.
Skip tasks where base is already strong to prevent catastrophic forgetting.

Base model results:
  NIAH: 60.0% → INCLUDE
  Multi-NIAH: 91.5% → SKIP (strong)
  Doc-Classify: 81.6% → SKIP (strong, but borderline)
  DataFrame QA: 54.0% → INCLUDE
  Code Debug: 25.6% → INCLUDE
  Multi-Hop QA: 85.0% → SKIP (strong)
  Notebook QA: 70.0% → SKIP (borderline)
  Hard NIAH: 93.3% → SKIP (strong)
  Verbatim: 100.0% → SKIP (strong)
  Oolong: 0.0% → INCLUDE
  Hard Multi-Hop: 40.0% → INCLUDE
  Event Counting: 57.2% → INCLUDE
  Cross-Doc: 43.0% → INCLUDE
  KV: 51.3% → INCLUDE
"""
import json
from collections import defaultdict
from pathlib import Path

INPUT = Path("/root/rlm/data/sft/sft_all_aggregated.jsonl")
OUTPUT = Path("/root/rlm/data/sft/sft_v6_weak_tasks_only.jsonl")

# Tasks where base model is WEAK (< 70% accuracy) — INCLUDE these
WEAK_TASKS = {
    "niah",            # 60.0%
    "dataframe_qa",    # 54.0%
    "code_debug",      # 25.6%
    "oolong",          # 0.0%
    "hard_multi_hop",  # 40.0%
    "event_counting",  # 57.2%
    "cross_doc_compare",  # 43.0%
    "key_value_retrieval",  # 51.3%
}

# For multi-step tasks, prefer multi-turn
MULTI_TURN_PREFERRED = {"hard_multi_hop", "code_debug"}

MAX_PER_TYPE = 80

def count_assistant_turns(sample):
    return sum(1 for m in sample["messages"] if m["role"] == "assistant")

def main():
    samples = []
    with open(INPUT) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} total samples")
    
    # Filter to weak tasks only
    by_type = defaultdict(list)
    for s in samples:
        if s["task_type"] in WEAK_TASKS:
            by_type[s["task_type"]].append(s)
    
    filtered = []
    for task_type in sorted(by_type.keys()):
        task_samples = by_type[task_type]
        
        if task_type in MULTI_TURN_PREFERRED:
            # Sort: multi-turn first, then by score
            task_samples.sort(key=lambda x: (-count_assistant_turns(x), -x.get("score", 0)))
        else:
            task_samples.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        selected = task_samples[:MAX_PER_TYPE]
        filtered.extend(selected)
        
        multi = sum(1 for s in selected if count_assistant_turns(s) >= 1)
        print(f"  {task_type}: {len(selected)}/{len(task_samples)} ({multi} multi-turn)")
    
    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for s in filtered:
            f.write(json.dumps(s) + "\n")
    
    print(f"\nSaved {len(filtered)} samples to {OUTPUT}")
    print("Skipped tasks (base >= 70%): multi_niah, doc_classify, multi_hop_qa, notebook_qa, hard_niah, verbatim_copy")

if __name__ == "__main__":
    main()
