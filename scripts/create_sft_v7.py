#!/usr/bin/env python3
"""SFT V7: Weak-only training FROM V11-s5 (best GRPO model).

V11-s5 is strong on many tasks but weak on DFQA, cross_doc, KV.
Train ONLY on V11-s5's weak tasks to improve them without interfering.

V11-s5 results:
  Strong (>= 60%): niah 80, multi_niah 87.8, doc_classify 99.2, multi_hop 80,
    hard_niah 100, verbatim 100, hard_multi_hop 50, event_counting 72.9, notebook 66.7
  Weak (< 60%): DFQA 40, code_debug 25.6, cross_doc 24.4, KV 36.1, oolong 20

Only train on the 5 weakest tasks.
"""
import json
from collections import defaultdict
from pathlib import Path

INPUT = Path("/root/rlm/data/sft/sft_all_aggregated.jsonl")
OUTPUT = Path("/root/rlm/data/sft/sft_v7_v11_weak_only.jsonl")

# V11-s5's weakest tasks (accuracy < 50%)
WEAK_TASKS = {
    "dataframe_qa",        # 40.0%
    "code_debug",          # 25.6%
    "cross_doc_compare",   # 24.4%
    "key_value_retrieval", # 36.1%
    "oolong",              # 20.0%
}

MAX_PER_TYPE = 80

def count_assistant_turns(sample):
    return sum(1 for m in sample["messages"] if m["role"] == "assistant")

def main():
    samples = []
    with open(INPUT) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} total samples")
    
    by_type = defaultdict(list)
    for s in samples:
        if s["task_type"] in WEAK_TASKS:
            by_type[s["task_type"]].append(s)
    
    filtered = []
    for task_type in sorted(by_type.keys()):
        task_samples = by_type[task_type]
        # For code_debug: prefer multi-turn
        if task_type == "code_debug":
            task_samples.sort(key=lambda x: (-count_assistant_turns(x), -x.get("score", 0)))
        else:
            task_samples.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        selected = task_samples[:MAX_PER_TYPE]
        filtered.extend(selected)
        multi = sum(1 for s in selected if count_assistant_turns(s) >= 1)
        print(f"  {task_type}: {len(selected)}/{len(task_samples)} ({multi} multi-turn)")
    
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for s in filtered:
            f.write(json.dumps(s) + "\n")
    
    print(f"\nSaved {len(filtered)} samples to {OUTPUT}")

if __name__ == "__main__":
    main()
