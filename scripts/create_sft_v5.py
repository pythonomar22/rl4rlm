#!/usr/bin/env python3
"""Create SFT V5: multi-turn filtered dataset.

Key change: For tasks that need multi-step reasoning (multi_hop_qa, notebook_qa, 
hard_multi_hop), ONLY include multi-turn trajectories (n_assistant >= 2 in messages).
This prevents the model from learning single-turn shortcuts that destroy multi-hop ability.
"""
import json
from collections import defaultdict
from pathlib import Path

INPUT = Path("/root/rlm/data/sft/sft_all_aggregated.jsonl")
OUTPUT = Path("/root/rlm/data/sft/sft_v5_multiturn_filtered.jsonl")

# Tasks that REQUIRE multi-turn trajectories (single-turn shortcuts are harmful)
MULTI_TURN_ONLY_TASKS = {"multi_hop_qa", "notebook_qa", "hard_multi_hop"}

# Max samples per task type
MAX_PER_TYPE = 80

def count_assistant_turns(sample):
    """Count assistant turns in messages (excluding completion)."""
    return sum(1 for m in sample["messages"] if m["role"] == "assistant")

def main():
    # Load all samples
    samples = []
    with open(INPUT) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {INPUT}")
    
    # Group by task type
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["task_type"]].append(s)
    
    # Apply filtering
    filtered = []
    stats = {}
    
    for task_type in sorted(by_type.keys()):
        task_samples = by_type[task_type]
        
        if task_type in MULTI_TURN_ONLY_TASKS:
            # ONLY keep multi-turn samples
            multi_turn = [s for s in task_samples if count_assistant_turns(s) >= 1]
            # count_assistant_turns counts turns in messages (before completion)
            # n_assistant=0 means single-turn (only system+user, then completion)
            # n_assistant>=1 means multi-turn (at least one prior assistant turn)
            multi_turn_actual = [s for s in task_samples if count_assistant_turns(s) >= 1]
            single_turn = [s for s in task_samples if count_assistant_turns(s) == 0]
            
            selected = multi_turn_actual[:MAX_PER_TYPE]
            stats[task_type] = {
                "total": len(task_samples),
                "single_filtered_out": len(single_turn),
                "multi_kept": len(selected),
            }
        else:
            # Keep all, sort by score descending, cap at MAX_PER_TYPE
            task_samples.sort(key=lambda x: x.get("score", 0), reverse=True)
            selected = task_samples[:MAX_PER_TYPE]
            stats[task_type] = {
                "total": len(task_samples),
                "selected": len(selected),
            }
        
        filtered.extend(selected)
    
    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for s in filtered:
            f.write(json.dumps(s) + "\n")
    
    print(f"\nSaved {len(filtered)} samples to {OUTPUT}")
    print("\nDataset composition:")
    for task_type in sorted(stats.keys()):
        s = stats[task_type]
        if task_type in MULTI_TURN_ONLY_TASKS:
            print(f"  {task_type}: {s['multi_kept']} multi-turn (filtered {s['single_filtered_out']} single-turn from {s['total']})")
        else:
            print(f"  {task_type}: {s['selected']}/{s['total']}")
    
    # Summary
    total_by_type = defaultdict(int)
    for s in filtered:
        total_by_type[s["task_type"]] += 1
    print(f"\nTotal: {len(filtered)} samples across {len(total_by_type)} task types")

if __name__ == "__main__":
    main()
