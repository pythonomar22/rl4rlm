#!/usr/bin/env python3
"""Create SFT V8 dataset: balanced, high-quality, all 14 task types.

Strategy:
1. Pull from all available sources (aggregated + balanced + merged)
2. Deduplicate by prompt hash
3. Balance: cap each type at 100, upsample rare types to minimum 30
4. Shuffle with fixed seed
"""

import json
import hashlib
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/root/rlm/data/sft")
OUTPUT = DATA_DIR / "sft_v8_balanced.jsonl"

# Target: 100 max per type, 30 min (upsample if needed)
MAX_PER_TYPE = 100
MIN_PER_TYPE = 30

def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def prompt_hash(sample):
    """Hash by messages content to dedup."""
    msgs = sample.get("messages", [])
    key = json.dumps(msgs[:2], sort_keys=True)  # system + first user
    return hashlib.md5(key.encode()).hexdigest()

def main():
    # Load all sources
    all_samples = []
    sources = [
        DATA_DIR / "sft_all_aggregated.jsonl",
        DATA_DIR / "sft_balanced_v3.jsonl",
        DATA_DIR / "sft_merged_eval.jsonl",
    ]

    for src in sources:
        if src.exists():
            samples = load_jsonl(src)
            print(f"Loaded {len(samples)} from {src.name}")
            all_samples.extend(samples)

    # Dedup by prompt hash
    seen = set()
    deduped = []
    for s in all_samples:
        h = prompt_hash(s)
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    print(f"\nAfter dedup: {len(deduped)} (removed {len(all_samples) - len(deduped)})")

    # Group by task type
    by_type = defaultdict(list)
    for s in deduped:
        tt = s.get("task_type", "unknown")
        by_type[tt].append(s)

    print("\nPer-type counts (raw):")
    for tt, samples in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {tt}: {len(samples)}")

    # Balance: cap at MAX, upsample to MIN
    rng = random.Random(42)
    final = []
    for tt, samples in by_type.items():
        if len(samples) > MAX_PER_TYPE:
            # Downsample
            selected = rng.sample(samples, MAX_PER_TYPE)
        elif len(samples) < MIN_PER_TYPE:
            # Upsample by repeating
            selected = list(samples)
            while len(selected) < MIN_PER_TYPE:
                selected.append(rng.choice(samples))
        else:
            selected = list(samples)
        final.extend(selected)

    # Shuffle
    rng.shuffle(final)

    # Write
    with open(OUTPUT, "w") as f:
        for s in final:
            f.write(json.dumps(s) + "\n")

    print(f"\nFinal dataset: {len(final)} samples → {OUTPUT}")

    # Final composition
    final_types = defaultdict(int)
    for s in final:
        final_types[s.get("task_type", "unknown")] += 1
    print("\nFinal per-type counts:")
    for tt, count in sorted(final_types.items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")

if __name__ == "__main__":
    main()
