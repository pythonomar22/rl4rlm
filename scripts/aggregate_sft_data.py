#!/usr/bin/env python3
"""
Aggregate all correct trajectories into a large, balanced RS-SFT dataset.

Reads from:
1. All trajectory directories (data/trajectories/*/correct_trajectories.json)
2. Existing filtered SFT files (data/filtered/*.jsonl)
3. Existing SFT datasets (data/sft/*.jsonl)

Outputs a single balanced JSONL file suitable for RS-SFT training.

Usage:
    uv run python scripts/aggregate_sft_data.py \
        --output data/sft/sft_large_balanced.jsonl \
        --max-per-type 200 \
        --min-per-type 30

    # Include new collection directories
    uv run python scripts/aggregate_sft_data.py \
        --output data/sft/sft_v16_balanced.jsonl \
        --extra-dirs data/trajectories/rs_sft_v10s40_weak_* \
        --max-per-type 200
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
from scripts.filter_trajectories import (
    filter_trajectories,
    trajectory_to_sft_samples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ordered from most specific to least specific to avoid prefix conflicts
TASK_TYPE_PREFIXES = [
    ("hard_multi_hop", "hard_multi_hop"),
    ("hard_niah", "hard_niah"),
    ("multi_niah", "multi_niah"),
    ("multi_hop", "multi_hop_qa"),
    ("cross_doc", "cross_doc_compare"),
    ("key_value", "key_value_retrieval"),
    ("doc_classify", "doc_classify"),
    ("dataframe_qa", "dataframe_qa"),
    ("code_debug", "code_debug"),
    ("notebook_qa", "notebook_qa"),
    ("verbatim_copy", "verbatim_copy"),
    ("event_counting", "event_counting"),
    ("oolong", "oolong"),
    ("niah", "niah"),
]


def infer_task_type(task_id: str) -> str | None:
    """Infer task type from task_id string."""
    for prefix, task_type in TASK_TYPE_PREFIXES:
        if prefix in task_id:
            return task_type
    return None


def load_trajectory_dirs(base_dir: str = "data/trajectories") -> list[dict]:
    """Load correct trajectories from all trajectory directories."""
    all_trajs = []
    base = Path(base_dir)

    for traj_dir in sorted(base.iterdir()):
        if not traj_dir.is_dir():
            continue

        # Skip legacy 1.7B trajectories
        if "1.7B" in traj_dir.name:
            logger.info(f"  Skipping legacy: {traj_dir.name}")
            continue

        correct_file = traj_dir / "correct_trajectories.json"
        all_file = traj_dir / "all_trajectories.json"

        if correct_file.exists():
            with open(correct_file) as f:
                trajs = json.load(f)
            logger.info(f"  {traj_dir.name}: {len(trajs)} correct trajectories")
            for t in trajs:
                t["_source"] = traj_dir.name
            all_trajs.extend(trajs)
        elif all_file.exists():
            with open(all_file) as f:
                trajs = json.load(f)
            correct = [t for t in trajs if t.get("score", 0) > 0]
            logger.info(f"  {traj_dir.name}: {len(correct)}/{len(trajs)} correct")
            for t in correct:
                t["_source"] = traj_dir.name
            all_trajs.extend(correct)

    return all_trajs


def load_extra_dirs(patterns: list[str]) -> list[dict]:
    """Load trajectories from extra directory patterns (glob)."""
    all_trajs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            if not p.is_dir():
                continue
            for fname in ["correct_trajectories.json", "all_trajectories.json"]:
                fpath = p / fname
                if fpath.exists():
                    with open(fpath) as f:
                        trajs = json.load(f)
                    if fname == "all_trajectories.json":
                        trajs = [t for t in trajs if t.get("score", 0) > 0]
                    logger.info(f"  Extra {p.name}: {len(trajs)} correct")
                    for t in trajs:
                        t["_source"] = f"extra_{p.name}"
                    all_trajs.extend(trajs)
                    break
    return all_trajs


def load_existing_sft(paths: list[str]) -> list[dict]:
    """Load existing SFT JSONL files (already in messages+completion format)."""
    all_samples = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        count = 0
        with open(p) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    sample["_source"] = f"sft_{p.stem}"
                    all_samples.append(sample)
                    count += 1
        logger.info(f"  Existing SFT {p.name}: {count} samples")
    return all_samples


def deduplicate_by_content(samples: list[dict]) -> list[dict]:
    """Remove exact-duplicate samples by comparing completion text.
    Keeps diverse trajectories for the same task (different code = different sample)."""
    seen = set()
    unique = []
    for s in samples:
        # Hash by task_type + completion text (the actual code)
        completion = s.get("completion", "")
        task_type = s.get("task_type", "unknown")
        # Use a content-based key: same code for same task type = duplicate
        key = f"{task_type}:{hash(completion)}"

        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def balance_by_type(
    samples: list[dict],
    max_per_type: int = 200,
    min_per_type: int = 30,
) -> list[dict]:
    """Balance samples across task types.

    - Cap each type at max_per_type
    - Oversample types below min_per_type (with replacement)
    """
    import numpy as np

    by_type = defaultdict(list)
    for s in samples:
        task_type = s.get("task_type", "unknown")
        by_type[task_type].append(s)

    logger.info(f"\nPre-balance distribution:")
    for t in sorted(by_type.keys()):
        logger.info(f"  {t}: {len(by_type[t])}")

    rng = np.random.RandomState(42)
    balanced = []

    for task_type in sorted(by_type.keys()):
        type_samples = by_type[task_type]
        n = len(type_samples)

        if n > max_per_type:
            # Downsample: pick best (by score) + random
            type_samples.sort(key=lambda x: x.get("score", 0), reverse=True)
            # Keep top half by score, random sample the rest
            top_half = max_per_type // 2
            selected = type_samples[:top_half]
            remaining = type_samples[top_half:]
            if remaining and max_per_type - top_half > 0:
                extra_idx = rng.choice(len(remaining), min(max_per_type - top_half, len(remaining)), replace=False)
                selected.extend(remaining[i] for i in extra_idx)
            balanced.extend(selected)
        elif n < min_per_type and n > 0:
            # Oversample with replacement
            balanced.extend(type_samples)
            extra_needed = min_per_type - n
            extra_idx = rng.choice(n, extra_needed, replace=True)
            balanced.extend(type_samples[i] for i in extra_idx)
        else:
            balanced.extend(type_samples)

    # Report
    final_by_type = defaultdict(int)
    for s in balanced:
        final_by_type[s.get("task_type", "unknown")] += 1

    logger.info(f"\nPost-balance distribution ({len(balanced)} total):")
    for t in sorted(final_by_type.keys()):
        logger.info(f"  {t}: {final_by_type[t]}")

    return balanced


def main():
    parser = argparse.ArgumentParser(description="Aggregate SFT data from all sources")
    parser.add_argument("--output", default="data/sft/sft_large_balanced.jsonl")
    parser.add_argument("--max-per-type", type=int, default=200,
                        help="Max samples per task type")
    parser.add_argument("--min-per-type", type=int, default=30,
                        help="Min samples per task type (oversample if needed)")
    parser.add_argument("--extra-dirs", nargs="*", default=[],
                        help="Additional trajectory directory glob patterns")
    parser.add_argument("--include-existing-sft", nargs="*", default=[],
                        help="Paths to existing SFT JSONL files to include")
    parser.add_argument("--skip-trajectory-dirs", action="store_true",
                        help="Skip loading from data/trajectories/")
    args = parser.parse_args()

    logger.info("=== Aggregating SFT Data ===\n")

    all_samples = []

    # 1. Load from trajectory directories
    if not args.skip_trajectory_dirs:
        logger.info("Loading trajectory directories...")
        trajs = load_trajectory_dirs()
        if trajs:
            logger.info(f"  Total raw correct trajectories: {len(trajs)}")

            # Build task_type lookup from trajectories before filtering
            task_type_lookup = {}
            for t in trajs:
                tid = t.get("task_id", "")
                tt = t.get("task_type", "")
                if tid and tt:
                    task_type_lookup[tid] = tt

            # Convert to SFT format
            samples, stats = filter_trajectories(
                trajs, QWEN35_35B_SYSTEM_PROMPT, max_turns=8, max_errors=2
            )
            logger.info(f"  After filtering: {stats['passed_filter']} trajs -> {len(samples)} SFT samples")

            # Propagate task_type from trajectory data
            for s in samples:
                tid = s.get("task_id", "")
                if tid in task_type_lookup:
                    s["task_type"] = task_type_lookup[tid]
                elif "task_type" not in s or s.get("task_type") == "unknown":
                    # Infer from task_id
                    inferred = infer_task_type(tid)
                    if inferred:
                        s["task_type"] = inferred
            all_samples.extend(samples)

    # 2. Load extra directories (that aren't already in data/trajectories/)
    if args.extra_dirs:
        logger.info(f"\nLoading extra directories: {args.extra_dirs}")
        extra_trajs = load_extra_dirs(args.extra_dirs)
        if extra_trajs:
            # Build task_type lookup
            extra_type_lookup = {}
            for t in extra_trajs:
                tid = t.get("task_id", "")
                tt = t.get("task_type", "")
                if tid and tt:
                    extra_type_lookup[tid] = tt

            samples, stats = filter_trajectories(
                extra_trajs, QWEN35_35B_SYSTEM_PROMPT, max_turns=8, max_errors=2
            )
            logger.info(f"  Extra: {stats['passed_filter']} trajs -> {len(samples)} SFT samples")
            for s in samples:
                tid = s.get("task_id", "")
                if tid in extra_type_lookup:
                    s["task_type"] = extra_type_lookup[tid]
                elif "task_type" not in s:
                    inferred = infer_task_type(tid)
                    if inferred:
                        s["task_type"] = inferred
            all_samples.extend(samples)

    # 3. Load existing SFT files
    if args.include_existing_sft:
        logger.info(f"\nLoading existing SFT files...")
        existing = load_existing_sft(args.include_existing_sft)
        all_samples.extend(existing)

    # 4. Also load the existing aggregated SFT as a baseline
    aggregated_path = Path("data/sft/sft_all_aggregated.jsonl")
    if aggregated_path.exists() and str(aggregated_path) not in (args.include_existing_sft or []):
        logger.info(f"\nLoading existing aggregated SFT...")
        existing = load_existing_sft([str(aggregated_path)])
        all_samples.extend(existing)

    logger.info(f"\nTotal raw samples before dedup: {len(all_samples)}")

    # 5. Deduplicate
    all_samples = deduplicate_by_content(all_samples)
    logger.info(f"After dedup: {len(all_samples)}")

    # 6. Balance
    balanced = balance_by_type(
        all_samples,
        max_per_type=args.max_per_type,
        min_per_type=args.min_per_type,
    )

    # 7. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(balanced))
    shuffled = [balanced[i] for i in indices]

    with open(output_path, "w") as f:
        for s in shuffled:
            # Clean internal fields
            s.pop("_source", None)
            f.write(json.dumps(s, default=str) + "\n")

    logger.info(f"\nSaved {len(shuffled)} samples to {output_path}")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    final_by_type = defaultdict(int)
    for s in shuffled:
        final_by_type[s.get("task_type", "unknown")] += 1

    stats = {
        "total_samples": len(shuffled),
        "by_type": dict(sorted(final_by_type.items())),
        "max_per_type": args.max_per_type,
        "min_per_type": args.min_per_type,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
