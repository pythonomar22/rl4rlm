#!/usr/bin/env python3
"""
Combine and filter trajectories from multiple collection runs into SFT training data.

This is the bridge between trajectory collection and SFT training:
1. Loads all_trajectories.json from each collection run
2. Applies quality filters (correct, terminated, reasonable turn count)
3. Converts to per-turn SFT samples via filter_trajectories.py
4. Outputs a single JSONL file ready for sft_tinker.py

Usage:
    uv run python scripts/prepare_sft_data.py \
        data/trajectories/star_r1_35b_*/all_trajectories.json \
        data/trajectories/star_r1_docclassify_*/all_trajectories.json \
        data/trajectories/star_r1_mniah_*/all_trajectories.json \
        --output data/filtered/sft_star_r1_35b.jsonl

    # Or with glob:
    uv run python scripts/prepare_sft_data.py \
        data/trajectories/star_r1_*/all_trajectories.json \
        --output data/filtered/sft_star_r1_35b.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.filter_trajectories import filter_trajectories, trajectory_to_sft_samples
from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(input_patterns: list[str]) -> list[dict]:
    """Load trajectories from multiple JSON files (supports glob patterns)."""
    all_trajs = []
    seen_files = set()

    for pattern in input_patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            logger.warning(f"No files match pattern: {pattern}")
            continue

        for filepath in matches:
            if filepath in seen_files:
                continue
            seen_files.add(filepath)

            try:
                with open(filepath) as f:
                    trajs = json.load(f)
                logger.info(f"Loaded {len(trajs)} trajectories from {filepath}")
                all_trajs.extend(trajs)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")

    return all_trajs


def analyze_trajectories(trajectories: list[dict]) -> dict:
    """Print detailed analysis of trajectory data."""
    stats = {
        "total": len(trajectories),
        "by_type": {},
        "by_score": {"correct": 0, "partial": 0, "zero": 0},
        "by_turns": {},
    }

    for t in trajectories:
        task_type = t.get("task_type", "unknown")
        score = t.get("score", 0)
        n_turns = len(t.get("turns", []))
        terminated = t.get("terminated", False)

        # By type
        if task_type not in stats["by_type"]:
            stats["by_type"][task_type] = {"total": 0, "correct": 0, "terminated": 0}
        stats["by_type"][task_type]["total"] += 1
        if score > 0:
            stats["by_type"][task_type]["correct"] += 1
        if terminated:
            stats["by_type"][task_type]["terminated"] += 1

        # By score
        if score >= 1.0:
            stats["by_score"]["correct"] += 1
        elif score > 0:
            stats["by_score"]["partial"] += 1
        else:
            stats["by_score"]["zero"] += 1

        # By turns
        turn_bucket = str(min(n_turns, 8))
        stats["by_turns"][turn_bucket] = stats["by_turns"].get(turn_bucket, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Combine and filter trajectories for SFT")
    parser.add_argument("inputs", nargs="+", help="Input trajectory JSON files (glob patterns OK)")
    parser.add_argument("--output", default="data/filtered/sft_star_r1_35b.jsonl",
                        help="Output SFT JSONL path")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-errors", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum score threshold (0 = any correct, 0.5 = above 50%)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Include partially correct trajectories (score > 0 but < 1)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze trajectories, don't produce SFT data")
    args = parser.parse_args()

    # Load all trajectories
    trajectories = load_trajectories(args.inputs)
    if not trajectories:
        logger.error("No trajectories loaded!")
        return

    # Analyze
    stats = analyze_trajectories(trajectories)
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAJECTORY ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"  Total trajectories: {stats['total']}")
    logger.info(f"  Correct (score=1.0): {stats['by_score']['correct']}")
    logger.info(f"  Partial (0<score<1): {stats['by_score']['partial']}")
    logger.info(f"  Zero (score=0): {stats['by_score']['zero']}")
    logger.info(f"\n  By task type:")
    for task_type, info in stats["by_type"].items():
        pct = info["correct"] / info["total"] * 100 if info["total"] > 0 else 0
        logger.info(f"    {task_type}: {info['correct']}/{info['total']} correct ({pct:.1f}%), "
                     f"{info['terminated']}/{info['total']} terminated")
    logger.info(f"\n  By turn count: {dict(sorted(stats['by_turns'].items()))}")

    if args.analyze_only:
        return

    # Apply score threshold
    min_score_filter = args.min_score if args.min_score > 0 else 0.001  # Default: any positive score
    if not args.include_partial:
        # For NIAH tasks, only keep score=1.0 (exact match)
        # For doc_classify and multi_niah, keep partial scores too
        filtered = []
        for t in trajectories:
            score = t.get("score", 0)
            task_type = t.get("task_type", "unknown")
            if task_type == "niah":
                if score >= 1.0:
                    filtered.append(t)
            else:
                if score >= min_score_filter:
                    filtered.append(t)
    else:
        filtered = [t for t in trajectories if t.get("score", 0) >= min_score_filter]

    logger.info(f"\n  After score filtering: {len(filtered)} trajectories")

    # Run through the filter pipeline
    system_prompt = QWEN35_35B_SYSTEM_PROMPT
    samples, filter_stats = filter_trajectories(
        filtered,
        system_prompt=system_prompt,
        max_turns=args.max_turns,
        max_errors=args.max_errors,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"FILTER RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Input to filter: {filter_stats['total']}")
    logger.info(f"  Passed filter: {filter_stats['passed_filter']}")
    logger.info(f"  SFT samples: {filter_stats['total_samples']}")
    logger.info(f"  Removed: {filter_stats['removed_reasons']}")

    if not samples:
        logger.error("No SFT samples produced!")
        return

    # Add task_type to each sample for analysis
    task_type_counts = {}
    for sample in samples:
        task_id = sample.get("task_id", "unknown")
        # Try to find the task type from the original trajectory
        task_type = "unknown"
        for t in filtered:
            if t.get("task_id") == task_id:
                task_type = t.get("task_type", "unknown")
                break
        sample["task_type"] = task_type
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    logger.info(f"\n  SFT samples by task type: {task_type_counts}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, default=str) + "\n")

    # Save stats
    all_stats = {
        "trajectory_stats": stats,
        "filter_stats": filter_stats,
        "task_type_counts": task_type_counts,
        "n_input_trajectories": len(trajectories),
        "n_filtered_trajectories": len(filtered),
        "n_sft_samples": len(samples),
        "args": vars(args),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\n  Saved {len(samples)} SFT samples to: {output_path}")
    logger.info(f"  Stats saved to: {stats_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
