#!/usr/bin/env python3
"""
Mine correct trajectories from evaluation result directories.

Each eval run saves per-task trajectories in results/*/timestamp/benchmark/trajectories/.
This script collects all correct trajectories (score > 0) across all eval runs
and converts them to SFT-ready format.

Usage:
    uv run python scripts/mine_eval_trajectories.py \
        --output data/trajectories/mined_from_evals/correct_trajectories.json

    # Only from V10-s40 evals
    uv run python scripts/mine_eval_trajectories.py \
        --filter v10 \
        --output data/trajectories/mined_v10_evals/correct_trajectories.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ALL_BENCHMARKS = [
    "niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug",
    "multi_hop_qa", "notebook_qa", "hard_niah", "verbatim_copy",
    "hard_multi_hop", "event_counting", "cross_doc_compare", "key_value_retrieval",
    "oolong",
]


def mine_eval_results(
    results_dir: str = "results",
    filter_pattern: str | None = None,
) -> list[dict]:
    """Mine correct trajectories from eval result directories."""
    all_correct = []
    results_base = Path(results_dir)

    # Find all result directories
    eval_dirs = sorted(results_base.iterdir())

    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            continue
        if filter_pattern and filter_pattern not in eval_dir.name:
            continue

        # Find timestamp subdirectories
        for timestamp_dir in eval_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue
            # Look for benchmark subdirectories
            for benchmark in ALL_BENCHMARKS:
                bench_dir = timestamp_dir / benchmark
                if not bench_dir.is_dir():
                    continue

                traj_dir = bench_dir / "trajectories"
                eval_results_file = bench_dir / "eval_results.json"

                if not traj_dir.is_dir() or not eval_results_file.exists():
                    continue

                # Load eval results for scores
                try:
                    with open(eval_results_file) as f:
                        eval_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                per_task = eval_data.get("per_task", [])
                if not per_task:
                    continue

                # Build score lookup by task index
                # Different benchmarks use different score fields:
                # - Most use "score"
                # - doc_classify uses "accuracy"
                # - multi_niah uses "recall"
                task_scores = {}
                for i, task_result in enumerate(per_task):
                    task_id = task_result.get("task_id", f"task_{i}")
                    # Try multiple score fields
                    score = task_result.get("score", None)
                    if score is None:
                        score = task_result.get("accuracy", None)
                    if score is None:
                        score = task_result.get("recall", None)
                    if score is None:
                        score = 0
                    task_scores[i] = {
                        "score": float(score),
                        "task_id": task_id,
                        "expected": task_result.get("expected", ""),
                    }

                # Load trajectory files
                traj_files = sorted(traj_dir.glob("trajectory_*.json"))
                for traj_file in traj_files:
                    try:
                        idx = int(traj_file.stem.split("_")[-1])
                    except ValueError:
                        continue

                    if idx not in task_scores:
                        continue

                    task_info = task_scores[idx]
                    if task_info["score"] <= 0:
                        continue  # Only correct trajectories

                    try:
                        with open(traj_file) as f:
                            traj = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue

                    # Add metadata
                    traj["score"] = task_info["score"]
                    traj["task_id"] = task_info["task_id"]
                    traj["task_type"] = benchmark
                    traj["expected_answer"] = task_info["expected"]
                    traj["_source_eval"] = eval_dir.name
                    traj["_source_timestamp"] = timestamp_dir.name

                    all_correct.append(traj)

    return all_correct


def main():
    parser = argparse.ArgumentParser(description="Mine eval trajectories for SFT data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--filter", default=None,
                        help="Only include eval dirs matching this pattern")
    parser.add_argument("--output", default="data/trajectories/mined_from_evals/correct_trajectories.json")
    args = parser.parse_args()

    logger.info("=== Mining Eval Results for Training Data ===\n")

    trajectories = mine_eval_results(args.results_dir, args.filter)

    # Stats
    by_type = defaultdict(int)
    by_source = defaultdict(int)
    for t in trajectories:
        by_type[t.get("task_type", "unknown")] += 1
        by_source[t.get("_source_eval", "unknown")] += 1

    logger.info(f"\nTotal correct trajectories mined: {len(trajectories)}")
    logger.info(f"\nBy benchmark:")
    for k in sorted(by_type.keys()):
        logger.info(f"  {k}: {by_type[k]}")

    logger.info(f"\nBy eval source (top 20):")
    for k, v in sorted(by_source.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {k}: {v}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2, default=str)
    logger.info(f"\nSaved to {output_path}")

    # Also save stats
    stats = {
        "total": len(trajectories),
        "by_type": dict(sorted(by_type.items())),
        "by_source": dict(sorted(by_source.items(), key=lambda x: -x[1])),
    }
    stats_path = output_path.parent / "mining_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
