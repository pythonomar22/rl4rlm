#!/usr/bin/env python3
"""
Collect RLM trajectories for SFT training.

Two modes:
1. Teacher mode: Use a larger model (e.g., Qwen3.5-9B) to generate trajectories
2. Self-bootstrap (STaR): Use the target model itself, filter correct ones

Trajectories are saved to data/trajectories/ and filtered to data/filtered/.

Usage:
    # Self-bootstrap with 1.7B
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/collect_trajectories.py \
        --model Qwen/Qwen3-1.7B \
        --n-tasks 50 \
        --trajectories-per-task 3

    # Teacher mode with 4B
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/collect_trajectories.py \
        --model Qwen/Qwen3-4B \
        --n-tasks 50 \
        --trajectories-per-task 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("6", "7", "6,7"), \
    "CUDA_VISIBLE_DEVICES must be set to 6, 7, or 6,7"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from scaffold.llm_query import HFModel
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("collect")


def collect_from_niah(
    model: HFModel,
    system_prompt: str,
    n_tasks: int = 50,
    trajectories_per_task: int = 3,
    max_iterations: int = 8,
    doc_lengths: list[int] | None = None,
) -> list[dict]:
    """Collect trajectories from NIAH tasks."""
    tasks = generate_niah_suite(n_tasks=n_tasks, doc_lengths=doc_lengths)

    all_trajectories = []
    correct_count = 0
    total_count = 0

    for task in tqdm(tasks, desc="Collecting"):
        for attempt in range(trajectories_per_task):
            traj = rlm(
                prompt=task.prompt,
                model=model,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                verbose=False,
            )

            score = score_niah(traj.answer, task.expected_answer)
            total_count += 1

            traj_dict = trajectory_to_dict(traj)
            traj_dict["task_id"] = task.task_id
            traj_dict["expected_answer"] = task.expected_answer
            traj_dict["score"] = score
            traj_dict["attempt"] = attempt

            all_trajectories.append(traj_dict)

            if score > 0:
                correct_count += 1

            logger.info(
                f"  {task.task_id} attempt {attempt}: "
                f"score={score} terminated={traj.terminated} "
                f"turns={len(traj.turns)} time={traj.total_time:.1f}s"
            )

    logger.info(f"\nCollection complete: {correct_count}/{total_count} correct "
                f"({correct_count / total_count * 100:.1f}%)")

    return all_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--trajectories-per-task", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--doc-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    assert torch.cuda.device_count() <= 2
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path("data/trajectories") / f"{model_short}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    t0 = time.time()
    model = HFModel(
        model_name=args.model,
        device="cuda:0",
        max_new_tokens=1024,
        temperature=0.8,  # Slightly higher for diversity
        do_sample=True,
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Collect
    t0 = time.time()
    trajectories = collect_from_niah(
        model=model,
        system_prompt=QWEN_2B_SYSTEM_PROMPT,
        n_tasks=args.n_tasks,
        trajectories_per_task=args.trajectories_per_task,
        max_iterations=args.max_iterations,
        doc_lengths=args.doc_lengths,
    )
    collection_time = time.time() - t0

    # Save all trajectories
    with open(output_dir / "all_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2, default=str)

    # Save correct trajectories separately
    correct = [t for t in trajectories if t["score"] > 0]
    with open(output_dir / "correct_trajectories.json", "w") as f:
        json.dump(correct, f, indent=2, default=str)

    # Save config
    config = {
        "model": args.model,
        "n_tasks": args.n_tasks,
        "trajectories_per_task": args.trajectories_per_task,
        "total_trajectories": len(trajectories),
        "correct_trajectories": len(correct),
        "accuracy": len(correct) / len(trajectories) if trajectories else 0,
        "collection_time": collection_time,
        "model_stats": model.total_stats(),
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"COLLECTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total: {len(trajectories)}")
    logger.info(f"Correct: {len(correct)} ({len(correct) / len(trajectories) * 100:.1f}%)")
    logger.info(f"Time: {collection_time:.0f}s")
    logger.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
