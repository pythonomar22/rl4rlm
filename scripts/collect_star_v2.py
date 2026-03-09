#!/usr/bin/env python3
"""
STaR-v2: Collect trajectories from SFT model on multi-NIAH and doc-classify tasks.

Uses the SFT-v2 checkpoint to generate trajectories on harder tasks,
filters correct ones, and combines with original NIAH trajectories
for a second round of SFT (SFT-v3).

Usage:
    CUDA_VISIBLE_DEVICES=6 uv run python scripts/collect_star_v2.py \
        --base-model Qwen/Qwen3-1.7B \
        --adapter data/sft/lora_v2/final \
        --n-tasks 30 \
        --output data/trajectories/star_v2
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

from training.rl import RLModel
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from eval.benchmarks.doc_classify import generate_doc_classify_suite, score_doc_classify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("star_v2")


def collect_multi_niah(model, system_prompt, n_tasks=24, max_iter=10):
    """Collect trajectories from multi-NIAH tasks."""
    tasks = generate_multi_niah_suite(n_tasks=n_tasks, seed_offset=30000)
    trajectories = []
    correct = 0

    for task in tqdm(tasks, desc="Multi-NIAH"):
        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iter,
            verbose=False,
        )

        scores = score_multi_niah(traj.answer, task.expected_answers)
        traj_dict = trajectory_to_dict(traj)
        traj_dict["task_id"] = task.task_id
        traj_dict["task_type"] = "multi_niah"
        traj_dict["expected_answers"] = task.expected_answers
        traj_dict["recall"] = scores["recall"]
        traj_dict["score"] = scores["recall"]
        trajectories.append(traj_dict)

        if scores["recall"] > 0.5:
            correct += 1

        logger.info(
            f"  {task.task_id}: recall={scores['recall']:.2f} "
            f"({scores['found']}/{scores['total']}) "
            f"turns={len(traj.turns)} time={traj.total_time:.1f}s"
        )

    logger.info(f"Multi-NIAH: {correct}/{len(tasks)} with recall > 0.5")
    return trajectories


def collect_doc_classify(model, system_prompt, n_tasks=20, max_iter=10):
    """Collect trajectories from doc-classify tasks."""
    tasks = generate_doc_classify_suite(n_tasks=n_tasks, seed_offset=30000)
    trajectories = []
    correct = 0

    for task in tqdm(tasks, desc="DocClassify"):
        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iter,
            verbose=False,
        )

        scores = score_doc_classify(traj.answer, task.expected_labels)
        traj_dict = trajectory_to_dict(traj)
        traj_dict["task_id"] = task.task_id
        traj_dict["task_type"] = "doc_classify"
        traj_dict["expected_labels"] = task.expected_labels
        traj_dict["accuracy"] = scores["accuracy"]
        traj_dict["score"] = scores["accuracy"]
        trajectories.append(traj_dict)

        if scores["accuracy"] > 0.5:
            correct += 1

        logger.info(
            f"  {task.task_id}: acc={scores['accuracy']:.2f} "
            f"({scores['correct']}/{scores['total']}) "
            f"turns={len(traj.turns)} time={traj.total_time:.1f}s"
        )

    logger.info(f"DocClassify: {correct}/{len(tasks)} with acc > 0.5")
    return trajectories


def collect_niah(model, system_prompt, n_tasks=30, max_iter=8):
    """Collect trajectories from NIAH tasks (harder ones: 50K, 100K)."""
    tasks = generate_niah_suite(
        n_tasks=n_tasks,
        doc_lengths=[50000, 100000],
        seed_offset=30000,
    )
    trajectories = []
    correct = 0

    for task in tqdm(tasks, desc="NIAH (hard)"):
        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iter,
            verbose=False,
        )

        score = score_niah(traj.answer, task.expected_answer)
        traj_dict = trajectory_to_dict(traj)
        traj_dict["task_id"] = task.task_id
        traj_dict["task_type"] = "niah"
        traj_dict["expected_answer"] = task.expected_answer
        traj_dict["score"] = score
        trajectories.append(traj_dict)

        if score > 0:
            correct += 1

        logger.info(
            f"  {task.task_id}: score={score} "
            f"turns={len(traj.turns)} time={traj.total_time:.1f}s"
        )

    logger.info(f"NIAH (hard): {correct}/{len(tasks)} correct")
    return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", required=True,
                        help="Path to SFT LoRA adapter")
    parser.add_argument("--n-tasks", type=int, default=30,
                        help="Tasks per benchmark type")
    parser.add_argument("--output", default="data/trajectories/star_v2")
    args = parser.parse_args()

    assert torch.cuda.device_count() <= 2

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SFT model
    logger.info(f"Loading SFT model: {args.base_model} + {args.adapter}")
    model = RLModel(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        device="cuda:0",
        max_new_tokens=1024,
    )

    t0 = time.time()
    all_trajectories = []

    # Collect from all task types
    niah_trajs = collect_niah(model, QWEN_2B_SYSTEM_PROMPT, n_tasks=args.n_tasks)
    all_trajectories.extend(niah_trajs)

    mniah_trajs = collect_multi_niah(model, QWEN_2B_SYSTEM_PROMPT, n_tasks=min(24, args.n_tasks))
    all_trajectories.extend(mniah_trajs)

    doc_trajs = collect_doc_classify(model, QWEN_2B_SYSTEM_PROMPT, n_tasks=min(20, args.n_tasks))
    all_trajectories.extend(doc_trajs)

    collection_time = time.time() - t0

    # Save all
    with open(output_dir / "all_trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2, default=str)

    # Filter: score > 0.5 for all task types
    correct = [t for t in all_trajectories if t["score"] > 0.5]
    with open(output_dir / "correct_trajectories.json", "w") as f:
        json.dump(correct, f, indent=2, default=str)

    # Save config
    config = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "total": len(all_trajectories),
        "correct": len(correct),
        "by_type": {
            "niah": {"total": len(niah_trajs), "correct": sum(1 for t in niah_trajs if t["score"] > 0.5)},
            "multi_niah": {"total": len(mniah_trajs), "correct": sum(1 for t in mniah_trajs if t["score"] > 0.5)},
            "doc_classify": {"total": len(doc_trajs), "correct": sum(1 for t in doc_trajs if t["score"] > 0.5)},
        },
        "collection_time": collection_time,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"STaR-v2 COLLECTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total: {len(all_trajectories)}")
    logger.info(f"Correct (score > 0.5): {len(correct)} ({len(correct)/len(all_trajectories)*100:.1f}%)")
    for task_type, stats in config["by_type"].items():
        logger.info(f"  {task_type}: {stats['correct']}/{stats['total']}")
    logger.info(f"Time: {collection_time:.0f}s ({collection_time/3600:.2f} GPU-hours)")
    logger.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
