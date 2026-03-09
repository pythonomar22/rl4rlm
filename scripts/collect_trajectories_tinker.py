#!/usr/bin/env python3
"""
Collect RLM trajectories using Tinker API for STaR self-bootstrap.

Generates trajectories by running the RLM scaffold on benchmark tasks
with a Tinker model (base or fine-tuned). Filters correct trajectories
for use in SFT training.

Usage:
    # Base model trajectories (STaR round 1)
    uv run python scripts/collect_trajectories_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --tasks niah \
        --n-tasks 150 \
        --experiment-name star_r1_35b

    # From fine-tuned checkpoint (STaR round 2+)
    uv run python scripts/collect_trajectories_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --model-path tinker://run-id/weights/sft-final \
        --tasks all \
        --n-tasks 100 \
        --experiment-name star_r2_35b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from scaffold.llm_query import TinkerModel
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from eval.benchmarks.doc_classify import generate_doc_classify_suite, score_doc_classify
from training.rewards import composite_reward, binary_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_niah_trajectories(
    model: TinkerModel,
    system_prompt: str,
    n_tasks: int = 150,
    trajectories_per_task: int = 1,
    max_iterations: int = 8,
    seed_offset: int = 0,
    doc_lengths: list[int] | None = None,
) -> list[dict]:
    """Collect trajectories on NIAH tasks."""
    doc_lengths = doc_lengths or [5000, 10000, 20000, 50000, 100000]
    tasks = generate_niah_suite(
        n_tasks=n_tasks, doc_lengths=doc_lengths, seed_offset=seed_offset
    )
    all_trajectories = []

    for i, task in enumerate(tasks):
        for k in range(trajectories_per_task):
            logger.info(
                f"NIAH [{i+1}/{len(tasks)}] k={k+1} | "
                f"len={task.doc_length} pos={task.needle_position:.2f} | "
                f"expected={task.expected_answer}"
            )
            t0 = time.time()
            try:
                traj = rlm(
                    prompt=task.prompt,
                    model=model,
                    system_prompt=system_prompt,
                    max_iterations=max_iterations,
                )
                elapsed = time.time() - t0

                score = score_niah(traj.answer, task.expected_answer)
                traj_dict = trajectory_to_dict(traj)
                traj_dict["task_id"] = task.task_id
                traj_dict["task_type"] = "niah"
                traj_dict["expected_answer"] = task.expected_answer
                traj_dict["score"] = score
                traj_dict["doc_length"] = task.doc_length
                traj_dict["needle_position"] = task.needle_position
                traj_dict["collection_time"] = elapsed

                all_trajectories.append(traj_dict)
                status = "CORRECT" if score > 0 else "WRONG"
                logger.info(f"  {status} ({score:.2f}) | answer={str(traj.answer)[:80]} | {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                all_trajectories.append({
                    "task_id": task.task_id,
                    "task_type": "niah",
                    "score": 0,
                    "error": str(e),
                })

    return all_trajectories


def collect_multi_niah_trajectories(
    model: TinkerModel,
    system_prompt: str,
    n_tasks: int = 24,
    max_iterations: int = 10,
    seed_offset: int = 50000,
) -> list[dict]:
    """Collect trajectories on multi-needle NIAH tasks."""
    tasks = generate_multi_niah_suite(n_tasks=n_tasks, seed_offset=seed_offset)
    all_trajectories = []

    for i, task in enumerate(tasks):
        logger.info(
            f"Multi-NIAH [{i+1}/{len(tasks)}] | "
            f"{task.n_needles} needles in {task.doc_length} chars"
        )
        t0 = time.time()
        try:
            traj = rlm(
                prompt=task.prompt,
                model=model,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
            )
            elapsed = time.time() - t0

            scores = score_multi_niah(traj.answer, task.expected_answers)
            traj_dict = trajectory_to_dict(traj)
            traj_dict["task_id"] = task.task_id
            traj_dict["task_type"] = "multi_niah"
            traj_dict["expected_answers"] = task.expected_answers
            traj_dict["score"] = scores["recall"]
            traj_dict["n_needles"] = task.n_needles
            traj_dict["doc_length"] = task.doc_length
            traj_dict["collection_time"] = elapsed

            all_trajectories.append(traj_dict)
            logger.info(
                f"  Found {scores['found']}/{scores['total']} (recall={scores['recall']:.2f}) | {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_trajectories.append({
                "task_id": task.task_id,
                "task_type": "multi_niah",
                "score": 0,
                "error": str(e),
            })

    return all_trajectories


def collect_doc_classify_trajectories(
    model: TinkerModel,
    system_prompt: str,
    n_tasks: int = 20,
    max_iterations: int = 10,
    seed_offset: int = 70000,
) -> list[dict]:
    """Collect trajectories on document classification tasks."""
    tasks = generate_doc_classify_suite(n_tasks=n_tasks, seed_offset=seed_offset)
    all_trajectories = []

    for i, task in enumerate(tasks):
        logger.info(
            f"DocClassify [{i+1}/{len(tasks)}] | {task.n_docs} docs, {task.doc_length} chars"
        )
        t0 = time.time()
        try:
            traj = rlm(
                prompt=task.prompt,
                model=model,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
            )
            elapsed = time.time() - t0

            scores = score_doc_classify(traj.answer, task.expected_labels)
            traj_dict = trajectory_to_dict(traj)
            traj_dict["task_id"] = task.task_id
            traj_dict["task_type"] = "doc_classify"
            traj_dict["expected_labels"] = task.expected_labels
            traj_dict["score"] = scores["accuracy"]
            traj_dict["n_docs"] = task.n_docs
            traj_dict["doc_length"] = task.doc_length
            traj_dict["collection_time"] = elapsed

            all_trajectories.append(traj_dict)
            logger.info(
                f"  Accuracy: {scores['correct']}/{scores['total']} ({scores['accuracy']:.1%}) | {elapsed:.1f}s"
            )

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            all_trajectories.append({
                "task_id": task.task_id,
                "task_type": "doc_classify",
                "score": 0,
                "error": str(e),
            })

    return all_trajectories


def main():
    parser = argparse.ArgumentParser(description="Collect RLM trajectories via Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--model-path", default=None,
                        help="Tinker model path for fine-tuned checkpoint")
    parser.add_argument("--tasks", default="niah",
                        choices=["niah", "multi_niah", "doc_classify", "all"])
    parser.add_argument("--n-tasks", type=int, default=150)
    parser.add_argument("--trajectories-per-task", type=int, default=1,
                        help="K trajectories per NIAH task")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--experiment-name", default="trajectories")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum score to include in correct trajectories")
    args = parser.parse_args()

    # Setup model
    model = TinkerModel(
        model_name=args.model,
        model_path=args.model_path,
        max_new_tokens=2048,
        temperature=0.7,
    )

    system_prompt = QWEN35_35B_SYSTEM_PROMPT

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("data/trajectories") / f"{args.experiment_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_trajectories = []
    t0 = time.time()

    # Collect trajectories
    task_types = ["niah", "multi_niah", "doc_classify"] if args.tasks == "all" else [args.tasks]

    for task_type in task_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting {task_type} trajectories")
        logger.info(f"{'='*60}")

        if task_type == "niah":
            trajs = collect_niah_trajectories(
                model, system_prompt,
                n_tasks=args.n_tasks,
                trajectories_per_task=args.trajectories_per_task,
                max_iterations=args.max_iterations,
            )
        elif task_type == "multi_niah":
            trajs = collect_multi_niah_trajectories(
                model, system_prompt,
                n_tasks=min(args.n_tasks, 24),
                max_iterations=args.max_iterations,
            )
        elif task_type == "doc_classify":
            trajs = collect_doc_classify_trajectories(
                model, system_prompt,
                n_tasks=min(args.n_tasks, 20),
                max_iterations=args.max_iterations,
            )

        all_trajectories.extend(trajs)

    total_time = time.time() - t0

    # Filter correct trajectories
    correct = [t for t in all_trajectories if t.get("score", 0) > args.min_score]

    # Save results
    with open(save_dir / "all_trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2, default=str)

    with open(save_dir / "correct_trajectories.json", "w") as f:
        json.dump(correct, f, indent=2, default=str)

    # Stats
    stats = {
        "model": args.model,
        "model_path": args.model_path,
        "task_types": task_types,
        "n_tasks_requested": args.n_tasks,
        "total_trajectories": len(all_trajectories),
        "correct_trajectories": len(correct),
        "success_rate": len(correct) / len(all_trajectories) if all_trajectories else 0,
        "total_time": total_time,
        "model_stats": model.total_stats(),
        "timestamp": timestamp,
        "per_type": {},
    }

    for task_type in task_types:
        type_trajs = [t for t in all_trajectories if t.get("task_type") == task_type]
        type_correct = [t for t in type_trajs if t.get("score", 0) > args.min_score]
        stats["per_type"][task_type] = {
            "total": len(type_trajs),
            "correct": len(type_correct),
            "success_rate": len(type_correct) / len(type_trajs) if type_trajs else 0,
            "avg_score": sum(t.get("score", 0) for t in type_trajs) / len(type_trajs) if type_trajs else 0,
        }

    with open(save_dir / "config.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAJECTORY COLLECTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total: {len(all_trajectories)} | Correct: {len(correct)} ({stats['success_rate']:.1%})")
    for task_type, info in stats["per_type"].items():
        logger.info(f"  {task_type}: {info['correct']}/{info['total']} ({info['success_rate']:.1%})")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Saved to: {save_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
