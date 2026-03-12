#!/usr/bin/env python3
"""
Collect RLM trajectories using Tinker API for STaR/RS-SFT training.

Generates trajectories by running the RLM scaffold on benchmark tasks
with a Tinker model (base or fine-tuned). Supports K attempts per task
for rejection sampling. Filters correct trajectories for use in SFT training.

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

    # RS-SFT: collect K=8 attempts per task across all benchmarks
    uv run python scripts/collect_trajectories_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --tasks all14 \
        --n-tasks 30 --trajectories-per-task 8 \
        --use-strategies \
        --experiment-name rs_sft_collection
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

import numpy as np

from scaffold.llm_query import TinkerModel, HybridTinkerModel
from scaffold.rlm import rlm, trajectory_to_dict, RLMTrajectory
from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
from training.rl_tinker_v6 import (
    STRATEGY_SUFFIXES, TASK_STRATEGY_WEIGHTS,
    score_trajectory, compute_reward, select_strategy,
)
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from eval.benchmarks.doc_classify import generate_doc_classify_suite, score_doc_classify
from eval.benchmarks.dataframe_qa import generate_dataframe_qa_suite, score_dataframe_qa
from eval.benchmarks.code_debug import generate_code_debug_suite, score_code_debug
from eval.benchmarks.multi_hop_qa import generate_multi_hop_suite, score_multi_hop
from eval.benchmarks.notebook_qa import generate_notebook_qa_suite, score_notebook_qa
from eval.benchmarks.hard_niah import generate_hard_niah_suite, score_hard_niah
from eval.benchmarks.verbatim_copy import generate_verbatim_copy_suite, score_verbatim_copy
from eval.benchmarks.multi_hop_hard import generate_hard_multi_hop_suite, score_hard_multi_hop
from eval.benchmarks.event_counting import generate_event_counting_suite, score_event_counting
from eval.benchmarks.cross_doc_compare import generate_cross_doc_suite, score_cross_doc
from eval.benchmarks.key_value_retrieval import generate_key_value_suite, score_key_value
from training.rewards import composite_reward, binary_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# All 14 benchmarks supported for RS-SFT collection
ALL14_BENCHMARKS = [
    "niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug",
    "multi_hop_qa", "notebook_qa", "hard_niah", "verbatim_copy",
    "hard_multi_hop", "event_counting", "cross_doc_compare", "key_value_retrieval",
]


def safe_rlm_call(prompt, model, system_prompt, max_iterations=8, verbose=False):
    """Run RLM with error handling — returns score-0 trajectory on failure."""
    try:
        return rlm(
            prompt=prompt, model=model, system_prompt=system_prompt,
            max_iterations=max_iterations, verbose=verbose,
        )
    except Exception as e:
        logger.error(f"RLM failed ({len(prompt)} chars): {type(e).__name__}: {e}")
        return RLMTrajectory(
            prompt=prompt, system_prompt=system_prompt,
            answer=None, terminated=False, total_time=0.0,
        )


def generate_tasks_for_benchmark(
    benchmark: str, n_tasks: int, seed_offset: int = 50000,
) -> list[dict]:
    """Generate task dicts for any of the 14 benchmarks."""
    tasks = []
    if benchmark == "niah":
        items = generate_niah_suite(n_tasks=n_tasks, doc_lengths=[50000, 100000], seed_offset=seed_offset)
        tasks = [{"task": t, "type": "niah"} for t in items]
    elif benchmark == "multi_niah":
        items = generate_multi_niah_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "multi_niah"} for t in items]
    elif benchmark == "doc_classify":
        items = generate_doc_classify_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "doc_classify"} for t in items]
    elif benchmark == "dataframe_qa":
        items = generate_dataframe_qa_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "dataframe_qa"} for t in items]
    elif benchmark == "code_debug":
        items = generate_code_debug_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "code_debug"} for t in items]
    elif benchmark == "multi_hop_qa":
        items = generate_multi_hop_suite(n_tasks=n_tasks, doc_lengths=[50000, 100000], seed_offset=seed_offset)
        tasks = [{"task": t, "type": "multi_hop_qa"} for t in items]
    elif benchmark == "notebook_qa":
        items = generate_notebook_qa_suite(n_tasks=n_tasks, doc_lengths=[50000, 100000], seed_offset=seed_offset)
        tasks = [{"task": t, "type": "notebook_qa"} for t in items]
    elif benchmark == "hard_niah":
        items = generate_hard_niah_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "hard_niah"} for t in items]
    elif benchmark == "verbatim_copy":
        items = generate_verbatim_copy_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "verbatim_copy"} for t in items]
    elif benchmark == "hard_multi_hop":
        items = generate_hard_multi_hop_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "hard_multi_hop"} for t in items]
    elif benchmark == "event_counting":
        items = generate_event_counting_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "event_counting"} for t in items]
    elif benchmark == "cross_doc_compare":
        items = generate_cross_doc_suite(n_tasks=n_tasks, seed_offset=seed_offset)
        tasks = [{"task": t, "type": "cross_doc_compare"} for t in items]
    elif benchmark == "key_value_retrieval":
        items = generate_key_value_suite(n_tasks=n_tasks, doc_lengths=[50000, 100000], seed_offset=seed_offset)
        tasks = [{"task": t, "type": "key_value_retrieval"} for t in items]
    return tasks


def collect_all14_trajectories(
    model,
    system_prompt: str,
    n_tasks: int = 30,
    trajectories_per_task: int = 8,
    max_iterations: int = 8,
    seed_offset: int = 50000,
    use_strategies: bool = False,
    benchmarks: list[str] | None = None,
) -> list[dict]:
    """Collect K trajectories per task across all 14 benchmarks.

    For RS-SFT: generates diverse trajectories by varying temperature and
    optionally strategy prompts. Scores each trajectory for later filtering.
    """
    if benchmarks is None:
        benchmarks = ALL14_BENCHMARKS

    temp_schedule = [0.3, 0.5, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1]
    if trajectories_per_task > len(temp_schedule):
        temp_schedule = temp_schedule + [0.7] * (trajectories_per_task - len(temp_schedule))

    all_trajectories = []

    for benchmark in benchmarks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting: {benchmark} ({n_tasks} tasks x {trajectories_per_task} attempts)")
        logger.info(f"{'='*60}")

        tasks = generate_tasks_for_benchmark(benchmark, n_tasks, seed_offset)
        if not tasks:
            logger.warning(f"No tasks generated for {benchmark}")
            continue

        benchmark_correct = 0
        benchmark_total = 0

        for task_idx, task_info in enumerate(tasks):
            task = task_info["task"]
            task_type = task_info["type"]
            task_id = getattr(task, "task_id", f"{task_type}_{task_idx}")
            strategy_rng = np.random.RandomState(seed_offset + task_idx)

            task_scores = []
            for k in range(trajectories_per_task):
                model.reset_stats() if hasattr(model, 'reset_stats') else None
                model.temperature = temp_schedule[k]

                if use_strategies:
                    strategy_name = select_strategy(task_type, strategy_rng)
                    traj_prompt = system_prompt + STRATEGY_SUFFIXES[strategy_name]
                else:
                    strategy_name = "standard"
                    traj_prompt = system_prompt

                traj = safe_rlm_call(
                    task.prompt, model, traj_prompt,
                    max_iterations=max_iterations, verbose=False,
                )
                traj_dict = trajectory_to_dict(traj)
                traj_dict["messages"] = traj.messages
                traj_dict["system_prompt"] = traj_prompt
                traj_dict["task_id"] = task_id
                traj_dict["task_type"] = task_type
                traj_dict["temperature"] = temp_schedule[k]
                traj_dict["strategy"] = strategy_name
                traj_dict["attempt"] = k

                score = score_trajectory(traj_dict, task_info)
                traj_dict["score"] = score
                traj_dict["reward"] = compute_reward(traj_dict, task_type, task_info)
                task_scores.append(score)

                all_trajectories.append(traj_dict)
                benchmark_total += 1
                if score > 0.5:
                    benchmark_correct += 1

            logger.info(
                f"  {task_id}: best={max(task_scores):.2f} "
                f"correct={sum(1 for s in task_scores if s > 0.5)}/{trajectories_per_task} "
                f"[{', '.join(f'{s:.2f}' for s in task_scores)}]"
            )

        logger.info(f"  {benchmark}: {benchmark_correct}/{benchmark_total} correct overall")

    return all_trajectories


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
                        choices=["niah", "multi_niah", "doc_classify", "all", "all14"])
    parser.add_argument("--benchmarks", default=None,
                        help="Comma-separated benchmark list (overrides --tasks)")
    parser.add_argument("--n-tasks", type=int, default=150)
    parser.add_argument("--trajectories-per-task", type=int, default=1,
                        help="K trajectories per task (use 8 for RS-SFT)")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--experiment-name", default="trajectories")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum score to include in correct trajectories")
    parser.add_argument("--use-strategies", action="store_true",
                        help="Use strategy-conditioned prompts for trajectory diversity")
    parser.add_argument("--seed-offset", type=int, default=50000,
                        help="Seed offset (avoid overlap with eval seeds at 10000)")
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

    # Determine which benchmarks to collect
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    elif args.tasks == "all14":
        benchmarks = ALL14_BENCHMARKS
    elif args.tasks == "all":
        benchmarks = ["niah", "multi_niah", "doc_classify"]
    else:
        benchmarks = [args.tasks]

    # Use the all14 collector for multi-attempt collection
    if args.trajectories_per_task > 1 or args.tasks == "all14" or args.benchmarks:
        all_trajectories = collect_all14_trajectories(
            model=model,
            system_prompt=system_prompt,
            n_tasks=args.n_tasks,
            trajectories_per_task=args.trajectories_per_task,
            max_iterations=args.max_iterations,
            seed_offset=args.seed_offset,
            use_strategies=args.use_strategies,
            benchmarks=benchmarks,
        )
    else:
        # Legacy single-attempt collection for backward compatibility
        for task_type in benchmarks:
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
            else:
                logger.warning(f"Legacy collection not supported for {task_type}, use --tasks all14")
                continue

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
    task_types_found = list(set(t.get("task_type", "unknown") for t in all_trajectories))
    stats = {
        "model": args.model,
        "model_path": args.model_path,
        "task_types": task_types_found,
        "benchmarks": benchmarks,
        "n_tasks_requested": args.n_tasks,
        "trajectories_per_task": args.trajectories_per_task,
        "use_strategies": args.use_strategies,
        "total_trajectories": len(all_trajectories),
        "correct_trajectories": len(correct),
        "success_rate": len(correct) / len(all_trajectories) if all_trajectories else 0,
        "total_time": total_time,
        "model_stats": model.total_stats(),
        "timestamp": timestamp,
        "per_type": {},
    }

    for task_type in sorted(task_types_found):
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
    for task_type in sorted(stats["per_type"].keys()):
        info = stats["per_type"][task_type]
        logger.info(f"  {task_type}: {info['correct']}/{info['total']} ({info['success_rate']:.1%})")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    logger.info(f"  Saved to: {save_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
