#!/usr/bin/env python3
"""
Evaluation harness for RLM models.

Runs the full RLM loop on benchmark tasks and scores results.
Saves detailed results, trajectories, and cost reports.

Usage:
    CUDA_VISIBLE_DEVICES=7 uv run python eval/run_eval.py \
        --model Qwen/Qwen3-1.7B \
        --benchmark niah \
        --n-tasks 10 \
        --experiment-name baseline_1.7b
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

from scaffold.llm_query import HFModel, strip_think_tags
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval")


def run_niah_eval(
    model: HFModel,
    system_prompt: str,
    n_tasks: int = 50,
    max_iterations: int = 8,
    doc_lengths: list[int] | None = None,
    verbose: bool = False,
) -> dict:
    """Run NIAH benchmark and return results."""
    tasks = generate_niah_suite(n_tasks=n_tasks, doc_lengths=doc_lengths)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="NIAH"):
        logger.info(f"\nTask: {task.task_id} | Q: {task.question} | Expected: {task.expected_answer}")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        score = score_niah(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "question": task.question,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": score,
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
            "needle_position": task.needle_position,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(f"  Score: {score} | Answer: {str(traj.answer)[:100]} | {traj.total_time:.1f}s")

    # Aggregate
    scores = [r["score"] for r in results]
    accuracy = sum(scores) / len(scores) if scores else 0

    # By doc length
    by_length = {}
    for r in results:
        dl = r["doc_length"]
        bucket = f"{dl // 1000}K"
        by_length.setdefault(bucket, []).append(r["score"])
    by_length_acc = {k: sum(v) / len(v) for k, v in by_length.items()}

    # By position
    by_position = {}
    for r in results:
        pos = f"{r['needle_position']:.1f}"
        by_position.setdefault(pos, []).append(r["score"])
    by_position_acc = {k: sum(v) / len(v) for k, v in by_position.items()}

    return {
        "benchmark": "niah",
        "accuracy": accuracy,
        "n_tasks": len(results),
        "by_doc_length": by_length_acc,
        "by_needle_position": by_position_acc,
        "results": results,
        "trajectories": trajectories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter (e.g., data/sft/lora_v1/final)")
    parser.add_argument("--benchmark", default="niah", choices=["niah"])
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--experiment-name", default="eval")
    parser.add_argument("--doc-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--system-prompt", default=None,
                        help="Path to custom system prompt file")
    args = parser.parse_args()

    # Verify GPU isolation
    assert torch.cuda.device_count() <= 2, "CUDA_VISIBLE_DEVICES not set correctly"
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

    # Results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / args.experiment_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    if args.adapter:
        logger.info(f"Loading LoRA adapter: {args.adapter}")
    t0 = time.time()

    if args.adapter:
        from training.rl import RLModel
        model = RLModel(
            base_model_name=args.model,
            adapter_path=args.adapter,
            device="cuda:0",
            max_new_tokens=1024,
        )
    else:
        model = HFModel(
            model_name=args.model,
            device="cuda:0",
            max_new_tokens=1024,
            temperature=0.7,
        )

    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # System prompt
    if args.system_prompt:
        system_prompt = Path(args.system_prompt).read_text()
    else:
        system_prompt = QWEN_2B_SYSTEM_PROMPT

    # Run eval
    logger.info(f"Running {args.benchmark} with {args.n_tasks} tasks")
    eval_start = time.time()

    if args.benchmark == "niah":
        eval_results = run_niah_eval(
            model=model,
            system_prompt=system_prompt,
            n_tasks=args.n_tasks,
            max_iterations=args.max_iterations,
            doc_lengths=args.doc_lengths,
            verbose=args.verbose,
        )

    eval_time = time.time() - eval_start

    # Model stats
    model_stats = model.total_stats()

    # Save results
    eval_output = {
        "benchmark": args.benchmark,
        "accuracy": eval_results["accuracy"],
        "n_tasks": eval_results["n_tasks"],
        "by_doc_length": eval_results.get("by_doc_length"),
        "by_needle_position": eval_results.get("by_needle_position"),
        "per_task": eval_results["results"],
    }

    config = {
        "model": args.model,
        "adapter": args.adapter,
        "benchmark": args.benchmark,
        "n_tasks": args.n_tasks,
        "max_iterations": args.max_iterations,
        "system_prompt": system_prompt[:500] + "...",
        "timestamp": timestamp,
        "git_hash": _get_git_hash(),
    }

    cost_report = {
        "model_load_time": load_time,
        "eval_time": eval_time,
        "total_time": load_time + eval_time,
        "model_stats": model_stats,
    }

    # Save files
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(eval_output, f, indent=2, default=str)

    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(results_dir / "cost_report.json", "w") as f:
        json.dump(cost_report, f, indent=2, default=str)

    # Save sample trajectories
    traj_dir = results_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for i, traj in enumerate(eval_results["trajectories"][:20]):
        with open(traj_dir / f"trajectory_{i:03d}.json", "w") as f:
            json.dump(traj, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVALUATION COMPLETE: {args.benchmark}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Accuracy: {eval_results['accuracy']:.1%}")
    logger.info(f"Tasks: {eval_results['n_tasks']}")
    logger.info(f"Time: {eval_time:.1f}s ({eval_time / eval_results['n_tasks']:.1f}s/task)")

    if eval_results.get("by_doc_length"):
        logger.info(f"\nBy doc length:")
        for k, v in sorted(eval_results["by_doc_length"].items()):
            logger.info(f"  {k}: {v:.1%}")

    if eval_results.get("by_needle_position"):
        logger.info(f"\nBy needle position:")
        for k, v in sorted(eval_results["by_needle_position"].items()):
            logger.info(f"  {k}: {v:.1%}")

    logger.info(f"\nModel stats: {model_stats}")
    logger.info(f"Results saved to: {results_dir}")


def _get_git_hash() -> str:
    """Get current git hash."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()[:8]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
