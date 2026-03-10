#!/usr/bin/env python3
"""
Teacher trajectory collection using Qwen3.5-397B-A17B.

Generates gold-standard RLM trajectories from a much larger teacher model.
These trajectories demonstrate genuine recursive strategies (multi-chunk,
multi-pass, aggregation) that can be used for:
1. SFT distillation into the 35B-A3B student model
2. DPO preference pairs (teacher trajectories as "chosen")
3. Analysis of what strategies work best on each task type

The teacher model (397B-A17B, 17B active params) is ~6x more expensive
per token than the student (35B-A3B, 3B active), but produces much
higher-quality code and strategies.

Usage:
    uv run python scripts/collect_teacher_trajectories.py \
        --n-tasks 10 --output data/trajectories/teacher_397b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
from eval.benchmarks.dataframe_qa import generate_dataframe_qa_suite, score_dataframe_qa
from eval.benchmarks.code_debug import generate_code_debug_suite, score_code_debug
from eval.benchmarks.multi_hop_qa import generate_multi_hop_suite, score_multi_hop
from eval.benchmarks.notebook_qa import generate_notebook_qa_suite, score_notebook_qa
from eval.benchmarks.multi_hop_hard import generate_hard_multi_hop_suite, score_hard_multi_hop
from eval.benchmarks.event_counting import generate_event_counting_suite, score_event_counting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("teacher_collect")


# Enhanced system prompt with strategy diversity hints for the teacher
TEACHER_STRATEGY_PROMPTS = {
    "standard": QWEN35_35B_SYSTEM_PROMPT,

    "extract_then_compute": QWEN35_35B_SYSTEM_PROMPT + """

## Strategy Preference: Extract-Then-Compute
For this task, prefer the extract-then-compute approach:
1. First pass: extract ALL relevant data items from the document (return raw items, not counts)
2. Then use Python to count, filter, aggregate, or compute the answer
Never delegate counting or arithmetic to llm_query — always do it in Python code.""",

    "binary_search": QWEN35_35B_SYSTEM_PROMPT + """

## Strategy Preference: Binary Search
For this task, try a binary-search-like approach:
1. First check the middle of the document for the target information
2. Based on the result, narrow your search to the first or second half
3. Continue subdividing until you find the answer
This is efficient for O(1) search tasks in long documents.""",

    "map_reduce": QWEN35_35B_SYSTEM_PROMPT + """

## Strategy Preference: Map-Reduce
For this task, use a map-reduce approach:
1. MAP: Process each chunk independently, extracting structured results
2. REDUCE: Combine all chunk results in Python (merge dicts, concatenate lists, etc.)
3. FINALIZE: Compute the final answer from the reduced results
Always accumulate results in Python data structures, never in natural language.""",

    "two_pass": QWEN35_35B_SYSTEM_PROMPT + """

## Strategy Preference: Two-Pass Verification
For this task, use a two-pass approach:
1. FIRST PASS: Quick scan of all chunks to identify candidate answers
2. SECOND PASS: Re-read only the chunks that contained candidates to verify
3. DECIDE: Pick the most confident answer from the verified candidates
This reduces errors from hallucination in single-pass approaches.""",

    "decompose": QWEN35_35B_SYSTEM_PROMPT + """

## Strategy Preference: Question Decomposition
For this task, decompose the question into sub-questions:
1. Identify what sub-facts you need (e.g., "Who is X?" then "What does X do?")
2. Search for each sub-fact independently across the document
3. Chain the results: use the answer from step 1 in your step 2 query
4. Combine sub-answers into the final answer
This is essential for multi-hop reasoning tasks.""",
}

# Which strategies to try for each task type
TASK_STRATEGIES = {
    "niah": ["standard", "binary_search"],
    "multi_niah": ["standard", "map_reduce"],
    "doc_classify": ["standard", "map_reduce"],
    "event_counting": ["extract_then_compute", "map_reduce"],
    "hard_multi_hop": ["decompose", "two_pass"],
    "multi_hop_qa": ["decompose", "standard"],
    "code_debug": ["standard", "two_pass"],
    "notebook_qa": ["extract_then_compute", "standard"],
    "dataframe_qa": ["extract_then_compute", "map_reduce"],
}


def generate_tasks(task_type: str, n_tasks: int, seed_offset: int = 50000):
    """Generate tasks for a given type with long contexts (anti-shortcut)."""
    if task_type == "niah":
        return generate_niah_suite(
            n_tasks=n_tasks,
            doc_lengths=[50000, 100000, 150000, 200000],
            seed_offset=seed_offset,
        )
    elif task_type == "multi_niah":
        return generate_multi_niah_suite(
            n_tasks=n_tasks,
            seed_offset=seed_offset,
        )
    elif task_type == "doc_classify":
        return generate_doc_classify_suite(
            n_tasks=n_tasks,
            seed_offset=seed_offset,
        )
    elif task_type == "event_counting":
        return generate_event_counting_suite(
            n_tasks=n_tasks,
            doc_lengths=[50000, 100000, 150000, 200000],
            seed_offset=seed_offset,
        )
    elif task_type == "hard_multi_hop":
        return generate_hard_multi_hop_suite(
            n_tasks=n_tasks,
            doc_lengths_2hop=[100000, 150000, 200000],
            doc_lengths_3hop=[150000, 200000],
            seed_offset=seed_offset,
        )
    elif task_type == "multi_hop_qa":
        return generate_multi_hop_suite(
            n_tasks=n_tasks,
            doc_lengths=[50000, 100000, 150000],
            seed_offset=seed_offset,
        )
    elif task_type == "code_debug":
        return generate_code_debug_suite(
            n_tasks=n_tasks,
            seed_offset=seed_offset,
        )
    elif task_type == "notebook_qa":
        return generate_notebook_qa_suite(
            n_tasks=n_tasks,
            doc_lengths=[50000, 100000],
            seed_offset=seed_offset,
        )
    elif task_type == "dataframe_qa":
        return generate_dataframe_qa_suite(
            n_tasks=n_tasks,
            seed_offset=seed_offset,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def score_task(task, answer, task_type):
    """Score a task answer."""
    if task_type == "niah":
        return score_niah(answer, task.expected_answer)
    elif task_type == "multi_niah":
        result = score_multi_niah(answer, task.expected_answers)
        return result["recall"]
    elif task_type == "doc_classify":
        result = score_doc_classify(answer, task.expected_labels)
        return result["accuracy"]
    elif task_type == "event_counting":
        result = score_event_counting(answer, task.expected_answer)
        return result["score"]
    elif task_type == "hard_multi_hop":
        result = score_hard_multi_hop(answer, task.expected_answer)
        return result["score"]
    elif task_type == "multi_hop_qa":
        result = score_multi_hop(answer, task.expected_answer)
        return result["score"]
    elif task_type == "code_debug":
        result = score_code_debug(answer, task.bugs)
        return result["score"]
    elif task_type == "notebook_qa":
        result = score_notebook_qa(answer, task.expected_answer)
        return result["score"]
    elif task_type == "dataframe_qa":
        result = score_dataframe_qa(answer, task.expected_answer, task.task_type)
        return result["score"]
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def collect_for_task_type(
    model: TinkerModel,
    task_type: str,
    n_tasks: int,
    max_iterations: int = 12,
    seed_offset: int = 50000,
) -> list[dict]:
    """Collect trajectories for a task type with strategy diversity."""
    tasks = generate_tasks(task_type, n_tasks, seed_offset)
    strategies = TASK_STRATEGIES.get(task_type, ["standard"])

    all_trajectories = []
    correct_count = 0

    for task_idx, task in enumerate(tasks):
        task_start = time.time()
        best_score = 0
        best_traj = None

        for strategy_name in strategies:
            system_prompt = TEACHER_STRATEGY_PROMPTS[strategy_name]

            logger.info(
                f"  [{task_type}] Task {task_idx+1}/{len(tasks)} "
                f"strategy={strategy_name} "
                f"prompt_len={len(task.prompt)}"
            )

            model.reset_stats()
            traj = rlm(
                prompt=task.prompt,
                model=model,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                verbose=True,
            )

            score = score_task(task, traj.answer, task_type)
            traj_dict = trajectory_to_dict(traj)
            traj_dict["task_id"] = task.task_id
            traj_dict["task_type"] = task_type
            traj_dict["score"] = score
            traj_dict["strategy"] = strategy_name
            traj_dict["teacher_model"] = "Qwen/Qwen3.5-397B-A17B"

            # Store expected answer for reference
            if hasattr(task, "expected_answer"):
                traj_dict["expected_answer"] = str(task.expected_answer)
            if hasattr(task, "expected_answers"):
                traj_dict["expected_answers"] = task.expected_answers
            if hasattr(task, "expected_labels"):
                traj_dict["expected_labels"] = task.expected_labels
            if hasattr(task, "bugs"):
                traj_dict["bugs"] = task.bugs
            if hasattr(task, "task_type"):
                traj_dict["subtask_type"] = task.task_type

            # Count sub-calls
            model_stats = traj.model_stats or {}
            n_subcalls = model_stats.get("sub_calls", 0)
            n_code_turns = sum(1 for t in traj.turns if t.get("parsed_code"))
            traj_dict["n_subcalls"] = n_subcalls
            traj_dict["n_code_turns"] = n_code_turns

            all_trajectories.append(traj_dict)

            elapsed = time.time() - task_start
            logger.info(
                f"    score={score:.2f} turns={len(traj.turns)} "
                f"subcalls={n_subcalls} time={elapsed:.1f}s "
                f"answer={str(traj.answer)[:100]}"
            )

            if score > best_score:
                best_score = score
                best_traj = traj_dict

            # If we got a perfect score, no need to try more strategies
            if score >= 1.0:
                break

        if best_score >= 0.5:
            correct_count += 1

    logger.info(
        f"  [{task_type}] Complete: {correct_count}/{len(tasks)} correct "
        f"({len(all_trajectories)} total trajectories)"
    )
    return all_trajectories


def main():
    parser = argparse.ArgumentParser(description="Collect teacher trajectories")
    parser.add_argument(
        "--teacher-model", default="Qwen/Qwen3.5-397B-A17B",
        help="Teacher model name",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=10,
        help="Tasks per benchmark type",
    )
    parser.add_argument(
        "--task-types", nargs="+",
        default=["event_counting", "hard_multi_hop", "code_debug",
                 "multi_hop_qa", "doc_classify"],
        help="Task types to collect",
    )
    parser.add_argument(
        "--output", default="data/trajectories/teacher_397b",
        help="Output directory",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=12,
        help="Max RLM iterations per trajectory",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Model state path (e.g., tinker://session:train:0/weights/state-0005). "
             "If provided, loads fine-tuned weights instead of base model.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Teacher Trajectory Collection")
    logger.info(f"  Model: {args.teacher_model}")
    if args.model_path:
        logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Tasks: {args.task_types}")
    logger.info(f"  N per type: {args.n_tasks}")
    logger.info(f"  Output: {args.output}")

    # Create model (base or fine-tuned)
    model = TinkerModel(
        model_name=args.teacher_model,
        max_new_tokens=2048,
        temperature=args.temperature,
        model_path=args.model_path,
    )

    t0 = time.time()
    all_trajectories = []
    stats = {}

    for task_type in args.task_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting {task_type} trajectories...")
        logger.info(f"{'='*60}")

        trajs = collect_for_task_type(
            model=model,
            task_type=task_type,
            n_tasks=args.n_tasks,
            max_iterations=args.max_iterations,
        )
        all_trajectories.extend(trajs)

        # Per-type stats
        correct = [t for t in trajs if t["score"] >= 0.5]
        perfect = [t for t in trajs if t["score"] >= 1.0]
        stats[task_type] = {
            "total": len(trajs),
            "correct": len(correct),
            "perfect": len(perfect),
            "avg_score": sum(t["score"] for t in trajs) / len(trajs) if trajs else 0,
            "avg_subcalls": sum(t.get("n_subcalls", 0) for t in trajs) / len(trajs) if trajs else 0,
            "avg_turns": sum(t.get("n_code_turns", 0) for t in trajs) / len(trajs) if trajs else 0,
            "strategies_used": list(set(t["strategy"] for t in trajs)),
        }

        # Save incrementally
        with open(output_dir / f"{task_type}_trajectories.json", "w") as f:
            json.dump(trajs, f, indent=2, default=str)

    total_time = time.time() - t0

    # Save all trajectories
    with open(output_dir / "all_trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=2, default=str)

    # Filter correct trajectories for SFT
    correct_trajs = [t for t in all_trajectories if t["score"] >= 0.5]
    with open(output_dir / "correct_trajectories.json", "w") as f:
        json.dump(correct_trajs, f, indent=2, default=str)

    # Filter high-quality trajectories (score >= 0.8 AND multi-turn)
    gold_trajs = [
        t for t in all_trajectories
        if t["score"] >= 0.8 and t.get("n_code_turns", 0) >= 2
    ]
    with open(output_dir / "gold_trajectories.json", "w") as f:
        json.dump(gold_trajs, f, indent=2, default=str)

    # Save config
    config = {
        "teacher_model": args.teacher_model,
        "task_types": args.task_types,
        "n_tasks_per_type": args.n_tasks,
        "max_iterations": args.max_iterations,
        "temperature": args.temperature,
        "total_trajectories": len(all_trajectories),
        "correct_trajectories": len(correct_trajs),
        "gold_trajectories": len(gold_trajs),
        "per_type_stats": stats,
        "total_time_seconds": total_time,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TEACHER TRAJECTORY COLLECTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {len(all_trajectories)} trajectories")
    logger.info(f"Correct (score >= 0.5): {len(correct_trajs)}")
    logger.info(f"Gold (score >= 0.8, multi-turn): {len(gold_trajs)}")
    logger.info(f"Time: {total_time:.0f}s ({total_time/3600:.2f} hours)")
    logger.info(f"")
    for task_type, s in stats.items():
        logger.info(
            f"  {task_type}: {s['correct']}/{s['total']} correct, "
            f"avg_score={s['avg_score']:.2f}, "
            f"avg_subcalls={s['avg_subcalls']:.1f}, "
            f"avg_turns={s['avg_turns']:.1f}"
        )
    logger.info(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
