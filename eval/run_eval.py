#!/usr/bin/env python3
"""
Evaluation harness for RLM models.

Runs the full RLM loop on benchmark tasks and scores results.
Saves detailed results, trajectories, and cost reports.

Usage (Tinker — no GPU needed):
    uv run python eval/run_eval.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --benchmark all \
        --experiment-name baseline_35b_a3b

    # With fine-tuned checkpoint:
    uv run python eval/run_eval.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --model-path tinker://run-id/weights/ckpt-100 \
        --benchmark all \
        --experiment-name sft_35b_v1

Legacy (local GPU):
    CUDA_VISIBLE_DEVICES=7 uv run python eval/run_eval.py \
        --model Qwen/Qwen3-1.7B --backend hf \
        --benchmark niah --n-tasks 10
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

from tqdm import tqdm

from scaffold.llm_query import TinkerModel, strip_think_tags
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from eval.benchmarks.doc_classify import generate_doc_classify_suite, score_doc_classify
from eval.benchmarks.dataframe_qa import generate_dataframe_qa_suite, score_dataframe_qa
from eval.benchmarks.code_debug import generate_code_debug_suite, score_code_debug
from eval.benchmarks.multi_hop_qa import generate_multi_hop_suite, score_multi_hop
from eval.benchmarks.notebook_qa import generate_notebook_qa_suite, score_notebook_qa
from eval.benchmarks.hard_niah import generate_hard_niah_suite, score_hard_niah
from eval.benchmarks.verbatim_copy import generate_verbatim_copy_suite, score_verbatim_copy
from eval.benchmarks.oolong import load_oolong_tasks, score_oolong
from eval.benchmarks.multi_hop_hard import generate_hard_multi_hop_suite, score_hard_multi_hop
from eval.benchmarks.event_counting import generate_event_counting_suite, score_event_counting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval")


def run_niah_eval(
    model,
    system_prompt: str,
    n_tasks: int = 50,
    max_iterations: int = 8,
    doc_lengths: list[int] | None = None,
    positions: list[float] | None = None,
    seed_offset: int = 10000,
    verbose: bool = False,
) -> dict:
    """Run NIAH benchmark and return results."""
    tasks = generate_niah_suite(
        n_tasks=n_tasks, doc_lengths=doc_lengths,
        positions=positions, seed_offset=seed_offset,
    )

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
        pos = f"{r['needle_position']:.2f}"
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


def run_multi_niah_eval(
    model,
    system_prompt: str,
    n_tasks: int = 24,
    max_iterations: int = 10,
    seed_offset: int = 50000,
    verbose: bool = False,
) -> dict:
    """Run multi-needle NIAH benchmark and return results."""
    tasks = generate_multi_niah_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="Multi-NIAH"):
        logger.info(
            f"\nTask: {task.task_id} | {task.n_needles} needles in {task.doc_length} chars"
        )

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_multi_niah(traj.answer, task.expected_answers)

        result = {
            "task_id": task.task_id,
            "n_needles": task.n_needles,
            "expected": task.expected_answers,
            "predicted": traj.answer,
            "recall": scores["recall"],
            "f1": scores["f1"],
            "found": scores["found"],
            "total": scores["total"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Found {scores['found']}/{scores['total']} | "
            f"Recall: {scores['recall']:.2f} | "
            f"Answer: {str(traj.answer)[:120]} | {traj.total_time:.1f}s"
        )

    # Aggregate
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0

    # By needle count
    by_n_needles = {}
    for r in results:
        k = f"{r['n_needles']} needles"
        by_n_needles.setdefault(k, []).append(r["recall"])
    by_n_needles_avg = {k: sum(v) / len(v) for k, v in by_n_needles.items()}

    # By doc length
    by_length = {}
    for r in results:
        dl = r["doc_length"]
        bucket = f"{dl // 1000}K"
        by_length.setdefault(bucket, []).append(r["recall"])
    by_length_avg = {k: sum(v) / len(v) for k, v in by_length.items()}

    return {
        "benchmark": "multi_niah",
        "accuracy": avg_recall,  # Use recall as primary metric
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "n_tasks": len(results),
        "by_n_needles": by_n_needles_avg,
        "by_doc_length": by_length_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_doc_classify_eval(
    model,
    system_prompt: str,
    n_tasks: int = 20,
    max_iterations: int = 10,
    seed_offset: int = 70000,
    verbose: bool = False,
) -> dict:
    """Run document classification benchmark and return results."""
    tasks = generate_doc_classify_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="DocClassify"):
        logger.info(
            f"\nTask: {task.task_id} | {task.n_docs} docs, {task.doc_length} chars"
        )

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_doc_classify(traj.answer, task.expected_labels)

        result = {
            "task_id": task.task_id,
            "n_docs": task.n_docs,
            "expected": task.expected_labels,
            "predicted": traj.answer,
            "accuracy": scores["accuracy"],
            "correct": scores["correct"],
            "total": scores["total"],
            "per_doc": scores["per_doc"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Correct: {scores['correct']}/{scores['total']} ({scores['accuracy']:.1%}) | "
            f"Answer: {str(traj.answer)[:120]} | {traj.total_time:.1f}s"
        )

    # Aggregate
    avg_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0

    # By n_docs
    by_n_docs = {}
    for r in results:
        k = f"{r['n_docs']} docs"
        by_n_docs.setdefault(k, []).append(r["accuracy"])
    by_n_docs_avg = {k: sum(v) / len(v) for k, v in by_n_docs.items()}

    return {
        "benchmark": "doc_classify",
        "accuracy": avg_accuracy,
        "n_tasks": len(results),
        "by_n_docs": by_n_docs_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_dataframe_qa_eval(
    model,
    system_prompt: str,
    n_tasks: int = 20,
    max_iterations: int = 10,
    seed_offset: int = 90000,
    verbose: bool = False,
) -> dict:
    """Run DataFrame QA benchmark."""
    tasks = generate_dataframe_qa_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="DataFrame QA"):
        logger.info(f"\nTask: {task.task_id} | {task.n_rows} rows, {len(task.prompt):,} chars")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_dataframe_qa(traj.answer, task.expected_answer, task.task_type)

        result = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "n_rows": task.n_rows,
            "n_tickers": task.n_tickers,
            "difficulty": task.difficulty,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "match_type": scores["match_type"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "prompt_chars": len(task.prompt),
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.2f} ({scores['match_type']}) | "
            f"Answer: {str(traj.answer)[:60]} | Expected: {str(task.expected_answer)[:60]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_difficulty = {}
    for r in results:
        by_difficulty.setdefault(r["difficulty"], []).append(r["score"])
    by_difficulty_avg = {k: sum(v) / len(v) for k, v in by_difficulty.items()}

    by_type = {}
    for r in results:
        by_type.setdefault(r["task_type"], []).append(r["score"])
    by_type_avg = {k: sum(v) / len(v) for k, v in by_type.items()}

    return {
        "benchmark": "dataframe_qa",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_difficulty": by_difficulty_avg,
        "by_type": by_type_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_code_debug_eval(
    model,
    system_prompt: str,
    n_tasks: int = 15,
    max_iterations: int = 10,
    seed_offset: int = 95000,
    verbose: bool = False,
) -> dict:
    """Run code debugging benchmark."""
    tasks = generate_code_debug_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="CodeDebug"):
        logger.info(f"\nTask: {task.task_id} | {task.n_bugs} bugs, {task.total_lines} lines")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_code_debug(traj.answer, task.bugs)

        result = {
            "task_id": task.task_id,
            "n_bugs": task.n_bugs,
            "n_functions": task.n_functions,
            "total_lines": task.total_lines,
            "expected_bugs": [b["function"] for b in task.bugs],
            "predicted": traj.answer,
            "score": scores["score"],
            "found": scores["found"],
            "total": scores["total"],
            "details": scores.get("details", []),
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "prompt_chars": len(task.prompt),
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Found: {scores['found']}/{scores['total']} ({scores['score']:.1%}) | "
            f"Answer: {str(traj.answer)[:100]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_n_bugs = {}
    for r in results:
        k = f"{r['n_bugs']} bugs"
        by_n_bugs.setdefault(k, []).append(r["score"])
    by_n_bugs_avg = {k: sum(v) / len(v) for k, v in by_n_bugs.items()}

    return {
        "benchmark": "code_debug",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_n_bugs": by_n_bugs_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_multi_hop_eval(
    model,
    system_prompt: str,
    n_tasks: int = 20,
    max_iterations: int = 10,
    seed_offset: int = 97000,
    verbose: bool = False,
) -> dict:
    """Run multi-hop QA benchmark."""
    tasks = generate_multi_hop_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="MultiHopQA"):
        logger.info(f"\nTask: {task.task_id} | {task.n_hops} hops, {task.doc_length} chars")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_multi_hop(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "n_hops": task.n_hops,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "partial": scores["partial"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "prompt_chars": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.1f} | "
            f"Expected: {task.expected_answer[:80]} | "
            f"Got: {str(traj.answer)[:80]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_hops = {}
    for r in results:
        k = f"{r['n_hops']}_hop"
        by_hops.setdefault(k, []).append(r["score"])
    by_hops_avg = {k: sum(v) / len(v) for k, v in by_hops.items()}

    return {
        "benchmark": "multi_hop_qa",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_hops": by_hops_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_oolong_eval(
    model,
    system_prompt: str,
    n_tasks: int = 10,
    max_iterations: int = 12,
    max_context_chars: int = 200000,
    verbose: bool = False,
) -> dict:
    """Run OOLONG benchmark (real D&D transcript aggregation tasks)."""
    tasks = load_oolong_tasks(
        n_tasks=n_tasks,
        max_context_chars=max_context_chars,
    )

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="OOLONG"):
        logger.info(f"\nTask: {task.task_id} | {task.question_type} | {task.context_length:,} chars")
        logger.info(f"  Q: {task.question[:100]}")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_oolong(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "question_type": task.question_type,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "match_type": scores["match_type"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "context_length": task.context_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.1f} ({scores['match_type']}) | "
            f"Expected: {task.expected_answer[:60]} | Got: {str(traj.answer)[:60]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_type = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["score"])
    by_type_avg = {k: sum(v) / len(v) for k, v in by_type.items()}

    return {
        "benchmark": "oolong",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_question_type": by_type_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_verbatim_copy_eval(
    model,
    system_prompt: str,
    n_tasks: int = 10,
    max_iterations: int = 10,
    seed_offset: int = 100000,
    verbose: bool = False,
) -> dict:
    """Run verbatim copy benchmark."""
    tasks = generate_verbatim_copy_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="VerbatimCopy"):
        logger.info(f"\nTask: {task.task_id} | {task.doc_length} chars, target {task.target_paragraph_length} chars")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        score = score_verbatim_copy(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "expected_length": task.target_paragraph_length,
            "predicted": traj.answer,
            "score": score,
            "target_position": task.target_paragraph_position,
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {score:.2f} | "
            f"Expected: {task.expected_answer[:60]}... | Got: {str(traj.answer)[:60]}..."
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    return {
        "benchmark": "verbatim_copy",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "results": results,
        "trajectories": trajectories,
    }


def run_hard_multi_hop_eval(
    model,
    system_prompt: str,
    n_tasks: int = 10,
    max_iterations: int = 10,
    verbose: bool = False,
) -> dict:
    """Run hard multi-hop QA benchmark (forces true multi-step decomposition)."""
    tasks = generate_hard_multi_hop_suite(n_tasks=n_tasks)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="HardMultiHop"):
        logger.info(f"\nTask: {task.task_id} | {task.n_hops}-hop | {task.doc_length:,} chars")
        logger.info(f"  Q: {task.question[:100]}")
        logger.info(f"  Decomposition: {' → '.join(task.decomposition)}")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_hard_multi_hop(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "n_hops": task.n_hops,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "match_type": scores["match_type"],
            "decomposition": task.decomposition,
            "n_distractors": len(task.distractors),
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.1f} ({scores['match_type']}) | "
            f"Expected: {task.expected_answer[:60]} | Got: {str(traj.answer)[:60]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_hops = {}
    for r in results:
        by_hops.setdefault(r["n_hops"], []).append(r["score"])
    by_hops_avg = {k: sum(v) / len(v) for k, v in by_hops.items()}

    return {
        "benchmark": "hard_multi_hop",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_hops": by_hops_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_event_counting_eval(
    model,
    system_prompt: str,
    n_tasks: int = 10,
    max_iterations: int = 12,
    verbose: bool = False,
) -> dict:
    """Run event counting benchmark (counting/aggregation over long docs)."""
    tasks = generate_event_counting_suite(n_tasks=n_tasks)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="EventCount"):
        logger.info(f"\nTask: {task.task_id} | {task.task_type} | {task.doc_length:,} chars | {task.n_events} events")
        logger.info(f"  Q: {task.question[:100]}")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_event_counting(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "match_type": scores["match_type"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
            "n_events": task.n_events,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.1f} ({scores['match_type']}) | "
            f"Expected: {task.expected_answer[:60]} | Got: {str(traj.answer)[:60]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_type = {}
    for r in results:
        by_type.setdefault(r["task_type"], []).append(r["score"])
    by_type_avg = {k: sum(v) / len(v) for k, v in by_type.items()}

    return {
        "benchmark": "event_counting",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_type": by_type_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_hard_niah_eval(
    model,
    system_prompt: str,
    n_tasks: int = 15,
    max_iterations: int = 10,
    seed_offset: int = 98000,
    verbose: bool = False,
) -> dict:
    """Run hard NIAH benchmark (adversarial distractors, extreme lengths, boundary positions)."""
    tasks = generate_hard_niah_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="HardNIAH"):
        logger.info(f"\nTask: {task.task_id} | {task.difficulty} | {task.doc_length} chars")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_hard_niah(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "match_type": scores["match_type"],
            "n_distractors": len(task.distractors),
            "needle_position": task.needle_position,
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "doc_length": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.1f} ({scores['match_type']}) | "
            f"Expected: {task.expected_answer} | Got: {str(traj.answer)[:80]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_difficulty = {}
    for r in results:
        by_difficulty.setdefault(r["difficulty"], []).append(r["score"])
    by_difficulty_avg = {k: sum(v) / len(v) for k, v in by_difficulty.items()}

    return {
        "benchmark": "hard_niah",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_difficulty": by_difficulty_avg,
        "results": results,
        "trajectories": trajectories,
    }


def run_notebook_qa_eval(
    model,
    system_prompt: str,
    n_tasks: int = 15,
    max_iterations: int = 10,
    seed_offset: int = 99000,
    verbose: bool = False,
) -> dict:
    """Run notebook QA benchmark."""
    tasks = generate_notebook_qa_suite(n_tasks=n_tasks, seed_offset=seed_offset)

    results = []
    trajectories = []

    for task in tqdm(tasks, desc="NotebookQA"):
        logger.info(f"\nTask: {task.task_id} | {task.question_type} | {task.n_cells} cells, {task.doc_length} chars")

        traj = rlm(
            prompt=task.prompt,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        scores = score_notebook_qa(traj.answer, task.expected_answer)

        result = {
            "task_id": task.task_id,
            "question_type": task.question_type,
            "n_cells": task.n_cells,
            "expected": task.expected_answer,
            "predicted": traj.answer,
            "score": scores["score"],
            "match_type": scores["match_type"],
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
            "prompt_chars": task.doc_length,
        }
        results.append(result)
        trajectories.append(trajectory_to_dict(traj))

        logger.info(
            f"  Score: {scores['score']:.1f} ({scores['match_type']}) | "
            f"Expected: {task.expected_answer[:60]} | Got: {str(traj.answer)[:60]}"
        )

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    by_type = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["score"])
    by_type_avg = {k: sum(v) / len(v) for k, v in by_type.items()}

    return {
        "benchmark": "notebook_qa",
        "accuracy": avg_score,
        "score": avg_score,
        "n_tasks": len(results),
        "by_question_type": by_type_avg,
        "results": results,
        "trajectories": trajectories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--model-path", default=None,
                        help="Tinker model path (e.g., tinker://run-id/weights/ckpt-100)")
    parser.add_argument("--adapter", default=None,
                        help="(Legacy) Path to local LoRA adapter")
    parser.add_argument("--backend", default="tinker", choices=["tinker", "hf"],
                        help="Backend: tinker (remote) or hf (local GPU)")
    parser.add_argument("--benchmark", default="niah",
                        choices=["niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug", "multi_hop_qa", "notebook_qa", "hard_niah", "verbatim_copy", "oolong", "hard_multi_hop", "event_counting", "all"])
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--experiment-name", default="eval")
    parser.add_argument("--doc-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--positions", nargs="+", type=float, default=None)
    parser.add_argument("--seed-offset", type=int, default=10000,
                        help="Seed offset for eval tasks (avoid training overlap)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--system-prompt", default=None,
                        help="Path to custom system prompt file")
    args = parser.parse_args()

    # Results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / args.experiment_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model} (backend={args.backend})")
    t0 = time.time()

    if args.backend == "tinker":
        model = TinkerModel(
            model_name=args.model,
            model_path=args.model_path,
            max_new_tokens=2048,
            temperature=0.7,
        )
    else:
        # Legacy: local HF model
        from scaffold.llm_query import HFModel
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
    elif "qwen3.5" in args.model.lower() or "Qwen3.5" in args.model:
        system_prompt = QWEN35_35B_SYSTEM_PROMPT
    else:
        system_prompt = QWEN_2B_SYSTEM_PROMPT

    # Run eval
    benchmarks_to_run = (
        ["niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug", "multi_hop_qa", "notebook_qa", "hard_niah", "verbatim_copy", "oolong", "hard_multi_hop", "event_counting"] if args.benchmark == "all"
        else [args.benchmark]
    )

    all_eval_results = {}
    eval_start = time.time()

    for bench in benchmarks_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running benchmark: {bench}")
        logger.info(f"{'=' * 60}")

        if bench == "niah":
            eval_results = run_niah_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=args.n_tasks,
                max_iterations=args.max_iterations,
                doc_lengths=args.doc_lengths,
                positions=args.positions,
                seed_offset=args.seed_offset,
                verbose=args.verbose,
            )
        elif bench == "multi_niah":
            eval_results = run_multi_niah_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 24),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "doc_classify":
            eval_results = run_doc_classify_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 20),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "dataframe_qa":
            eval_results = run_dataframe_qa_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 20),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "code_debug":
            eval_results = run_code_debug_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 15),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "multi_hop_qa":
            eval_results = run_multi_hop_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 20),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "notebook_qa":
            eval_results = run_notebook_qa_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 15),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "hard_niah":
            eval_results = run_hard_niah_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 15),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "verbatim_copy":
            eval_results = run_verbatim_copy_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 10),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "oolong":
            eval_results = run_oolong_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 10),
                max_iterations=12,  # OOLONG tasks are complex, need more iterations
                verbose=args.verbose,
            )
        elif bench == "hard_multi_hop":
            eval_results = run_hard_multi_hop_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=min(args.n_tasks, 10),
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        elif bench == "event_counting":
            eval_results = run_event_counting_eval(
                model=model,
                system_prompt=system_prompt,
                n_tasks=args.n_tasks,
                max_iterations=12,  # Counting tasks need more iterations
                verbose=args.verbose,
            )

        all_eval_results[bench] = eval_results
        _print_benchmark_summary(bench, eval_results)

    eval_time = time.time() - eval_start

    # Model stats
    model_stats = model.total_stats()

    config = {
        "model": args.model,
        "backend": args.backend,
        "model_path": args.model_path,
        "adapter": args.adapter,
        "benchmarks": benchmarks_to_run,
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

    # Save files for each benchmark
    for bench, eval_results in all_eval_results.items():
        bench_dir = results_dir / bench if len(benchmarks_to_run) > 1 else results_dir
        bench_dir.mkdir(parents=True, exist_ok=True)

        eval_output = {
            "benchmark": bench,
            "accuracy": eval_results["accuracy"],
            "n_tasks": eval_results["n_tasks"],
            "per_task": eval_results["results"],
        }
        # Add benchmark-specific aggregates
        for key in ["by_doc_length", "by_needle_position", "by_n_needles", "by_n_docs",
                     "avg_recall", "avg_f1"]:
            if key in eval_results:
                eval_output[key] = eval_results[key]

        with open(bench_dir / "eval_results.json", "w") as f:
            json.dump(eval_output, f, indent=2, default=str)

        # Save sample trajectories
        traj_dir = bench_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        for i, traj in enumerate(eval_results["trajectories"][:20]):
            with open(traj_dir / f"trajectory_{i:03d}.json", "w") as f:
                json.dump(traj, f, indent=2, default=str)

    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(results_dir / "cost_report.json", "w") as f:
        json.dump(cost_report, f, indent=2, default=str)

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL EVALUATIONS COMPLETE")
    logger.info(f"{'=' * 60}")
    for bench, res in all_eval_results.items():
        logger.info(f"  {bench}: {res['accuracy']:.1%} ({res['n_tasks']} tasks)")
    logger.info(f"Total time: {eval_time:.1f}s")
    logger.info(f"Model stats: {model_stats}")
    logger.info(f"Results saved to: {results_dir}")


def _print_benchmark_summary(bench: str, eval_results: dict):
    """Print summary for a single benchmark."""
    logger.info(f"\n--- {bench} Results ---")
    logger.info(f"Accuracy/Recall: {eval_results['accuracy']:.1%}")
    logger.info(f"Tasks: {eval_results['n_tasks']}")

    if eval_results.get("by_doc_length"):
        logger.info(f"By doc length:")
        for k, v in sorted(eval_results["by_doc_length"].items()):
            logger.info(f"  {k}: {v:.1%}")

    if eval_results.get("by_needle_position"):
        logger.info(f"By needle position:")
        for k, v in sorted(eval_results["by_needle_position"].items()):
            logger.info(f"  {k}: {v:.1%}")

    if eval_results.get("by_n_needles"):
        logger.info(f"By needle count:")
        for k, v in sorted(eval_results["by_n_needles"].items()):
            logger.info(f"  {k}: {v:.1%}")

    if eval_results.get("by_n_docs"):
        logger.info(f"By document count:")
        for k, v in sorted(eval_results["by_n_docs"].items()):
            logger.info(f"  {k}: {v:.1%}")

    if eval_results.get("avg_recall") is not None:
        logger.info(f"Avg recall: {eval_results['avg_recall']:.1%}")
    if eval_results.get("avg_f1") is not None:
        logger.info(f"Avg F1: {eval_results['avg_f1']:.1%}")


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
