#!/usr/bin/env python3
"""
Critical smoke test: Can base Qwen3.5-2B write REPL code?

This is THE most important experiment before anything else.
The paper says "models without sufficient coding capabilities struggle as RLMs."
At 2B, this is our biggest risk.

We run 5 simple tasks through the full RLM loop and examine raw output.
Questions to answer:
1. Can it write `context[:1000]`?
2. Can it call `llm_query(...)`?
3. Can it use `FINAL()` / `FINAL_VAR()` correctly?
4. Does it produce valid Python at all?
5. Does it wrap code in ```repl blocks or something else?

Usage:
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/smoke_test_2b.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Safety check
assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("6", "7", "6,7"), \
    "CUDA_VISIBLE_DEVICES must be set to 6, 7, or 6,7"

sys.path.insert(0, ".")

import torch
from scaffold.llm_query import HFModel
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT

# Verify GPU isolation
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
assert torch.cuda.device_count() <= 2, "CUDA_VISIBLE_DEVICES not set correctly"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("smoke_test")

# ============================================================
# Test tasks — simple, with known answers
# ============================================================

TASKS = [
    {
        "name": "needle_in_haystack",
        "prompt": ("X" * 2000 + "The secret code is ALPHA-7." + "X" * 2000),
        "question": "Find the secret code hidden in the context.",
        "expected": "ALPHA-7",
    },
    {
        "name": "count_lines",
        "prompt": "\n".join([f"Line {i}: data_{i}" for i in range(100)]),
        "question": "How many lines are in the context?",
        "expected": "100",
    },
    {
        "name": "first_and_last",
        "prompt": "FIRST_WORD " + "middle " * 500 + "LAST_WORD",
        "question": "What is the first word and the last word of the context?",
        "expected": "FIRST_WORD",  # Partial match OK
    },
    {
        "name": "simple_extraction",
        "prompt": "Name: Alice\nAge: 30\nCity: Boston\n" + "Other data\n" * 200,
        "question": "What is the person's name and city?",
        "expected": "Alice",  # Partial match OK
    },
    {
        "name": "length_check",
        "prompt": "A" * 5000,
        "question": "How many characters are in the context?",
        "expected": "5000",
    },
]


def make_task_prompt(task: dict) -> str:
    """Build the full prompt that goes into the REPL as `context`."""
    return f"QUESTION: {task['question']}\n\nDOCUMENT:\n{task['prompt']}"


def main():
    logger.info("=" * 70)
    logger.info("SMOKE TEST: Can Qwen3.5-2B write REPL code?")
    logger.info("=" * 70)

    # Check if model is already downloaded
    model_name = "Qwen/Qwen3-1.7B"
    logger.info(f"Loading {model_name}...")

    t0 = time.time()
    model = HFModel(
        model_name=model_name,
        device="cuda:0",
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Results storage
    results_dir = Path("results/smoke_test") / time.strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, task in enumerate(TASKS):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"Task {i+1}/{len(TASKS)}: {task['name']}")
        logger.info(f"Question: {task['question']}")
        logger.info(f"Expected (partial): {task['expected']}")
        logger.info(f"{'#' * 70}")

        prompt = make_task_prompt(task)

        traj = rlm(
            prompt=prompt,
            model=model,
            system_prompt=QWEN_2B_SYSTEM_PROMPT,
            max_iterations=6,
            stdout_max_chars=1000,
            code_timeout=30,
            verbose=True,
        )

        # Analyze results
        task_result = {
            "task": task["name"],
            "question": task["question"],
            "expected": task["expected"],
            "answer": traj.answer,
            "terminated": traj.terminated,
            "num_turns": len(traj.turns),
            "total_time": traj.total_time,
        }

        # Check quality indicators
        had_valid_code = any(t.get("parsed_code") for t in traj.turns)
        had_context_access = any(
            "context" in (t.get("parsed_code") or "")
            for t in traj.turns
        )
        had_llm_query = any(
            "llm_query" in (t.get("parsed_code") or "")
            for t in traj.turns
        )
        had_final = any(
            "FINAL" in (t.get("parsed_code") or "")
            for t in traj.turns
        )
        had_errors = any(t.get("error") for t in traj.turns)

        task_result["quality"] = {
            "valid_code": had_valid_code,
            "uses_context": had_context_access,
            "uses_llm_query": had_llm_query,
            "uses_FINAL": had_final,
            "has_errors": had_errors,
        }

        # Check if answer is roughly correct
        if traj.answer and task["expected"] in str(traj.answer):
            task_result["correct"] = True
            logger.info(f"CORRECT: Answer contains '{task['expected']}'")
        else:
            task_result["correct"] = False
            logger.info(f"INCORRECT/NO ANSWER: got '{traj.answer}', expected '{task['expected']}'")

        all_results.append(task_result)

        # Save individual trajectory
        traj_path = results_dir / f"trajectory_{task['name']}.json"
        with open(traj_path, "w") as f:
            json.dump(trajectory_to_dict(traj), f, indent=2, default=str)

        logger.info(f"Trajectory saved to {traj_path}")

    # ============================================================
    # Summary
    # ============================================================
    logger.info(f"\n{'=' * 70}")
    logger.info("SMOKE TEST SUMMARY")
    logger.info(f"{'=' * 70}")

    for r in all_results:
        q = r["quality"]
        status = "CORRECT" if r["correct"] else ("TERMINATED" if r["terminated"] else "FAILED")
        logger.info(
            f"  {r['task']:20s} | {status:10s} | "
            f"code={q['valid_code']} ctx={q['uses_context']} "
            f"llm_q={q['uses_llm_query']} final={q['uses_FINAL']} "
            f"err={q['has_errors']} | {r['num_turns']} turns | {r['total_time']:.1f}s"
        )

    # Key diagnostic questions
    n = len(all_results)
    pct_valid_code = sum(r["quality"]["valid_code"] for r in all_results) / n * 100
    pct_context = sum(r["quality"]["uses_context"] for r in all_results) / n * 100
    pct_llm_query = sum(r["quality"]["uses_llm_query"] for r in all_results) / n * 100
    pct_final = sum(r["quality"]["uses_FINAL"] for r in all_results) / n * 100
    pct_correct = sum(r["correct"] for r in all_results) / n * 100

    logger.info(f"\nDIAGNOSTICS:")
    logger.info(f"  Produces valid Python:    {pct_valid_code:.0f}%")
    logger.info(f"  Accesses context:         {pct_context:.0f}%")
    logger.info(f"  Uses llm_query():         {pct_llm_query:.0f}%")
    logger.info(f"  Uses FINAL/FINAL_VAR:     {pct_final:.0f}%")
    logger.info(f"  Correct answers:          {pct_correct:.0f}%")

    logger.info(f"\nModel stats: {model.total_stats()}")

    # Save summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "results": all_results,
            "diagnostics": {
                "pct_valid_code": pct_valid_code,
                "pct_context": pct_context,
                "pct_llm_query": pct_llm_query,
                "pct_final": pct_final,
                "pct_correct": pct_correct,
            },
            "model_stats": model.total_stats(),
        }, f, indent=2, default=str)

    logger.info(f"\nFull results saved to {results_dir}/")

    # Decision guidance
    logger.info(f"\n{'=' * 70}")
    logger.info("DECISION GUIDANCE:")
    if pct_valid_code >= 80:
        logger.info("  Model CAN produce valid Python. Proceed with SFT/RL.")
    elif pct_valid_code >= 40:
        logger.info("  Model SOMETIMES produces valid Python. SFT on trajectories should help.")
        logger.info("  Consider also: prompt engineering, constrained decoding.")
    else:
        logger.info("  Model STRUGGLES with Python. Options:")
        logger.info("    1. Intensive SFT on synthetic REPL trajectories")
        logger.info("    2. Scale up to 4B+ model")
        logger.info("    3. Constrained decoding to force valid syntax")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
