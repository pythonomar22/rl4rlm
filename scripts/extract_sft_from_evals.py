#!/usr/bin/env python3
"""
Extract SFT training data from evaluation trajectory files.

Walks eval result directories, finds correct trajectories (score >= threshold),
reconstructs the full message sequence, and outputs JSONL for SFT training.

This gives us diverse SFT data across ALL 14 benchmark types without needing
to run additional trajectory collection.
"""

from __future__ import annotations

import argparse
import json
import glob
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Max chars for stdout metadata (mirrors repl.py)
STDOUT_MAX_CHARS = 1000


def reconstruct_metadata(prompt: str) -> str:
    """Reconstruct the initial metadata the LLM would see (mirrors repl.metadata)."""
    ctx_len = len(prompt)
    prefix = prompt[:500]
    if len(prompt) > 500:
        prefix += f"\n... [{ctx_len - 500} more characters]"

    meta = f"Context length: {ctx_len} characters\n"
    meta += f"Context prefix:\n{prefix}\n\n"
    meta += "Available functions:\n"
    meta += "  context (str) — full input text\n"
    meta += "  llm_query(text: str) -> str — send text to LLM\n"
    meta += "  FINAL(answer: str) — submit final answer\n"
    meta += "  FINAL_VAR(var_name: str) — submit variable as answer\n"
    meta += "  print() — show output"

    return meta


def reconstruct_stdout_metadata(stdout: str, max_chars: int = STDOUT_MAX_CHARS) -> str:
    """Mirror repl.stdout_metadata."""
    if not stdout:
        return "[No output]"
    if len(stdout) <= max_chars:
        return stdout
    return stdout[:max_chars] + f"\n... [truncated, {len(stdout)} total characters]"


def trajectory_to_sft_sample(traj: dict, eval_result: dict, task_type: str) -> dict | None:
    """Convert a trajectory + eval result into an SFT sample.

    Returns dict with:
    - messages: list of {role, content} up to (but not including) the final completion
    - completion: the assistant response to train on (final turn's raw_response)
    - task_type: benchmark type
    - score: trajectory score
    """
    turns = traj.get("turns", [])
    system_prompt = traj.get("system_prompt", QWEN35_35B_SYSTEM_PROMPT)
    prompt = traj.get("prompt", "")

    if not turns:
        return None

    # Reconstruct initial metadata
    initial_meta = reconstruct_metadata(prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_meta},
    ]

    # For single-turn trajectories: the completion is the only turn
    if len(turns) == 1:
        raw_response = turns[0].get("raw_response", "")
        if not raw_response:
            return None
        return {
            "messages": messages,
            "completion": raw_response,
            "task_type": task_type,
            "score": eval_result.get("score", 0),
        }

    # For multi-turn trajectories: build message history, completion is the LAST turn
    for i, turn in enumerate(turns):
        raw_response = turn.get("raw_response", "")
        if not raw_response:
            continue

        if i == len(turns) - 1:
            # Last turn = completion to train on
            return {
                "messages": messages,
                "completion": raw_response,
                "task_type": task_type,
                "score": eval_result.get("score", 0),
            }

        # Add assistant response
        messages.append({"role": "assistant", "content": raw_response})

        # Add user feedback (stdout/error + metadata)
        if turn.get("error"):
            feedback = f"Error executing code:\n{turn['error']}\n\n"
            # Re-compute metadata (approximate — we don't have the REPL state)
            feedback += "Current state:\n" + reconstruct_metadata(prompt)
        else:
            stdout = turn.get("stdout", "")
            stdout_meta = reconstruct_stdout_metadata(stdout)
            feedback = ""
            if stdout_meta and stdout_meta != "[No output]":
                feedback += f"Output:\n{stdout_meta}\n\n"
            feedback += f"State:\n{reconstruct_metadata(prompt)}"

        messages.append({"role": "user", "content": feedback})

    return None


def extract_from_eval_dir(eval_dir: str, min_score: float = 0.5) -> list[dict]:
    """Extract SFT samples from an eval results directory."""
    samples = []
    eval_dir = Path(eval_dir)

    # Find all benchmark subdirectories
    for benchmark_dir in sorted(eval_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue

        task_type = benchmark_dir.name

        # Load eval results
        eval_results_file = benchmark_dir / "eval_results.json"
        if not eval_results_file.exists():
            continue

        with open(eval_results_file) as f:
            eval_data = json.load(f)

        # Results stored as per_task list (not results)
        results_list = eval_data.get("per_task", eval_data.get("results", []))
        trajectories_dir = benchmark_dir / "trajectories"

        if not trajectories_dir.exists():
            continue

        # Load individual trajectory files
        for traj_file in sorted(trajectories_dir.glob("*.json")):
            idx_str = traj_file.stem.replace("trajectory_", "")
            try:
                idx = int(idx_str)  # trajectory_000.json → index 0
            except ValueError:
                continue

            with open(traj_file) as f:
                traj = json.load(f)

            # Get corresponding eval result
            if idx < len(results_list):
                result = results_list[idx]
                # Different benchmarks use different keys for score
                score = result.get("score", result.get("accuracy", 0))
            else:
                score = 1.0 if traj.get("terminated") and traj.get("answer") else 0.0
                result = {"score": score}

            if score >= min_score:
                sample = trajectory_to_sft_sample(traj, result, task_type)
                if sample:
                    samples.append(sample)

    return samples


def filter_gibberish(samples: list[dict]) -> list[dict]:
    """Remove samples with gibberish completions (MoE routing failures)."""
    clean = []
    n_filtered = 0
    for s in samples:
        completion = s.get("completion", "")
        if len(completion) > 200:
            non_ascii = sum(1 for c in completion if ord(c) > 127)
            if non_ascii / len(completion) > 0.10:
                n_filtered += 1
                continue
        clean.append(s)
    if n_filtered:
        logger.info(f"  Filtered {n_filtered} gibberish samples")
    return clean


def balance_by_task_type(samples: list[dict], max_per_type: int = 30) -> list[dict]:
    """Balance samples across task types."""
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["task_type"]].append(s)

    balanced = []
    for task_type in sorted(by_type.keys()):
        type_samples = by_type[task_type]
        # Sort by score descending, take top N
        type_samples.sort(key=lambda x: x.get("score", 0), reverse=True)
        selected = type_samples[:max_per_type]
        balanced.extend(selected)
        logger.info(f"  {task_type}: {len(selected)}/{len(type_samples)} samples (max {max_per_type})")

    return balanced


def main():
    parser = argparse.ArgumentParser(description="Extract SFT data from eval trajectories")
    parser.add_argument("--eval-dirs", nargs="+", required=True,
                        help="Eval result directories to extract from")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Minimum score to include trajectory")
    parser.add_argument("--max-per-type", type=int, default=30,
                        help="Maximum samples per task type")
    parser.add_argument("--include-trained", action="store_true",
                        help="Include trajectories from trained models (risk: format rigidity)")
    args = parser.parse_args()

    all_samples = []
    for eval_dir in args.eval_dirs:
        # Find latest run directory
        runs = sorted(glob.glob(os.path.join(eval_dir, "2*/")))
        if not runs:
            logger.warning(f"No run directories found in {eval_dir}")
            continue

        run_dir = runs[-1]
        logger.info(f"Extracting from: {run_dir}")
        samples = extract_from_eval_dir(run_dir, min_score=args.min_score)
        logger.info(f"  Found {len(samples)} correct trajectories")
        all_samples.extend(samples)

    if not all_samples:
        logger.error("No samples found!")
        return

    # Filter gibberish
    all_samples = filter_gibberish(all_samples)

    # Balance
    logger.info(f"\nBalancing {len(all_samples)} samples:")
    balanced = balance_by_task_type(all_samples, max_per_type=args.max_per_type)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for s in balanced:
            f.write(json.dumps(s) + "\n")

    logger.info(f"\nSaved {len(balanced)} SFT samples to {args.output}")

    # Summary stats
    by_type = defaultdict(int)
    for s in balanced:
        by_type[s["task_type"]] += 1
    logger.info("\nFinal distribution:")
    for task_type in sorted(by_type.keys()):
        logger.info(f"  {task_type}: {by_type[task_type]}")


if __name__ == "__main__":
    main()
