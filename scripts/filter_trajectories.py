#!/usr/bin/env python3
"""
Filter collected trajectories for SFT training.

Criteria:
1. Correct answer (score > 0)
2. Terminated properly (FINAL/FINAL_VAR called)
3. Not flagged for template removal
4. Reasonable number of turns (1-8)
5. No excessive errors

Outputs per-turn SFT samples: (system_prompt, metadata, code) triples.

Usage:
    uv run python scripts/filter_trajectories.py \
        data/trajectories/model_timestamp/all_trajectories.json \
        --output data/filtered/filtered_YYYYMMDD.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filter")


def trajectory_to_sft_samples(trajectory: dict, system_prompt: str) -> list[dict]:
    """
    Convert a trajectory to per-turn SFT training samples.

    Each sample is a (messages, completion) pair where:
    - messages: conversation up to this turn
    - completion: the model's code for this turn

    This is how the paper trains: per-turn supervised learning.
    """
    samples = []

    # Build conversation incrementally
    messages = [{"role": "system", "content": system_prompt}]

    for i, turn in enumerate(trajectory.get("turns", [])):
        code = turn.get("parsed_code")
        if not code:
            continue

        raw_response = turn.get("raw_response", "")

        # The "input" is all messages so far
        # The "output" is the model's response (with code block)
        sample = {
            "messages": list(messages),  # Copy
            "completion": f"```repl\n{code}\n```",
            "turn_index": i,
            "task_id": trajectory.get("task_id", "unknown"),
        }
        samples.append(sample)

        # Add this turn to the conversation for the next sample
        messages.append({"role": "assistant", "content": f"```repl\n{code}\n```"})

        # Add feedback (stdout or error)
        stdout = turn.get("stdout", "")
        error = turn.get("error", "")
        if error:
            feedback = f"Error: {error}"
        elif stdout:
            feedback = f"Output:\n{stdout[:1000]}"
        else:
            feedback = "Code executed successfully."

        if not turn.get("terminated"):
            messages.append({"role": "user", "content": feedback})

    return samples


def filter_trajectories(
    trajectories: list[dict],
    system_prompt: str,
    max_turns: int = 8,
    max_errors: int = 2,
) -> tuple[list[dict], dict]:
    """
    Filter trajectories and convert to SFT samples.

    Returns (samples, stats).
    """
    stats = {
        "total": len(trajectories),
        "correct": 0,
        "terminated": 0,
        "passed_filter": 0,
        "total_samples": 0,
        "removed_reasons": {},
    }

    all_samples = []

    for traj in trajectories:
        score = traj.get("score", 0)
        terminated = traj.get("terminated", False)
        turns = traj.get("turns", [])
        n_turns = len(turns)
        n_errors = sum(1 for t in turns if t.get("error"))
        flagged = traj.get("flagged_for_removal", False)

        if score > 0:
            stats["correct"] += 1
        if terminated:
            stats["terminated"] += 1

        # Apply filters
        reason = None
        if score <= 0:
            reason = "incorrect"
        elif not terminated:
            reason = "not_terminated"
        elif flagged:
            reason = "template_flagged"
        elif n_turns > max_turns:
            reason = "too_many_turns"
        elif n_errors > max_errors:
            reason = "too_many_errors"

        if reason:
            stats["removed_reasons"][reason] = stats["removed_reasons"].get(reason, 0) + 1
            continue

        stats["passed_filter"] += 1

        # Convert to SFT samples
        samples = trajectory_to_sft_samples(traj, system_prompt)
        all_samples.extend(samples)
        stats["total_samples"] += len(samples)

    return all_samples, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to trajectories JSON file")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-errors", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path) as f:
        trajectories = json.load(f)

    logger.info(f"Loaded {len(trajectories)} trajectories from {input_path}")

    from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT

    samples, stats = filter_trajectories(
        trajectories,
        system_prompt=QWEN_2B_SYSTEM_PROMPT,
        max_turns=args.max_turns,
        max_errors=args.max_errors,
    )

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/filtered") / f"sft_samples_{input_path.parent.name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, default=str) + "\n")

    logger.info(f"\nFilter results:")
    logger.info(f"  Total trajectories: {stats['total']}")
    logger.info(f"  Correct: {stats['correct']}")
    logger.info(f"  Terminated: {stats['terminated']}")
    logger.info(f"  Passed filter: {stats['passed_filter']}")
    logger.info(f"  SFT samples: {stats['total_samples']}")
    logger.info(f"  Removed: {stats['removed_reasons']}")
    logger.info(f"  Saved to: {output_path}")

    # Also save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
