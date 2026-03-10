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


def reconstruct_initial_metadata(trajectory: dict) -> str:
    """
    Reconstruct the initial metadata message from stored trajectory data.

    During RLM execution, the model sees metadata (context length, prefix,
    available functions) instead of the full prompt. We reconstruct this
    from the stored prompt prefix and length.
    """
    prompt = trajectory.get("prompt", "")
    prompt_length = trajectory.get("prompt_length", len(prompt))

    # Remove trailing "..." from truncated prompts
    prefix = prompt.rstrip(".")
    if len(prefix) < len(prompt):
        prefix = prefix  # Was truncated
    else:
        prefix = prompt[:500]

    if prompt_length > 500:
        prefix_display = prefix + f"\n... [{prompt_length - 500} more characters]"
    else:
        prefix_display = prefix

    parts = [
        f"Context length: {prompt_length} characters",
        f"Context prefix:\n{prefix_display}",
        f"\nAvailable functions: llm_query(prompt_str), FINAL(answer), FINAL_VAR(variable_name)",
        f"Available variable: context (the full input, {prompt_length} chars)",
    ]
    return "\n".join(parts)


def trajectory_to_sft_samples(trajectory: dict, system_prompt: str) -> list[dict]:
    """
    Convert a trajectory to per-turn SFT training samples.

    Each sample is a (messages, completion) pair where:
    - messages: conversation up to this turn (matching actual RLM conversation)
    - completion: the model's code for this turn

    The initial metadata message is reconstructed from stored prompt data,
    matching what the model sees during actual RLM execution.
    """
    samples = []

    # Build conversation incrementally — matches build_initial_message() in rlm.py
    initial_metadata = reconstruct_initial_metadata(trajectory)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_metadata},
    ]

    for i, turn in enumerate(trajectory.get("turns", [])):
        code = turn.get("parsed_code")
        if not code:
            continue

        error = turn.get("error", "")
        stdout = turn.get("stdout", "")

        # Only create training samples from non-error turns.
        # Error turns teach bad patterns (failed code as positive examples).
        if not error:
            sample = {
                "messages": list(messages),  # Copy
                "completion": f"```repl\n{code}\n```",
                "turn_index": i,
                "task_id": trajectory.get("task_id", "unknown"),
            }
            samples.append(sample)

        # Still add ALL turns to conversation context so later turns
        # have the full history (including error recovery)
        messages.append({"role": "assistant", "content": f"```repl\n{code}\n```"})

        if not turn.get("terminated"):
            if error:
                feedback = f"Error executing code:\n{error}"
            elif stdout:
                stdout_display = stdout[:1000]
                if len(stdout) > 1000:
                    stdout_display += f"\n... [{len(stdout) - 1000} more chars]"
                feedback = f"Output:\n{stdout_display}"
            else:
                feedback = "Code executed successfully."
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

    from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
    from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT

    # Auto-detect prompt based on trajectory content
    system_prompt = QWEN35_35B_SYSTEM_PROMPT  # Default to newer prompt

    samples, stats = filter_trajectories(
        trajectories,
        system_prompt=system_prompt,
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
