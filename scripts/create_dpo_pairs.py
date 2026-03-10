#!/usr/bin/env python3
"""
Create DPO preference pairs from collected trajectory files.

Takes one or more trajectory JSON files (all_trajectories.json from collection
runs), groups trajectories by task_id, and for each task with both correct and
incorrect trajectories, creates (chosen, rejected) pairs.

Output format (JSONL, one pair per line):
{
  "chosen": {
    "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...],
    "score": 1.0,
    "reward": 0.95,
    "task_id": "niah_30000_50000_0.1",
    "task_type": "niah",
    "num_turns": 2,
    "terminated": true
  },
  "rejected": {
    "messages": [...],
    "score": 0.0,
    "reward": 0.02,
    "task_id": "niah_30000_50000_0.1",
    "task_type": "niah",
    "num_turns": 3,
    "terminated": false
  },
  "task_id": "niah_30000_50000_0.1",
  "task_type": "niah",
  "reward_gap": 0.93,
  "source_files": ["data/trajectories/run1/all_trajectories.json"]
}

Usage:
    # Single file
    uv run python scripts/create_dpo_pairs.py \
        data/trajectories/star_v2/all_trajectories.json \
        --output data/dpo/pairs.jsonl

    # Multiple files (merged)
    uv run python scripts/create_dpo_pairs.py \
        data/trajectories/run1/all_trajectories.json \
        data/trajectories/run2/all_trajectories.json \
        --output data/dpo/pairs_merged.jsonl

    # With filters
    uv run python scripts/create_dpo_pairs.py \
        data/trajectories/star_v2/all_trajectories.json \
        --min-reward-gap 0.2 \
        --max-pairs-per-task 3 \
        --require-termination \
        --output data/dpo/pairs_filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reconstruct_messages(trajectory: dict, system_prompt: str) -> list[dict]:
    """Reconstruct the full message history from a stored trajectory.

    The trajectory stores turns with raw_response, stdout, stderr, error, etc.
    We reconstruct the message list as it would appear during RLM execution:
    - system prompt
    - initial metadata (user message)
    - alternating assistant (code) / user (feedback) messages
    """
    messages = []

    # System prompt
    traj_system = trajectory.get("system_prompt", system_prompt)
    # The stored system_prompt may be truncated — use the full one if available
    if traj_system and len(traj_system) > 50:
        messages.append({"role": "system", "content": traj_system})
    else:
        messages.append({"role": "system", "content": system_prompt})

    # Initial metadata (user message)
    prompt = trajectory.get("prompt", "")
    prompt_length = trajectory.get("prompt_length", len(prompt))

    # Remove trailing "..." from truncated prompts
    prefix = prompt.rstrip(".")
    if len(prefix) == len(prompt):
        prefix = prompt[:500]

    if prompt_length > 500:
        prefix_display = prefix + f"\n... [{prompt_length - 500} more characters]"
    else:
        prefix_display = prefix

    initial_metadata = "\n".join([
        f"Context length: {prompt_length} characters",
        f"Context prefix:\n{prefix_display}",
        f"\nAvailable functions: llm_query(prompt_str), FINAL(answer), FINAL_VAR(variable_name)",
        f"Available variable: context (the full input, {prompt_length} chars)",
    ])
    messages.append({"role": "user", "content": initial_metadata})

    # Turns
    turns = trajectory.get("turns", [])
    for i, turn in enumerate(turns):
        code = turn.get("parsed_code", "")
        raw_response = turn.get("raw_response", "")

        # Assistant message: use raw_response if available (preserves ```repl blocks)
        # otherwise wrap parsed_code
        if raw_response:
            messages.append({"role": "assistant", "content": raw_response})
        elif code:
            messages.append({"role": "assistant", "content": f"```repl\n{code}\n```"})

        # User feedback (for non-terminal turns)
        if not turn.get("terminated", False):
            error = turn.get("error")
            stdout = turn.get("stdout", "")

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

    return messages


def compute_trajectory_reward(trajectory: dict) -> float:
    """Compute a reward for a trajectory.

    Uses the stored score if available, otherwise computes a simple reward
    based on score + format quality.
    """
    # Use pre-computed reward if available
    if "reward" in trajectory:
        return float(trajectory["reward"])

    score = float(trajectory.get("score", 0))
    terminated = trajectory.get("terminated", False)
    turns = trajectory.get("turns", [])

    # Simple composite reward
    format_bonus = 0.0
    if terminated:
        format_bonus += 0.05
    n_errors = sum(1 for t in turns if t.get("error"))
    format_bonus -= 0.02 * n_errors

    return 0.85 * score + 0.15 * max(format_bonus + 0.1, 0)


def load_trajectories(file_paths: list[str]) -> tuple[list[dict], list[str]]:
    """Load trajectories from one or more JSON files.

    Returns (trajectories_list, source_files_list).
    Each trajectory gets a '_source_file' field added.
    """
    all_trajectories = []
    source_files = []

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {path}: {e}")
            continue

        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, dict) and "trajectories" in data:
            trajectories = data["trajectories"]
        else:
            logger.warning(f"Unexpected format in {path}, skipping")
            continue

        for traj in trajectories:
            traj["_source_file"] = str(path)

        all_trajectories.extend(trajectories)
        source_files.append(str(path))
        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")

    logger.info(f"Total: {len(all_trajectories)} trajectories from {len(source_files)} files")
    return all_trajectories, source_files


def group_by_task(trajectories: list[dict]) -> dict[str, list[dict]]:
    """Group trajectories by task_id."""
    groups = defaultdict(list)
    n_no_id = 0

    for traj in trajectories:
        task_id = traj.get("task_id")
        if task_id is None:
            # Generate a pseudo task_id from prompt hash if missing
            prompt = traj.get("prompt", "")
            task_id = f"auto_{hash(prompt[:200]) % 100000}"
            n_no_id += 1
        groups[task_id].append(traj)

    if n_no_id > 0:
        logger.warning(f"{n_no_id} trajectories had no task_id (auto-assigned)")

    logger.info(f"Grouped into {len(groups)} unique tasks")
    return dict(groups)


def create_pairs(
    groups: dict[str, list[dict]],
    system_prompt: str,
    min_reward_gap: float = 0.1,
    max_pairs_per_task: int = 5,
    require_termination: bool = False,
    require_correct_chosen: bool = True,
) -> tuple[list[dict], dict]:
    """Create (chosen, rejected) pairs from grouped trajectories.

    For each task with mixed outcomes:
    1. Split trajectories into "good" (reward > threshold) and "bad" (reward <= threshold)
    2. Pair each good trajectory with a bad one
    3. Ensure minimum reward gap between chosen and rejected

    Returns (pairs, stats).
    """
    all_pairs = []
    stats = {
        "n_tasks": len(groups),
        "n_tasks_with_pairs": 0,
        "n_tasks_all_correct": 0,
        "n_tasks_all_incorrect": 0,
        "n_tasks_mixed": 0,
        "n_pairs_total": 0,
        "n_pairs_skipped_gap": 0,
        "n_pairs_skipped_termination": 0,
        "avg_reward_gap": 0,
        "reward_gap_histogram": {},
    }

    reward_gaps = []

    for task_id, trajectories in groups.items():
        # Compute rewards
        for traj in trajectories:
            traj["_reward"] = compute_trajectory_reward(traj)

        # Split into correct (score > 0) and incorrect
        correct = [t for t in trajectories if t.get("score", 0) > 0]
        incorrect = [t for t in trajectories if t.get("score", 0) <= 0]

        if len(correct) == len(trajectories):
            stats["n_tasks_all_correct"] += 1
        elif len(incorrect) == len(trajectories):
            stats["n_tasks_all_incorrect"] += 1
        else:
            stats["n_tasks_mixed"] += 1

        # Need both correct and incorrect for pairs
        if not correct or not incorrect:
            # Fallback: if require_correct_chosen is False, we can also pair
            # high-reward vs low-reward even if both are technically "correct"
            # or both "incorrect"
            if not require_correct_chosen and len(trajectories) >= 2:
                sorted_trajs = sorted(trajectories, key=lambda t: t["_reward"], reverse=True)
                best = sorted_trajs[:len(sorted_trajs)//2]
                worst = sorted_trajs[len(sorted_trajs)//2:]
                if not best or not worst:
                    continue
                correct = best
                incorrect = worst
            else:
                continue

        stats["n_tasks_with_pairs"] += 1

        # Sort by reward (best first for correct, worst first for incorrect)
        correct_sorted = sorted(correct, key=lambda t: t["_reward"], reverse=True)
        incorrect_sorted = sorted(incorrect, key=lambda t: t["_reward"])

        # Create pairs: pair best correct with worst incorrect, etc.
        n_pairs = min(max_pairs_per_task, len(correct_sorted), len(incorrect_sorted))
        task_pairs = []

        for i in range(n_pairs):
            chosen = correct_sorted[i % len(correct_sorted)]
            rejected = incorrect_sorted[i % len(incorrect_sorted)]

            # Check reward gap
            gap = chosen["_reward"] - rejected["_reward"]
            if gap < min_reward_gap:
                stats["n_pairs_skipped_gap"] += 1
                continue

            # Check termination requirement
            if require_termination and not chosen.get("terminated", False):
                stats["n_pairs_skipped_termination"] += 1
                continue

            # Reconstruct messages
            chosen_messages = reconstruct_messages(chosen, system_prompt)
            rejected_messages = reconstruct_messages(rejected, system_prompt)

            # Skip if messages are empty or trivial
            if len(chosen_messages) < 3 or len(rejected_messages) < 3:
                continue

            pair = {
                "chosen": {
                    "messages": chosen_messages,
                    "score": float(chosen.get("score", 0)),
                    "reward": float(chosen["_reward"]),
                    "task_id": task_id,
                    "task_type": chosen.get("task_type", "unknown"),
                    "num_turns": len(chosen.get("turns", [])),
                    "terminated": chosen.get("terminated", False),
                },
                "rejected": {
                    "messages": rejected_messages,
                    "score": float(rejected.get("score", 0)),
                    "reward": float(rejected["_reward"]),
                    "task_id": task_id,
                    "task_type": rejected.get("task_type", "unknown"),
                    "num_turns": len(rejected.get("turns", [])),
                    "terminated": rejected.get("terminated", False),
                },
                "task_id": task_id,
                "task_type": chosen.get("task_type", "unknown"),
                "reward_gap": float(gap),
                "source_files": list(set(
                    [chosen.get("_source_file", ""), rejected.get("_source_file", "")]
                )),
            }

            task_pairs.append(pair)
            reward_gaps.append(gap)

        all_pairs.extend(task_pairs)

    stats["n_pairs_total"] = len(all_pairs)
    stats["avg_reward_gap"] = float(sum(reward_gaps) / len(reward_gaps)) if reward_gaps else 0

    # Reward gap histogram (buckets of 0.1)
    for gap in reward_gaps:
        bucket = f"{int(gap * 10) / 10:.1f}-{int(gap * 10 + 1) / 10:.1f}"
        stats["reward_gap_histogram"][bucket] = stats["reward_gap_histogram"].get(bucket, 0) + 1

    return all_pairs, stats


def main():
    parser = argparse.ArgumentParser(
        description="Create DPO preference pairs from trajectory files"
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Trajectory JSON files (all_trajectories.json)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSONL path (default: data/dpo/pairs.jsonl)"
    )
    parser.add_argument(
        "--min-reward-gap", type=float, default=0.1,
        help="Minimum reward gap between chosen and rejected (default: 0.1)"
    )
    parser.add_argument(
        "--max-pairs-per-task", type=int, default=5,
        help="Maximum pairs to create per task (default: 5)"
    )
    parser.add_argument(
        "--require-termination", action="store_true",
        help="Only use chosen trajectories that terminated properly"
    )
    parser.add_argument(
        "--no-require-correct", action="store_true",
        help="Allow pairing high-reward vs low-reward even if both correct/incorrect"
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="System prompt to use (default: auto-detect from trajectory or use Qwen3.5-35B)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Shuffle output pairs"
    )
    parser.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false",
        help="Do not shuffle output pairs"
    )
    args = parser.parse_args()

    # Load system prompt
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        try:
            from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
            system_prompt = QWEN35_35B_SYSTEM_PROMPT
        except ImportError:
            try:
                from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
                system_prompt = QWEN_2B_SYSTEM_PROMPT
            except ImportError:
                system_prompt = "You are a code execution agent."
                logger.warning("Could not load system prompt, using default")

    # Load trajectories
    trajectories, source_files = load_trajectories(args.inputs)
    if not trajectories:
        logger.error("No trajectories loaded!")
        return

    # Group by task
    groups = group_by_task(trajectories)

    # Log trajectory stats
    scores = [t.get("score", 0) for t in trajectories]
    n_correct = sum(1 for s in scores if s > 0)
    n_incorrect = sum(1 for s in scores if s <= 0)
    logger.info(f"Trajectory stats: {n_correct} correct, {n_incorrect} incorrect, "
                f"{n_correct / len(scores):.1%} success rate")

    # Create pairs
    pairs, stats = create_pairs(
        groups,
        system_prompt=system_prompt,
        min_reward_gap=args.min_reward_gap,
        max_pairs_per_task=args.max_pairs_per_task,
        require_termination=args.require_termination,
        require_correct_chosen=not args.no_require_correct,
    )

    if not pairs:
        logger.error("No valid pairs created! Try lowering --min-reward-gap or "
                      "collecting more diverse trajectories.")
        return

    # Shuffle
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(pairs)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/dpo/pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pairs as JSONL
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, default=str) + "\n")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    stats["source_files"] = source_files
    stats["output_path"] = str(output_path)
    stats["args"] = vars(args)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"DPO PAIR CREATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Source files: {len(source_files)}")
    logger.info(f"  Total trajectories: {len(trajectories)}")
    logger.info(f"  Unique tasks: {stats['n_tasks']}")
    logger.info(f"  Tasks with pairs: {stats['n_tasks_with_pairs']}")
    logger.info(f"    All correct: {stats['n_tasks_all_correct']}")
    logger.info(f"    All incorrect: {stats['n_tasks_all_incorrect']}")
    logger.info(f"    Mixed: {stats['n_tasks_mixed']}")
    logger.info(f"  Total pairs: {stats['n_pairs_total']}")
    logger.info(f"  Skipped (gap too small): {stats['n_pairs_skipped_gap']}")
    logger.info(f"  Skipped (not terminated): {stats['n_pairs_skipped_termination']}")
    logger.info(f"  Avg reward gap: {stats['avg_reward_gap']:.3f}")
    logger.info(f"  Reward gap distribution:")
    for bucket, count in sorted(stats["reward_gap_histogram"].items()):
        bar = "#" * min(count, 40)
        logger.info(f"    {bucket}: {count:4d} {bar}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Stats:  {stats_path}")
    logger.info(f"{'='*60}")

    # Print sample pair for verification
    if pairs:
        sample = pairs[0]
        logger.info(f"\nSample pair (task: {sample['task_id']}):")
        logger.info(f"  Chosen:   score={sample['chosen']['score']:.2f} "
                     f"reward={sample['chosen']['reward']:.3f} "
                     f"turns={sample['chosen']['num_turns']} "
                     f"msgs={len(sample['chosen']['messages'])}")
        logger.info(f"  Rejected: score={sample['rejected']['score']:.2f} "
                     f"reward={sample['rejected']['reward']:.3f} "
                     f"turns={sample['rejected']['num_turns']} "
                     f"msgs={len(sample['rejected']['messages'])}")
        logger.info(f"  Gap: {sample['reward_gap']:.3f}")


if __name__ == "__main__":
    main()
