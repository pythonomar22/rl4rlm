#!/usr/bin/env python3
"""
Enhanced trajectory filtering for Rejection-Sampling SFT (RS-SFT).

Goes beyond basic score/termination filtering to ensure high-quality
training data that avoids the GRPO failure modes:
  1. Format rigidity (strict parsing that fails on format variation)
  2. Shortcut learning (single-pass on long docs, missing chunking)
  3. Template convergence (identical code across tasks)

Usage:
    # Filter collected trajectories
    uv run python scripts/filter_rs_sft.py \
        data/trajectories/base_rs_sft/all_trajectories.json \
        --output data/filtered/rs_sft_v1.jsonl \
        --min-score 0.9 \
        --balance-tasks \
        --max-per-task-type 50

    # Strict filtering for highest quality
    uv run python scripts/filter_rs_sft.py \
        data/trajectories/base_rs_sft/all_trajectories.json \
        --output data/filtered/rs_sft_strict.jsonl \
        --min-score 0.95 \
        --require-multi-turn \
        --check-format-robustness \
        --balance-tasks
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
from scripts.filter_trajectories import trajectory_to_sft_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filter_rs_sft")


# Tasks where multi-turn is essential for correct behavior
# (single-turn solutions are shortcuts that won't generalize)
MULTI_TURN_REQUIRED = {
    "hard_multi_hop",  # Must decompose: find entity A, then search for A's property
    "cross_doc_compare",  # Must process docs separately, then compare
    "multi_hop_qa",  # Chain of queries needed
}

# Tasks where llm_query sub-calls are essential
# (tasks with long context that can't fit in one call)
SUBCALL_REQUIRED = {
    "niah", "multi_niah", "doc_classify", "dataframe_qa",
    "multi_hop_qa", "notebook_qa", "hard_niah",
    "hard_multi_hop", "event_counting", "cross_doc_compare",
    "key_value_retrieval",
}

# Patterns indicating format-rigid parsing (the root cause of GRPO regression)
FORMAT_RIGID_PATTERNS = [
    # Strict format-specific extraction
    r'if\s+"[A-Z]+:\s*"\s+in\s+',           # if "ORG: " in line
    r'\.split\("[A-Z]+:\s*"\s*,',             # .split("ORG: ", 1)
    r'if\s+line\.startswith\("[A-Z]+:',       # if line.startswith("Category:")
    r'\.split\(":\s*"\)\[1\]',               # .split(": ")[1] -- brittle indexing
    r'parts\s*=\s*line\.split\("[^"]*"\)\s*\n.*parts\[',  # split then index
    # Overly specific prompt instructions that demand format
    r'Format:\s*[\'"]?[A-Z]+:\s+',            # Format: "ORG: Name"
    r'Return\s+as:\s+[\'"]?[A-Z]+:',          # Return as: "CATEGORY: ..."
]

# Patterns indicating format-ROBUST parsing (base model's approach, which we want)
FORMAT_ROBUST_PATTERNS = [
    r'for\s+line\s+in\s+\w+\.strip\(\)\.split',  # loose line splitting
    r'\.strip\(\)\.split\("\\n"\)',               # strip+split on newlines
    r'if\s+\w+\.strip\(\):',                      # skip empty lines
    r'\w+\.add\(\w+\.strip\(\)\)',                 # add stripped items to set
    r'Counter\(\w+\)\.most_common',               # majority voting
    r'try:.*except',                               # error handling
]


def check_format_robustness(trajectory: dict) -> tuple[bool, str]:
    """Check if trajectory code uses format-robust (not rigid) parsing.

    Returns (is_robust, reason).

    Format rigidity is the root cause of GRPO regressions:
    - RL trains code that demands "ORG: Name" format from sub-calls
    - When sub-calls return data in a different format, the code silently drops it
    - The base model uses loose parsing (line splitting) which handles any format

    We want RS-SFT to train on trajectories with ROBUST parsing.
    """
    turns = trajectory.get("turns", [])
    all_code = "\n".join(t.get("parsed_code") or "" for t in turns)

    if not all_code:
        return False, "no_code"

    # Check for rigid patterns
    rigid_count = 0
    for pattern in FORMAT_RIGID_PATTERNS:
        if re.search(pattern, all_code):
            rigid_count += 1

    # Check for robust patterns
    robust_count = 0
    for pattern in FORMAT_ROBUST_PATTERNS:
        if re.search(pattern, all_code):
            robust_count += 1

    if rigid_count >= 2:
        return False, f"rigid_parsing ({rigid_count} rigid patterns)"

    if rigid_count > 0 and robust_count == 0:
        return False, f"rigid_parsing (rigid without fallback)"

    return True, "ok"


def check_uses_subcalls(trajectory: dict) -> bool:
    """Check if trajectory uses llm_query sub-calls."""
    turns = trajectory.get("turns", [])
    for turn in turns:
        code = turn.get("parsed_code") or ""
        if "llm_query" in code:
            return True
    return False


def check_reasonable_chunk_size(trajectory: dict) -> tuple[bool, str]:
    """Check that chunk sizes are reasonable (5K-25K, not oversized).

    GRPO teaches oversized chunks (30K+) which cause boundary misses.
    """
    turns = trajectory.get("turns", [])
    for turn in turns:
        code = turn.get("parsed_code") or ""
        # Look for chunk_size = NNNN assignments
        for match in re.finditer(r'chunk_size\s*=\s*(\d+)', code):
            size = int(match.group(1))
            if size > 30000:
                return False, f"chunk_size={size} (too large, boundary misses)"
            if size < 1000:
                return False, f"chunk_size={size} (too small, excessive calls)"
    return True, "ok"


def check_no_fstring_bug(trajectory: dict) -> bool:
    """Check for the f-string bug: llm_query("{context}") without f-string."""
    turns = trajectory.get("turns", [])
    for turn in turns:
        code = turn.get("parsed_code") or ""
        # Pattern: llm_query("{...}") without f-string prefix
        if re.search(r'llm_query\(\s*".*\{', code) and not re.search(r'llm_query\(\s*f"', code):
            return False
    return True


def check_proper_termination(trajectory: dict) -> bool:
    """Check that trajectory terminates with FINAL/FINAL_VAR."""
    turns = trajectory.get("turns", [])
    for turn in turns:
        code = turn.get("parsed_code") or ""
        if "FINAL(" in code or "FINAL_VAR(" in code:
            if turn.get("terminated", False):
                return True
    return trajectory.get("terminated", False)


def count_code_turns(trajectory: dict) -> int:
    """Count turns that contain parsed code."""
    return sum(
        1 for t in trajectory.get("turns", [])
        if t.get("parsed_code")
    )


def count_error_turns(trajectory: dict) -> int:
    """Count turns with errors."""
    return sum(
        1 for t in trajectory.get("turns", [])
        if t.get("error")
    )


def filter_rs_sft(
    trajectories: list[dict],
    system_prompt: str,
    min_score: float = 0.9,
    max_turns: int = 8,
    max_errors: int = 1,
    require_multi_turn: bool = False,
    require_subcalls: bool = True,
    check_format: bool = False,
    check_chunks: bool = False,
    max_per_task_type: int | None = None,
    balance_tasks: bool = False,
) -> tuple[list[dict], dict]:
    """Filter trajectories for RS-SFT training with enhanced quality checks.

    RS-SFT filtering is STRICTER than standard SFT filtering because
    we're selecting the BEST trajectories as exemplars. The model will
    learn to reproduce exactly these patterns.

    Args:
        trajectories: Raw trajectory dicts from collection
        system_prompt: System prompt for SFT sample conversion
        min_score: Minimum task score to include (0.9 = near-perfect)
        max_turns: Maximum turns allowed
        max_errors: Maximum error turns allowed (stricter than regular: 1 not 2)
        require_multi_turn: Require 2+ code turns for complex tasks
        require_subcalls: Require llm_query usage
        check_format: Check for format-robust parsing (anti-rigidity filter)
        check_chunks: Check for reasonable chunk sizes
        max_per_task_type: Cap trajectories per task type
        balance_tasks: Equalize representation across task types

    Returns:
        (sft_samples, stats)
    """
    stats = {
        "total_input": len(trajectories),
        "passed": 0,
        "removed": defaultdict(int),
        "by_task_type": defaultdict(lambda: {"input": 0, "passed": 0}),
        "filter_details": [],
    }

    passed_trajectories = []

    for traj in trajectories:
        task_type = traj.get("task_type", "unknown")
        stats["by_task_type"][task_type]["input"] += 1

        # --- Core filters ---
        score = traj.get("score", 0)
        if score < min_score:
            stats["removed"]["low_score"] += 1
            continue

        if not check_proper_termination(traj):
            stats["removed"]["not_terminated"] += 1
            continue

        n_turns = count_code_turns(traj)
        if n_turns > max_turns:
            stats["removed"]["too_many_turns"] += 1
            continue

        n_errors = count_error_turns(traj)
        if n_errors > max_errors:
            stats["removed"]["too_many_errors"] += 1
            continue

        # --- RS-SFT quality filters ---

        # Multi-turn requirement for complex tasks
        if require_multi_turn and task_type in MULTI_TURN_REQUIRED:
            if n_turns < 2:
                stats["removed"]["needs_multi_turn"] += 1
                continue

        # Sub-call requirement
        if require_subcalls and task_type in SUBCALL_REQUIRED:
            if not check_uses_subcalls(traj):
                stats["removed"]["no_subcalls"] += 1
                continue

        # F-string bug check
        if not check_no_fstring_bug(traj):
            stats["removed"]["fstring_bug"] += 1
            continue

        # Format robustness check (anti-rigidity)
        if check_format:
            is_robust, reason = check_format_robustness(traj)
            if not is_robust:
                stats["removed"][f"format_rigid:{reason}"] += 1
                continue

        # Chunk size check
        if check_chunks:
            ok, reason = check_reasonable_chunk_size(traj)
            if not ok:
                stats["removed"][f"bad_chunks:{reason}"] += 1
                continue

        passed_trajectories.append(traj)
        stats["by_task_type"][task_type]["passed"] += 1

    # --- Task balancing ---
    if balance_tasks or max_per_task_type:
        by_type = defaultdict(list)
        for traj in passed_trajectories:
            by_type[traj.get("task_type", "unknown")].append(traj)

        if balance_tasks:
            # Equalize: take min(available, target) per type
            # Target = total / n_types, rounded up
            n_types = len(by_type)
            if max_per_task_type:
                target_per_type = max_per_task_type
            else:
                target_per_type = max(10, len(passed_trajectories) // n_types)

            balanced = []
            for task_type in sorted(by_type.keys()):
                type_trajs = by_type[task_type]
                # Sort by score (descending) to keep best trajectories
                type_trajs.sort(key=lambda t: t.get("score", 0), reverse=True)
                selected = type_trajs[:target_per_type]
                balanced.extend(selected)
                stats["by_task_type"][task_type]["balanced"] = len(selected)

            passed_trajectories = balanced

        elif max_per_task_type:
            capped = []
            for task_type in sorted(by_type.keys()):
                type_trajs = by_type[task_type]
                type_trajs.sort(key=lambda t: t.get("score", 0), reverse=True)
                selected = type_trajs[:max_per_task_type]
                capped.extend(selected)
                stats["by_task_type"][task_type]["capped"] = len(selected)

            passed_trajectories = capped

    # --- Convert to SFT samples ---
    all_samples = []
    for traj in passed_trajectories:
        samples = trajectory_to_sft_samples(traj, system_prompt)
        # Tag each sample with task type for analysis
        for s in samples:
            s["task_type"] = traj.get("task_type", "unknown")
            s["trajectory_score"] = traj.get("score", 0)
            s["strategy"] = traj.get("strategy", "standard")
        all_samples.extend(samples)

    stats["passed"] = len(passed_trajectories)
    stats["total_sft_samples"] = len(all_samples)

    return all_samples, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="RS-SFT trajectory filtering")
    parser.add_argument("input", help="Path to trajectories JSON file")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--min-score", type=float, default=0.9,
                        help="Minimum score threshold (default: 0.9)")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-errors", type=int, default=1,
                        help="Max error turns (stricter than regular SFT)")
    parser.add_argument("--require-multi-turn", action="store_true",
                        help="Require 2+ turns for complex tasks")
    parser.add_argument("--check-format-robustness", action="store_true",
                        help="Filter out format-rigid parsing code")
    parser.add_argument("--check-chunk-sizes", action="store_true",
                        help="Filter out oversized chunk patterns")
    parser.add_argument("--balance-tasks", action="store_true",
                        help="Equalize representation across task types")
    parser.add_argument("--max-per-task-type", type=int, default=None,
                        help="Cap trajectories per task type")
    args = parser.parse_args()

    # Load trajectories
    input_path = Path(args.input)
    with open(input_path) as f:
        trajectories = json.load(f)

    logger.info(f"Loaded {len(trajectories)} trajectories from {input_path}")

    # Check if these are grouped by task or flat
    # (collection may return per-attempt trajectories)
    task_types = set(t.get("task_type", "unknown") for t in trajectories)
    logger.info(f"Task types found: {sorted(task_types)}")

    # Filter
    samples, stats = filter_rs_sft(
        trajectories,
        system_prompt=QWEN35_35B_SYSTEM_PROMPT,
        min_score=args.min_score,
        max_turns=args.max_turns,
        max_errors=args.max_errors,
        require_multi_turn=args.require_multi_turn,
        require_subcalls=True,
        check_format=args.check_format_robustness,
        check_chunks=args.check_chunk_sizes,
        max_per_task_type=args.max_per_task_type,
        balance_tasks=args.balance_tasks,
    )

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/filtered") / f"rs_sft_{input_path.parent.name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save SFT samples
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, default=str) + "\n")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("RS-SFT FILTERING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Input trajectories: {stats['total_input']}")
    logger.info(f"  Passed filter: {stats['passed']}")
    logger.info(f"  SFT samples generated: {stats['total_sft_samples']}")
    logger.info(f"\n  Removal reasons:")
    for reason, count in sorted(stats["removed"].items(), key=lambda x: -x[1]):
        logger.info(f"    {reason}: {count}")
    logger.info(f"\n  Per task type:")
    for task_type in sorted(stats["by_task_type"].keys()):
        info = stats["by_task_type"][task_type]
        logger.info(
            f"    {task_type}: {info['passed']}/{info['input']} passed"
            + (f" (balanced: {info.get('balanced', info.get('capped', '?'))})"
               if 'balanced' in info or 'capped' in info else "")
        )
    logger.info(f"\n  Saved to: {output_path}")
    logger.info(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
