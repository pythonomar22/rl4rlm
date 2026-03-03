#!/usr/bin/env python3
"""
Fix common template mistakes in collected trajectories.

From the paper:
- 16% of turns: FINAL() with a plan/explanation as the answer instead of the actual answer
- 13% of turns: FINAL_VAR() with literal text instead of a variable name

This script programmatically fixes these where possible.

Usage:
    uv run python scripts/fix_templates.py data/trajectories/model_timestamp/
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_templates")


def fix_final_var_literal(code: str, available_vars: list[str]) -> tuple[str, bool]:
    """
    Fix FINAL_VAR("literal text") → FINAL("literal text") or
    FINAL_VAR("close_match") → FINAL_VAR("correct_var").

    Returns (fixed_code, was_fixed).
    """
    # Match FINAL_VAR("something")
    match = re.search(r'FINAL_VAR\(["\'](.+?)["\']\)', code)
    if not match:
        return code, False

    var_name = match.group(1)

    # If it's a valid variable name and exists, it's fine
    if var_name.isidentifier() and var_name in available_vars:
        return code, False

    # If it looks like a variable name but doesn't exist, find closest match
    if var_name.isidentifier():
        # Check for close matches
        for v in available_vars:
            if v.lower() == var_name.lower():
                fixed = code.replace(f'FINAL_VAR("{var_name}")', f'FINAL_VAR("{v}")')
                fixed = fixed.replace(f"FINAL_VAR('{var_name}')", f'FINAL_VAR("{v}")')
                logger.info(f"  Fixed FINAL_VAR case: {var_name} → {v}")
                return fixed, True

    # If it's literal text (not a valid variable name), convert to FINAL()
    if not var_name.isidentifier() or var_name not in available_vars:
        fixed = code.replace(f'FINAL_VAR("{var_name}")', f'FINAL("{var_name}")')
        fixed = fixed.replace(f"FINAL_VAR('{var_name}')", f"FINAL('{var_name}')")
        logger.info(f"  Fixed FINAL_VAR with literal: '{var_name}' → FINAL()")
        return fixed, True

    return code, False


def fix_final_with_plan(code: str) -> tuple[str, bool]:
    """
    Detect FINAL() with a plan/explanation instead of an answer.

    Heuristics:
    - FINAL("I will..." / "Let me..." / "First..." / "The approach is...")
    - These are plans, not answers. Hard to fix automatically — flag for removal.

    Returns (code, should_remove).
    """
    match = re.search(r'FINAL\(["\'](.+?)["\']\)', code, re.DOTALL)
    if not match:
        return code, False

    answer = match.group(1)

    # Plan-like patterns
    plan_patterns = [
        r"^(I will|Let me|First|The approach|To solve|I need to|Here's my plan)",
        r"^(Step \d|1\.|Phase \d)",
        r"(I'll|we should|we need to|the next step)",
    ]

    for pattern in plan_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            logger.info(f"  Flagged FINAL with plan: '{answer[:80]}...'")
            return code, True  # Flag for removal

    return code, False


def fix_missing_fstring(code: str) -> tuple[str, bool]:
    """
    Fix llm_query("{context}") → llm_query(f"{context}").

    The model sometimes writes string literals with {variable} without the f-prefix.
    """
    # Match llm_query("...{something}...") without f-prefix
    pattern = r'llm_query\("([^"]*\{[^}]+\}[^"]*)"'
    matches = list(re.finditer(pattern, code))

    if not matches:
        return code, False

    fixed = code
    was_fixed = False
    for m in matches:
        full_match = m.group(0)
        # Check if already has f-prefix
        pos = m.start()
        if pos > 0 and fixed[pos - 1] == 'f':
            continue
        # Check it actually contains variable references
        content = m.group(1)
        if re.search(r'\{(context|chunk|result|answer|data|text|doc|summary)', content):
            new = full_match.replace('llm_query("', 'llm_query(f"')
            fixed = fixed.replace(full_match, new)
            was_fixed = True
            logger.info(f"  Fixed missing f-string in llm_query")

    return fixed, was_fixed


def fix_trajectory(trajectory: dict) -> dict:
    """Fix template mistakes in a single trajectory."""
    fixes = {"final_var_literal": 0, "final_plan": 0, "missing_fstring": 0}
    flagged_for_removal = False

    for turn in trajectory.get("turns", []):
        code = turn.get("parsed_code", "")
        if not code:
            continue

        # Track available variables (rough approximation)
        # In practice, we'd need to actually parse the REPL state
        available_vars = []
        for line in code.split("\n"):
            if "=" in line and not line.strip().startswith("#"):
                var = line.split("=")[0].strip()
                if var.isidentifier():
                    available_vars.append(var)

        # Fix FINAL_VAR with literal
        code, fixed = fix_final_var_literal(code, available_vars)
        if fixed:
            fixes["final_var_literal"] += 1
            turn["parsed_code"] = code

        # Check FINAL with plan
        _, is_plan = fix_final_with_plan(code)
        if is_plan:
            fixes["final_plan"] += 1
            flagged_for_removal = True

        # Fix missing f-strings
        code, fixed = fix_missing_fstring(code)
        if fixed:
            fixes["missing_fstring"] += 1
            turn["parsed_code"] = code

    trajectory["template_fixes"] = fixes
    trajectory["flagged_for_removal"] = flagged_for_removal

    return trajectory


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_templates.py <trajectory_dir>")
        sys.exit(1)

    traj_dir = Path(sys.argv[1])

    for json_file in sorted(traj_dir.glob("*trajectories*.json")):
        logger.info(f"Processing: {json_file}")

        with open(json_file) as f:
            trajectories = json.load(f)

        total_fixes = {"final_var_literal": 0, "final_plan": 0, "missing_fstring": 0}
        flagged = 0

        for traj in trajectories:
            fix_trajectory(traj)
            for k, v in traj.get("template_fixes", {}).items():
                total_fixes[k] += v
            if traj.get("flagged_for_removal"):
                flagged += 1

        # Save fixed version
        fixed_path = json_file.with_stem(json_file.stem + "_fixed")
        with open(fixed_path, "w") as f:
            json.dump(trajectories, f, indent=2, default=str)

        logger.info(f"  Total fixes: {total_fixes}")
        logger.info(f"  Flagged for removal: {flagged}/{len(trajectories)}")
        logger.info(f"  Saved to: {fixed_path}")


if __name__ == "__main__":
    main()
