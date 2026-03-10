#!/usr/bin/env python3
"""
Compare evaluation results between base model and fine-tuned checkpoints.

Usage:
    uv run python scripts/compare_eval.py \
        results/baseline_35b_a3b/ \
        results/grpo_35b_v1_step10/ \
        results/grpo_35b_v1_step20/ \
        results/grpo_35b_v1_final/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_results(result_dir: str) -> dict:
    """Load evaluation results from a directory."""
    result_dir = Path(result_dir)
    results = {}

    for f in result_dir.glob("*.json"):
        data = json.loads(f.read_text())
        bench = data.get("benchmark", f.stem)
        results[bench] = data

    return results


def print_comparison(result_dirs: list[str]):
    """Print a comparison table of evaluation results."""
    all_results = {}
    names = []

    for d in result_dirs:
        name = Path(d).name
        names.append(name)
        all_results[name] = load_results(d)

    benchmarks = set()
    for name, results in all_results.items():
        benchmarks.update(results.keys())

    benchmarks = sorted(benchmarks)

    # Header
    name_width = max(len(n) for n in names) + 2
    print(f"\n{'Model':<{name_width}}", end="")
    for bench in benchmarks:
        print(f"  {bench:>15}", end="")
    print(f"  {'Average':>10}")
    print("-" * (name_width + 17 * len(benchmarks) + 12))

    # Rows
    for name in names:
        results = all_results[name]
        print(f"{name:<{name_width}}", end="")
        scores = []
        for bench in benchmarks:
            if bench in results:
                data = results[bench]
                # Try different keys for the main score
                score = (
                    data.get("accuracy")
                    or data.get("recall")
                    or data.get("score")
                    or 0
                )
                scores.append(score)
                print(f"  {score:>14.1%}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {avg:>9.1%}")

    print()

    # Delta from first model
    if len(names) > 1:
        base_name = names[0]
        base_results = all_results[base_name]
        print(f"\nDeltas vs {base_name}:")
        print("-" * (name_width + 17 * len(benchmarks) + 12))

        for name in names[1:]:
            results = all_results[name]
            print(f"{name:<{name_width}}", end="")
            deltas = []
            for bench in benchmarks:
                if bench in results and bench in base_results:
                    score = (
                        results[bench].get("accuracy")
                        or results[bench].get("recall")
                        or results[bench].get("score")
                        or 0
                    )
                    base_score = (
                        base_results[bench].get("accuracy")
                        or base_results[bench].get("recall")
                        or base_results[bench].get("score")
                        or 0
                    )
                    delta = score - base_score
                    deltas.append(delta)
                    sign = "+" if delta >= 0 else ""
                    print(f"  {sign}{delta:>13.1%}", end="")
                else:
                    print(f"  {'N/A':>15}", end="")
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            sign = "+" if avg_delta >= 0 else ""
            print(f"  {sign}{avg_delta:>8.1%}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: compare_eval.py <result_dir1> [result_dir2] ...")
        sys.exit(1)

    print_comparison(sys.argv[1:])
