#!/usr/bin/env python3
"""
Compare head-to-head evaluation results between base model and fine-tuned model.

Generates a formatted comparison table for the paper.

Usage:
    uv run python scripts/compare_headtohead.py \
        --baseline results/baseline_headtohead/ \
        --finetuned results/v4s5_headtohead/ \
        --output results/headtohead_comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compare")


def load_results(results_dir: str) -> dict[str, dict]:
    """Load evaluation results from a directory.

    Looks for JSON result files in timestamped subdirectories.
    """
    results_path = Path(results_dir)
    all_results = {}

    # Find the results subdirectory (timestamped)
    for subdir in sorted(results_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("202"):
            # Look for JSON files in here
            for f in sorted(subdir.iterdir()):
                if f.suffix == ".json":
                    try:
                        with open(f) as fp:
                            data = json.load(fp)
                        bench = data.get("benchmark", f.stem)
                        all_results[bench] = data
                        logger.info(f"  Loaded {bench}: {data.get('accuracy', data.get('score', 'N/A'))}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {f}: {e}")

    return all_results


def format_comparison(baseline: dict, finetuned: dict) -> str:
    """Format a comparison table in Markdown."""

    all_benchmarks = sorted(set(list(baseline.keys()) + list(finetuned.keys())))

    lines = []
    lines.append("# Head-to-Head Comparison: Base vs Post-Trained RLM")
    lines.append("")
    lines.append("| Benchmark | Base Model | Post-Trained (V4-s5) | Delta | Improvement |")
    lines.append("|-----------|-----------|---------------------|-------|-------------|")

    total_base = 0
    total_ft = 0
    n_benchmarks = 0

    for bench in all_benchmarks:
        base_score = None
        ft_score = None

        if bench in baseline:
            base_score = baseline[bench].get("accuracy", baseline[bench].get("score"))
        if bench in finetuned:
            ft_score = finetuned[bench].get("accuracy", finetuned[bench].get("score"))

        base_str = f"{base_score*100:.1f}%" if base_score is not None else "N/A"
        ft_str = f"{ft_score*100:.1f}%" if ft_score is not None else "N/A"

        if base_score is not None and ft_score is not None:
            delta = (ft_score - base_score) * 100
            delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

            if base_score > 0:
                improvement = (ft_score - base_score) / base_score * 100
                imp_str = f"+{improvement:.0f}%" if improvement >= 0 else f"{improvement:.0f}%"
            else:
                imp_str = "N/A"

            total_base += base_score
            total_ft += ft_score
            n_benchmarks += 1
        else:
            delta_str = "N/A"
            imp_str = "N/A"

        # Format benchmark name nicely
        bench_display = bench.replace("_", " ").title()

        lines.append(f"| {bench_display} | {base_str} | {ft_str} | {delta_str} | {imp_str} |")

    if n_benchmarks > 0:
        avg_base = total_base / n_benchmarks
        avg_ft = total_ft / n_benchmarks
        avg_delta = (avg_ft - avg_base) * 100
        avg_imp = (avg_ft - avg_base) / avg_base * 100 if avg_base > 0 else 0

        lines.append(f"| **Average** | **{avg_base*100:.1f}%** | **{avg_ft*100:.1f}%** | **+{avg_delta:.1f}%** | **+{avg_imp:.0f}%** |")

    lines.append("")
    lines.append(f"Evaluated on {n_benchmarks} benchmarks with 20 tasks each.")
    lines.append("")

    # Per-benchmark detail section
    lines.append("## Per-Benchmark Details")
    lines.append("")

    for bench in all_benchmarks:
        lines.append(f"### {bench.replace('_', ' ').title()}")

        for label, results in [("Base", baseline), ("V4-s5", finetuned)]:
            if bench in results:
                data = results[bench]
                n_tasks = data.get("n_tasks", "?")
                score = data.get("accuracy", data.get("score", "?"))

                lines.append(f"- **{label}:** {score*100:.1f}% ({n_tasks} tasks)")

                # Show by-type breakdown if available
                by_type = data.get("by_type", {})
                if by_type:
                    for ttype, tscore in sorted(by_type.items()):
                        lines.append(f"  - {ttype}: {tscore*100:.1f}%")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline results directory")
    parser.add_argument("--finetuned", required=True, help="Path to fine-tuned results directory")
    parser.add_argument("--output", default="results/headtohead_comparison.md")
    args = parser.parse_args()

    logger.info(f"Loading baseline results from {args.baseline}")
    baseline = load_results(args.baseline)

    logger.info(f"Loading fine-tuned results from {args.finetuned}")
    finetuned = load_results(args.finetuned)

    if not baseline and not finetuned:
        logger.error("No results found in either directory!")
        sys.exit(1)

    comparison = format_comparison(baseline, finetuned)

    with open(args.output, "w") as f:
        f.write(comparison)

    print(comparison)
    logger.info(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
