#!/usr/bin/env python3
"""
Analyze comprehensive evaluation results.

Loads results from multiple model evaluations and generates comparison tables,
confidence intervals, and summary statistics.

Usage:
    uv run python scripts/analyze_results.py \
        --results results/comprehensive_base/latest \
                  results/comprehensive_sft_v2/latest \
                  results/comprehensive_rl_v1/latest \
                  results/comprehensive_rl_v2/latest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load eval results from a results directory."""
    results = {}

    # Handle single-benchmark or multi-benchmark directory structures
    for bench_dir in [results_dir] + list(results_dir.iterdir()):
        eval_file = bench_dir / "eval_results.json" if bench_dir.is_dir() else None
        if eval_file and eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            benchmark = data.get("benchmark", bench_dir.name)
            results[benchmark] = data

    return results


def ci_95(scores: list[float]) -> tuple[float, float, float]:
    """Compute mean and 95% confidence interval."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(scores) / n
    if n == 1:
        return mean, 0.0, 1.0
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    se = math.sqrt(variance / n)
    ci = 1.96 * se
    return mean, ci, se


def print_niah_table(models: dict[str, dict]):
    """Print NIAH comparison table with confidence intervals."""
    print("\n" + "=" * 80)
    print("NIAH RESULTS (100 tasks, 4 per cell)")
    print("=" * 80)

    # Overall accuracy
    print(f"\n{'Model':<20} {'Overall':>10} {'95% CI':>12} {'N':>5}")
    print("-" * 50)
    for name, data in models.items():
        if "niah" not in data:
            continue
        tasks = data["niah"].get("per_task", [])
        scores = [t["score"] for t in tasks]
        mean, ci, se = ci_95(scores)
        print(f"{name:<20} {mean:>9.1%} {f'±{ci:.1%}':>12} {len(scores):>5}")

    # By doc length
    print(f"\n{'Model':<20}", end="")
    lengths = ["5K", "10K", "20K", "50K", "100K"]
    for l in lengths:
        print(f" {l:>8}", end="")
    print()
    print("-" * (20 + 8 * len(lengths)))
    for name, data in models.items():
        if "niah" not in data:
            continue
        by_len = data["niah"].get("by_doc_length", {})
        print(f"{name:<20}", end="")
        for l in lengths:
            acc = by_len.get(l, -1)
            if acc >= 0:
                print(f" {acc:>7.0%}", end="")
            else:
                print(f" {'N/A':>7}", end="")
        print()

    # By position
    print(f"\n{'Model':<20}", end="")
    positions = sorted(set(
        pos for data in models.values() if "niah" in data
        for pos in data["niah"].get("by_needle_position", {}).keys()
    ))
    for p in positions:
        print(f" {p:>8}", end="")
    print()
    print("-" * (20 + 8 * len(positions)))
    for name, data in models.items():
        if "niah" not in data:
            continue
        by_pos = data["niah"].get("by_needle_position", {})
        print(f"{name:<20}", end="")
        for p in positions:
            acc = by_pos.get(p, -1)
            if acc >= 0:
                print(f" {acc:>7.0%}", end="")
            else:
                print(f" {'N/A':>7}", end="")
        print()


def print_multi_niah_table(models: dict[str, dict]):
    """Print multi-needle NIAH comparison table."""
    print("\n" + "=" * 80)
    print("MULTI-NEEDLE NIAH RESULTS (O(K) complexity)")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Recall':>10} {'F1':>10} {'N':>5}")
    print("-" * 50)
    for name, data in models.items():
        if "multi_niah" not in data:
            continue
        res = data["multi_niah"]
        print(f"{name:<20} {res['accuracy']:>9.1%} {res.get('avg_f1', 0):>9.1%} {res['n_tasks']:>5}")

    # By needle count
    has_by_needles = any("multi_niah" in d and "by_n_needles" in d["multi_niah"] for d in models.values())
    if has_by_needles:
        needle_counts = sorted(set(
            k for data in models.values() if "multi_niah" in data
            for k in data["multi_niah"].get("by_n_needles", {}).keys()
        ))
        print(f"\n{'Model':<20}", end="")
        for nc in needle_counts:
            print(f" {nc:>12}", end="")
        print()
        print("-" * (20 + 12 * len(needle_counts)))
        for name, data in models.items():
            if "multi_niah" not in data:
                continue
            by_nc = data["multi_niah"].get("by_n_needles", {})
            print(f"{name:<20}", end="")
            for nc in needle_counts:
                acc = by_nc.get(nc, -1)
                if acc >= 0:
                    print(f" {acc:>11.0%}", end="")
                else:
                    print(f" {'N/A':>11}", end="")
            print()


def print_doc_classify_table(models: dict[str, dict]):
    """Print document classification comparison table."""
    print("\n" + "=" * 80)
    print("DOCUMENT CLASSIFICATION RESULTS (O(N) complexity)")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Accuracy':>10} {'N':>5}")
    print("-" * 40)
    for name, data in models.items():
        if "doc_classify" not in data:
            continue
        res = data["doc_classify"]
        print(f"{name:<20} {res['accuracy']:>9.1%} {res['n_tasks']:>5}")

    # By n_docs
    has_by_docs = any("doc_classify" in d and "by_n_docs" in d["doc_classify"] for d in models.values())
    if has_by_docs:
        doc_counts = sorted(set(
            k for data in models.values() if "doc_classify" in data
            for k in data["doc_classify"].get("by_n_docs", {}).keys()
        ))
        print(f"\n{'Model':<20}", end="")
        for dc in doc_counts:
            print(f" {dc:>10}", end="")
        print()
        print("-" * (20 + 10 * len(doc_counts)))
        for name, data in models.items():
            if "doc_classify" not in data:
                continue
            by_dc = data["doc_classify"].get("by_n_docs", {})
            print(f"{name:<20}", end="")
            for dc in doc_counts:
                acc = by_dc.get(dc, -1)
                if acc >= 0:
                    print(f" {acc:>9.0%}", end="")
                else:
                    print(f" {'N/A':>9}", end="")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to results directories")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Model names (same order as results)")
    args = parser.parse_args()

    # Load all results
    models = {}
    for i, result_path in enumerate(args.results):
        path = Path(result_path)
        if not path.exists():
            # Try finding latest timestamp directory
            parent = path.parent if path.is_file() else path
            if parent.exists():
                subdirs = sorted(parent.iterdir())
                if subdirs:
                    path = subdirs[-1]  # Latest timestamp

        if not path.exists():
            print(f"WARNING: {result_path} not found, skipping")
            continue

        name = args.names[i] if args.names and i < len(args.names) else path.parent.name
        name = name.replace("comprehensive_", "")
        models[name] = load_results(path)

    if not models:
        print("No results found!")
        sys.exit(1)

    print(f"Loaded results for: {', '.join(models.keys())}")

    # Print tables
    print_niah_table(models)
    print_multi_niah_table(models)
    print_doc_classify_table(models)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<20} {'NIAH':>8} {'M-NIAH':>8} {'Classify':>8}")
    print("-" * 48)
    for name, data in models.items():
        niah = f"{data['niah']['accuracy']:.0%}" if "niah" in data else "N/A"
        mniah = f"{data['multi_niah']['accuracy']:.0%}" if "multi_niah" in data else "N/A"
        classify = f"{data['doc_classify']['accuracy']:.0%}" if "doc_classify" in data else "N/A"
        print(f"{name:<20} {niah:>8} {mniah:>8} {classify:>8}")


if __name__ == "__main__":
    main()
