#!/usr/bin/env python3
"""Summarize all evaluation results into a single comparison table."""

import json
import os
from pathlib import Path

BENCHMARKS = [
    "niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug",
    "multi_hop_qa", "notebook_qa", "hard_niah", "verbatim_copy", "oolong",
    "hard_multi_hop", "event_counting", "cross_doc_compare", "key_value_retrieval",
]

def load_eval_results(results_dir: str) -> dict[str, float]:
    """Load eval results from a directory, trying multiple sub-dirs."""
    results = {}
    path = Path(results_dir)

    if not path.exists():
        return results

    # Check for eval_results.json in subdirectories
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir():
            results_file = subdir / "eval_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                bench = data.get("benchmark", "")
                if bench:
                    score = data.get("accuracy", data.get("score", 0))
                    results[bench] = score

    # Also check for direct eval_results.json
    direct = path / "eval_results.json"
    if direct.exists():
        with open(direct) as f:
            data = json.load(f)
        bench = data.get("benchmark", "")
        if bench:
            score = data.get("accuracy", data.get("score", 0))
            results[bench] = score

    return results


def load_from_log(log_pattern: str) -> dict[str, float]:
    """Load results from individual benchmark log files."""
    results = {}
    results_dir = Path("results")

    for log_file in sorted(results_dir.glob(log_pattern)):
        # Extract benchmark name from filename
        name = log_file.stem
        # Try to find accuracy in the log
        with open(log_file) as f:
            for line in f:
                if "Accuracy/Recall:" in line or "accuracy:" in line.lower():
                    # Extract percentage
                    import re
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        bench_name = None
                        for b in BENCHMARKS:
                            if b in name:
                                bench_name = b
                                break
                        if bench_name:
                            results[bench_name] = float(match.group(1)) / 100
    return results


def print_comparison(configs: dict[str, dict[str, float]]):
    """Print a comparison table."""
    # Header
    names = list(configs.keys())
    header = f"| {'Benchmark':25s} |"
    for name in names:
        header += f" {name:>12s} |"
    print(header)
    print("|" + "-" * 27 + "|" + ("|".join(["-" * 14] * len(names))) + "|")

    # Rows
    totals = {name: [] for name in names}
    for bench in BENCHMARKS:
        row = f"| {bench:25s} |"
        scores = []
        for name in names:
            score = configs[name].get(bench)
            if score is not None:
                row += f" {score:11.1%} |"
                scores.append(score)
                totals[name].append(score)
            else:
                row += f" {'---':>11s} |"
        # Bold the best
        print(row)

    # Average
    print("|" + "-" * 27 + "|" + ("|".join(["-" * 14] * len(names))) + "|")
    avg_row = f"| {'**Average**':25s} |"
    for name in names:
        if totals[name]:
            avg = sum(totals[name]) / len(totals[name])
            avg_row += f" {avg:11.1%} |"
        else:
            avg_row += f" {'---':>11s} |"
    print(avg_row)


if __name__ == "__main__":
    configs = {}

    # Load baseline results
    baseline = load_eval_results("results/baseline_headtohead") or load_eval_results("results/clean_headtohead_base")
    if baseline:
        configs["Base"] = baseline

    # Load V4-s5 results
    v4s5 = load_eval_results("results/v4s5_headtohead") or load_eval_results("results/clean_headtohead_v4s5")
    if v4s5:
        configs["V4-s5"] = v4s5

    # Load V4-s5 hybrid results
    v4s5h = load_eval_results("results/v4s5_hybrid_headtohead") or load_eval_results("results/clean_headtohead_v4s5_hybrid")
    if v4s5h:
        configs["V4-s5-Hybrid"] = v4s5h

    if configs:
        print("# Evaluation Results Comparison\n")
        print_comparison(configs)
    else:
        print("No results found. Check results/ directory.")
