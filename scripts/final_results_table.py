#!/usr/bin/env python3
"""Generate the final results comparison table for the paper.

Aggregates results from all evaluation runs and produces:
1. Full 14-benchmark comparison table (markdown)
2. Oracle best-of-all analysis
3. LaTeX table for the paper
"""

import json
import os
from pathlib import Path
from collections import defaultdict

BENCHMARKS = [
    "niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug",
    "multi_hop_qa", "notebook_qa", "hard_niah", "verbatim_copy", "oolong",
    "hard_multi_hop", "event_counting", "cross_doc_compare", "key_value_retrieval",
]

BENCHMARK_SHORT = {
    "niah": "NIAH",
    "multi_niah": "Multi-NIAH",
    "doc_classify": "Doc-Classify",
    "dataframe_qa": "DataFrame QA",
    "code_debug": "Code Debug",
    "multi_hop_qa": "Multi-Hop QA",
    "notebook_qa": "Notebook QA",
    "hard_niah": "Hard NIAH",
    "verbatim_copy": "Verbatim Copy",
    "oolong": "OOLONG",
    "hard_multi_hop": "Hard Multi-Hop",
    "event_counting": "Event Counting",
    "cross_doc_compare": "Cross-Doc Compare",
    "key_value_retrieval": "KV Retrieval",
}

RESULTS_DIR = Path("/root/rlm/results")


def load_eval_dir(path: Path) -> dict[str, float]:
    """Load eval results from a directory containing benchmark subdirs."""
    results = {}
    if not path.exists():
        return results

    for subdir in path.iterdir():
        if not subdir.is_dir():
            continue
        results_file = subdir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            bench = data.get("benchmark", subdir.name)
            score = data.get("accuracy", data.get("score", None))
            if score is not None:
                results[bench] = score
    return results


def find_run_dir(experiment_dir: Path) -> Path | None:
    """Find the actual run directory (dated subdir) inside an experiment dir."""
    if not experiment_dir.exists():
        return None
    for d in sorted(experiment_dir.iterdir()):
        if d.is_dir() and d.name != "log.txt":
            # Check if it contains benchmark subdirs
            if any((d / b).exists() for b in BENCHMARKS):
                return d
    return None


def load_experiment(name: str) -> dict[str, float]:
    """Load results from an experiment by name."""
    exp_dir = RESULTS_DIR / name
    run_dir = find_run_dir(exp_dir)
    if run_dir:
        return load_eval_dir(run_dir)
    return load_eval_dir(exp_dir)


def print_markdown_table(configs: dict[str, dict[str, float]], title: str = ""):
    """Print a markdown comparison table."""
    if title:
        print(f"\n## {title}\n")

    names = list(configs.keys())

    # Header
    header = f"| {'Benchmark':20s} | {'N':>3s} |"
    for name in names:
        header += f" {name:>14s} |"
    print(header)
    sep = f"|{'-'*22}|{'-'*5}|"
    for _ in names:
        sep += f"{'-'*16}|"
    print(sep)

    # Find best per benchmark
    totals = {name: [] for name in names}
    n_tasks = {}

    for bench in BENCHMARKS:
        scores = {}
        for name in names:
            s = configs[name].get(bench)
            if s is not None:
                scores[name] = s

        if not scores:
            continue

        best_score = max(scores.values()) if scores else -1

        # Get N from any config that has it
        n = "?"
        for name in names:
            exp_dir = None
            # Try to find the actual results file
            for exp_name in os.listdir(RESULTS_DIR):
                if name.lower().replace("+", "_").replace(" ", "_") in exp_name.lower():
                    exp_dir = RESULTS_DIR / exp_name
                    break

        short = BENCHMARK_SHORT.get(bench, bench)
        row = f"| {short:20s} | {n:>3s} |"
        for name in names:
            s = configs[name].get(bench)
            if s is not None:
                pct = f"{s*100:.1f}%"
                if s == best_score and len([v for v in scores.values() if v == best_score]) < len(scores):
                    pct = f"**{pct}**"
                row += f" {pct:>14s} |"
                totals[name].append(s)
            else:
                row += f" {'—':>14s} |"
        print(row)

    # Average
    print(sep)
    avg_row = f"| {'**Average**':20s} | {'':>3s} |"
    best_avg = -1
    avgs = {}
    for name in names:
        if totals[name]:
            avg = sum(totals[name]) / len(totals[name])
            avgs[name] = avg
            if avg > best_avg:
                best_avg = avg

    for name in names:
        if name in avgs:
            pct = f"{avgs[name]*100:.1f}%"
            if avgs[name] == best_avg:
                pct = f"**{pct}**"
            n_bench = len(totals[name])
            avg_row += f" {pct:>14s} |"
        else:
            avg_row += f" {'—':>14s} |"
    print(avg_row)

    # Count
    count_row = f"| {'Benchmarks':20s} | {'':>3s} |"
    for name in names:
        count_row += f" {len(totals[name]):>14d} |"
    print(count_row)


def print_latex_table(configs: dict[str, dict[str, float]]):
    """Print a LaTeX table for the paper."""
    names = list(configs.keys())
    n_cols = len(names) + 1

    print("\n% LaTeX table")
    print(f"\\begin{{tabular}}{{l{'r' * len(names)}}}")
    print("\\toprule")
    header = "Benchmark"
    for name in names:
        header += f" & {name}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    totals = {name: [] for name in names}

    for bench in BENCHMARKS:
        scores = {name: configs[name].get(bench) for name in names if configs[name].get(bench) is not None}
        if not scores:
            continue
        best = max(scores.values())

        short = BENCHMARK_SHORT.get(bench, bench)
        row = short
        for name in names:
            s = configs[name].get(bench)
            if s is not None:
                pct = f"{s*100:.1f}\\%"
                if s == best and len(scores) > 1 and list(scores.values()).count(best) < len(scores):
                    pct = f"\\textbf{{{pct}}}"
                row += f" & {pct}"
                totals[name].append(s)
            else:
                row += " & —"
        row += " \\\\"
        print(row)

    print("\\midrule")
    avgs = {}
    for name in names:
        if totals[name]:
            avgs[name] = sum(totals[name]) / len(totals[name])
    best_avg = max(avgs.values()) if avgs else -1

    row = "\\textbf{Average}"
    for name in names:
        if name in avgs:
            pct = f"{avgs[name]*100:.1f}\\%"
            if avgs[name] == best_avg:
                pct = f"\\textbf{{{pct}}}"
            row += f" & {pct}"
        else:
            row += " & —"
    row += " \\\\"
    print(row)
    print("\\bottomrule")
    print("\\end{tabular}")


def oracle_analysis(configs: dict[str, dict[str, float]]):
    """Compute oracle best-of-all for each benchmark."""
    print("\n## Oracle Best-of-All Analysis\n")

    oracle = {}
    oracle_source = {}
    for bench in BENCHMARKS:
        best_score = -1
        best_name = ""
        for name, scores in configs.items():
            s = scores.get(bench)
            if s is not None and s > best_score:
                best_score = s
                best_name = name
        if best_score >= 0:
            oracle[bench] = best_score
            oracle_source[bench] = best_name

    for bench in BENCHMARKS:
        if bench in oracle:
            short = BENCHMARK_SHORT.get(bench, bench)
            print(f"  {short:20s}: {oracle[bench]*100:.1f}% (from {oracle_source[bench]})")

    if oracle:
        avg = sum(oracle.values()) / len(oracle)
        print(f"\n  Oracle Average: {avg*100:.1f}% ({len(oracle)} benchmarks)")


def main():
    print("# RLM Comprehensive Results\n")

    # ---- Deterministic (temp=0, seed_offset=10000) ----
    det_configs = {}
    for name, exp in [
        ("Base", "deterministic_base"),
        ("Base+Strat", "deterministic_base_strategy"),
        ("V11-s5", "deterministic_v11s5"),
        ("V11-s5+Strat", "deterministic_v11s5_strategy"),
    ]:
        results = load_experiment(exp)
        if results:
            det_configs[name] = results

    if det_configs:
        print_markdown_table(det_configs, "Deterministic (temp=0, seed_offset=10000)")
        oracle_analysis(det_configs)

    # ---- Standard (temp=0.7, seed_offset=10000) ----
    std_configs = {}
    for name, exp in [
        ("Base+Strat", "FINAL_base_with_strategies"),
        ("V11-s5+Strat", "FINAL_v11s5_with_strategies"),
    ]:
        results = load_experiment(exp)
        if results:
            std_configs[name] = results

    if std_configs:
        print_markdown_table(std_configs, "Standard (temp=0.7, seed_offset=10000) — FINAL")

    # ---- Original Head-to-Head (temp=0.7, seed_offset=0) ----
    orig_configs = {}
    for name, exp in [
        ("Base", "clean_headtohead_base"),
        ("V4-s5", "clean_headtohead_v4s5"),
        ("V11-s5", "clean_headtohead_v11s5"),
    ]:
        results = load_experiment(exp)
        if results:
            orig_configs[name] = results

    # Also check baseline_headtohead
    if "Base" not in orig_configs:
        base = load_experiment("baseline_headtohead")
        if base:
            orig_configs["Base"] = base

    if orig_configs:
        print_markdown_table(orig_configs, "Original Head-to-Head (temp=0.7, seed_offset=0)")

    # ---- SFT results (if available) ----
    sft_configs = {}
    for name, exp in [
        ("SFT-V1-ep2", "clean_headtohead_sft_v1_ep2"),
        ("SFT-V1-ep5", "clean_headtohead_sft_v1_ep5"),
        ("SFT-V7-ep1", "clean_headtohead_sft_v7_ep1"),
    ]:
        results = load_experiment(exp)
        if results:
            sft_configs[name] = results

    if sft_configs:
        print_markdown_table(sft_configs, "SFT Checkpoints")

    # ---- Strategy eval results ----
    strat_configs = {}
    for name, exp in [
        ("Base+Strat", "base_with_strategy"),
        ("V11-s5+Strat", "clean_headtohead_v11s5_strategy"),
    ]:
        results = load_experiment(exp)
        if results:
            strat_configs[name] = results

    if strat_configs:
        print_markdown_table(strat_configs, "Strategy Evaluation")

    # ---- Combined Oracle ----
    all_configs = {}
    all_configs.update(det_configs)
    all_configs.update({f"std_{k}": v for k, v in std_configs.items()})
    all_configs.update({f"orig_{k}": v for k, v in orig_configs.items()})
    all_configs.update({f"sft_{k}": v for k, v in sft_configs.items()})
    all_configs.update({f"strat_{k}": v for k, v in strat_configs.items()})

    if all_configs:
        oracle_analysis(all_configs)

    # ---- LaTeX for paper ----
    if det_configs:
        print_latex_table(det_configs)


if __name__ == "__main__":
    main()
