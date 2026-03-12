#!/usr/bin/env python3
"""
Aggregate ALL available correct trajectory data into a unified SFT dataset.

Data sources:
1. Existing SFT files (sft_from_base_eval.jsonl, sft_merged_eval.jsonl)
2. GRPO training run trajectories (reward > 0.5)
3. Teacher trajectories (score > 0.5)
4. STaR R1 trajectories
5. Student V4-s5 RFT trajectories
6. Eval run trajectories from results/ dirs (with per_task scores)

Output: /root/rlm/data/sft/sft_all_aggregated.jsonl
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path("/root/rlm")
OUTPUT_PATH = ROOT / "data" / "sft" / "sft_all_aggregated.jsonl"

# System prompt from scaffold/prompts/qwen35_35b.py
sys.path.insert(0, str(ROOT / "scaffold" / "prompts"))
from qwen35_35b import QWEN35_35B_SYSTEM_PROMPT

SYSTEM_PROMPT = QWEN35_35B_SYSTEM_PROMPT

# Score thresholds
MIN_REWARD = 0.5
MIN_SCORE = 0.5

# Counters
stats = defaultdict(lambda: defaultdict(int))  # source -> {loaded, filtered, kept}


def is_gibberish(text: str, threshold: float = 0.10) -> bool:
    """Check if >threshold fraction of text is non-ASCII."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > threshold


def build_metadata(prompt: str) -> str:
    """Build RLM metadata string matching scaffold/repl.py format."""
    prefix = prompt[:500]
    remaining = len(prompt) - 500
    meta = f"Context length: {len(prompt)} characters\n"
    meta += f"Context prefix:\n{prefix}\n"
    if remaining > 0:
        meta += f"... [{remaining} more characters]\n"
    meta += "\nAvailable functions:\n"
    meta += '  context (str) — full input text\n'
    meta += '  llm_query(text: str) -> str — send text to LLM\n'
    meta += '  FINAL(answer: str) — submit final answer\n'
    meta += '  FINAL_VAR(var_name: str) — submit variable as answer\n'
    meta += '  print() — show output'
    return meta


def build_turn_observation(turn: dict, prompt: str) -> str:
    """Build the user observation after a code turn."""
    stdout = turn.get("stdout", "")
    error = turn.get("error", "")
    stderr = turn.get("stderr", "")
    metadata = build_metadata(prompt)

    if error:
        err_text = error
        if stderr and stderr not in error:
            err_text += "\n" + stderr
        return f"Error executing code:\n{err_text}\n\nCurrent state:\n{metadata}"
    else:
        return f"Output:\n{stdout}\n\nState:\n{metadata}"


def trajectory_to_sft(traj: dict, task_type: str, score: float, source: str) -> dict | None:
    """Convert a trajectory dict (with turns) to SFT format.

    Returns None if the trajectory is invalid.
    """
    prompt = traj.get("prompt", "")
    system_prompt = traj.get("system_prompt", SYSTEM_PROMPT)
    turns = traj.get("turns", [])

    if not turns:
        return None

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_metadata(prompt)},
    ]

    # For multi-turn: add all turns except the last as context
    for i, turn in enumerate(turns[:-1]):
        raw_response = turn.get("raw_response", "")
        if not raw_response:
            return None
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": build_turn_observation(turn, prompt)})

    # The last turn's raw_response is the completion
    last_turn = turns[-1]
    completion = last_turn.get("raw_response", "")
    if not completion:
        return None

    if is_gibberish(completion):
        return None

    return {
        "messages": messages,
        "completion": completion,
        "task_type": task_type,
        "score": score,
        "source": source,
    }


def messages_to_sft(traj: dict, task_type: str, score: float, source: str) -> dict | None:
    """Convert a trajectory that already has 'messages' list to SFT format.

    The messages list has system, user, assistant, ... The last assistant message
    becomes the completion.
    """
    msgs = traj.get("messages", [])
    if len(msgs) < 3:  # need at least system + user + assistant
        return None

    # Find the last assistant message
    last_asst_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return None

    completion = msgs[last_asst_idx]["content"]
    if not completion or is_gibberish(completion):
        return None

    context_messages = msgs[:last_asst_idx]

    return {
        "messages": context_messages,
        "completion": completion,
        "task_type": task_type,
        "score": score,
        "source": source,
    }


def get_task_score(per_task_entry: dict, benchmark: str) -> float:
    """Extract a 0-1 score from an eval_results per_task entry."""
    # Different benchmarks use different score keys
    if "score" in per_task_entry:
        return float(per_task_entry["score"])
    elif benchmark == "multi_niah":
        return float(per_task_entry.get("recall", per_task_entry.get("f1", 0)))
    elif benchmark == "doc_classify":
        return float(per_task_entry.get("accuracy", 0))
    elif "accuracy" in per_task_entry:
        return float(per_task_entry["accuracy"])
    elif "correct" in per_task_entry and "total" in per_task_entry:
        total = per_task_entry["total"]
        if total > 0:
            return per_task_entry["correct"] / total
        return 0.0
    return 0.0


def infer_task_type_from_prompt(prompt: str) -> str:
    """Try to infer task type from the prompt content."""
    prompt_lower = prompt[:500].lower()
    if any(k in prompt_lower for k in ["classify", "categories", "categorize"]):
        return "doc_classify"
    elif any(k in prompt_lower for k in ["secret code", "password is", "needle"]):
        return "niah"
    elif any(k in prompt_lower for k in ["how many events", "how many times", "count the"]):
        return "event_counting"
    elif any(k in prompt_lower for k in ["two budget reports", "two organizational",
                                          "two event timelines", "two performance",
                                          "compare the following", "two directories",
                                          "two annual reports"]):
        return "cross_doc_compare"
    elif any(k in prompt_lower for k in ["notebook", "jupyter"]):
        return "notebook_qa"
    elif any(k in prompt_lower for k in ["dataframe", "csv", "ticker", "stock",
                                          "closing price", "highest average"]):
        return "dataframe_qa"
    elif any(k in prompt_lower for k in ["which function contains the bug", "find the bug",
                                          "codebase contains"]):
        return "code_debug"
    elif any(k in prompt_lower for k in ["verbatim", "copy exactly", "reproduce"]):
        return "verbatim_copy"
    elif any(k in prompt_lower for k in ["key:", "value:", "lookup", "retrieve the value"]):
        return "key_value_retrieval"
    elif any(k in prompt_lower for k in ["multi-hop", "multihop"]):
        return "multi_hop_qa"
    elif any(k in prompt_lower for k in ["which city", "what is the project",
                                          "who leads", "who is the", "what is the budget",
                                          "where did", "what award",
                                          "based in", "leader of project"]):
        return "multi_hop_qa"
    elif any(k in prompt_lower for k in ["what is the project codename", "what is the code",
                                          "vault password", "secret password"]):
        return "niah"
    elif any(k in prompt_lower for k in ["registry entry", "find the entry",
                                          "return its category", "return only the"]):
        return "key_value_retrieval"
    elif any(k in prompt_lower for k in ["headquartered", "approved budget",
                                          "sponsored by", "project led by",
                                          "earned", "rising star award",
                                          "completed on time"]):
        return "multi_hop_qa"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Source 1 & 2: Existing SFT files
# ---------------------------------------------------------------------------
def load_existing_sft():
    """Load from sft_from_base_eval.jsonl and sft_merged_eval.jsonl."""
    samples = []

    for filename, src in [
        ("sft_from_base_eval.jsonl", "existing_base_eval"),
        ("sft_merged_eval.jsonl", "existing_merged_eval"),
    ]:
        path = ROOT / "data" / "sft" / filename
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats[src]["loaded"] += 1
                score = d.get("score", 1.0)
                if score is not None and score < MIN_SCORE:
                    stats[src]["filtered_low_score"] += 1
                    continue

                # Check for gibberish in completion
                completion = d.get("completion", "")
                if is_gibberish(completion):
                    stats[src]["filtered_gibberish"] += 1
                    continue

                d["source"] = src
                if "task_type" not in d or d["task_type"] is None:
                    d["task_type"] = "unknown"
                samples.append(d)
                stats[src]["kept"] += 1
                count += 1

        print(f"  [{src}] loaded {count} samples from {path.name}")

    return samples


# ---------------------------------------------------------------------------
# Source 3: GRPO training run trajectories
# ---------------------------------------------------------------------------
def load_grpo_trajectories():
    """Load from data/rl/grpo_35b_v{9,10,10h,11,12}/ sample files."""
    samples = []

    for version in ["v9", "v10", "v10h", "v11", "v12", "v13"]:
        dirpath = ROOT / "data" / "rl" / f"grpo_35b_{version}"
        if not dirpath.is_dir():
            continue

        src = f"grpo_{version}"
        count = 0

        for step_dir in sorted(dirpath.iterdir()):
            if not step_dir.is_dir() or not step_dir.name.startswith("samples"):
                continue

            for fn in sorted(step_dir.iterdir()):
                if not fn.name.startswith("group") or not fn.name.endswith(".json"):
                    continue

                try:
                    with open(fn) as f:
                        d = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                stats[src]["loaded"] += 1
                reward = d.get("reward", 0)
                if reward < MIN_REWARD:
                    stats[src]["filtered_low_reward"] += 1
                    continue

                # Infer task type from prompt
                prompt = d.get("prompt", "")
                task_type = d.get("task_type") or infer_task_type_from_prompt(prompt)

                # These trajectories have both 'messages' and 'turns'
                # Use 'messages' directly when available (already formatted)
                if d.get("messages") and len(d["messages"]) >= 3:
                    sample = messages_to_sft(d, task_type, reward, src)
                elif d.get("turns"):
                    sample = trajectory_to_sft(d, task_type, reward, src)
                else:
                    stats[src]["filtered_no_data"] += 1
                    continue

                if sample is None:
                    stats[src]["filtered_invalid"] += 1
                    continue

                samples.append(sample)
                stats[src]["kept"] += 1
                count += 1

        if count > 0:
            print(f"  [{src}] loaded {count} samples")

    return samples


# ---------------------------------------------------------------------------
# Source 4: Teacher trajectories
# ---------------------------------------------------------------------------
def load_teacher_trajectories():
    """Load from data/trajectories/teacher_merged.json."""
    samples = []
    src = "teacher"

    path = ROOT / "data" / "trajectories" / "teacher_merged.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return samples

    with open(path) as f:
        trajs = json.load(f)

    for traj in trajs:
        stats[src]["loaded"] += 1
        score = traj.get("score", 0)
        if score < MIN_SCORE:
            stats[src]["filtered_low_score"] += 1
            continue

        task_type = traj.get("task_type", "unknown") or "unknown"
        sample = trajectory_to_sft(traj, task_type, score, src)
        if sample is None:
            stats[src]["filtered_invalid"] += 1
            continue

        samples.append(sample)
        stats[src]["kept"] += 1

    print(f"  [{src}] loaded {stats[src]['kept']} samples from {path.name}")
    return samples


# ---------------------------------------------------------------------------
# Source 5: STaR R1 trajectories
# ---------------------------------------------------------------------------
def load_star_r1_trajectories():
    """Load from data/trajectories/star_r1_35b_*/correct_trajectories.json."""
    samples = []
    src = "star_r1"

    for traj_dir in ROOT.glob("data/trajectories/star_r1_*/"):
        path = traj_dir / "correct_trajectories.json"
        if not path.exists():
            continue

        with open(path) as f:
            trajs = json.load(f)

        for traj in trajs:
            stats[src]["loaded"] += 1
            score = traj.get("score", 0)
            if score < MIN_SCORE:
                stats[src]["filtered_low_score"] += 1
                continue

            task_type = traj.get("task_type", "unknown") or "unknown"
            sample = trajectory_to_sft(traj, task_type, score, src)
            if sample is None:
                stats[src]["filtered_invalid"] += 1
                continue

            samples.append(sample)
            stats[src]["kept"] += 1

    # Also check star_r1_docclassify and star_r1_mniah
    for sub_dir in ROOT.glob("data/trajectories/star_r1_*_*/"):
        path = sub_dir / "correct_trajectories.json"
        if not path.exists():
            continue

        with open(path) as f:
            trajs = json.load(f)

        subsrc = sub_dir.name.split("_", 3)[-1] if "_" in sub_dir.name else sub_dir.name
        for traj in trajs:
            stats[src]["loaded"] += 1
            score = traj.get("score", 0)
            if score < MIN_SCORE:
                stats[src]["filtered_low_score"] += 1
                continue

            task_type = traj.get("task_type", "unknown") or "unknown"
            sample = trajectory_to_sft(traj, task_type, score, src)
            if sample is None:
                stats[src]["filtered_invalid"] += 1
                continue

            samples.append(sample)
            stats[src]["kept"] += 1

    print(f"  [{src}] loaded {stats[src]['kept']} samples")
    return samples


# ---------------------------------------------------------------------------
# Source 6: Student V4-s5 RFT trajectories
# ---------------------------------------------------------------------------
def load_student_rft_trajectories():
    """Load from data/trajectories/student_v4s5_rft/correct_trajectories.json."""
    samples = []
    src = "student_v4s5_rft"

    path = ROOT / "data" / "trajectories" / "student_v4s5_rft" / "correct_trajectories.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return samples

    with open(path) as f:
        trajs = json.load(f)

    for traj in trajs:
        stats[src]["loaded"] += 1
        score = traj.get("score", 0)
        if score < MIN_SCORE:
            stats[src]["filtered_low_score"] += 1
            continue

        task_type = traj.get("task_type", "unknown") or "unknown"
        sample = trajectory_to_sft(traj, task_type, score, src)
        if sample is None:
            stats[src]["filtered_invalid"] += 1
            continue

        samples.append(sample)
        stats[src]["kept"] += 1

    print(f"  [{src}] loaded {stats[src]['kept']} samples")
    return samples


# ---------------------------------------------------------------------------
# Source 7: Eval run trajectories from results/ directories
# ---------------------------------------------------------------------------
def load_eval_trajectories():
    """Load trajectories from results/ directories.

    These pair eval_results.json (which has scores) with trajectory files.
    Handles two structures:
      - Flat: results/<name>/<timestamp>/eval_results.json + trajectories/
      - Nested: results/<name>/<timestamp>/<benchmark>/eval_results.json + trajectories/
    """
    samples = []

    # Find all eval_results.json files
    results_root = ROOT / "results"
    eval_files = list(results_root.rglob("eval_results.json"))
    print(f"  Found {len(eval_files)} eval_results.json files to scan")

    for eval_path in sorted(eval_files):
        parent = eval_path.parent
        traj_dir = parent / "trajectories"
        config_path = parent / "config.json"

        if not traj_dir.is_dir():
            continue

        try:
            with open(eval_path) as f:
                eval_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        benchmark = eval_results.get("benchmark", "unknown")
        per_task = eval_results.get("per_task", [])
        if not per_task:
            continue

        # Determine source name from path
        # e.g., results/clean_headtohead_base/20260311_085930/niah
        rel_parts = eval_path.relative_to(results_root).parts
        src = f"eval_{rel_parts[0]}"

        # Get model info from config if available
        model_info = ""
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model_info = config.get("model_path", config.get("model", ""))
            except (json.JSONDecodeError, IOError):
                pass

        # Map trajectory files to per_task entries
        # trajectory_001.json corresponds to per_task[0], etc.
        traj_files = sorted(
            [f for f in traj_dir.iterdir() if f.name.startswith("trajectory_") and f.name.endswith(".json")]
        )

        for i, traj_file in enumerate(traj_files):
            if i >= len(per_task):
                break

            pt = per_task[i]
            score = get_task_score(pt, benchmark)

            stats[src]["loaded"] += 1

            if score < MIN_SCORE:
                stats[src]["filtered_low_score"] += 1
                continue

            try:
                with open(traj_file) as f:
                    traj = json.load(f)
            except (json.JSONDecodeError, IOError):
                stats[src]["filtered_invalid"] += 1
                continue

            task_type = traj.get("task_type") or benchmark

            sample = trajectory_to_sft(traj, task_type, score, src)
            if sample is None:
                stats[src]["filtered_invalid"] += 1
                continue

            samples.append(sample)
            stats[src]["kept"] += 1

    total_kept = sum(s.get("kept", 0) for s in stats.values() if any(
        k.startswith("eval_") for k in [""]))
    eval_sources = {k: v for k, v in stats.items() if k.startswith("eval_")}
    eval_total = sum(v.get("kept", 0) for v in eval_sources.values())
    print(f"  [eval_runs] loaded {eval_total} total samples from {len(eval_sources)} result dirs")
    return samples


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicates: same task_type + first 200 chars of completion match."""
    seen = set()
    unique = []
    n_dupes = 0

    for s in samples:
        task_type = s.get("task_type", "unknown")
        completion = s.get("completion", "")
        dedup_key = (task_type, completion[:200])

        if dedup_key in seen:
            n_dupes += 1
            continue

        seen.add(dedup_key)
        unique.append(s)

    print(f"\n  Deduplication: {n_dupes} duplicates removed, {len(unique)} unique samples remain")
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SFT Data Aggregation Pipeline")
    print("=" * 70)

    all_samples = []

    # Source 1 & 2: Existing SFT files
    print("\n[1/6] Loading existing SFT files...")
    all_samples.extend(load_existing_sft())

    # Source 3: GRPO training trajectories
    print("\n[2/6] Loading GRPO training trajectories (reward > 0.5)...")
    all_samples.extend(load_grpo_trajectories())

    # Source 4: Teacher trajectories
    print("\n[3/6] Loading teacher trajectories...")
    all_samples.extend(load_teacher_trajectories())

    # Source 5: STaR R1 trajectories
    print("\n[4/6] Loading STaR R1 trajectories...")
    all_samples.extend(load_star_r1_trajectories())

    # Source 6: Student V4-s5 RFT trajectories
    print("\n[5/6] Loading student V4-s5 RFT trajectories...")
    all_samples.extend(load_student_rft_trajectories())

    # Source 7: Eval run trajectories
    print("\n[6/6] Loading eval run trajectories from results/...")
    all_samples.extend(load_eval_trajectories())

    print(f"\n{'=' * 70}")
    print(f"Total samples before dedup: {len(all_samples)}")

    # Deduplicate
    all_samples = deduplicate(all_samples)

    # Report per source
    print(f"\n{'=' * 70}")
    print("Per-source breakdown:")
    print(f"{'Source':<35} {'Loaded':>8} {'Kept':>8} {'Filtered':>8}")
    print("-" * 70)
    for src in sorted(stats.keys()):
        s = stats[src]
        loaded = s.get("loaded", 0)
        kept = s.get("kept", 0)
        filtered = loaded - kept
        print(f"{src:<35} {loaded:>8} {kept:>8} {filtered:>8}")

    # Count per source in final deduplicated set
    source_counts = defaultdict(int)
    for s in all_samples:
        source_counts[s.get("source", "unknown")] += 1

    print(f"\n{'=' * 70}")
    print("After dedup, per-source counts:")
    print(f"{'Source':<35} {'Count':>8}")
    print("-" * 45)
    for src in sorted(source_counts.keys()):
        print(f"{src:<35} {source_counts[src]:>8}")

    # Report per task type
    task_counts = defaultdict(int)
    for s in all_samples:
        task_counts[s.get("task_type", "unknown")] += 1

    print(f"\n{'=' * 70}")
    print("Per-task-type breakdown (after dedup):")
    print(f"{'Task Type':<30} {'Count':>8}")
    print("-" * 40)
    for tt in sorted(task_counts.keys()):
        print(f"{tt:<30} {task_counts[tt]:>8}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL UNIQUE SAMPLES: {len(all_samples)}")
    print(f"{'=' * 70}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved to: {OUTPUT_PATH}")
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
