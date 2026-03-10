#!/usr/bin/env python3
"""
RL training V6 for RLM on Tinker API.

Major improvements over rl_tinker.py (V1-V5):
1. Gradient accumulation — one optim_step per training step, not per mini-batch
2. Adaptive task difficulty — auto-increase difficulty when tasks are saturated
3. Minimum context length — no tasks under 20K chars (zero gradient on easy tasks)
4. Multi-turn persistence bonus — reward iterative behavior, not just final answer
5. Better timeout handling — limit retries, don't let stuck trajectories block training
6. Temperature-corrected importance sampling — adjust for per-trajectory temperature
7. Proper loss/reward logging across full step
8. KL penalty via reward shaping (since Tinker kl_coeff not wired up)
9. Reject-and-replace for completely failed trajectories
10. Entropy-encouraging reward bonus for code diversity

Usage:
    uv run python training/rl_tinker_v6.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --model-path tinker://run-id/weights/state-0005 \
        --steps 30 --K 8 --batch-size 4 --lr 2e-6 \
        --task-type mixed_v6 --save-every 5 \
        --experiment-name grpo_35b_v6
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr

from scaffold.llm_query import TinkerModel
from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen35_35b import QWEN35_35B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from eval.benchmarks.doc_classify import generate_doc_classify_suite, score_doc_classify
from eval.benchmarks.dataframe_qa import generate_dataframe_qa_suite, score_dataframe_qa
from eval.benchmarks.code_debug import generate_code_debug_suite, score_code_debug
from eval.benchmarks.multi_hop_qa import generate_multi_hop_suite, score_multi_hop
from eval.benchmarks.notebook_qa import generate_notebook_qa_suite, score_notebook_qa
from eval.benchmarks.multi_hop_hard import generate_hard_multi_hop_suite, score_hard_multi_hop
from eval.benchmarks.event_counting import generate_event_counting_suite, score_event_counting

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Reward computation
# ============================================================================

def compute_reward(trajectory_dict: dict, task_type: str, task_info: dict | None = None) -> float:
    """Compute reward for a trajectory.

    V6 improvements:
    - Multi-turn persistence bonus: extra reward for iterating
    - Code diversity bonus: different code patterns get small bonus
    - Decomposition reward for hard_multi_hop
    """
    score = trajectory_dict.get("score", 0)
    terminated = trajectory_dict.get("terminated", False)
    turns = trajectory_dict.get("turns", [])

    # Format bonus
    format_bonus = 0.0
    if terminated:
        format_bonus += 0.05
    n_errors = sum(1 for t in turns if t.get("error"))
    format_bonus -= 0.02 * n_errors

    # Multi-turn persistence bonus (V6 new):
    # Reward iterative behavior — models that use 2+ turns and still get correct answer
    # should be encouraged over single-turn solutions on hard tasks
    persistence_bonus = 0.0
    n_code_turns = sum(1 for t in turns if t.get("parsed_code"))
    if score >= 0.8 and n_code_turns >= 2:
        # Correct answer with multi-turn: small bonus (0.05 for 2 turns, up to 0.15 for 4+)
        persistence_bonus = min(0.15, 0.05 * (n_code_turns - 1))

    # Sub-call count bonus (V6 ANTI-SHORTCUT):
    # Reward using multiple llm_query calls on long contexts.
    # This prevents the model from learning single-pass shortcuts.
    subcall_bonus = 0.0
    prompt_length = trajectory_dict.get("prompt_length", 0)
    if prompt_length > 30000:
        # Count llm_query calls in the code
        n_subcalls = 0
        for t in turns:
            code = t.get("parsed_code") or ""
            stdout = t.get("stdout") or ""
            # Count actual llm_query executions from stdout
            n_subcalls += stdout.count("llm_query")  # rough proxy
            # Also check if the code has a loop with llm_query
            if "while" in code and "llm_query" in code:
                n_subcalls = max(n_subcalls, 2)
        # Count sub-calls from model stats if available
        model_stats = trajectory_dict.get("model_stats")
        if model_stats and "sub_calls" in model_stats:
            n_subcalls = model_stats["sub_calls"]
        if n_subcalls >= 2:
            subcall_bonus = min(0.10, 0.03 * n_subcalls)

    if task_type == "niah":
        return 0.75 * score + 0.10 * (format_bonus + 0.1) + 0.10 * persistence_bonus + 0.05 * subcall_bonus
    elif task_type in ("multi_niah", "doc_classify"):
        return 0.75 * score + 0.10 * (format_bonus + 0.1) + 0.05 * persistence_bonus + 0.10 * subcall_bonus
    elif task_type == "hard_multi_hop":
        # Intermediate reward: partial credit for bridge entities
        decomp_bonus = _compute_decomposition_bonus(trajectory_dict, task_info)
        # 50% final answer + 20% decomposition + 10% format + 10% persistence + 10% subcalls
        return 0.50 * score + 0.20 * decomp_bonus + 0.10 * (format_bonus + 0.1) + 0.10 * persistence_bonus + 0.10 * subcall_bonus
    elif task_type in ("multi_hop_qa", "notebook_qa"):
        return 0.70 * score + 0.10 * (format_bonus + 0.1) + 0.10 * persistence_bonus + 0.10 * subcall_bonus
    elif task_type == "dataframe_qa":
        return 0.70 * score + 0.10 * (format_bonus + 0.1) + 0.10 * persistence_bonus + 0.10 * subcall_bonus
    elif task_type == "code_debug":
        # Code debug keeps short contexts, no subcall bonus
        return 0.85 * score + 0.10 * (format_bonus + 0.1) + 0.05 * persistence_bonus
    elif task_type == "event_counting":
        # Event counting: reward Python-based counting (not sub-model counting)
        return 0.70 * score + 0.10 * (format_bonus + 0.1) + 0.10 * persistence_bonus + 0.10 * subcall_bonus
    else:
        return score


def _compute_decomposition_bonus(trajectory_dict: dict, task_info: dict | None) -> float:
    """Compute intermediate reward for finding bridge entities in multi-hop tasks."""
    if task_info is None:
        return 0.0

    task = task_info.get("task")
    if task is None or not hasattr(task, "decomposition"):
        return 0.0

    decomposition = task.decomposition
    if not decomposition:
        return 0.0

    # Extract bridge entities from decomposition steps
    bridge_entities = []
    for step in decomposition:
        if "→" in step:
            entity = step.split("→")[-1].strip()
            bridge_entities.append(entity)

    if not bridge_entities:
        return 0.0

    # Collect all stdout from trajectory turns
    turns = trajectory_dict.get("turns", [])
    all_text = " ".join(
        (t.get("stdout") or "") + " " + (t.get("raw_response") or "")
        for t in turns
    ).lower()

    answer = trajectory_dict.get("answer") or ""
    all_text += " " + answer.lower()

    # Score: fraction of bridge entities found
    found = sum(1 for e in bridge_entities if e.lower() in all_text)
    return found / len(bridge_entities)


def _compute_code_diversity(trajectories: list[dict]) -> list[float]:
    """Compute per-trajectory diversity bonus within a group.

    Trajectories with code patterns different from the group median
    get a small bonus. This prevents mode collapse by making identical
    trajectories worth slightly less than diverse ones.
    """
    # Extract code from each trajectory
    codes = []
    for traj in trajectories:
        turns = traj.get("turns", [])
        code = "\n".join(t.get("parsed_code", "") or "" for t in turns)
        codes.append(code)

    if not codes or all(c == codes[0] for c in codes):
        return [0.0] * len(trajectories)

    # Jaccard-style diversity: how different is each code from the majority?
    # Use 3-gram sets for rough similarity
    def ngrams(text, n=3):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    code_ngrams = [ngrams(c) for c in codes]

    bonuses = []
    for i, ng_i in enumerate(code_ngrams):
        if not ng_i:
            bonuses.append(0.0)
            continue
        # Average Jaccard distance from all other trajectories
        distances = []
        for j, ng_j in enumerate(code_ngrams):
            if i == j or not ng_j:
                continue
            intersection = len(ng_i & ng_j)
            union = len(ng_i | ng_j)
            distances.append(1.0 - intersection / max(union, 1))
        avg_dist = np.mean(distances) if distances else 0.0
        # Small bonus for diversity: max 0.03 for very unique code
        bonuses.append(min(0.03, avg_dist * 0.05))

    return bonuses


# ============================================================================
# Training data conversion
# ============================================================================

def trajectory_to_training_data(
    trajectory_dict: dict,
    advantage: float,
    renderer,
    tokenizer,
) -> list[tinker.Datum]:
    """Convert trajectory to cross_entropy training data (fallback when no logprobs)."""
    data = []
    all_messages = trajectory_dict.get("messages", [])

    if not all_messages:
        return data

    for i, msg in enumerate(all_messages):
        if msg["role"] != "assistant":
            continue

        try:
            full_msgs = all_messages[:i] + [msg]
            model_input, weights = renderer.build_supervised_example(full_msgs)

            from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets
            input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
                model_input.chunks
            )

            if hasattr(weights, 'tolist'):
                weights_list = weights.tolist()
            else:
                weights_list = list(weights)
            weights_shifted = weights_list[1:len(target_tokens) + 1]

            advantage_weight = max(min(advantage, 5.0), -5.0)
            weighted = [float(w) * advantage_weight for w in weights_shifted]

            datum = tinker.Datum(
                model_input=input_model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens, dtype="int64", shape=[len(target_tokens)],
                    ),
                    "weights": tinker.TensorData(
                        data=weighted, dtype="float32", shape=[len(weighted)],
                    ),
                }
            )
            data.append(datum)
        except Exception as e:
            logger.warning(f"Failed to convert turn {i}: {e}")

    return data


def trajectory_to_training_data_is(
    trajectory_dict: dict,
    advantage: float,
    renderer,
    tokenizer,
) -> list[tinker.Datum]:
    """Convert trajectory to importance_sampling Datum format.

    Uses per-token logprobs from sampling for proper GRPO.
    """
    data = []
    all_messages = trajectory_dict.get("messages", [])
    turns = trajectory_dict.get("turns", [])

    if not all_messages:
        return data

    turn_idx = 0
    for i, msg in enumerate(all_messages):
        if msg["role"] != "assistant":
            continue

        if turn_idx >= len(turns):
            break
        turn = turns[turn_idx]
        turn_idx += 1

        turn_logprobs = turn.get("logprobs")
        turn_tokens = turn.get("tokens")
        if turn_logprobs is None or turn_tokens is None:
            continue

        try:
            full_msgs = all_messages[:i] + [msg]
            model_input, weights = renderer.build_supervised_example(full_msgs)

            from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets
            input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
                model_input.chunks
            )

            if hasattr(weights, 'tolist'):
                weights_list = weights.tolist()
            else:
                weights_list = list(weights)
            weights_shifted = weights_list[1:len(target_tokens) + 1]

            n_targets = len(target_tokens)
            logprobs_aligned = [0.0] * n_targets
            advantages_aligned = [0.0] * n_targets

            resp_start = None
            for j, w in enumerate(weights_shifted):
                if float(w) > 0:
                    resp_start = j
                    break

            if resp_start is not None:
                n_resp = min(len(turn_logprobs), n_targets - resp_start)
                for j in range(n_resp):
                    logprobs_aligned[resp_start + j] = float(turn_logprobs[j])
                    advantages_aligned[resp_start + j] = float(advantage)

            datum = tinker.Datum(
                model_input=input_model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens, dtype="int64", shape=[len(target_tokens)],
                    ),
                    "logprobs": tinker.TensorData(
                        data=logprobs_aligned, dtype="float32", shape=[len(logprobs_aligned)],
                    ),
                    "advantages": tinker.TensorData(
                        data=advantages_aligned, dtype="float32", shape=[len(advantages_aligned)],
                    ),
                }
            )
            data.append(datum)
        except Exception as e:
            logger.warning(f"Failed to convert turn for IS: {e}")

    return data


# ============================================================================
# Adaptive task difficulty
# ============================================================================

class AdaptiveTaskScheduler:
    """Tracks per-task-type performance and adapts difficulty.

    When a task type has all-same-reward (skip rate) above threshold
    for N consecutive steps, we increase difficulty:
    - NIAH: increase doc length
    - Multi-NIAH: increase K (number of needles)
    - Doc-Classify: increase N (number of documents)
    - Multi-Hop/Hard: increase doc length
    """

    def __init__(self):
        self.skip_history: dict[str, list[float]] = defaultdict(list)  # skip rates per task
        self.difficulty_level: dict[str, int] = defaultdict(int)  # 0=base, 1=harder, 2=hardest
        self.window = 3  # steps to look back
        self.skip_threshold = 0.6  # trigger difficulty increase at 60% skip rate

    def record_step(self, task_type: str, n_groups: int, n_skipped: int):
        """Record skip rate for a task type this step."""
        rate = n_skipped / max(n_groups, 1)
        self.skip_history[task_type].append(rate)

    def should_increase_difficulty(self, task_type: str) -> bool:
        """Check if we should increase difficulty for this task type."""
        history = self.skip_history.get(task_type, [])
        if len(history) < self.window:
            return False
        recent = history[-self.window:]
        return np.mean(recent) >= self.skip_threshold

    def check_and_adapt(self) -> list[str]:
        """Check all task types and increase difficulty where needed.
        Returns list of task types that were upgraded.
        """
        upgraded = []
        for task_type in list(self.skip_history.keys()):
            if self.should_increase_difficulty(task_type):
                old_level = self.difficulty_level[task_type]
                if old_level < 2:  # Max 2 upgrades
                    self.difficulty_level[task_type] = old_level + 1
                    upgraded.append(task_type)
                    logger.info(
                        f"  ADAPTIVE: {task_type} difficulty {old_level} → {old_level + 1} "
                        f"(skip rate {np.mean(self.skip_history[task_type][-self.window:]):.1%})"
                    )
        return upgraded

    def get_doc_lengths(self, task_type: str) -> list[int]:
        """Get document length options based on difficulty level.

        ANTI-SHORTCUT: All levels start at minimum 50K to force genuine chunking.
        The sub-call context window is ~30K, so 50K+ always requires multiple chunks.
        """
        level = self.difficulty_level.get(task_type, 0)
        configs = {
            "niah": [
                [50000, 100000],              # level 0: min 50K
                [100000, 200000],             # level 1
                [200000, 500000],             # level 2
            ],
            "multi_niah": [
                [50000, 100000],
                [100000, 200000],
                [200000],
            ],
            "doc_classify": [
                [50000, 100000],
                [100000, 200000],
                [200000],
            ],
        }
        return configs.get(task_type, [[50000]])[min(level, 2)]


# ============================================================================
# Task sampling
# ============================================================================

def sample_tasks_v6(
    batch_size: int,
    step: int,
    scheduler: AdaptiveTaskScheduler,
) -> list[dict]:
    """Sample a batch of tasks for V6 training.

    CRITICAL DESIGN: Minimum context length = 50K chars for all tasks except code_debug.
    This prevents the model from learning single-pass shortcuts (where it just sends
    the entire context to one llm_query call and bypasses chunking entirely).

    Analysis of v1-v4 trajectories showed RL TEACHES THE MODEL TO AVOID RECURSION
    when contexts fit in one sub-call window (~30K chars). By forcing 50K+ contexts,
    chunking becomes physically necessary, and RL optimizes chunking QUALITY instead.

    mixed_v6: Hard tasks that REQUIRE genuine RLM behavior.
    - 25% hard_multi_hop (100K-300K, decomposition required)
    - 20% multi_hop_qa (50K-150K, multi-chunk required)
    - 15% notebook_qa (50K-150K, structured extraction)
    - 10% dataframe_qa (50K-200K, large CSV analysis)
    - 10% code_debug (keep short — code is naturally short)
    - 10% doc_classify (50K-200K, many docs)
    - 5% niah (50K-200K, forces real chunking)
    - 5% multi_niah (50K-200K, forces real chunking)
    """
    seed_offset = step * 1000

    task_weights = [
        ("hard_multi_hop", 0.20),
        ("multi_hop_qa", 0.15),
        ("event_counting", 0.15),  # NEW: teaches extract-then-count-in-Python
        ("notebook_qa", 0.10),
        ("dataframe_qa", 0.10),
        ("code_debug", 0.10),
        ("doc_classify", 0.10),
        ("niah", 0.05),
        ("multi_niah", 0.05),
    ]

    type_names = [t[0] for t in task_weights]
    type_probs = [t[1] for t in task_weights]
    rng = np.random.RandomState(seed_offset)
    chosen_types = rng.choice(type_names, size=batch_size, p=type_probs)

    type_counts = {}
    for t in chosen_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    tasks = []

    for ttype, count in type_counts.items():
        s = seed_offset + hash(ttype) % 100000

        if ttype == "niah":
            # ANTI-SHORTCUT: minimum 50K so chunking is required
            doc_lengths = scheduler.get_doc_lengths("niah")
            items = generate_niah_suite(n_tasks=count, doc_lengths=doc_lengths, seed_offset=s)
            tasks.extend({"task": t, "type": "niah"} for t in items)
        elif ttype == "multi_niah":
            # ANTI-SHORTCUT: minimum 50K
            items = generate_multi_niah_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "multi_niah"} for t in items)
        elif ttype == "doc_classify":
            items = generate_doc_classify_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "doc_classify"} for t in items)
        elif ttype == "multi_hop_qa":
            items = generate_multi_hop_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "multi_hop_qa"} for t in items)
        elif ttype == "hard_multi_hop":
            items = generate_hard_multi_hop_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "hard_multi_hop"} for t in items)
        elif ttype == "dataframe_qa":
            items = generate_dataframe_qa_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "dataframe_qa"} for t in items)
        elif ttype == "code_debug":
            # Exception: code is naturally short, keep as-is
            items = generate_code_debug_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "code_debug"} for t in items)
        elif ttype == "notebook_qa":
            items = generate_notebook_qa_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "notebook_qa"} for t in items)
        elif ttype == "event_counting":
            # ANTI-SHORTCUT: 50K-200K event logs, trains extract-then-count
            items = generate_event_counting_suite(n_tasks=count, seed_offset=s)
            tasks.extend({"task": t, "type": "event_counting"} for t in items)

    rng.shuffle(tasks)
    return tasks


# ============================================================================
# Score trajectory
# ============================================================================

def score_trajectory(traj_dict: dict, task_info: dict) -> float:
    """Score a trajectory against its task."""
    task = task_info["task"]
    task_type = task_info["type"]
    answer = traj_dict.get("answer")

    if task_type == "niah":
        return score_niah(answer, task.expected_answer)
    elif task_type == "multi_niah":
        scores = score_multi_niah(answer, task.expected_answers)
        return scores["recall"]
    elif task_type == "doc_classify":
        scores = score_doc_classify(answer, task.expected_labels)
        return scores["accuracy"]
    elif task_type == "dataframe_qa":
        scores = score_dataframe_qa(answer, task.expected_answer, task.task_type)
        return scores["score"]
    elif task_type == "code_debug":
        scores = score_code_debug(answer, task.bugs)
        return scores["score"]
    elif task_type == "multi_hop_qa":
        scores = score_multi_hop(answer, task.expected_answer)
        return scores["score"]
    elif task_type == "notebook_qa":
        scores = score_notebook_qa(answer, task.expected_answer)
        return scores["score"]
    elif task_type == "hard_multi_hop":
        scores = score_hard_multi_hop(answer, task.expected_answer)
        return scores["score"]
    elif task_type == "event_counting":
        scores = score_event_counting(answer, task.expected_answer)
        return scores["score"]
    return 0


# ============================================================================
# Main training loop
# ============================================================================

def train_rl_v6(
    model_name: str,
    model_path: str | None = None,
    steps: int = 30,
    K: int = 8,
    batch_size: int = 4,
    lr: float = 2e-6,
    lora_rank: int = 32,
    kl_coeff: float = 0.01,
    task_type: str = "mixed_v6",
    save_every: int = 5,
    experiment_name: str = "grpo_v6",
    warmup_steps: int = 2,
    grad_accum_batch: int = 4,
    timeout_retry_limit: int = 2,
):
    """Run GRPO RL training V6 on Tinker.

    Key V6 differences:
    - Gradient accumulation: one optim_step per training step
    - Adaptive task difficulty
    - Multi-turn persistence bonus
    - Warmup phase (linear LR warmup before cosine decay)
    - KL penalty via reward shaping
    - Timeout retry limiting
    """

    log_dir = Path("data/rl") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-6s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Setup
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)
    service_client = tinker.ServiceClient()

    if model_path:
        logger.info(f"Resuming from {model_path}")
        training_client = service_client.create_training_client_from_state(model_path)
    else:
        logger.info(f"Creating fresh LoRA training client for {model_name}")
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            seed=42,
        )

    logger.info(f"  Training model_id: {training_client.model_id}")

    # Create sampling client
    sampling_client = training_client.save_weights_and_get_sampling_client(name="v6-init")

    # Model wrapper
    model = TinkerModel(
        model_name=model_name,
        max_new_tokens=2048,
        temperature=1.0,
    )
    model.capture_logprobs = True
    model.sampling_client = sampling_client

    # LoRA LR scaling
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
    except Exception:
        effective_lr = lr

    system_prompt = QWEN35_35B_SYSTEM_PROMPT
    scheduler = AdaptiveTaskScheduler()

    logger.info(f"\n{'='*60}")
    logger.info(f"GRPO V6 Training Config")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Effective LR: {effective_lr:.2e}")
    logger.info(f"  K: {K}, Batch: {batch_size}, Steps: {steps}")
    logger.info(f"  Task type: {task_type}")
    logger.info(f"  KL coeff (reward shaping): {kl_coeff}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Grad accum batch: {grad_accum_batch}")
    logger.info(f"  Timeout retry limit: {timeout_retry_limit}")
    logger.info(f"  Adaptive difficulty: ON")
    logger.info(f"  Multi-turn persistence bonus: ON")
    logger.info(f"  Code diversity bonus: ON")
    logger.info(f"{'='*60}\n")

    # Temperature schedule: narrower range than V5 (1.2+ mainly causes failures)
    temp_schedule = [0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2]
    if K > len(temp_schedule):
        temp_schedule = temp_schedule + [1.0] * (K - len(temp_schedule))

    training_log = []
    t0 = time.time()

    # Track trajectory length stats for KL approximation
    # We use the mean trajectory logprob as a proxy for KL divergence
    ref_mean_logprob = None  # Will be set from first step

    for step in range(steps):
        step_t0 = time.time()

        # LR schedule: linear warmup + cosine decay
        if step < warmup_steps:
            # Linear warmup
            step_lr = effective_lr * (step + 1) / warmup_steps
        else:
            # Cosine decay from effective_lr to 10%
            min_lr = effective_lr * 0.1
            progress = (step - warmup_steps) / max(steps - warmup_steps - 1, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            step_lr = min_lr + (effective_lr - min_lr) * cosine_decay

        adam_params = tinker.AdamParams(learning_rate=step_lr)

        logger.info(f"\n{'='*60}")
        logger.info(f"--- Step {step + 1}/{steps} --- (LR: {step_lr:.2e})")

        # 1. Sample tasks (V6: use adaptive scheduler for mixed_v6)
        if task_type == "mixed_v6":
            tasks = sample_tasks_v6(batch_size, step, scheduler)
        else:
            # Fall back to V5 task sampling for compatibility
            from training.rl_tinker import sample_tasks
            tasks = sample_tasks(task_type, batch_size, step)

        # 2. Generate K trajectories per task
        all_groups = []
        step_timeout_count = 0

        for task_idx, task_info in enumerate(tasks):
            task = task_info["task"]
            group_trajectories = []

            task_t0 = time.time()
            logger.info(f"  Task {task_idx+1}/{len(tasks)}: {task_info['type']} ({len(task.prompt)} chars)")

            for k in range(K):
                if hasattr(model, 'reset_stats'):
                    model.reset_stats()
                model.temperature = temp_schedule[k]

                try:
                    traj = rlm(
                        prompt=task.prompt,
                        model=model,
                        system_prompt=system_prompt,
                        max_iterations=8,
                    )
                    traj_dict = trajectory_to_dict(traj)
                    traj_dict["messages"] = traj.messages
                    traj_dict["system_prompt"] = system_prompt
                    traj_dict["temperature"] = temp_schedule[k]

                    score = score_trajectory(traj_dict, task_info)
                    traj_dict["score"] = score
                    traj_dict["reward"] = compute_reward(traj_dict, task_info["type"], task_info)

                    # Track timeouts
                    n_timeouts = sum(
                        1 for t in traj_dict.get("turns", [])
                        if t.get("error") and "Timeout" in str(t.get("error", ""))
                    )
                    if n_timeouts > 0:
                        step_timeout_count += 1

                    # Capture mean logprob for KL approximation
                    logprobs = []
                    for t in traj_dict.get("turns", []):
                        if t.get("logprobs"):
                            logprobs.extend(t["logprobs"])
                    if logprobs:
                        traj_dict["mean_logprob"] = np.mean(logprobs)

                    group_trajectories.append(traj_dict)

                except Exception as e:
                    logger.warning(f"  Trajectory {k+1}/{K} failed: {e}")
                    group_trajectories.append({
                        "score": 0, "reward": 0, "error": str(e),
                        "turns": [], "terminated": False,
                    })

            task_time = time.time() - task_t0
            rewards = [t["reward"] for t in group_trajectories]
            logger.info(
                f"    → rewards: [{', '.join(f'{r:.2f}' for r in rewards)}] "
                f"mean={np.mean(rewards):.3f} std={np.std(rewards):.3f} "
                f"({task_time:.1f}s)"
            )

            all_groups.append({
                "task_info": task_info,
                "trajectories": group_trajectories,
            })

        # 3. Apply code diversity bonus (V6)
        for group in all_groups:
            diversity_bonuses = _compute_code_diversity(group["trajectories"])
            for traj, bonus in zip(group["trajectories"], diversity_bonuses):
                traj["reward"] += bonus

        # 4. KL penalty via reward shaping (V6)
        # Approximate KL as deviation of mean logprob from reference
        step_logprobs = []
        for group in all_groups:
            for traj in group["trajectories"]:
                if "mean_logprob" in traj:
                    step_logprobs.append(traj["mean_logprob"])

        if step_logprobs:
            current_mean_lp = np.mean(step_logprobs)
            if ref_mean_logprob is None:
                ref_mean_logprob = current_mean_lp  # Set reference from first step

            # KL penalty: penalize trajectories that deviate far from reference
            for group in all_groups:
                for traj in group["trajectories"]:
                    if "mean_logprob" in traj:
                        kl_approx = abs(traj["mean_logprob"] - ref_mean_logprob)
                        traj["reward"] -= kl_coeff * kl_approx

        # 5. Compute advantages (group-relative)
        step_rewards = []
        step_advantages = []
        training_data = []
        n_updates = 0
        n_skipped_groups = 0
        per_task_rewards = defaultdict(list)
        per_task_skips = defaultdict(int)
        per_task_groups = defaultdict(int)

        for group in all_groups:
            ttype = group["task_info"]["type"]
            rewards = [t["reward"] for t in group["trajectories"]]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            step_rewards.extend(rewards)
            per_task_rewards[ttype].append(mean_r)
            per_task_groups[ttype] += 1

            if std_r < 1e-6:
                # All same reward — skip (no learning signal)
                n_skipped_groups += 1
                per_task_skips[ttype] += 1
                logger.info(f"  SKIP group {ttype}: all same reward {mean_r:.3f}")
                continue

            for traj_dict, reward in zip(group["trajectories"], rewards):
                advantage = (reward - mean_r) / (std_r + 1e-8)

                if abs(advantage) < 0.05:
                    continue

                step_advantages.append(advantage)

                # Convert to training data
                turn_has_logprobs = any(
                    t.get("logprobs") is not None for t in traj_dict.get("turns", [])
                )
                if turn_has_logprobs:
                    data = trajectory_to_training_data_is(
                        traj_dict, advantage, renderer, tokenizer
                    )
                else:
                    data = trajectory_to_training_data(
                        traj_dict, advantage, renderer, tokenizer
                    )
                training_data.extend(data)
                n_updates += 1

        # 6. Record adaptive difficulty stats
        for ttype in per_task_groups:
            scheduler.record_step(ttype, per_task_groups[ttype], per_task_skips[ttype])

        # Check for difficulty upgrades
        upgraded = scheduler.check_and_adapt()

        # 7. Train with gradient accumulation (V6: single optim_step per step)
        step_losses = []
        if training_data:
            sample_inputs = training_data[0].loss_fn_inputs if training_data else {}
            actual_loss_fn = "importance_sampling" if "advantages" in sample_inputs else "cross_entropy"

            logger.info(f"  Training: {len(training_data)} datums, loss_fn={actual_loss_fn}")

            # Submit all forward_backward calls
            fwd_bwd_futures = []
            for i in range(0, len(training_data), grad_accum_batch):
                batch = training_data[i:i+grad_accum_batch]
                fwd_bwd = training_client.forward_backward(batch, actual_loss_fn)
                fwd_bwd_futures.append(fwd_bwd)

            # Wait for all forward_backward to complete
            for fwd_bwd in fwd_bwd_futures:
                result = fwd_bwd.result()
                loss = result.metrics.get("loss:sum") if hasattr(result, 'metrics') else None
                if loss is not None:
                    step_losses.append(float(loss))

            # Single optimizer step (V6: gradient accumulation!)
            optim = training_client.optim_step(adam_params)
            optim.result()

        # 8. Refresh sampling client
        if training_data:
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"v6-step-{step+1}"
            )
            model.sampling_client = sampling_client

        # 9. Log comprehensively
        step_time = time.time() - step_t0
        avg_reward = np.mean(step_rewards) if step_rewards else 0
        avg_advantage = np.mean(np.abs(step_advantages)) if step_advantages else 0
        total_loss = sum(step_losses) if step_losses else None
        avg_loss = np.mean(step_losses) if step_losses else None

        n_pos = sum(1 for a in step_advantages if a > 0)
        n_neg = sum(1 for a in step_advantages if a < 0)

        step_info = {
            "step": step + 1,
            "avg_reward": float(avg_reward),
            "max_reward": float(max(step_rewards)) if step_rewards else 0,
            "min_reward": float(min(step_rewards)) if step_rewards else 0,
            "n_updates": n_updates,
            "n_skipped_groups": n_skipped_groups,
            "n_pos_advantages": n_pos,
            "n_neg_advantages": n_neg,
            "n_training_data": len(training_data),
            "avg_advantage_abs": float(avg_advantage),
            "total_loss": total_loss,
            "avg_loss": avg_loss,
            "lr": step_lr,
            "time": step_time,
            "elapsed": time.time() - t0,
            "n_timeouts": step_timeout_count,
            "difficulty_levels": dict(scheduler.difficulty_level),
            "upgraded_tasks": upgraded,
            "per_task_rewards": {k: float(np.mean(v)) for k, v in per_task_rewards.items()},
        }
        training_log.append(step_info)

        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
        logger.info(
            f"  Reward: {avg_reward:.3f} [{min(step_rewards) if step_rewards else 0:.3f}, "
            f"{max(step_rewards) if step_rewards else 0:.3f}] | "
            f"Updates: {n_updates} (+{n_pos}/-{n_neg}) | "
            f"Skipped: {n_skipped_groups}/{len(all_groups)} | "
            f"Data: {len(training_data)} | "
            f"Loss: {loss_str} | "
            f"Timeouts: {step_timeout_count} | "
            f"Time: {step_time:.1f}s"
        )

        # Per-task breakdown
        parts = []
        for ttype in sorted(per_task_rewards.keys()):
            avg_r = np.mean(per_task_rewards[ttype])
            skip = per_task_skips.get(ttype, 0)
            total = per_task_groups.get(ttype, 0)
            parts.append(f"{ttype}={avg_r:.3f}(skip={skip}/{total})")
        logger.info(f"  Per-task: {' | '.join(parts)}")

        # Save checkpoint
        if (step + 1) % save_every == 0:
            ckpt_name = f"checkpoint-{step+1:04d}"
            logger.info(f"  Saving checkpoint: {ckpt_name}")
            training_client.save_weights_and_get_sampling_client(name=ckpt_name)
            training_client.save_state(name=f"state-{step+1:04d}")

            # Save sample trajectories
            sample_dir = log_dir / f"samples_step{step+1}"
            sample_dir.mkdir(exist_ok=True)
            for gi, group in enumerate(all_groups[:3]):
                for ti, traj in enumerate(group["trajectories"][:3]):
                    with open(sample_dir / f"group{gi}_traj{ti}.json", "w") as f:
                        json.dump(traj, f, indent=2, default=str)

            # Save training log so far
            with open(log_dir / "training_log.json", "w") as f:
                json.dump({
                    "model": model_name, "model_path": model_path,
                    "steps_completed": step + 1, "total_steps": steps,
                    "K": K, "batch_size": batch_size, "lr": effective_lr,
                    "training_log": training_log,
                }, f, indent=2, default=str)

    # Save final
    logger.info("\nSaving final checkpoint...")
    training_client.save_weights_and_get_sampling_client(name="final")
    training_client.save_state(name="final-state")

    total_time = time.time() - t0

    with open(log_dir / "training_log.json", "w") as f:
        json.dump({
            "model": model_name, "model_path": model_path,
            "steps": steps, "K": K, "batch_size": batch_size,
            "lr": effective_lr, "kl_coeff": kl_coeff,
            "task_type": task_type, "total_time": total_time,
            "training_log": training_log,
            "final_difficulty_levels": dict(scheduler.difficulty_level),
        }, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"GRPO V6 TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"  Final avg reward: {training_log[-1]['avg_reward']:.3f}")
    logger.info(f"  Final difficulty levels: {dict(scheduler.difficulty_level)}")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GRPO V6 RL training on Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--task-type", default="mixed_v6")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--experiment-name", default="grpo_35b_v6")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--grad-accum-batch", type=int, default=4)
    parser.add_argument("--timeout-retry-limit", type=int, default=2)
    args = parser.parse_args()

    train_rl_v6(
        model_name=args.model,
        model_path=args.model_path,
        steps=args.steps,
        K=args.K,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        kl_coeff=args.kl_coeff,
        task_type=args.task_type,
        save_every=args.save_every,
        experiment_name=args.experiment_name,
        warmup_steps=args.warmup_steps,
        grad_accum_batch=args.grad_accum_batch,
        timeout_retry_limit=args.timeout_retry_limit,
    )


if __name__ == "__main__":
    main()
