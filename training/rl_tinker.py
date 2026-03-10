#!/usr/bin/env python3
"""
RL training for RLM on Tinker API.

Implements GRPO (Group Relative Policy Optimization) using Tinker's
importance_sampling loss function. Each training step:
1. Sample K trajectories per prompt via RLM scaffold
2. Score each trajectory
3. Compute group-relative advantages
4. Train with importance_sampling loss on Tinker

Usage:
    uv run python training/rl_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --model-path tinker://run-id/weights/sft-final \
        --steps 30 \
        --K 8 \
        --experiment-name grpo_35b_v1

Key improvements over CS234 (rl_v4.py):
- Conditioned log-probs (fixed from v1-v3 bug)
- Token-level KL via Tinker's built-in KL tracking
- K=8 for stable advantage estimation
- Tinker handles distributed training (no local GPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_reward(trajectory_dict: dict, task_type: str) -> float:
    """Compute reward for a trajectory based on task type."""
    score = trajectory_dict.get("score", 0)
    terminated = trajectory_dict.get("terminated", False)
    turns = trajectory_dict.get("turns", [])

    # Format bonus
    format_bonus = 0.0
    if terminated:
        format_bonus += 0.05
    n_errors = sum(1 for t in turns if t.get("error"))
    format_bonus -= 0.02 * n_errors

    if task_type == "niah":
        return 0.85 * score + 0.15 * (format_bonus + 0.1)
    elif task_type == "multi_niah":
        return 0.90 * score + 0.10 * (format_bonus + 0.1)
    elif task_type == "doc_classify":
        return 0.90 * score + 0.10 * (format_bonus + 0.1)
    elif task_type == "dataframe_qa":
        return 0.90 * score + 0.10 * (format_bonus + 0.1)
    elif task_type == "code_debug":
        return 0.90 * score + 0.10 * (format_bonus + 0.1)
    elif task_type in ("multi_hop_qa", "hard_multi_hop", "notebook_qa"):
        return 0.90 * score + 0.10 * (format_bonus + 0.1)
    else:
        return score


def trajectory_to_training_data(
    trajectory_dict: dict,
    advantage: float,
    renderer,
    tokenizer,
    sampling_logprobs: list | None = None,
) -> list[tinker.Datum]:
    """Convert a trajectory to Tinker training data for importance_sampling loss.

    Each turn in the trajectory becomes a training datum with:
    - Full conversation context (conditioned log-probs!)
    - Advantage signal
    - Sampling log-probs (from generation)
    """
    messages_so_far = []
    data = []

    # Add system prompt
    system_prompt = trajectory_dict.get("system_prompt", QWEN35_35B_SYSTEM_PROMPT)
    messages_so_far.append({"role": "system", "content": system_prompt})

    turns = trajectory_dict.get("turns", [])
    all_messages = trajectory_dict.get("messages", [])

    # Reconstruct conversation from messages
    if all_messages:
        # Use stored messages for accurate reconstruction
        for i, msg in enumerate(all_messages):
            if msg["role"] == "assistant":
                # This is a turn we want to train on
                full_context = all_messages[:i]  # Everything before this assistant turn
                assistant_msg = msg

                try:
                    full_msgs = full_context + [assistant_msg]
                    model_input, weights = renderer.build_supervised_example(full_msgs)

                    # Use cookbook utility for proper token shifting
                    from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets
                    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
                        model_input.chunks
                    )

                    # Shift weights to match target tokens
                    if hasattr(weights, 'tolist'):
                        weights_list = weights.tolist()
                    else:
                        weights_list = list(weights)
                    weights_shifted = weights_list[1:len(target_tokens) + 1]

                    # Scale weights by advantage (allow BOTH positive AND negative)
                    # Positive advantage → reinforce these tokens
                    # Negative advantage → push away from these tokens
                    advantage_weight = max(min(advantage, 5.0), -5.0)  # Clip for stability
                    weighted = [float(w) * advantage_weight for w in weights_shifted]

                    datum = tinker.Datum(
                        model_input=input_model_input,
                        loss_fn_inputs={
                            "target_tokens": tinker.TensorData(
                                data=target_tokens,
                                dtype="int64",
                                shape=[len(target_tokens)],
                            ),
                            "weights": tinker.TensorData(
                                data=weighted,
                                dtype="float32",
                                shape=[len(weighted)],
                            ),
                        }
                    )
                    data.append(datum)

                except Exception as e:
                    logger.warning(f"Failed to convert turn {i}: {e}")
                    continue

    return data


def trajectory_to_training_data_is(
    trajectory_dict: dict,
    advantage: float,
    renderer,
    tokenizer,
) -> list[tinker.Datum]:
    """Convert trajectory to importance_sampling Datum format.

    Uses per-token logprobs from sampling for proper GRPO.
    Requires logprobs to be stored in turn records.
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

        # Find matching turn record with logprobs
        if turn_idx >= len(turns):
            break
        turn = turns[turn_idx]
        turn_idx += 1

        turn_logprobs = turn.get("logprobs")
        turn_tokens = turn.get("tokens")
        if turn_logprobs is None or turn_tokens is None:
            continue

        try:
            # Build full conversation context for this turn
            full_msgs = all_messages[:i] + [msg]
            model_input, weights = renderer.build_supervised_example(full_msgs)

            from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets
            input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
                model_input.chunks
            )

            # Shift weights to align with targets
            if hasattr(weights, 'tolist'):
                weights_list = weights.tolist()
            else:
                weights_list = list(weights)
            weights_shifted = weights_list[1:len(target_tokens) + 1]

            # Build logprobs array aligned with target_tokens
            # weights_shifted has 0 for prompt positions, 1 for response positions
            # We need to map sampling logprobs to the response positions
            n_targets = len(target_tokens)
            logprobs_aligned = [0.0] * n_targets
            advantages_aligned = [0.0] * n_targets

            # Find where response tokens start (first non-zero weight)
            resp_start = None
            for j, w in enumerate(weights_shifted):
                if float(w) > 0:
                    resp_start = j
                    break

            if resp_start is not None:
                # Map sampling logprobs to response positions
                n_resp = min(len(turn_logprobs), n_targets - resp_start)
                for j in range(n_resp):
                    logprobs_aligned[resp_start + j] = float(turn_logprobs[j])
                    advantages_aligned[resp_start + j] = float(advantage)

            datum = tinker.Datum(
                model_input=input_model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=target_tokens,
                        dtype="int64",
                        shape=[len(target_tokens)],
                    ),
                    "logprobs": tinker.TensorData(
                        data=logprobs_aligned,
                        dtype="float32",
                        shape=[len(logprobs_aligned)],
                    ),
                    "advantages": tinker.TensorData(
                        data=advantages_aligned,
                        dtype="float32",
                        shape=[len(advantages_aligned)],
                    ),
                }
            )
            data.append(datum)

        except Exception as e:
            logger.warning(f"Failed to convert turn for importance_sampling: {e}")
            continue

    return data


def sample_tasks(task_type: str, batch_size: int, step: int) -> list[dict]:
    """Sample a batch of tasks for training."""
    seed_offset = step * 1000  # Different tasks each step

    if task_type == "niah":
        tasks = generate_niah_suite(
            n_tasks=batch_size,
            doc_lengths=[5000, 10000, 20000, 50000, 100000],
            seed_offset=seed_offset,
        )
        return [{"task": t, "type": "niah"} for t in tasks]
    elif task_type == "multi_niah":
        tasks = generate_multi_niah_suite(
            n_tasks=batch_size, seed_offset=seed_offset
        )
        return [{"task": t, "type": "multi_niah"} for t in tasks]
    elif task_type == "doc_classify":
        tasks = generate_doc_classify_suite(
            n_tasks=batch_size, seed_offset=seed_offset
        )
        return [{"task": t, "type": "doc_classify"} for t in tasks]
    elif task_type == "dataframe_qa":
        tasks = generate_dataframe_qa_suite(
            n_tasks=batch_size, seed_offset=seed_offset
        )
        return [{"task": t, "type": "dataframe_qa"} for t in tasks]
    elif task_type == "code_debug":
        tasks = generate_code_debug_suite(
            n_tasks=batch_size, seed_offset=seed_offset
        )
        return [{"task": t, "type": "code_debug"} for t in tasks]
    elif task_type == "mixed":
        # Allocate evenly: ~1/3 each of niah, multi_niah, doc_classify
        n_niah = max(1, batch_size // 3)
        n_mniah = max(1, batch_size // 3)
        n_docclass = max(1, batch_size - n_niah - n_mniah)
        niah_tasks = generate_niah_suite(
            n_tasks=n_niah,
            doc_lengths=[5000, 10000, 20000, 50000, 100000],
            seed_offset=seed_offset,
        )
        mniah_tasks = generate_multi_niah_suite(
            n_tasks=n_mniah,
            seed_offset=seed_offset + 50000,
        )
        docclass_tasks = generate_doc_classify_suite(
            n_tasks=n_docclass,
            seed_offset=seed_offset + 80000,
        )
        tasks = [{"task": t, "type": "niah"} for t in niah_tasks]
        tasks += [{"task": t, "type": "multi_niah"} for t in mniah_tasks]
        tasks += [{"task": t, "type": "doc_classify"} for t in docclass_tasks]
        np.random.shuffle(tasks)
        return tasks
    elif task_type == "mixed_all":
        # Include all 5 task types — 1 each minimum
        n_per = max(1, batch_size // 5)
        niah_tasks = generate_niah_suite(
            n_tasks=n_per,
            doc_lengths=[5000, 10000, 20000, 50000, 100000],
            seed_offset=seed_offset,
        )
        mniah_tasks = generate_multi_niah_suite(
            n_tasks=n_per, seed_offset=seed_offset + 50000,
        )
        docclass_tasks = generate_doc_classify_suite(
            n_tasks=n_per, seed_offset=seed_offset + 80000,
        )
        dfqa_tasks = generate_dataframe_qa_suite(
            n_tasks=n_per, seed_offset=seed_offset + 90000,
        )
        debug_tasks = generate_code_debug_suite(
            n_tasks=max(1, batch_size - 4 * n_per),
            seed_offset=seed_offset + 95000,
        )
        tasks = [{"task": t, "type": "niah"} for t in niah_tasks]
        tasks += [{"task": t, "type": "multi_niah"} for t in mniah_tasks]
        tasks += [{"task": t, "type": "doc_classify"} for t in docclass_tasks]
        tasks += [{"task": t, "type": "dataframe_qa"} for t in dfqa_tasks]
        tasks += [{"task": t, "type": "code_debug"} for t in debug_tasks]
        np.random.shuffle(tasks)
        return tasks
    elif task_type == "mixed_v2":
        # Weighted random sampling of task types per slot
        # 40% NIAH, 20% Multi-NIAH, 20% Doc-Classify, 10% DataFrame QA, 10% Code Debug
        # This prevents NIAH regression by overweighting NIAH
        task_weights = [
            ("niah", 0.40),
            ("multi_niah", 0.20),
            ("doc_classify", 0.20),
            ("dataframe_qa", 0.10),
            ("code_debug", 0.10),
        ]
        type_names = [t[0] for t in task_weights]
        type_probs = [t[1] for t in task_weights]
        rng = np.random.RandomState(seed_offset)
        chosen_types = rng.choice(type_names, size=batch_size, p=type_probs)

        # Count how many of each type
        type_counts = {}
        for t in chosen_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        tasks = []
        generators = {
            "niah": lambda n, s: [{"task": t, "type": "niah"} for t in
                generate_niah_suite(n_tasks=n, doc_lengths=[5000, 10000, 20000, 50000], seed_offset=s)],
            "multi_niah": lambda n, s: [{"task": t, "type": "multi_niah"} for t in
                generate_multi_niah_suite(n_tasks=n, seed_offset=s)],
            "doc_classify": lambda n, s: [{"task": t, "type": "doc_classify"} for t in
                generate_doc_classify_suite(n_tasks=n, seed_offset=s)],
            "dataframe_qa": lambda n, s: [{"task": t, "type": "dataframe_qa"} for t in
                generate_dataframe_qa_suite(n_tasks=n, seed_offset=s)],
            "code_debug": lambda n, s: [{"task": t, "type": "code_debug"} for t in
                generate_code_debug_suite(n_tasks=n, seed_offset=s)],
        }
        for i, (ttype, count) in enumerate(type_counts.items()):
            tasks.extend(generators[ttype](count, seed_offset + i * 10000))
        np.random.shuffle(tasks)
        return tasks
    elif task_type == "mixed_v3":
        # v3: Adds multi-hop QA + hard NIAH (100K docs)
        # 30% NIAH, 15% Multi-NIAH, 15% Doc-Classify, 15% Multi-Hop QA, 10% DFQA, 10% CodeDebug, 5% Hard NIAH
        task_weights = [
            ("niah", 0.30),
            ("multi_niah", 0.15),
            ("doc_classify", 0.15),
            ("multi_hop_qa", 0.15),
            ("dataframe_qa", 0.10),
            ("code_debug", 0.10),
            ("niah_hard", 0.05),
        ]
        type_names = [t[0] for t in task_weights]
        type_probs = [t[1] for t in task_weights]
        rng = np.random.RandomState(seed_offset)
        chosen_types = rng.choice(type_names, size=batch_size, p=type_probs)

        type_counts = {}
        for t in chosen_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        tasks = []
        generators = {
            "niah": lambda n, s: [{"task": t, "type": "niah"} for t in
                generate_niah_suite(n_tasks=n, doc_lengths=[5000, 10000, 20000, 50000], seed_offset=s)],
            "niah_hard": lambda n, s: [{"task": t, "type": "niah"} for t in
                generate_niah_suite(n_tasks=n, doc_lengths=[100000], seed_offset=s)],
            "multi_niah": lambda n, s: [{"task": t, "type": "multi_niah"} for t in
                generate_multi_niah_suite(n_tasks=n, seed_offset=s)],
            "doc_classify": lambda n, s: [{"task": t, "type": "doc_classify"} for t in
                generate_doc_classify_suite(n_tasks=n, seed_offset=s)],
            "multi_hop_qa": lambda n, s: [{"task": t, "type": "multi_hop_qa"} for t in
                generate_multi_hop_suite(n_tasks=n, seed_offset=s)],
            "dataframe_qa": lambda n, s: [{"task": t, "type": "dataframe_qa"} for t in
                generate_dataframe_qa_suite(n_tasks=n, seed_offset=s)],
            "code_debug": lambda n, s: [{"task": t, "type": "code_debug"} for t in
                generate_code_debug_suite(n_tasks=n, seed_offset=s)],
        }
        for i, (ttype, count) in enumerate(type_counts.items()):
            tasks.extend(generators[ttype](count, seed_offset + i * 10000))
        np.random.shuffle(tasks)
        return tasks
    elif task_type == "mixed_v4":
        # v4: Heavy multi-hop focus for decomposition learning
        # 15% NIAH, 10% Multi-NIAH, 10% Doc-Classify, 20% Hard Multi-Hop, 10% Multi-Hop QA,
        # 10% DFQA, 10% CodeDebug, 10% Notebook QA, 5% Hard NIAH
        task_weights = [
            ("niah", 0.15),
            ("multi_niah", 0.10),
            ("doc_classify", 0.10),
            ("hard_multi_hop", 0.20),
            ("multi_hop_qa", 0.10),
            ("dataframe_qa", 0.10),
            ("code_debug", 0.10),
            ("notebook_qa", 0.10),
            ("niah_hard", 0.05),
        ]
        type_names = [t[0] for t in task_weights]
        type_probs = [t[1] for t in task_weights]
        rng = np.random.RandomState(seed_offset)
        chosen_types = rng.choice(type_names, size=batch_size, p=type_probs)

        type_counts = {}
        for t in chosen_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        tasks = []
        generators = {
            "niah": lambda n, s: [{"task": t, "type": "niah"} for t in
                generate_niah_suite(n_tasks=n, doc_lengths=[5000, 10000, 20000, 50000], seed_offset=s)],
            "niah_hard": lambda n, s: [{"task": t, "type": "niah"} for t in
                generate_niah_suite(n_tasks=n, doc_lengths=[100000], seed_offset=s)],
            "multi_niah": lambda n, s: [{"task": t, "type": "multi_niah"} for t in
                generate_multi_niah_suite(n_tasks=n, seed_offset=s)],
            "doc_classify": lambda n, s: [{"task": t, "type": "doc_classify"} for t in
                generate_doc_classify_suite(n_tasks=n, seed_offset=s)],
            "multi_hop_qa": lambda n, s: [{"task": t, "type": "multi_hop_qa"} for t in
                generate_multi_hop_suite(n_tasks=n, seed_offset=s)],
            "hard_multi_hop": lambda n, s: [{"task": t, "type": "hard_multi_hop"} for t in
                generate_hard_multi_hop_suite(n_tasks=n, seed_offset=s)],
            "dataframe_qa": lambda n, s: [{"task": t, "type": "dataframe_qa"} for t in
                generate_dataframe_qa_suite(n_tasks=n, seed_offset=s)],
            "code_debug": lambda n, s: [{"task": t, "type": "code_debug"} for t in
                generate_code_debug_suite(n_tasks=n, seed_offset=s)],
            "notebook_qa": lambda n, s: [{"task": t, "type": "notebook_qa"} for t in
                generate_notebook_qa_suite(n_tasks=n, seed_offset=s)],
        }
        for i, (ttype, count) in enumerate(type_counts.items()):
            tasks.extend(generators[ttype](count, seed_offset + i * 10000))
        np.random.shuffle(tasks)
        return tasks

    return []


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
    return 0


def train_rl(
    model_name: str,
    model_path: str | None = None,
    steps: int = 30,
    K: int = 8,
    batch_size: int = 2,
    lr: float = 5e-6,
    lora_rank: int = 32,
    kl_coeff: float = 0.05,
    task_type: str = "mixed",
    save_every: int = 10,
    experiment_name: str = "grpo",
):
    """Run GRPO RL training on Tinker."""

    # Setup logging
    log_dir = Path("data/rl") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for persistent logging
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-6s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Setup tokenizer and renderer
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    # Create Tinker training client (resume from SFT checkpoint)
    service_client = tinker.ServiceClient()

    if model_path:
        logger.info(f"Resuming training from {model_path}")
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

    # Log model_id for checkpoint access
    logger.info(f"  Training model_id: {training_client.model_id}")
    logger.info(f"  State checkpoint format: tinker://{training_client.model_id}/weights/state-XXXX")

    # Create sampling client for trajectory generation
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="rl-init"
    )

    # Create model wrapper for RLM scaffold
    model = TinkerModel(
        model_name=model_name,
        max_new_tokens=2048,
        temperature=1.0,  # Higher temp for exploration (v2: was 0.8)
    )
    model.capture_logprobs = True  # Enable logprob capture for importance_sampling
    # Override the sampling client with our training one
    model.sampling_client = sampling_client

    # LoRA LR scaling
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
    except Exception:
        effective_lr = lr

    system_prompt = QWEN35_35B_SYSTEM_PROMPT

    logger.info(f"\n{'='*60}")
    logger.info(f"GRPO RL Training Config")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  LR: {effective_lr:.2e}")
    logger.info(f"  K (trajectories/prompt): {K}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Task type: {task_type}")
    logger.info(f"  KL coeff: {kl_coeff}")
    logger.info(f"  LoRA rank: {lora_rank}")
    logger.info(f"  LR schedule: cosine (min_lr = 10% of base)")
    logger.info(f"{'='*60}\n")

    # Determine loss function (v2: try importance_sampling if logprobs available)
    use_importance_sampling = True  # Will fall back to cross_entropy if no logprobs
    loss_fn = "importance_sampling"  # Default for v2

    training_log = []
    t0 = time.time()

    for step in range(steps):
        step_t0 = time.time()

        # Cosine LR schedule: decay from effective_lr to 10% over training
        import math
        min_lr = effective_lr * 0.1
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / max(steps - 1, 1)))
        step_lr = min_lr + (effective_lr - min_lr) * cosine_decay
        adam_params = tinker.AdamParams(learning_rate=step_lr)

        logger.info(f"\n--- Step {step + 1}/{steps} --- (LR: {step_lr:.2e})")

        # 1. Sample tasks
        tasks = sample_tasks(task_type, batch_size, step)

        # 2. Generate K trajectories per task
        all_groups = []
        for task_info in tasks:
            task = task_info["task"]
            group_trajectories = []

            for k in range(K):
                # Reset model stats between trajectories to prevent accumulation
                if hasattr(model, 'reset_stats'):
                    model.reset_stats()
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
                    score = score_trajectory(traj_dict, task_info)
                    traj_dict["score"] = score
                    traj_dict["reward"] = compute_reward(traj_dict, task_info["type"])
                    group_trajectories.append(traj_dict)
                except Exception as e:
                    logger.warning(f"Trajectory {k+1} failed: {e}")
                    group_trajectories.append({"score": 0, "reward": 0, "error": str(e)})

            all_groups.append({
                "task_info": task_info,
                "trajectories": group_trajectories,
            })

        # 3. Compute advantages (group-relative)
        step_rewards = []
        step_advantages = []
        training_data = []
        n_updates = 0
        per_task_rewards = {}  # Track rewards by task type

        for group in all_groups:
            rewards = [t["reward"] for t in group["trajectories"]]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            step_rewards.extend(rewards)

            # Track per-task-type rewards
            ttype = group["task_info"]["type"]
            if ttype not in per_task_rewards:
                per_task_rewards[ttype] = []
            per_task_rewards[ttype].append(mean_r)

            if std_r < 1e-6:
                # All same reward — skip this group (no learning signal)
                logger.info(f"  Skipping group (all same reward: {mean_r:.3f})")
                continue

            for traj_dict, reward in zip(group["trajectories"], rewards):
                advantage = (reward - mean_r) / (std_r + 1e-8)

                if abs(advantage) < 0.05:
                    continue  # Skip near-zero advantages (tighter threshold)

                step_advantages.append(advantage)

                # Convert to training data — try importance_sampling first, fall back to cross_entropy
                turn_has_logprobs = any(
                    t.get("logprobs") is not None for t in traj_dict.get("turns", [])
                )
                if use_importance_sampling and turn_has_logprobs:
                    data = trajectory_to_training_data_is(
                        traj_dict, advantage, renderer, tokenizer
                    )
                else:
                    data = trajectory_to_training_data(
                        traj_dict, advantage, renderer, tokenizer
                    )
                training_data.extend(data)
                n_updates += 1

        # 4. Train on collected data
        step_loss = None
        if training_data:
            # Determine loss function based on data format
            # If data has "logprobs" and "advantages" keys → importance_sampling
            # If data has "weights" key → cross_entropy
            sample_inputs = training_data[0].loss_fn_inputs if training_data else {}
            actual_loss_fn = "importance_sampling" if "advantages" in sample_inputs else "cross_entropy"

            # Batch the training data
            for i in range(0, len(training_data), 4):
                batch = training_data[i:i+4]
                fwd_bwd = training_client.forward_backward(batch, actual_loss_fn)
                optim = training_client.optim_step(adam_params)
                result = fwd_bwd.result()
                optim.result()

                loss = result.metrics.get("loss:sum") if hasattr(result, 'metrics') else None
                if loss is not None:
                    step_loss = float(loss)

        # 5. Refresh sampling client with updated weights (every step for v2)
        if training_data:  # Only refresh if we actually trained
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"rl-step-{step+1}"
            )
            model.sampling_client = sampling_client

        # 6. Log
        step_time = time.time() - step_t0
        avg_reward = np.mean(step_rewards) if step_rewards else 0
        avg_advantage = np.mean(np.abs(step_advantages)) if step_advantages else 0

        # Log negative/positive advantage counts
        n_pos = sum(1 for a in step_advantages if a > 0)
        n_neg = sum(1 for a in step_advantages if a < 0)

        step_info = {
            "step": step + 1,
            "avg_reward": float(avg_reward),
            "max_reward": float(max(step_rewards)) if step_rewards else 0,
            "min_reward": float(min(step_rewards)) if step_rewards else 0,
            "n_updates": n_updates,
            "n_pos_advantages": n_pos,
            "n_neg_advantages": n_neg,
            "n_training_data": len(training_data),
            "avg_advantage_abs": float(avg_advantage),
            "loss": step_loss,
            "lr": step_lr,
            "time": step_time,
            "elapsed": time.time() - t0,
        }
        training_log.append(step_info)

        logger.info(
            f"  Reward: {avg_reward:.3f} [{min(step_rewards) if step_rewards else 0:.3f}, "
            f"{max(step_rewards) if step_rewards else 0:.3f}] | "
            f"Updates: {n_updates} (+{n_pos}/-{n_neg}) | Data: {len(training_data)} | "
            f"Loss: {step_loss if step_loss is not None else 'N/A'} | "
            f"Time: {step_time:.1f}s"
        )

        # Log per-task-type rewards for monitoring task interference
        if per_task_rewards:
            parts = []
            for ttype, rewards in sorted(per_task_rewards.items()):
                avg_r = np.mean(rewards)
                parts.append(f"{ttype}={avg_r:.3f}")
            logger.info(f"  Per-task rewards: {' | '.join(parts)}")

        # Save checkpoint
        if (step + 1) % save_every == 0:
            ckpt_name = f"checkpoint-{step+1:04d}"
            logger.info(f"  Saving checkpoint: {ckpt_name}")
            training_client.save_weights_and_get_sampling_client(name=ckpt_name)
            training_client.save_state(name=f"state-{step+1:04d}")

            # Save sample trajectories
            sample_dir = log_dir / f"samples_step{step+1}"
            sample_dir.mkdir(exist_ok=True)
            for gi, group in enumerate(all_groups[:2]):  # Save first 2 groups
                for ti, traj in enumerate(group["trajectories"][:3]):  # First 3 per group
                    with open(sample_dir / f"group{gi}_traj{ti}.json", "w") as f:
                        json.dump(traj, f, indent=2, default=str)

    # Save final model
    logger.info("\nSaving final checkpoint...")
    training_client.save_weights_and_get_sampling_client(name="final")
    training_client.save_state(name="final-state")

    total_time = time.time() - t0

    # Save full log
    log_info = {
        "model": model_name,
        "model_path": model_path,
        "steps": steps,
        "K": K,
        "batch_size": batch_size,
        "lr": effective_lr,
        "lora_rank": lora_rank,
        "kl_coeff": kl_coeff,
        "task_type": task_type,
        "total_time": total_time,
        "training_log": training_log,
    }

    with open(log_dir / "training_log.json", "w") as f:
        json.dump(log_info, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"RL TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"  Final avg reward: {training_log[-1]['avg_reward']:.3f}")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GRPO RL training on Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--model-path", default=None,
                        help="Tinker state path (from SFT checkpoint)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--kl-coeff", type=float, default=0.05)
    parser.add_argument("--task-type", default="mixed",
                        choices=["niah", "multi_niah", "doc_classify", "dataframe_qa", "code_debug", "mixed", "mixed_all", "mixed_v2", "mixed_v3", "mixed_v4"])
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--experiment-name", default="grpo_tinker")
    args = parser.parse_args()

    train_rl(
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
    )


if __name__ == "__main__":
    main()
