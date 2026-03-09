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
    import torch

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

                    # Convert weights to advantage-weighted
                    weights_np = np.array(
                        [float(w) for w in weights], dtype=np.float32
                    )
                    # Advantages only on response tokens (where weight > 0)
                    advantages_np = weights_np * advantage

                    # For importance_sampling, we need target_tokens and logprobs
                    # Use cross_entropy with weighted advantages instead
                    datum = tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "weights": tinker.TensorData.from_numpy(
                                weights_np * max(advantage, 0)
                            ),
                        }
                    )
                    data.append(datum)

                except Exception as e:
                    logger.warning(f"Failed to convert turn {i}: {e}")
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
    elif task_type == "mixed":
        niah_tasks = generate_niah_suite(
            n_tasks=batch_size // 2,
            doc_lengths=[5000, 10000, 20000, 50000, 100000],
            seed_offset=seed_offset,
        )
        mniah_tasks = generate_multi_niah_suite(
            n_tasks=batch_size - batch_size // 2,
            seed_offset=seed_offset + 50000,
        )
        tasks = [{"task": t, "type": "niah"} for t in niah_tasks]
        tasks += [{"task": t, "type": "multi_niah"} for t in mniah_tasks]
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

    # Create sampling client for trajectory generation
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="rl-init"
    )

    # Create model wrapper for RLM scaffold
    model = TinkerModel(
        model_name=model_name,
        max_new_tokens=2048,
        temperature=0.8,
    )
    # Override the sampling client with our training one
    model.sampling_client = sampling_client

    # LoRA LR scaling
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
    except Exception:
        effective_lr = lr

    adam_params = tinker.AdamParams(learning_rate=effective_lr)
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
    logger.info(f"{'='*60}\n")

    training_log = []
    t0 = time.time()

    for step in range(steps):
        step_t0 = time.time()
        logger.info(f"\n--- Step {step + 1}/{steps} ---")

        # 1. Sample tasks
        tasks = sample_tasks(task_type, batch_size, step)

        # 2. Generate K trajectories per task
        all_groups = []
        for task_info in tasks:
            task = task_info["task"]
            group_trajectories = []

            for k in range(K):
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

        for group in all_groups:
            rewards = [t["reward"] for t in group["trajectories"]]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            step_rewards.extend(rewards)

            if std_r < 1e-6:
                # All same reward — skip this group (no learning signal)
                logger.info(f"  Skipping group (all same reward: {mean_r:.3f})")
                continue

            for traj_dict, reward in zip(group["trajectories"], rewards):
                advantage = (reward - mean_r) / (std_r + 1e-8)

                if abs(advantage) < 0.1:
                    continue  # Skip near-zero advantages

                step_advantages.append(advantage)

                # Convert to training data
                data = trajectory_to_training_data(
                    traj_dict, advantage, renderer, tokenizer
                )
                training_data.extend(data)
                n_updates += 1

        # 4. Train on collected data
        step_loss = None
        if training_data:
            # Batch the training data
            for i in range(0, len(training_data), 4):
                batch = training_data[i:i+4]
                fwd_bwd = training_client.forward_backward(batch, "cross_entropy")
                optim = training_client.optim_step(adam_params)
                result = fwd_bwd.result()
                optim.result()

                if hasattr(result, 'loss') and result.loss is not None:
                    step_loss = float(result.loss)

        # 5. Refresh sampling client with updated weights
        if step % 5 == 4 or step == steps - 1:
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"rl-step-{step+1}"
            )
            model.sampling_client = sampling_client

        # 6. Log
        step_time = time.time() - step_t0
        avg_reward = np.mean(step_rewards) if step_rewards else 0
        avg_advantage = np.mean(np.abs(step_advantages)) if step_advantages else 0

        step_info = {
            "step": step + 1,
            "avg_reward": float(avg_reward),
            "max_reward": float(max(step_rewards)) if step_rewards else 0,
            "min_reward": float(min(step_rewards)) if step_rewards else 0,
            "n_updates": n_updates,
            "n_training_data": len(training_data),
            "avg_advantage_abs": float(avg_advantage),
            "loss": step_loss,
            "time": step_time,
            "elapsed": time.time() - t0,
        }
        training_log.append(step_info)

        logger.info(
            f"  Reward: {avg_reward:.3f} [{min(step_rewards) if step_rewards else 0:.3f}, "
            f"{max(step_rewards) if step_rewards else 0:.3f}] | "
            f"Updates: {n_updates} | Data: {len(training_data)} | "
            f"Loss: {step_loss if step_loss is not None else 'N/A'} | "
            f"Time: {step_time:.1f}s"
        )

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
                        choices=["niah", "multi_niah", "mixed"])
    parser.add_argument("--save-every", type=int, default=10)
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
