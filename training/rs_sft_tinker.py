#!/usr/bin/env python3
"""
Rejection-Sampling SFT (RS-SFT) training on Tinker API.

This addresses the zero-sum specialization-generalization tradeoff in GRPO
by training ONLY on the best trajectories from rejection sampling, using
standard cross-entropy loss (no negative gradients).

Key differences from standard SFT (sft_tinker.py):
1. Task-balanced batching: equal representation across task types per batch
2. LR schedule: linear warmup + cosine decay (not constant)
3. Curriculum: optionally start with easy tasks, progress to hard ones
4. Diversity weighting: de-duplicate similar code patterns within each task type
5. Gradient accumulation: accumulate over mini-batches before optimizer step
6. Periodic eval checkpoints: save every N steps for comparison

RS-SFT hypothesis: By eliminating negative gradients (which GRPO uses to push
the model AWAY from bad trajectories), we avoid the format rigidity regression
where RL training teaches overly strict parsing code.

Usage:
    # Standard RS-SFT
    uv run python training/rs_sft_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --data data/filtered/rs_sft_v1.jsonl \
        --epochs 3 --lr 2e-5 \
        --save-every 5 \
        --experiment-name rs_sft_v1

    # With curriculum learning (easy first, then hard)
    uv run python training/rs_sft_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --data data/filtered/rs_sft_v1.jsonl \
        --epochs 5 --lr 2e-5 \
        --curriculum \
        --save-every 5 \
        --experiment-name rs_sft_curriculum

    # Resume from GRPO checkpoint (RS-SFT as regularization)
    uv run python training/rs_sft_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --data data/filtered/rs_sft_v1.jsonl \
        --resume-from tinker://run-id/weights/state-0005 \
        --epochs 2 --lr 5e-6 \
        --experiment-name rs_sft_on_grpo
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Task difficulty ordering for curriculum learning
TASK_DIFFICULTY = {
    # Easy: high base model accuracy, simple search pattern
    "verbatim_copy": 0,
    "hard_niah": 0,
    "niah": 1,
    "multi_niah": 1,
    # Medium: requires multi-step or aggregation
    "doc_classify": 2,
    "event_counting": 2,
    "multi_hop_qa": 2,
    "key_value_retrieval": 3,
    # Hard: complex reasoning, comparison, structured data
    "notebook_qa": 3,
    "dataframe_qa": 3,
    "hard_multi_hop": 4,
    "cross_doc_compare": 4,
    "code_debug": 4,
}


def load_sft_data(data_path: str) -> list[dict]:
    """Load SFT samples from JSONL file."""
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} SFT samples from {data_path}")

    # Report distribution
    by_type = defaultdict(int)
    for s in samples:
        by_type[s.get("task_type", "unknown")] += 1
    for task_type in sorted(by_type.keys()):
        logger.info(f"  {task_type}: {by_type[task_type]} samples")

    return samples


def sample_to_datum(sample: dict, renderer) -> tinker.Datum | None:
    """Convert an SFT sample to a Tinker Datum.

    Each sample has:
    - messages: conversation history (system + user + assistant turns)
    - completion: the assistant response to train on
    """
    messages = sample["messages"]
    completion = sample.get("completion", "")

    # Build the full message sequence including the completion
    full_messages = list(messages)
    if completion:
        full_messages.append({"role": "assistant", "content": completion})

    try:
        model_input, weights = renderer.build_supervised_example(full_messages)
    except Exception as e:
        logger.warning(f"Failed to render sample: {e}")
        return None

    # Use cookbook's utility to create properly shifted input/target pairs
    from tinker_cookbook.supervised.common import (
        create_rightshifted_model_input_and_leftshifted_targets,
    )
    input_model_input, target_tokens = (
        create_rightshifted_model_input_and_leftshifted_targets(model_input.chunks)
    )

    # Weights need same shift: drop first, keep up to len(target_tokens)
    if hasattr(weights, "tolist"):
        weights_list = weights.tolist()
    else:
        weights_list = list(weights)
    weights_shifted = weights_list[1 : len(target_tokens) + 1]

    datum = tinker.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
            "weights": tinker.TensorData(
                data=[float(w) for w in weights_shifted],
                dtype="float32",
                shape=[len(weights_shifted)],
            ),
        },
    )
    return datum


def create_balanced_batches(
    data_with_types: list[tuple[tinker.Datum, str]],
    batch_size: int,
    rng: np.random.RandomState,
) -> list[list[tinker.Datum]]:
    """Create task-balanced batches.

    Each batch tries to include samples from as many task types as possible,
    rather than letting dominant task types monopolize batches.
    """
    # Group by task type
    by_type = defaultdict(list)
    for datum, task_type in data_with_types:
        by_type[task_type].append(datum)

    # Shuffle within each type
    for task_type in by_type:
        rng.shuffle(by_type[task_type])

    # Round-robin: cycle through task types
    type_names = sorted(by_type.keys())
    type_indices = {t: 0 for t in type_names}

    batches = []
    current_batch = []
    type_cycle = 0

    total_data = sum(len(v) for v in by_type.values())

    while sum(type_indices[t] < len(by_type[t]) for t in type_names) > 0:
        # Pick from each type in round-robin fashion
        for t in type_names:
            idx = type_indices[t]
            if idx < len(by_type[t]):
                current_batch.append(by_type[t][idx])
                type_indices[t] += 1

                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []

        type_cycle += 1

    # Final partial batch
    if current_batch:
        batches.append(current_batch)

    return batches


def create_curriculum_batches(
    data_with_types: list[tuple[tinker.Datum, str]],
    batch_size: int,
    epoch: int,
    total_epochs: int,
    rng: np.random.RandomState,
) -> list[list[tinker.Datum]]:
    """Create curriculum-ordered batches.

    Early epochs focus on easy tasks, later epochs include all tasks.
    This prevents the model from being overwhelmed by hard tasks before
    it learns basic RLM patterns.
    """
    # Determine max difficulty for this epoch
    max_difficulty = min(4, int((epoch / max(total_epochs - 1, 1)) * 5))

    # Filter to allowed difficulty levels
    filtered = [
        (datum, task_type)
        for datum, task_type in data_with_types
        if TASK_DIFFICULTY.get(task_type, 2) <= max_difficulty
    ]

    if not filtered:
        filtered = data_with_types  # Fallback: use all

    types_included = set(t for _, t in filtered)
    logger.info(
        f"  Curriculum epoch {epoch+1}: max_difficulty={max_difficulty}, "
        f"types={sorted(types_included)}, samples={len(filtered)}"
    )

    return create_balanced_batches(filtered, batch_size, rng)


def train_rs_sft(
    model_name: str,
    data_path: str,
    epochs: int = 3,
    lr: float = 2e-5,
    lora_rank: int = 32,
    batch_size: int = 4,
    grad_accum: int = 4,
    warmup_frac: float = 0.1,
    save_every: int = 5,
    experiment_name: str = "rs_sft",
    resume_from: str | None = None,
    curriculum: bool = False,
    balanced: bool = True,
):
    """Run RS-SFT training on Tinker.

    Args:
        model_name: Base model identifier
        data_path: Path to filtered JSONL from filter_rs_sft.py
        epochs: Number of training epochs
        lr: Learning rate (before LoRA scaling)
        lora_rank: LoRA rank for adapter
        batch_size: Mini-batch size for forward_backward
        grad_accum: Gradient accumulation steps before optim_step
        warmup_frac: Fraction of total steps for linear LR warmup
        save_every: Save checkpoint every N optimizer steps
        experiment_name: Name for logging directory
        resume_from: Tinker state path to resume from
        curriculum: Use curriculum learning (easy tasks first)
        balanced: Use task-balanced batching
    """
    log_dir = Path("data/sft") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-6s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Load data
    samples = load_sft_data(data_path)
    if not samples:
        logger.error("No training data loaded!")
        return

    # Setup tokenizer and renderer
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    # Convert samples to Tinker datums
    logger.info("Converting samples to Tinker format...")
    data_with_types = []
    failed = 0
    for sample in samples:
        datum = sample_to_datum(sample, renderer)
        if datum:
            task_type = sample.get("task_type", "unknown")
            data_with_types.append((datum, task_type))
        else:
            failed += 1
    logger.info(f"Converted {len(data_with_types)}/{len(samples)} samples ({failed} failed)")

    if not data_with_types:
        logger.error("No valid training data!")
        return

    # Create Tinker training client
    service_client = tinker.ServiceClient()

    if resume_from:
        logger.info(f"Resuming from {resume_from}")
        training_client = service_client.create_training_client_from_state(resume_from)
    else:
        logger.info(f"Creating LoRA training client for {model_name} (rank={lora_rank})")
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            seed=42,
        )

    # Adjust LR for LoRA
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
        logger.info(f"LoRA LR scaling: {lr} x {lr_factor:.1f} = {effective_lr:.2e}")
    except Exception:
        effective_lr = lr
        logger.info(f"Using base LR: {effective_lr:.2e}")

    # Compute schedule parameters
    steps_per_epoch = len(data_with_types) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))
    min_lr = effective_lr * 0.1

    logger.info(f"\n{'='*60}")
    logger.info(f"RS-SFT Training Config")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Resume from: {resume_from or 'scratch'}")
    logger.info(f"  LoRA rank: {lora_rank}")
    logger.info(f"  Learning rate: {effective_lr:.2e}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    logger.info(f"  Training samples: {len(data_with_types)}")
    logger.info(f"  Steps per epoch: ~{steps_per_epoch}")
    logger.info(f"  Total steps: ~{total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Balanced batching: {balanced}")
    logger.info(f"  Curriculum learning: {curriculum}")
    logger.info(f"  Save every: {save_every} steps")
    logger.info(f"{'='*60}\n")

    # Training loop
    training_log = []
    global_step = 0
    accum_losses = []
    t0 = time.time()

    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        rng = np.random.RandomState(42 + epoch)

        # Create batches
        if curriculum:
            batches = create_curriculum_batches(
                data_with_types, batch_size, epoch, epochs, rng
            )
        elif balanced:
            batches = create_balanced_batches(data_with_types, batch_size, rng)
        else:
            # Simple shuffle
            indices = rng.permutation(len(data_with_types))
            all_datums = [data_with_types[i][0] for i in indices]
            batches = [
                all_datums[i : i + batch_size]
                for i in range(0, len(all_datums), batch_size)
            ]

        epoch_losses = []

        for batch_idx, batch in enumerate(batches):
            if not batch:
                continue

            # LR schedule: linear warmup + cosine decay
            if global_step < warmup_steps:
                step_lr = effective_lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                step_lr = min_lr + (effective_lr - min_lr) * cosine_decay

            # Forward-backward
            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")

            # Gradient accumulation: only step every grad_accum batches
            if (batch_idx + 1) % grad_accum == 0:
                adam_params = tinker.AdamParams(learning_rate=step_lr)
                optim_future = training_client.optim_step(adam_params)

                # Wait for results
                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                loss = (
                    fwd_bwd_result.metrics.get("loss:sum")
                    if hasattr(fwd_bwd_result, "metrics")
                    else None
                )
                if loss is not None:
                    epoch_losses.append(float(loss))
                    accum_losses.append(float(loss))

                global_step += 1

                # Log periodically
                if global_step % 5 == 0:
                    recent_loss = np.mean(accum_losses[-5:]) if accum_losses else 0
                    elapsed = time.time() - t0
                    logger.info(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {recent_loss:.4f} | LR: {step_lr:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                step_info = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(loss) if loss is not None else None,
                    "lr": step_lr,
                    "time": time.time() - t0,
                }
                training_log.append(step_info)

                # Save checkpoint
                if global_step % save_every == 0:
                    ckpt_name = f"state-{global_step:04d}"
                    logger.info(f"  Saving checkpoint: {ckpt_name}")
                    training_client.save_state(name=ckpt_name)
                    sc = training_client.save_weights_and_get_sampling_client(
                        name=f"weights-{global_step:04d}"
                    )
                    ckpt_info = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "avg_loss": float(np.mean(accum_losses[-save_every:])) if accum_losses else None,
                        "lr": step_lr,
                        "model_path": sc.model_path if hasattr(sc, "model_path") else None,
                    }
                    with open(log_dir / f"{ckpt_name}_info.json", "w") as f:
                        json.dump(ckpt_info, f, indent=2)

            else:
                # Just wait for forward_backward (gradient accumulating)
                fwd_bwd_result = fwd_bwd_future.result()
                loss = (
                    fwd_bwd_result.metrics.get("loss:sum")
                    if hasattr(fwd_bwd_result, "metrics")
                    else None
                )
                if loss is not None:
                    epoch_losses.append(float(loss))
                    accum_losses.append(float(loss))

        # End of epoch
        if epoch_losses:
            logger.info(
                f"  Epoch {epoch + 1} complete | "
                f"Avg loss: {np.mean(epoch_losses):.4f} | "
                f"Steps: {global_step}"
            )

    # Save final checkpoint
    logger.info("\nSaving final checkpoint...")
    training_client.save_state(name="final-state")
    final_sc = training_client.save_weights_and_get_sampling_client(name="final")
    final_model_path = final_sc.model_path if hasattr(final_sc, "model_path") else None

    total_time = time.time() - t0

    # Save training log
    log_info = {
        "model": model_name,
        "data_path": data_path,
        "epochs": epochs,
        "lr": effective_lr,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "balanced": balanced,
        "curriculum": curriculum,
        "total_steps": global_step,
        "total_time": total_time,
        "n_samples": len(data_with_types),
        "warmup_steps": warmup_steps,
        "final_model_path": final_model_path,
        "final_loss": float(np.mean(accum_losses[-10:])) if accum_losses else None,
        "resume_from": resume_from,
        "training_log": training_log,
    }
    with open(log_dir / "training_log.json", "w") as f:
        json.dump(log_info, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("RS-SFT TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Final loss: {accum_losses[-1]:.4f}" if accum_losses else "  No loss data")
    logger.info(f"  Total time: {total_time:.0f}s ({total_time/3600:.2f}h)")
    logger.info(f"  Final model: {final_model_path}")
    logger.info(f"  Logs: {log_dir}")
    logger.info(f"{'='*60}")

    return final_sc


def main():
    parser = argparse.ArgumentParser(description="RS-SFT training on Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--data", required=True, help="Path to filtered RS-SFT JSONL")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-frac", type=float, default=0.1,
                        help="Fraction of steps for warmup")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--experiment-name", default="rs_sft")
    parser.add_argument("--resume-from", default=None,
                        help="Tinker state path to resume from")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum learning")
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable task-balanced batching")
    args = parser.parse_args()

    train_rs_sft(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_frac=args.warmup_frac,
        save_every=args.save_every,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from,
        curriculum=args.curriculum,
        balanced=not args.no_balance,
    )


if __name__ == "__main__":
    main()
