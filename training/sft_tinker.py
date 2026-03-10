#!/usr/bin/env python3
"""
SFT training on Tinker API.

Replaces training/sft.py for remote training on Tinker's GPU cluster.
Uses tinker's forward_backward("cross_entropy") + optim_step() API.

Usage:
    uv run python training/sft_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --data data/filtered/sft_samples.jsonl \
        --epochs 5 \
        --lr 2e-4 \
        --lora-rank 32 \
        --experiment-name sft_35b_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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


def load_sft_data(data_path: str) -> list[dict]:
    """Load SFT samples from JSONL file."""
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logger.info(f"Loaded {len(samples)} SFT samples from {data_path}")
    return samples


def sample_to_datum(sample: dict, renderer, tokenizer) -> tinker.Datum | None:
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
    from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets
    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        model_input.chunks
    )
    # Weights need same shift: drop first, keep up to len(target_tokens)
    if hasattr(weights, 'tolist'):
        weights_list = weights.tolist()
    else:
        weights_list = list(weights)
    weights_shifted = weights_list[1:len(target_tokens) + 1]

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
        }
    )
    return datum


def train_sft(
    model_name: str,
    data_path: str,
    epochs: int = 5,
    lr: float = 2e-4,
    lora_rank: int = 32,
    batch_size: int = 4,
    save_every: int = 50,
    experiment_name: str = "sft",
    resume_from: str | None = None,
):
    """Run SFT training on Tinker."""

    # Setup
    log_dir = Path("data/sft") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    samples = load_sft_data(data_path)

    # Setup tokenizer and renderer
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

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

    # Adjust LR for LoRA (LoRA needs much higher LR)
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
        logger.info(f"LoRA LR scaling: {lr} × {lr_factor:.1f} = {effective_lr:.2e}")
    except Exception:
        effective_lr = lr
        logger.info(f"Using base LR: {effective_lr:.2e} (no LoRA scaling available)")

    # Convert samples to Tinker data
    logger.info("Converting samples to Tinker format...")
    all_data = []
    for sample in samples:
        datum = sample_to_datum(sample, renderer, tokenizer)
        if datum:
            all_data.append(datum)
    logger.info(f"Converted {len(all_data)}/{len(samples)} samples successfully")

    if not all_data:
        logger.error("No valid training data!")
        return

    # Training loop
    adam_params = tinker.AdamParams(learning_rate=effective_lr)
    total_steps = 0
    training_log = []

    logger.info(f"\n{'='*60}")
    logger.info(f"SFT Training Config")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  LoRA rank: {lora_rank}")
    logger.info(f"  Learning rate: {effective_lr:.2e}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Samples: {len(all_data)}")
    logger.info(f"  Total steps: ~{len(all_data) * epochs // batch_size}")
    logger.info(f"{'='*60}\n")

    t0 = time.time()

    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        # Shuffle data each epoch
        np.random.seed(42 + epoch)
        indices = np.random.permutation(len(all_data))

        epoch_losses = []

        for batch_start in range(0, len(all_data), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [all_data[i] for i in batch_indices]

            if not batch:
                continue

            # Pipeline: submit forward_backward + optim_step together
            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            # Wait for results
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            # Extract loss from metrics dict (Tinker uses "loss:sum" key)
            loss = fwd_bwd_result.metrics.get("loss:sum") if hasattr(fwd_bwd_result, 'metrics') else None

            step_info = {
                "epoch": epoch + 1,
                "step": total_steps,
                "batch_size": len(batch),
                "time": time.time() - t0,
            }
            if loss is not None:
                step_info["loss"] = float(loss)
                epoch_losses.append(float(loss))

            training_log.append(step_info)
            total_steps += 1

            # Log periodically
            if total_steps % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:]) if epoch_losses else 0
                logger.info(
                    f"  Step {total_steps} | Loss: {avg_loss:.4f} | "
                    f"Time: {time.time() - t0:.1f}s"
                )

            # Save checkpoint
            if total_steps % save_every == 0:
                ckpt_name = f"checkpoint-{total_steps:04d}"
                logger.info(f"  Saving checkpoint: {ckpt_name}")
                sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=ckpt_name
                )
                # Log the model path for later use
                with open(log_dir / f"{ckpt_name}_info.json", "w") as f:
                    json.dump({
                        "step": total_steps,
                        "epoch": epoch + 1,
                        "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    }, f, indent=2)

        # End of epoch stats
        if epoch_losses:
            logger.info(
                f"  Epoch {epoch + 1} complete | "
                f"Avg loss: {np.mean(epoch_losses):.4f} | "
                f"Steps: {total_steps}"
            )

    # Save final checkpoint
    logger.info("\nSaving final checkpoint...")
    final_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="final"
    )
    final_model_path = final_sampling_client.model_path if hasattr(final_sampling_client, 'model_path') else None
    logger.info(f"  Final model path: {final_model_path}")
    training_client.save_state(name="final-state")

    total_time = time.time() - t0

    # Save training log
    log_info = {
        "model": model_name,
        "data_path": data_path,
        "epochs": epochs,
        "lr": effective_lr,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "total_time": total_time,
        "n_samples": len(all_data),
        "final_model_path": final_model_path,
        "training_log": training_log,
    }

    with open(log_dir / "training_log.json", "w") as f:
        json.dump(log_info, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"SFT TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info(f"{'='*60}")

    return final_sampling_client


def main():
    parser = argparse.ArgumentParser(description="SFT training on Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--data", required=True, help="Path to SFT JSONL file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--experiment-name", default="sft_tinker")
    parser.add_argument("--resume-from", default=None,
                        help="Tinker state path to resume from")
    args = parser.parse_args()

    train_sft(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        save_every=args.save_every,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
