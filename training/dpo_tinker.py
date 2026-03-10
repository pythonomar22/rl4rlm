#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training on Tinker API.

Replaces training/dpo.py for remote training on Tinker's GPU cluster.
Uses Tinker's forward_backward_custom with a custom DPO loss function,
following the exact pattern from tinker_cookbook.preference.train_dpo.

Two training modes:
1. True DPO (--mode dpo): Uses forward_backward_custom with chosen/rejected pairs
   and a frozen reference model for KL-constrained preference optimization.
   This is the gold standard but costs 1.5x FLOPs per step.

2. Rejection sampling SFT (--mode rejection_sft): Trains only on chosen (correct)
   trajectories using standard cross_entropy, optionally weighted by reward score.
   Equivalent to "best-of-N rejection sampling" — a strong DPO approximation
   that is cheaper and simpler.

Data format (JSONL, created by scripts/create_dpo_pairs.py):
  {"chosen": {"messages": [...], "score": 1.0}, "rejected": {"messages": [...], "score": 0.0}, "task_id": "..."}

Usage:
    # True DPO
    uv run python training/dpo_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --data data/dpo/pairs.jsonl \
        --mode dpo \
        --model-path tinker://run-id/weights/sft-final \
        --experiment-name dpo_35b_v1

    # Rejection sampling SFT (simpler, cheaper)
    uv run python training/dpo_tinker.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --data data/dpo/pairs.jsonl \
        --mode rejection_sft \
        --experiment-name rejection_sft_35b_v1
"""

from __future__ import annotations

import argparse
import asyncio
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
import torch
import torch.nn.functional as F
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_pairs(data_path: str) -> list[dict]:
    """Load DPO pairs from JSONL file.

    Each line: {"chosen": {...}, "rejected": {...}, "task_id": "...", ...}
    Both chosen and rejected have a "messages" field (list of message dicts)
    and optionally a "score" field.
    """
    pairs = []
    with open(data_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                if "chosen" not in pair or "rejected" not in pair:
                    logger.warning(f"Line {line_no}: missing chosen/rejected, skipping")
                    continue
                if "messages" not in pair["chosen"] or "messages" not in pair["rejected"]:
                    logger.warning(f"Line {line_no}: missing messages in chosen/rejected, skipping")
                    continue
                pairs.append(pair)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_no}: invalid JSON: {e}")
    logger.info(f"Loaded {len(pairs)} DPO pairs from {data_path}")
    return pairs


# ---------------------------------------------------------------------------
# Conversion: messages -> Tinker Datum
# ---------------------------------------------------------------------------

def messages_to_datum(
    messages: list[dict],
    renderer,
    tokenizer,
    weight_scale: float = 1.0,
    max_length: int | None = None,
) -> tinker.Datum | None:
    """Convert a message list to a Tinker Datum for cross_entropy or DPO.

    The renderer's build_supervised_example gives per-token weights (0 for
    prompt tokens, 1 for completion tokens). We optionally scale them by
    weight_scale (for quality weighting in rejection_sft mode).
    """
    try:
        model_input, weights = renderer.build_supervised_example(messages)
    except Exception as e:
        logger.warning(f"Failed to render messages: {e}")
        return None

    # Convert weights to numpy
    if hasattr(weights, 'numpy'):
        weights_np = weights.float().numpy().astype(np.float32)
    elif isinstance(weights, torch.Tensor):
        weights_np = weights.float().numpy().astype(np.float32)
    else:
        weights_np = np.array(weights, dtype=np.float32)

    # Apply quality weight scaling
    if weight_scale != 1.0:
        weights_np = weights_np * weight_scale

    # Optional max_length truncation
    if max_length is not None and len(weights_np) > max_length:
        # Truncate from the end
        model_input_ints = list(model_input.to_ints())[:max_length]
        weights_np = weights_np[:max_length]
        model_input = tinker.ModelInput.from_ints(model_input_ints)

    datum = tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData.from_numpy(weights_np),
        }
    )
    return datum


def pair_to_data(
    pair: dict,
    renderer,
    tokenizer,
    max_length: int | None = None,
) -> tuple[tinker.Datum | None, tinker.Datum | None]:
    """Convert a preference pair to (chosen_datum, rejected_datum).

    Returns (None, None) if either side fails to render.
    """
    chosen_datum = messages_to_datum(
        pair["chosen"]["messages"], renderer, tokenizer,
        max_length=max_length,
    )
    rejected_datum = messages_to_datum(
        pair["rejected"]["messages"], renderer, tokenizer,
        max_length=max_length,
    )
    return chosen_datum, rejected_datum


# ---------------------------------------------------------------------------
# DPO loss (for forward_backward_custom)
# ---------------------------------------------------------------------------

def make_dpo_loss_fn(
    ref_logprobs_chosen: list[list[float | None]],
    ref_logprobs_rejected: list[list[float | None]],
    chosen_data: list[tinker.Datum],
    rejected_data: list[tinker.Datum],
    beta: float = 0.1,
):
    """Create a DPO loss function closure with pre-computed reference logprobs.

    Follows the exact pattern from tinker_cookbook.preference.train_dpo:
    - data is interleaved [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    - logprobs_list matches the same ordering
    - Reference logprobs are pre-computed and captured in the closure
    """
    def dpo_loss_fn(
        data: list[tinker.Datum],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Split logprobs into chosen and rejected (interleaved ordering)
        chosen_logprobs_list = [logprobs_list[i] for i in range(0, len(data), 2)]
        rejected_logprobs_list = [logprobs_list[i] for i in range(1, len(data), 2)]

        n_pairs = len(chosen_data)
        chosen_log_ratios = []
        rejected_log_ratios = []

        for i in range(n_pairs):
            # Get per-token weights (only completion tokens have weight > 0)
            chosen_weights = torch.tensor(
                chosen_data[i].loss_fn_inputs["weights"].data, dtype=torch.float32
            )
            rejected_weights = torch.tensor(
                rejected_data[i].loss_fn_inputs["weights"].data, dtype=torch.float32
            )

            # Policy log-probs (from forward pass, weighted by completion mask)
            chosen_policy_lp = torch.dot(
                chosen_logprobs_list[i].float(), chosen_weights
            )
            rejected_policy_lp = torch.dot(
                rejected_logprobs_list[i].float(), rejected_weights
            )

            # Reference log-probs (pre-computed, weighted by completion mask)
            ref_chosen_raw = ref_logprobs_chosen[i]
            ref_rejected_raw = ref_logprobs_rejected[i]

            # Replace None with 0.0 for tokens where ref logprobs weren't computed
            ref_chosen_tensor = torch.tensor(
                [lp if lp is not None else 0.0 for lp in ref_chosen_raw],
                dtype=torch.float32,
            )
            ref_rejected_tensor = torch.tensor(
                [lp if lp is not None else 0.0 for lp in ref_rejected_raw],
                dtype=torch.float32,
            )

            # Align lengths (ref logprobs may differ slightly from policy)
            min_len_c = min(len(ref_chosen_tensor), len(chosen_weights))
            chosen_ref_lp = torch.dot(
                ref_chosen_tensor[:min_len_c], chosen_weights[:min_len_c]
            )

            min_len_r = min(len(ref_rejected_tensor), len(rejected_weights))
            rejected_ref_lp = torch.dot(
                ref_rejected_tensor[:min_len_r], rejected_weights[:min_len_r]
            )

            # Log-ratio: log(pi/pi_ref) = log_pi - log_pi_ref
            chosen_log_ratios.append(chosen_policy_lp - chosen_ref_lp)
            rejected_log_ratios.append(rejected_policy_lp - rejected_ref_lp)

        # Stack and compute DPO loss
        chosen_log_ratio = torch.stack(chosen_log_ratios)
        rejected_log_ratio = torch.stack(rejected_log_ratios)

        # L = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        losses = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio))
        loss = losses.mean()

        # Metrics
        accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
        chosen_rewards = beta * chosen_log_ratio
        rejected_rewards = beta * rejected_log_ratio
        margin = (chosen_rewards - rejected_rewards).mean().item()

        metrics = {
            "dpo_loss": loss.item(),
            "accuracy": accuracy,
            "margin": margin,
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

        return loss, metrics

    return dpo_loss_fn


# ---------------------------------------------------------------------------
# Reference logprob computation
# ---------------------------------------------------------------------------

def compute_reference_logprobs(
    reference_client: tinker.SamplingClient,
    data_list: list[tinker.Datum],
) -> list[list[float | None]]:
    """Compute reference model logprobs for a list of Datums.

    Uses reference_client.compute_logprobs (sync) for each datum.
    Returns a list of per-token logprob lists.
    """
    all_logprobs = []

    async def _compute_all():
        tasks = [
            reference_client.compute_logprobs_async(datum.model_input)
            for datum in data_list
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_compute_all())
    for result in results:
        all_logprobs.append(list(result))

    return all_logprobs


# ---------------------------------------------------------------------------
# Mode 1: True DPO training
# ---------------------------------------------------------------------------

def train_dpo(
    model_name: str,
    data_path: str,
    epochs: int = 3,
    lr: float = 1e-5,
    lora_rank: int = 32,
    batch_size: int = 2,
    beta: float = 0.1,
    max_length: int | None = None,
    save_every: int = 20,
    experiment_name: str = "dpo",
    resume_from: str | None = None,
    model_path: str | None = None,
):
    """Run true DPO training on Tinker using forward_backward_custom.

    1. Load preference pairs
    2. Create training client and reference model (frozen snapshot)
    3. For each batch of pairs:
       a. Compute reference logprobs via reference_client.compute_logprobs
       b. Run forward_backward_custom with DPO loss
       c. optim_step
    """

    # Setup
    log_dir = Path("data/dpo") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pairs = load_dpo_pairs(data_path)
    if not pairs:
        logger.error("No valid DPO pairs loaded!")
        return

    # Setup tokenizer and renderer
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    # Create Tinker training client
    service_client = tinker.ServiceClient()

    if resume_from:
        logger.info(f"Resuming from state: {resume_from}")
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

        # If starting from an SFT checkpoint, load those weights
        if model_path:
            logger.info(f"Loading weights from SFT checkpoint: {model_path}")
            training_client.load_state(model_path).result()

    # Create reference model snapshot (frozen at init)
    logger.info("Creating reference model snapshot...")
    reference_client = training_client.save_weights_and_get_sampling_client(
        name="dpo-reference"
    )

    # LoRA LR scaling
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
        logger.info(f"LoRA LR scaling: {lr} x {lr_factor:.1f} = {effective_lr:.2e}")
    except Exception:
        effective_lr = lr
        logger.info(f"Using base LR: {effective_lr:.2e} (no LoRA scaling available)")

    # Convert all pairs to Tinker data
    logger.info("Converting pairs to Tinker format...")
    valid_pairs = []
    chosen_data_all = []
    rejected_data_all = []

    for pair in pairs:
        chosen_datum, rejected_datum = pair_to_data(
            pair, renderer, tokenizer, max_length=max_length,
        )
        if chosen_datum is not None and rejected_datum is not None:
            valid_pairs.append(pair)
            chosen_data_all.append(chosen_datum)
            rejected_data_all.append(rejected_datum)

    logger.info(f"Converted {len(valid_pairs)}/{len(pairs)} pairs successfully")

    if not valid_pairs:
        logger.error("No valid training data after conversion!")
        return

    # Training loop
    adam_params = tinker.AdamParams(learning_rate=effective_lr)
    total_steps = 0
    training_log = []

    n_pairs = len(valid_pairs)
    steps_per_epoch = (n_pairs + batch_size - 1) // batch_size
    total_expected_steps = steps_per_epoch * epochs

    logger.info(f"\n{'='*60}")
    logger.info(f"DPO Training Config")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Mode: True DPO (forward_backward_custom)")
    logger.info(f"  LoRA rank: {lora_rank}")
    logger.info(f"  Learning rate: {effective_lr:.2e}")
    logger.info(f"  DPO beta: {beta}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Pairs: {n_pairs}")
    logger.info(f"  Expected steps: {total_expected_steps}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"{'='*60}\n")

    t0 = time.time()

    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        # Shuffle indices each epoch
        np.random.seed(42 + epoch)
        indices = np.random.permutation(n_pairs)

        epoch_metrics = []

        for batch_start in range(0, n_pairs, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_chosen = [chosen_data_all[i] for i in batch_indices]
            batch_rejected = [rejected_data_all[i] for i in batch_indices]
            n_batch = len(batch_chosen)

            # 1. Compute reference logprobs for this batch
            ref_t0 = time.time()
            all_batch_data = []
            for c, r in zip(batch_chosen, batch_rejected):
                all_batch_data.append(c)
                all_batch_data.append(r)

            ref_logprobs_all = compute_reference_logprobs(reference_client, all_batch_data)
            ref_logprobs_chosen = [ref_logprobs_all[i] for i in range(0, len(all_batch_data), 2)]
            ref_logprobs_rejected = [ref_logprobs_all[i] for i in range(1, len(all_batch_data), 2)]
            ref_time = time.time() - ref_t0

            # 2. Build interleaved data for forward_backward_custom
            # Pattern: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
            interleaved_data = []
            for c, r in zip(batch_chosen, batch_rejected):
                interleaved_data.append(c)
                interleaved_data.append(r)

            # 3. Create DPO loss function with captured reference logprobs
            loss_fn = make_dpo_loss_fn(
                ref_logprobs_chosen=ref_logprobs_chosen,
                ref_logprobs_rejected=ref_logprobs_rejected,
                chosen_data=batch_chosen,
                rejected_data=batch_rejected,
                beta=beta,
            )

            # 4. Forward-backward with custom DPO loss + optimizer step
            fwd_bwd_future = training_client.forward_backward_custom(
                interleaved_data, loss_fn
            )
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            # 5. Extract metrics
            dpo_metrics = fwd_bwd_result.metrics if hasattr(fwd_bwd_result, 'metrics') else {}

            step_info = {
                "epoch": epoch + 1,
                "step": total_steps,
                "n_pairs": n_batch,
                "ref_time": ref_time,
                "time": time.time() - t0,
            }
            step_info.update(dpo_metrics)
            training_log.append(step_info)
            epoch_metrics.append(step_info)
            total_steps += 1

            # Log periodically
            if total_steps % 5 == 0 or total_steps == 1:
                dpo_loss = dpo_metrics.get("dpo_loss", "N/A")
                accuracy = dpo_metrics.get("accuracy", "N/A")
                margin = dpo_metrics.get("margin", "N/A")
                logger.info(
                    f"  Step {total_steps} | "
                    f"Loss: {dpo_loss} | "
                    f"Acc: {accuracy} | "
                    f"Margin: {margin} | "
                    f"Ref time: {ref_time:.1f}s | "
                    f"Elapsed: {time.time() - t0:.1f}s"
                )

            # Save checkpoint
            if total_steps % save_every == 0:
                ckpt_name = f"checkpoint-{total_steps:04d}"
                logger.info(f"  Saving checkpoint: {ckpt_name}")
                training_client.save_weights_and_get_sampling_client(name=ckpt_name)
                training_client.save_state(name=f"state-{total_steps:04d}")

                with open(log_dir / f"{ckpt_name}_info.json", "w") as f:
                    json.dump(step_info, f, indent=2)

        # End of epoch stats
        if epoch_metrics:
            avg_loss = np.mean([
                m.get("dpo_loss", 0) for m in epoch_metrics
                if isinstance(m.get("dpo_loss"), (int, float))
            ]) if epoch_metrics else 0
            avg_acc = np.mean([
                m.get("accuracy", 0) for m in epoch_metrics
                if isinstance(m.get("accuracy"), (int, float))
            ]) if epoch_metrics else 0
            logger.info(
                f"  Epoch {epoch + 1} complete | "
                f"Avg loss: {avg_loss:.4f} | "
                f"Avg accuracy: {avg_acc:.1%} | "
                f"Steps: {total_steps}"
            )

    # Save final checkpoint
    logger.info("\nSaving final checkpoint...")
    training_client.save_weights_and_get_sampling_client(name="final")
    training_client.save_state(name="final-state")

    total_time = time.time() - t0

    # Save training log
    log_info = {
        "mode": "dpo",
        "model": model_name,
        "model_path": model_path,
        "data_path": data_path,
        "epochs": epochs,
        "lr": effective_lr,
        "beta": beta,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "n_pairs": n_pairs,
        "total_steps": total_steps,
        "total_time": total_time,
        "training_log": training_log,
    }

    with open(log_dir / "training_log.json", "w") as f:
        json.dump(log_info, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"DPO TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Mode: True DPO")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info(f"{'='*60}")

    return training_client


# ---------------------------------------------------------------------------
# Mode 2: Rejection sampling SFT (train on chosen only, optionally weighted)
# ---------------------------------------------------------------------------

def train_rejection_sft(
    model_name: str,
    data_path: str,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_rank: int = 32,
    batch_size: int = 4,
    max_length: int | None = None,
    save_every: int = 50,
    experiment_name: str = "rejection_sft",
    resume_from: str | None = None,
    model_path: str | None = None,
    weight_by_score: bool = True,
):
    """Train on chosen (correct) trajectories only, using standard cross_entropy.

    This is equivalent to "best-of-N rejection sampling" — a strong and cheap
    approximation to DPO. Optionally weight each sample by its score so
    higher-quality trajectories contribute more to the gradient.
    """

    # Setup
    log_dir = Path("data/dpo") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data (only use chosen side)
    pairs = load_dpo_pairs(data_path)
    if not pairs:
        logger.error("No valid DPO pairs loaded!")
        return

    # Setup tokenizer and renderer
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    # Create Tinker training client
    service_client = tinker.ServiceClient()

    if resume_from:
        logger.info(f"Resuming from state: {resume_from}")
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

        if model_path:
            logger.info(f"Loading weights from SFT checkpoint: {model_path}")
            training_client.load_state(model_path).result()

    # LoRA LR scaling
    try:
        lr_factor = get_lora_lr_over_full_finetune_lr(model_name)
        effective_lr = lr * lr_factor
        logger.info(f"LoRA LR scaling: {lr} x {lr_factor:.1f} = {effective_lr:.2e}")
    except Exception:
        effective_lr = lr
        logger.info(f"Using base LR: {effective_lr:.2e} (no LoRA scaling available)")

    # Convert chosen trajectories to training data
    logger.info("Converting chosen trajectories to Tinker format...")
    all_data = []
    for pair in pairs:
        chosen = pair["chosen"]
        score = chosen.get("score", 1.0)

        # Quality weighting: scale loss weights by score
        weight_scale = score if weight_by_score else 1.0

        datum = messages_to_datum(
            chosen["messages"], renderer, tokenizer,
            weight_scale=weight_scale,
            max_length=max_length,
        )
        if datum is not None:
            all_data.append(datum)

    logger.info(f"Converted {len(all_data)}/{len(pairs)} chosen trajectories")

    if not all_data:
        logger.error("No valid training data!")
        return

    # Training loop (same pattern as sft_tinker.py)
    adam_params = tinker.AdamParams(learning_rate=effective_lr)
    total_steps = 0
    training_log = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Rejection Sampling SFT Config")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Mode: Rejection sampling SFT (chosen only)")
    logger.info(f"  Weight by score: {weight_by_score}")
    logger.info(f"  LoRA rank: {lora_rank}")
    logger.info(f"  Learning rate: {effective_lr:.2e}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Samples: {len(all_data)}")
    logger.info(f"  Expected steps: ~{len(all_data) * epochs // batch_size}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"{'='*60}\n")

    t0 = time.time()

    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        np.random.seed(42 + epoch)
        indices = np.random.permutation(len(all_data))

        epoch_losses = []

        for batch_start in range(0, len(all_data), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [all_data[i] for i in batch_indices]

            if not batch:
                continue

            # Pipeline: forward_backward + optim_step on same clock cycle
            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            loss = fwd_bwd_result.loss if hasattr(fwd_bwd_result, 'loss') else None

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
                training_client.save_weights_and_get_sampling_client(name=ckpt_name)

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
    training_client.save_state(name="final-state")

    total_time = time.time() - t0

    # Save training log
    log_info = {
        "mode": "rejection_sft",
        "model": model_name,
        "model_path": model_path,
        "data_path": data_path,
        "epochs": epochs,
        "lr": effective_lr,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "weight_by_score": weight_by_score,
        "n_samples": len(all_data),
        "total_steps": total_steps,
        "total_time": total_time,
        "training_log": training_log,
    }

    with open(log_dir / "training_log.json", "w") as f:
        json.dump(log_info, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"REJECTION SAMPLING SFT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"  Logs saved to: {log_dir}")
    logger.info(f"{'='*60}")

    return final_sampling_client


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO training on Tinker")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--data", required=True,
                        help="Path to DPO pairs JSONL file (from create_dpo_pairs.py)")
    parser.add_argument("--mode", default="dpo", choices=["dpo", "rejection_sft"],
                        help="Training mode: 'dpo' (true DPO with reference model) "
                             "or 'rejection_sft' (train on chosen only)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Base learning rate (LoRA scaling applied automatically). "
                             "DPO typically uses 1e-5 to 1e-6; rejection_sft uses 2e-4.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Number of pairs per batch (DPO mode) or samples (rejection_sft)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (KL constraint strength). Only for dpo mode.")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Max sequence length (truncate longer sequences)")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--experiment-name", default="dpo_tinker")
    parser.add_argument("--resume-from", default=None,
                        help="Tinker state path to resume training from")
    parser.add_argument("--model-path", default=None,
                        help="Tinker weights path to initialize from (e.g. SFT checkpoint)")
    parser.add_argument("--weight-by-score", action="store_true", default=True,
                        help="Weight samples by score in rejection_sft mode")
    parser.add_argument("--no-weight-by-score", dest="weight_by_score", action="store_false",
                        help="Equal weight for all samples in rejection_sft mode")
    args = parser.parse_args()

    if args.mode == "dpo":
        train_dpo(
            model_name=args.model,
            data_path=args.data,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size,
            beta=args.beta,
            max_length=args.max_length,
            save_every=args.save_every,
            experiment_name=args.experiment_name,
            resume_from=args.resume_from,
            model_path=args.model_path,
        )
    elif args.mode == "rejection_sft":
        train_rejection_sft(
            model_name=args.model,
            data_path=args.data,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size,
            max_length=args.max_length,
            save_every=args.save_every,
            experiment_name=args.experiment_name,
            resume_from=args.resume_from,
            model_path=args.model_path,
            weight_by_score=args.weight_by_score,
        )


if __name__ == "__main__":
    main()
