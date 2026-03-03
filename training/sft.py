#!/usr/bin/env python3
"""
SFT warm-start: LoRA fine-tune Qwen3-1.7B on filtered trajectories.

This creates the starting checkpoint for RL training.
Per-turn SFT: each sample is (conversation_so_far, model_code_response).

Usage:
    CUDA_VISIBLE_DEVICES=7 uv run python training/sft.py \
        --data data/filtered/sft_samples_model_timestamp.jsonl \
        --output data/sft/lora_v1 \
        --epochs 3 \
        --batch-size 4 \
        --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("6", "7", "6,7"), \
    "CUDA_VISIBLE_DEVICES must be set to 6, 7, or 6,7"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("sft")


class SFTDataset(Dataset):
    """Dataset of (messages, completion) pairs for SFT."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} SFT samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        completion = sample["completion"]

        # Build the full conversation with completion
        full_messages = messages + [{"role": "assistant", "content": completion}]

        # Tokenize with chat template (disable thinking to avoid extra tokens)
        try:
            text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # Tokenize
        full_tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_tensors="pt", padding=False,
        )
        prompt_tokens = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length,
            return_tensors="pt", padding=False,
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        # Labels: -100 for prompt tokens (don't compute loss), actual tokens for completion
        labels = input_ids.clone()
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        labels[i, :seq_len] = b["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", required=True, help="Path to SFT samples JSONL")
    parser.add_argument("--output", required=True, help="Output dir for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    assert torch.cuda.device_count() <= 2
    device = "cuda:0"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = SFTDataset(args.data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model with LoRA
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
    )

    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    logger.info(f"Starting SFT training:")
    logger.info(f"  Samples: {len(dataset)}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  LoRA rank: {args.lora_rank}")

    model.train()
    global_step = 0
    total_loss = 0
    log_losses = []
    t0 = time.time()

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            total_loss += loss.item()
            log_losses.append(loss.item() * args.gradient_accumulation)

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = sum(log_losses[-args.log_interval * args.gradient_accumulation:]) / \
                               len(log_losses[-args.log_interval * args.gradient_accumulation:])
                    elapsed = time.time() - t0
                    logger.info(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )

                if global_step % args.save_interval == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    logger.info(f"  Saved checkpoint: {checkpoint_dir}")

    # Save final
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SFT TRAINING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Final loss: {log_losses[-1]:.4f}")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 3600:.2f} GPU-hours)")
    logger.info(f"Saved to: {output_dir / 'final'}")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "losses": log_losses,
            "total_steps": global_step,
            "training_time": elapsed,
            "gpu_hours": elapsed / 3600,
        }, f, indent=2)


if __name__ == "__main__":
    main()
