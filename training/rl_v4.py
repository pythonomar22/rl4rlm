#!/usr/bin/env python3
"""
GRPO-v4: Proper Group Relative Policy Optimization for RLM.

Fixes over previous GRPO (rl.py):
1. CONDITIONED LOG-PROBS: Policy loss computed on full conversation context,
   not raw code text. The old code trained on code_text alone, which meant
   the model learned to produce/avoid code patterns unconditionally.
2. TOKEN-LEVEL KL: Proper KL(π_θ || π_ref) using frozen reference model,
   not weight-space L2 (which was ~0 due to normalization by 17.4M params).
3. K=8: Better advantage estimation (K=4 was too small for stable advantages).

Usage:
    CUDA_VISIBLE_DEVICES=7 uv run python training/rl_v4.py \
        --model data/sft/lora_v3/final \
        --base-model Qwen/Qwen3-1.7B \
        --output data/sft/rl_v4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("6", "7", "6,7"), \
    "CUDA_VISIBLE_DEVICES must be set to 6, 7, or 6,7"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from training.rewards import composite_reward
from training.logprobs import compute_turn_kl_logits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl_v4")


class RLModelV4:
    """Model wrapper for GRPO-v4 with separate policy and reference models."""

    def __init__(self, base_model_name, adapter_path=None, device="cuda:0",
                 max_new_tokens=1024):
        logger.info(f"Loading policy model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model (trainable)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        if adapter_path:
            logger.info(f"Loading policy LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=True,
            )

        # Reference model (frozen) — separate copy
        logger.info(f"Loading reference model (frozen)")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        if adapter_path:
            logger.info(f"Loading reference LoRA adapter: {adapter_path}")
            self.ref_model = PeftModel.from_pretrained(
                self.ref_model, adapter_path, is_trainable=False,
            )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.device = device
        self.max_new_tokens = max_new_tokens

        # Memory report
        policy_mem = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9
        ref_mem = sum(p.numel() * p.element_size() for p in self.ref_model.parameters()) / 1e9
        logger.info(f"Policy model: {policy_mem:.2f} GB")
        logger.info(f"Reference model: {ref_mem:.2f} GB")
        logger.info(f"Total model memory: {policy_mem + ref_mem:.2f} GB")

    def generate(self, messages, temperature=0.8):
        """Generate for RLM root turns."""
        from scaffold.llm_query import strip_think_tags

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return strip_think_tags(response)

    def sub_query(self, prompt_str):
        from scaffold.llm_query import strip_think_tags

        messages = [{"role": "user", "content": prompt_str}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return strip_think_tags(response)


def multi_niah_reward(predicted, expected_answers, trajectory_dict):
    if predicted is None:
        return 0.0
    scores = score_multi_niah(predicted, expected_answers)
    recall = scores["recall"]
    fmt_bonus = 0.1 if trajectory_dict.get("terminated") else 0.0
    return 0.9 * recall + 0.1 * fmt_bonus


def grpo_step_v4(
    rl_model: RLModelV4,
    optimizer: torch.optim.Optimizer,
    task_prompts: list[str],
    expected_answers: list,
    system_prompt: str,
    reward_fn: str = "composite",
    reward_fn_per_task: list[str] | None = None,
    k: int = 8,
    kl_coeff: float = 0.05,
) -> dict:
    """
    GRPO-v4 step with conditioned log-probs and token-level KL.

    Key differences from grpo_step (rl.py):
    1. Log-probs computed conditioned on full conversation context
    2. Token-level KL via reference model
    3. No more raw code_text training
    """
    all_rewards = []
    all_trajectories = []
    stats = {"correct": 0, "total": 0}

    # 1. Generate K trajectories per prompt
    rl_model.model.eval()
    for idx, (prompt, expected) in enumerate(zip(task_prompts, expected_answers)):
        group_rewards = []
        group_trajs = []

        task_reward_fn = reward_fn
        if reward_fn_per_task is not None and idx < len(reward_fn_per_task):
            task_reward_fn = reward_fn_per_task[idx]

        for ki in range(k):
            traj = rlm(
                prompt=prompt,
                model=rl_model,
                system_prompt=system_prompt,
                max_iterations=6,
                verbose=False,
            )

            traj_dict = trajectory_to_dict(traj)
            if task_reward_fn == "multi_niah":
                reward = multi_niah_reward(traj.answer, expected, traj_dict)
            else:
                reward = composite_reward(traj.answer, expected, traj_dict)

            group_rewards.append(reward)
            group_trajs.append(traj)

            stats["total"] += 1
            if reward > 0.5:
                stats["correct"] += 1

        all_rewards.append(group_rewards)
        all_trajectories.append(group_trajs)

    # 2. Compute group-relative advantages
    flat_rewards = [r for group in all_rewards for r in group]
    stats["avg_reward"] = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0

    advantages = []
    for group_rewards in all_rewards:
        mean_r = sum(group_rewards) / len(group_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_r = max(std_r, 1e-8)
        group_advantages = [(r - mean_r) / std_r for r in group_rewards]
        advantages.append(group_advantages)

    # 3. Policy gradient update with conditioned log-probs + token KL
    rl_model.model.train()
    total_policy_loss = 0.0
    total_kl = 0.0
    n_updates = 0

    optimizer.zero_grad()

    for group_idx, (group_trajs, group_advs) in enumerate(zip(all_trajectories, advantages)):
        for traj, adv in zip(group_trajs, group_advs):
            if abs(adv) < 0.1:
                continue

            if not traj.messages:
                continue

            # Process each assistant turn with proper conditioning
            turn_policy_loss = torch.tensor(0.0, device=rl_model.device)
            turn_kl = torch.tensor(0.0, device=rl_model.device)
            n_turns = 0

            for i, msg in enumerate(traj.messages):
                if msg["role"] != "assistant":
                    continue

                messages_before = traj.messages[:i]

                # Compute KL and policy/ref log-probs together
                kl_mean, policy_lp, ref_lp = compute_turn_kl_logits(
                    rl_model.model, rl_model.ref_model,
                    rl_model.tokenizer,
                    messages_before, msg,
                    device=rl_model.device,
                )

                turn_policy_loss = turn_policy_loss + policy_lp
                turn_kl = turn_kl + kl_mean
                n_turns += 1

            if n_turns == 0:
                continue

            # Average over turns
            mean_policy_lp = turn_policy_loss / n_turns
            mean_kl = turn_kl / n_turns

            # GRPO loss: -advantage * mean(log_prob) + kl_coeff * KL
            # Note: mean_policy_lp is sum of token log-probs (negative for typical tokens)
            # We want to maximize policy_lp for positive advantage → minimize -adv * policy_lp
            loss = -adv * mean_policy_lp + kl_coeff * mean_kl

            # Normalize by total number of update trajectories (approximate)
            loss = loss / max(1, k)  # Rough normalization
            loss.backward()

            total_policy_loss += (-adv * mean_policy_lp).item()
            total_kl += mean_kl.item()
            n_updates += 1

    if n_updates > 0:
        torch.nn.utils.clip_grad_norm_(rl_model.model.parameters(), 1.0)
        optimizer.step()

    optimizer.zero_grad()

    avg_turns = sum(
        len(t.turns)
        for group in all_trajectories for t in group
    ) / max(1, sum(len(g) for g in all_trajectories))

    stats["avg_turns"] = avg_turns
    stats["policy_loss"] = total_policy_loss / max(1, n_updates)
    stats["kl"] = total_kl / max(1, n_updates)
    stats["n_updates"] = n_updates

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model", default=None,
                        help="Path to LoRA adapter (SFT checkpoint)")
    parser.add_argument("--output", default="data/sft/rl_v4")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch-prompts", type=int, default=2,
                        help="Prompts per GRPO step")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-coeff", type=float, default=0.05,
                        help="Token-level KL coefficient")
    parser.add_argument("--task-type", default="mixed",
                        choices=["niah", "multi_niah", "mixed"])
    parser.add_argument("--doc-lengths", nargs="+", type=int,
                        default=[5000, 10000, 20000, 50000])
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=10,
                        help="Save sample trajectories every N steps")
    args = parser.parse_args()

    assert torch.cuda.device_count() <= 2
    device = "cuda:0"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
    config["version"] = "grpo_v4"
    config["fixes"] = [
        "conditioned_log_probs",
        "token_level_kl",
        "k=8",
    ]
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load model (policy + reference)
    rl_model = RLModelV4(
        base_model_name=args.base_model,
        adapter_path=args.model,
        device=device,
    )

    trainable_params = [p for p in rl_model.model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Generate task pool
    if args.task_type == "mixed":
        niah_tasks = generate_niah_suite(n_tasks=args.n_tasks, doc_lengths=args.doc_lengths)
        mniah_tasks = generate_multi_niah_suite(n_tasks=max(12, args.n_tasks // 2))
        logger.info(f"Task pool: {len(niah_tasks)} NIAH + {len(mniah_tasks)} multi-NIAH")
    elif args.task_type == "multi_niah":
        mniah_tasks = generate_multi_niah_suite(n_tasks=args.n_tasks)
        niah_tasks = []
        logger.info(f"Task pool: {len(mniah_tasks)} multi-NIAH only")
    else:
        niah_tasks = generate_niah_suite(n_tasks=args.n_tasks, doc_lengths=args.doc_lengths)
        mniah_tasks = []
        logger.info(f"Task pool: {len(niah_tasks)} NIAH only")

    # Training loop
    logger.info(f"\nStarting GRPO-v4 training:")
    logger.info(f"  K={args.k} trajectories per prompt")
    logger.info(f"  {args.batch_prompts} prompts per step")
    logger.info(f"  {args.steps} steps")
    logger.info(f"  KL coefficient: {args.kl_coeff}")
    logger.info(f"  Learning rate: {args.lr}")

    training_log = []
    t0 = time.time()

    for step in range(args.steps):
        step_t0 = time.time()

        # Sample batch of tasks
        if args.task_type == "mixed":
            n_niah = max(1, args.batch_prompts // 2)
            n_mniah = args.batch_prompts - n_niah
            batch_niah = random.sample(niah_tasks, min(n_niah, len(niah_tasks)))
            batch_mniah = random.sample(mniah_tasks, min(n_mniah, len(mniah_tasks)))

            batch_prompts = [t.prompt for t in batch_niah] + [t.prompt for t in batch_mniah]
            batch_expected = [t.expected_answer for t in batch_niah] + \
                             [t.expected_answers for t in batch_mniah]
            reward_fn_per_task = ["composite"] * len(batch_niah) + \
                                 ["multi_niah"] * len(batch_mniah)
            reward_fn = "composite"
        elif args.task_type == "multi_niah":
            batch_tasks = random.sample(mniah_tasks, min(args.batch_prompts, len(mniah_tasks)))
            batch_prompts = [t.prompt for t in batch_tasks]
            batch_expected = [t.expected_answers for t in batch_tasks]
            reward_fn = "multi_niah"
            reward_fn_per_task = None
        else:
            batch_tasks = random.sample(niah_tasks, min(args.batch_prompts, len(niah_tasks)))
            batch_prompts = [t.prompt for t in batch_tasks]
            batch_expected = [t.expected_answer for t in batch_tasks]
            reward_fn = "composite"
            reward_fn_per_task = None

        # GRPO step
        stats = grpo_step_v4(
            rl_model=rl_model,
            optimizer=optimizer,
            task_prompts=batch_prompts,
            expected_answers=batch_expected,
            system_prompt=QWEN_2B_SYSTEM_PROMPT,
            reward_fn=reward_fn,
            reward_fn_per_task=reward_fn_per_task,
            k=args.k,
            kl_coeff=args.kl_coeff,
        )

        step_time = time.time() - step_t0
        stats["step_time"] = step_time
        training_log.append({"step": step, **stats})

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Step {step + 1}/{args.steps} | "
                f"Reward: {stats['avg_reward']:.3f} | "
                f"Correct: {stats['correct']}/{stats['total']} | "
                f"Loss: {stats['policy_loss']:.3f} | "
                f"KL: {stats['kl']:.4f} | "
                f"Updates: {stats['n_updates']} | "
                f"Time: {step_time:.0f}s (total {elapsed:.0f}s)"
            )

        # Save sample trajectories
        if (step + 1) % args.sample_interval == 0:
            sample_dir = output_dir / "sample_trajectories" / f"step_{step + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            sample_tasks = []
            if niah_tasks:
                sample_tasks.extend(niah_tasks[:2])
            if mniah_tasks:
                sample_tasks.extend(mniah_tasks[:1])

            for i, task in enumerate(sample_tasks[:3]):
                traj = rlm(
                    prompt=task.prompt,
                    model=rl_model,
                    system_prompt=QWEN_2B_SYSTEM_PROMPT,
                    max_iterations=6,
                    verbose=False,
                )
                with open(sample_dir / f"traj_{i}.json", "w") as f:
                    json.dump(trajectory_to_dict(traj), f, indent=2, default=str)

                if hasattr(task, "expected_answers"):
                    scores = score_multi_niah(traj.answer, task.expected_answers)
                    logger.info(
                        f"  Sample {i} (multi-NIAH): recall={scores['recall']:.2f} "
                        f"({scores['found']}/{scores['total']}) "
                        f"answer={str(traj.answer)[:80]}"
                    )
                else:
                    logger.info(
                        f"  Sample {i} (NIAH): correct={score_niah(traj.answer, task.expected_answer)} "
                        f"answer={str(traj.answer)[:80]}"
                    )

        if (step + 1) % args.save_interval == 0:
            ckpt_dir = output_dir / f"checkpoint-{step + 1}"
            rl_model.model.save_pretrained(ckpt_dir)
            logger.info(f"  Saved checkpoint: {ckpt_dir}")

    # Save final
    rl_model.model.save_pretrained(output_dir / "final")
    rl_model.tokenizer.save_pretrained(output_dir / "final")

    elapsed = time.time() - t0
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"GRPO-V4 TRAINING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Final avg reward: {training_log[-1]['avg_reward']:.3f}")
    logger.info(f"Final KL: {training_log[-1]['kl']:.4f}")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 3600:.2f} GPU-hours)")
    logger.info(f"Saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
