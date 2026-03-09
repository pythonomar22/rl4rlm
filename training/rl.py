#!/usr/bin/env python3
"""
GRPO training for RLM.

Group Relative Policy Optimization (DeepSeek-R1):
1. For each prompt, sample K trajectories
2. Score each with reward function
3. Compute advantage: (reward - mean) / std within the group
4. Update policy with clipped objective + KL penalty vs reference

This is trajectory-level GRPO: the entire RLM execution is one "response".

Usage:
    CUDA_VISIBLE_DEVICES=7 uv run python training/rl.py \
        --model data/sft/lora_v1/final \
        --base-model Qwen/Qwen3-1.7B \
        --k 4 \
        --n-tasks 20 \
        --steps 100
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from scaffold.rlm import rlm, trajectory_to_dict
from scaffold.prompts.qwen2b import QWEN_2B_SYSTEM_PROMPT
from eval.benchmarks.niah import generate_niah_suite, score_niah
from eval.benchmarks.multi_niah import generate_multi_niah_suite, score_multi_niah
from training.rewards import binary_reward, composite_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl")


class RLModel:
    """
    Wrapper for RL training that handles both generation (RLM loop)
    and training (gradient updates).

    The same model is used for:
    1. Root LLM generation (the RLM's main agent)
    2. Sub-call generation (llm_query inside REPL)
    3. Policy gradient updates
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str | None = None,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
    ):
        logger.info(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        if adapter_path:
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, is_trainable=True,
            )

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.generation_log: list[dict] = []

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
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
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = strip_think_tags(response)

        self.generation_log.append({
            "type": "root",
            "input_tokens": input_len,
            "output_tokens": len(new_tokens),
        })

        return response

    def sub_query(self, prompt_str: str) -> str:
        """Sub-call for llm_query() inside REPL."""
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

        self.generation_log.append({
            "type": "sub_query",
            "input_tokens": input_len,
            "output_tokens": len(new_tokens),
        })

        return strip_think_tags(response)

    def total_stats(self) -> dict:
        root = [g for g in self.generation_log if g["type"] == "root"]
        sub = [g for g in self.generation_log if g["type"] == "sub_query"]
        return {
            "root_calls": len(root),
            "sub_calls": len(sub),
            "total_input_tokens": sum(g["input_tokens"] for g in self.generation_log),
            "total_output_tokens": sum(g["output_tokens"] for g in self.generation_log),
        }


def multi_niah_reward(predicted: str | None, expected_answers: list[str],
                      trajectory: dict) -> float:
    """Reward for multi-needle NIAH: recall + format bonus."""
    if predicted is None:
        return 0.0
    scores = score_multi_niah(predicted, expected_answers)
    recall = scores["recall"]
    # Small format bonus for proper termination
    fmt_bonus = 0.1 if trajectory.get("terminated") else 0.0
    return 0.9 * recall + 0.1 * fmt_bonus


def compute_anchor_regularization(model: RLModel, ref_weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute anchor regularization: L2 distance between current and reference LoRA weights.

    This is a first-order approximation to KL(pi_theta || pi_ref) for LoRA,
    analogous to the elastic weight consolidation term. It prevents the policy
    from drifting too far from the SFT reference, addressing the training
    instability observed in RL-v2.
    """
    l2_sum = torch.tensor(0.0, device=next(iter(ref_weights.values())).device)
    n_params = 0
    for name, p in model.model.named_parameters():
        if p.requires_grad and name in ref_weights:
            l2_sum = l2_sum + (p - ref_weights[name]).pow(2).sum()
            n_params += p.numel()
    return l2_sum / max(n_params, 1)


def grpo_step(
    model: RLModel,
    optimizer: torch.optim.Optimizer,
    task_prompts: list[str],
    expected_answers: list,  # list[str] for NIAH, list[list[str]] for multi-NIAH
    system_prompt: str,
    reward_fn: str = "composite",  # "composite" or "multi_niah"
    reward_fn_per_task: list[str] | None = None,  # per-task reward fn for mixed mode
    k: int = 4,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.0,
    ref_weights: dict[str, torch.Tensor] | None = None,
) -> dict:
    """
    One GRPO update step.

    For each prompt:
    1. Generate K trajectories
    2. Score with reward
    3. Compute group-relative advantages
    4. Update policy with optional KL regularization

    When kl_coeff > 0, adds anchor regularization to prevent policy drift
    from the SFT reference (addresses RL-v2 instability).
    """
    all_rewards = []
    all_trajectories = []
    step_stats = {"correct": 0, "total": 0, "avg_reward": 0, "avg_turns": 0}

    # 1. Generate K trajectories per prompt
    for idx, (prompt, expected) in enumerate(zip(task_prompts, expected_answers)):
        group_rewards = []
        group_trajs = []

        # Determine reward function for this task
        task_reward_fn = reward_fn
        if reward_fn_per_task is not None and idx < len(reward_fn_per_task):
            task_reward_fn = reward_fn_per_task[idx]

        for _ in range(k):
            traj = rlm(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_iterations=6,
                verbose=False,
            )

            if task_reward_fn == "multi_niah":
                reward = multi_niah_reward(
                    traj.answer, expected, trajectory_to_dict(traj),
                )
            else:
                reward = composite_reward(
                    traj.answer, expected, trajectory_to_dict(traj),
                )
            group_rewards.append(reward)
            group_trajs.append(traj)

            step_stats["total"] += 1
            if reward > 0.5:
                step_stats["correct"] += 1

        all_rewards.append(group_rewards)
        all_trajectories.append(group_trajs)

    # 2. Compute group-relative advantages
    flat_rewards = [r for group in all_rewards for r in group]
    step_stats["avg_reward"] = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0

    # For each group, normalize rewards
    advantages = []
    for group_rewards in all_rewards:
        mean_r = sum(group_rewards) / len(group_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_r = max(std_r, 1e-8)
        group_advantages = [(r - mean_r) / std_r for r in group_rewards]
        advantages.append(group_advantages)

    # 3. Policy gradient update with optional KL regularization
    model.model.train()
    total_policy_loss = 0.0
    total_kl_loss = 0.0
    n_updates = 0

    for group_idx, (group_trajs, group_advs) in enumerate(zip(all_trajectories, advantages)):
        for traj, adv in zip(group_trajs, group_advs):
            if abs(adv) < 0.1:
                continue  # Skip near-zero advantage trajectories

            # Build training sequence from trajectory turns
            for turn in traj.turns:
                code = turn.get("parsed_code")
                if not code:
                    continue

                # Tokenize the code as a training target
                code_text = f"```repl\n{code}\n```"
                tokens = model.tokenizer(
                    code_text, return_tensors="pt", truncation=True,
                    max_length=512,
                ).to(model.device)

                # Forward pass to get log probs
                outputs = model.model(**tokens, labels=tokens["input_ids"])
                loss = outputs.loss

                # Scale by advantage (positive = reinforce, negative = discourage)
                policy_loss = -adv * loss

                # Add KL regularization if enabled
                if kl_coeff > 0 and ref_weights is not None:
                    kl_loss = compute_anchor_regularization(model, ref_weights)
                    total_step_loss = policy_loss + kl_coeff * kl_loss
                    total_kl_loss += kl_loss.item()
                else:
                    total_step_loss = policy_loss

                total_step_loss.backward()
                total_policy_loss += policy_loss.item()
                n_updates += 1

    if n_updates > 0:
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_turns = sum(
        len(t.turns)
        for group in all_trajectories for t in group
    ) / max(1, sum(len(g) for g in all_trajectories))

    step_stats["avg_turns"] = avg_turns
    step_stats["policy_loss"] = total_policy_loss / max(1, n_updates)
    step_stats["kl_loss"] = total_kl_loss / max(1, n_updates)
    step_stats["n_updates"] = n_updates

    return step_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model", default=None,
                        help="Path to LoRA adapter (SFT checkpoint)")
    parser.add_argument("--k", type=int, default=4,
                        help="Trajectories per prompt for GRPO")
    parser.add_argument("--n-tasks", type=int, default=20,
                        help="Tasks per training step")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-prompts", type=int, default=4,
                        help="Prompts per GRPO step")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output", default="data/sft/rl_v1")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--sample-trajectories-interval", type=int, default=10,
                        help="How often to log sample trajectories for inspection")
    parser.add_argument("--doc-lengths", nargs="+", type=int,
                        default=[5000, 10000, 20000],
                        help="Document lengths for NIAH training tasks")
    parser.add_argument("--task-type", default="niah",
                        choices=["niah", "multi_niah", "mixed"],
                        help="Task type for training (mixed = NIAH + multi-NIAH)")
    parser.add_argument("--kl-coeff", type=float, default=0.0,
                        help="KL regularization coefficient (0 = disabled). "
                             "Penalizes weight drift from SFT reference to prevent "
                             "policy oscillation observed in RL-v2.")
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

    # Load model
    rl_model = RLModel(
        base_model_name=args.base_model,
        adapter_path=args.model,
        device=device,
    )

    # Optimizer (only LoRA parameters if adapter loaded)
    trainable_params = [p for p in rl_model.model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Generate task pool
    import random as _random
    if args.task_type == "mixed":
        niah_tasks = generate_niah_suite(n_tasks=args.n_tasks, doc_lengths=args.doc_lengths)
        mniah_tasks = generate_multi_niah_suite(n_tasks=max(12, args.n_tasks // 2))
        reward_fn = "mixed"
        logger.info(f"  Task type: mixed (NIAH + multi-NIAH)")
        logger.info(f"  NIAH tasks: {len(niah_tasks)}, multi-NIAH tasks: {len(mniah_tasks)}")
    elif args.task_type == "multi_niah":
        tasks = generate_multi_niah_suite(n_tasks=args.n_tasks)
        reward_fn = "multi_niah"
        logger.info(f"  Task type: multi_niah (O(K) needle search)")
    else:
        tasks = generate_niah_suite(n_tasks=args.n_tasks, doc_lengths=args.doc_lengths)
        reward_fn = "composite"
        logger.info(f"  Task type: niah")
        logger.info(f"  Doc lengths: {args.doc_lengths}")

    # Save reference weights for KL regularization
    ref_weights = None
    if args.kl_coeff > 0:
        ref_weights = {
            name: p.data.clone()
            for name, p in rl_model.model.named_parameters()
            if p.requires_grad
        }
        logger.info(f"  KL regularization: coeff={args.kl_coeff}, "
                     f"saved {len(ref_weights)} reference weight tensors")

    # Training loop
    logger.info(f"\nStarting GRPO training:")
    logger.info(f"  K={args.k} trajectories per prompt")
    logger.info(f"  {args.batch_prompts} prompts per step")
    logger.info(f"  {args.steps} steps")
    if args.task_type != "mixed":
        logger.info(f"  {args.n_tasks} total tasks in pool")
    logger.info(f"  Reward function: {reward_fn}")
    logger.info(f"  KL coeff: {args.kl_coeff}")

    training_log = []
    t0 = time.time()

    for step in range(args.steps):
        # Sample batch of tasks
        if args.task_type == "mixed":
            # Mix NIAH and multi-NIAH tasks
            n_niah = max(1, args.batch_prompts // 2)
            n_mniah = args.batch_prompts - n_niah
            batch_niah = _random.sample(niah_tasks, min(n_niah, len(niah_tasks)))
            batch_mniah = _random.sample(mniah_tasks, min(n_mniah, len(mniah_tasks)))

            batch_prompts = [t.prompt for t in batch_niah] + [t.prompt for t in batch_mniah]
            batch_expected = [t.expected_answer for t in batch_niah] + \
                             [t.expected_answers for t in batch_mniah]
            reward_fn_per_task = ["composite"] * len(batch_niah) + \
                                 ["multi_niah"] * len(batch_mniah)
        else:
            batch_tasks = _random.sample(tasks, min(args.batch_prompts, len(tasks)))
            batch_prompts = [t.prompt for t in batch_tasks]
            reward_fn_per_task = None

            if args.task_type == "multi_niah":
                batch_expected = [t.expected_answers for t in batch_tasks]
            else:
                batch_expected = [t.expected_answer for t in batch_tasks]

        # GRPO step
        stats = grpo_step(
            model=rl_model,
            optimizer=optimizer,
            task_prompts=batch_prompts,
            expected_answers=batch_expected,
            system_prompt=QWEN_2B_SYSTEM_PROMPT,
            reward_fn=reward_fn if reward_fn != "mixed" else "composite",
            reward_fn_per_task=reward_fn_per_task,
            k=args.k,
            kl_coeff=args.kl_coeff,
            ref_weights=ref_weights,
        )

        training_log.append({"step": step, **stats})

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            kl_str = f" | KL: {stats.get('kl_loss', 0):.6f}" if args.kl_coeff > 0 else ""
            logger.info(
                f"Step {step + 1}/{args.steps} | "
                f"Reward: {stats['avg_reward']:.3f} | "
                f"Correct: {stats['correct']}/{stats['total']} | "
                f"Loss: {stats['policy_loss']:.4f}{kl_str} | "
                f"Updates: {stats['n_updates']} | "
                f"Turns: {stats['avg_turns']:.1f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Sample and log trajectories for inspection
        if (step + 1) % args.sample_trajectories_interval == 0:
            sample_dir = output_dir / "sample_trajectories" / f"step_{step + 1}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Run 3 tasks and save full trajectories
            sample_tasks = (niah_tasks[:2] + mniah_tasks[:1]) if args.task_type == "mixed" else tasks[:3]
            for i, task in enumerate(sample_tasks):
                traj = rlm(
                    prompt=task.prompt,
                    model=rl_model,
                    system_prompt=QWEN_2B_SYSTEM_PROMPT,
                    max_iterations=6,
                    verbose=False,
                )
                with open(sample_dir / f"traj_{i}.json", "w") as f:
                    json.dump(trajectory_to_dict(traj), f, indent=2, default=str)

                if hasattr(task, 'expected_answers'):
                    scores = score_multi_niah(traj.answer, task.expected_answers)
                    logger.info(
                        f"  Sample {i}: recall={scores['recall']:.2f} "
                        f"({scores['found']}/{scores['total']}) "
                        f"answer={str(traj.answer)[:80]}"
                    )
                else:
                    logger.info(
                        f"  Sample {i}: score={score_niah(traj.answer, task.expected_answer)} "
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
    logger.info(f"RL TRAINING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Final avg reward: {training_log[-1]['avg_reward']:.3f}")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 3600:.2f} GPU-hours)")
    logger.info(f"Saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
