#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training for RLM.

Approach:
1. Generate K trajectories per task using the SFT model
2. Form preference pairs: best trajectory (highest reward) vs worst (lowest reward)
3. Train with DPO loss, which implicitly constrains KL from reference

The key advantage over GRPO: DPO has a built-in KL constraint (the reference model
is implicit in the loss function), so we don't need separate KL regularization.

Usage:
    CUDA_VISIBLE_DEVICES=6 uv run python training/dpo.py \
        --model data/sft/lora_v3/final \
        --base-model Qwen/Qwen3-1.7B \
        --output data/sft/dpo_v1

Phase 1: Generates trajectory pairs (slow, ~60 min)
Phase 2: DPO training on pairs (fast, ~15 min)
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
from training.logprobs import compute_trajectory_logprobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("dpo")


class DPOModel:
    """Model wrapper for DPO (same as RLModel but with reference log-prob caching)."""

    def __init__(self, base_model_name, adapter_path=None, device="cuda:0",
                 max_new_tokens=1024):
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
        return strip_think_tags(response)


def multi_niah_reward(predicted, expected_answers, trajectory_dict):
    """Reward for multi-needle NIAH."""
    if predicted is None:
        return 0.0
    scores = score_multi_niah(predicted, expected_answers)
    recall = scores["recall"]
    fmt_bonus = 0.1 if trajectory_dict.get("terminated") else 0.0
    return 0.9 * recall + 0.1 * fmt_bonus


def generate_trajectory_pairs(
    model, tasks, task_type, system_prompt, k=8,
    min_reward_gap=0.2, temperature=0.8,
):
    """
    Generate K trajectories per task, form preference pairs.

    Returns list of dicts:
      {task, task_type, winner_traj, loser_traj, winner_reward, loser_reward}
    """
    pairs = []
    all_trajs_info = []

    for task_idx, task in enumerate(tqdm(tasks, desc=f"Generating {task_type} trajectories")):
        group_trajs = []
        group_rewards = []

        for ki in range(k):
            traj = rlm(
                prompt=task.prompt,
                model=model,
                system_prompt=system_prompt,
                max_iterations=6,
                verbose=False,
            )

            traj_dict = trajectory_to_dict(traj)
            if task_type == "multi_niah":
                reward = multi_niah_reward(traj.answer, task.expected_answers, traj_dict)
            else:
                reward = composite_reward(traj.answer, task.expected_answer, traj_dict)

            group_trajs.append(traj)
            group_rewards.append(reward)

        # Log group stats
        avg_r = sum(group_rewards) / len(group_rewards)
        max_r = max(group_rewards)
        min_r = min(group_rewards)
        logger.info(
            f"Task {task_idx}: {task_type} | "
            f"rewards=[{min_r:.2f}, {avg_r:.2f}, {max_r:.2f}] | "
            f"gap={max_r - min_r:.2f}"
        )

        all_trajs_info.append({
            "task_idx": task_idx,
            "task_type": task_type,
            "rewards": group_rewards,
            "answers": [str(t.answer)[:100] for t in group_trajs],
        })

        # Form preference pair if sufficient gap
        if max_r - min_r >= min_reward_gap:
            best_idx = group_rewards.index(max_r)
            worst_idx = group_rewards.index(min_r)

            pairs.append({
                "task_idx": task_idx,
                "task_type": task_type,
                "winner_traj": group_trajs[best_idx],
                "loser_traj": group_trajs[worst_idx],
                "winner_reward": max_r,
                "loser_reward": min_r,
            })
            logger.info(
                f"  -> Pair formed: winner={max_r:.2f} loser={min_r:.2f} "
                f"gap={max_r - min_r:.2f}"
            )
        else:
            logger.info(f"  -> No pair (gap {max_r - min_r:.2f} < {min_reward_gap})")

    return pairs, all_trajs_info


def dpo_loss(
    policy_logprob_winner: torch.Tensor,
    policy_logprob_loser: torch.Tensor,
    ref_logprob_winner: torch.Tensor,
    ref_logprob_loser: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute DPO loss for one preference pair.

    L = -log σ(β * ((log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x))))
    """
    policy_diff = policy_logprob_winner - policy_logprob_loser
    ref_diff = ref_logprob_winner - ref_logprob_loser
    logit = beta * (policy_diff - ref_diff)
    loss = -F.logsigmoid(logit)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model", default=None,
                        help="Path to LoRA adapter (SFT checkpoint)")
    parser.add_argument("--output", default="data/sft/dpo_v1")
    parser.add_argument("--k", type=int, default=8,
                        help="Trajectories per prompt for pair generation")
    parser.add_argument("--n-niah-tasks", type=int, default=15,
                        help="Number of NIAH tasks")
    parser.add_argument("--n-mniah-tasks", type=int, default=20,
                        help="Number of multi-NIAH tasks")
    parser.add_argument("--min-reward-gap", type=float, default=0.15,
                        help="Minimum reward gap to form a pair")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL constraint strength)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs over preference pairs")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N training steps")
    parser.add_argument("--doc-lengths", nargs="+", type=int,
                        default=[5000, 10000, 20000, 50000],
                        help="Document lengths for NIAH tasks")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip pair generation, load from --pairs-file")
    parser.add_argument("--pairs-file", default=None,
                        help="Load pre-generated pairs from this file")
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
    dpo_model = DPOModel(
        base_model_name=args.base_model,
        adapter_path=args.model,
        device=device,
    )

    trainable_params = [p for p in dpo_model.model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    # ================================================================
    # Phase 1: Generate trajectory pairs
    # ================================================================
    if not args.skip_generation:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Generating trajectory pairs")
        logger.info("=" * 60)

        t0_gen = time.time()

        # Generate tasks
        niah_tasks = generate_niah_suite(
            n_tasks=args.n_niah_tasks, doc_lengths=args.doc_lengths,
        )
        mniah_tasks = generate_multi_niah_suite(n_tasks=args.n_mniah_tasks)

        logger.info(f"Generated {len(niah_tasks)} NIAH + {len(mniah_tasks)} multi-NIAH tasks")
        logger.info(f"K={args.k} trajectories per task, min_gap={args.min_reward_gap}")

        # Generate pairs for both task types
        niah_pairs, niah_info = generate_trajectory_pairs(
            dpo_model, niah_tasks, "niah", QWEN_2B_SYSTEM_PROMPT,
            k=args.k, min_reward_gap=args.min_reward_gap,
        )

        mniah_pairs, mniah_info = generate_trajectory_pairs(
            dpo_model, mniah_tasks, "multi_niah", QWEN_2B_SYSTEM_PROMPT,
            k=args.k, min_reward_gap=args.min_reward_gap,
        )

        all_pairs = niah_pairs + mniah_pairs
        random.shuffle(all_pairs)

        gen_time = time.time() - t0_gen
        logger.info(f"\nPair generation complete in {gen_time:.0f}s")
        logger.info(f"NIAH pairs: {len(niah_pairs)}/{len(niah_tasks)} tasks")
        logger.info(f"Multi-NIAH pairs: {len(mniah_pairs)}/{len(mniah_tasks)} tasks")
        logger.info(f"Total pairs: {len(all_pairs)}")

        # Save pair info for analysis
        pair_info = []
        for p in all_pairs:
            pair_info.append({
                "task_idx": p["task_idx"],
                "task_type": p["task_type"],
                "winner_reward": p["winner_reward"],
                "loser_reward": p["loser_reward"],
                "winner_answer": str(p["winner_traj"].answer)[:200],
                "loser_answer": str(p["loser_traj"].answer)[:200],
                "winner_turns": len(p["winner_traj"].turns),
                "loser_turns": len(p["loser_traj"].turns),
            })
        with open(output_dir / "pair_info.json", "w") as f:
            json.dump(pair_info, f, indent=2)

        # Save generation stats
        with open(output_dir / "generation_stats.json", "w") as f:
            json.dump({
                "niah_info": niah_info,
                "mniah_info": mniah_info,
                "gen_time_seconds": gen_time,
                "n_niah_pairs": len(niah_pairs),
                "n_mniah_pairs": len(mniah_pairs),
            }, f, indent=2, default=str)

        if len(all_pairs) < 5:
            logger.error(
                f"Only {len(all_pairs)} pairs generated! Need more tasks or lower min_reward_gap."
            )
            return
    else:
        logger.info("Skipping generation, loading pairs from file...")
        raise NotImplementedError("--skip-generation not yet implemented")

    # ================================================================
    # Phase 2: Compute reference log-probs (before any training)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Computing reference log-probs")
    logger.info("=" * 60)

    t0_ref = time.time()
    dpo_model.model.eval()

    ref_logprobs = []  # List of (winner_lp, loser_lp)
    for pair_idx, pair in enumerate(tqdm(all_pairs, desc="Computing ref log-probs")):
        with torch.no_grad():
            winner_lp, _, _ = compute_trajectory_logprobs(
                dpo_model.model, dpo_model.tokenizer,
                pair["winner_traj"], device,
            )
            loser_lp, _, _ = compute_trajectory_logprobs(
                dpo_model.model, dpo_model.tokenizer,
                pair["loser_traj"], device,
            )
        ref_logprobs.append((winner_lp.item(), loser_lp.item()))

        if pair_idx < 3:
            logger.info(
                f"  Pair {pair_idx}: ref_lp(winner)={winner_lp.item():.1f} "
                f"ref_lp(loser)={loser_lp.item():.1f} "
                f"diff={winner_lp.item() - loser_lp.item():.1f}"
            )

    ref_time = time.time() - t0_ref
    logger.info(f"Reference log-probs computed in {ref_time:.0f}s")

    # ================================================================
    # Phase 3: DPO training
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: DPO training")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    training_log = []
    t0_train = time.time()

    total_steps = 0
    for epoch in range(args.epochs):
        # Shuffle pairs each epoch
        indices = list(range(len(all_pairs)))
        random.shuffle(indices)

        epoch_losses = []
        epoch_accs = []  # DPO accuracy: does policy prefer the winner?

        for idx in indices:
            pair = all_pairs[idx]
            ref_w_lp, ref_l_lp = ref_logprobs[idx]

            # Compute policy log-probs (with gradients)
            dpo_model.model.train()
            policy_w_lp, _, n_w = compute_trajectory_logprobs(
                dpo_model.model, dpo_model.tokenizer,
                pair["winner_traj"], device,
            )
            policy_l_lp, _, n_l = compute_trajectory_logprobs(
                dpo_model.model, dpo_model.tokenizer,
                pair["loser_traj"], device,
            )

            # Normalize by length to prevent length bias
            if n_w > 0:
                policy_w_lp_norm = policy_w_lp / n_w
            else:
                policy_w_lp_norm = policy_w_lp
            if n_l > 0:
                policy_l_lp_norm = policy_l_lp / n_l
            else:
                policy_l_lp_norm = policy_l_lp

            ref_w_lp_norm = ref_w_lp / max(n_w, 1)
            ref_l_lp_norm = ref_l_lp / max(n_l, 1)

            # DPO loss
            loss = dpo_loss(
                policy_w_lp_norm, policy_l_lp_norm,
                torch.tensor(ref_w_lp_norm, device=device),
                torch.tensor(ref_l_lp_norm, device=device),
                beta=args.beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Track accuracy: does policy prefer winner over loser?
            with torch.no_grad():
                policy_diff = policy_w_lp_norm - policy_l_lp_norm
                ref_diff = ref_w_lp_norm - ref_l_lp_norm
                implicit_reward_diff = (policy_diff - ref_diff).item()
                acc = 1.0 if implicit_reward_diff > 0 else 0.0

            epoch_losses.append(loss.item())
            epoch_accs.append(acc)
            total_steps += 1

            step_info = {
                "step": total_steps,
                "epoch": epoch,
                "loss": loss.item(),
                "acc": acc,
                "policy_w_lp": policy_w_lp_norm.item(),
                "policy_l_lp": policy_l_lp_norm.item(),
                "task_type": pair["task_type"],
                "winner_reward": pair["winner_reward"],
                "loser_reward": pair["loser_reward"],
            }
            training_log.append(step_info)

            if total_steps % 10 == 0:
                recent_losses = epoch_losses[-10:]
                recent_accs = epoch_accs[-10:]
                logger.info(
                    f"Step {total_steps} (epoch {epoch+1}) | "
                    f"Loss: {sum(recent_losses)/len(recent_losses):.4f} | "
                    f"Acc: {sum(recent_accs)/len(recent_accs):.1%} | "
                    f"π(w): {policy_w_lp_norm.item():.2f} "
                    f"π(l): {policy_l_lp_norm.item():.2f}"
                )

            if total_steps % args.save_interval == 0:
                ckpt_dir = output_dir / f"checkpoint-{total_steps}"
                dpo_model.model.save_pretrained(ckpt_dir)
                logger.info(f"  Saved checkpoint: {ckpt_dir}")

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_accs) / len(epoch_accs)
        logger.info(
            f"\nEpoch {epoch+1}/{args.epochs}: "
            f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.1%} "
            f"({len(epoch_losses)} steps)"
        )

    # Save final checkpoint
    dpo_model.model.save_pretrained(output_dir / "final")
    dpo_model.tokenizer.save_pretrained(output_dir / "final")

    train_time = time.time() - t0_train
    total_time = time.time() - t0_gen if not args.skip_generation else train_time

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"DPO TRAINING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Pairs: {len(all_pairs)} ({len(niah_pairs)} NIAH + {len(mniah_pairs)} multi-NIAH)")
    logger.info(f"Steps: {total_steps} ({args.epochs} epochs)")
    logger.info(f"Final loss: {training_log[-1]['loss']:.4f}")
    logger.info(f"Final acc: {sum(e['acc'] for e in training_log[-20:]) / min(20, len(training_log)):.1%}")
    logger.info(f"Generation time: {gen_time:.0f}s")
    logger.info(f"Training time: {train_time:.0f}s")
    logger.info(f"Total time: {total_time:.0f}s ({total_time / 3600:.2f} GPU-hours)")
    logger.info(f"Saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
