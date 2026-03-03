"""
Shared log-probability computation for DPO and GRPO-v4.

Key insight: the old GRPO (rl.py) computed log P(code_text) unconditionally,
without the conversation context. This meant the model learned to
produce/avoid code patterns in general, not conditioned on the task.

This module computes proper conditioned log-probs:
  log P(assistant_response | system_prompt, metadata, prior_turns)

Used by both DPO (preference pairs) and GRPO-v4 (policy gradient + KL).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any


def _apply_chat_template(tokenizer, messages, add_generation_prompt=False):
    """Apply chat template with enable_thinking=False (Qwen3 specific)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def compute_turn_logprobs(
    model,
    tokenizer,
    messages_before: list[dict[str, str]],
    assistant_message: dict[str, str],
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute per-token log-probs of an assistant response conditioned on context.

    Args:
        model: HuggingFace model (caller manages train/eval and grad context)
        tokenizer: Tokenizer
        messages_before: All messages before this assistant turn
        assistant_message: The assistant message (role="assistant")
        device: Device

    Returns:
        (total_logprob, per_token_logprobs [1, n_response_tokens], n_response_tokens)
    """
    # Context text: everything up to where the assistant starts generating
    context_text = _apply_chat_template(
        tokenizer, messages_before, add_generation_prompt=True,
    )

    # Full text: context + assistant response
    full_messages = list(messages_before) + [assistant_message]
    full_text = _apply_chat_template(
        tokenizer, full_messages, add_generation_prompt=False,
    )

    # Tokenize
    context_ids = tokenizer(context_text, return_tensors="pt").to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)

    prompt_len = context_ids["input_ids"].shape[1]
    total_len = full_ids["input_ids"].shape[1]

    if total_len <= prompt_len:
        zero = torch.tensor(0.0, device=device)
        return zero, torch.zeros(1, 0, device=device), 0

    # Forward pass (caller manages gradient context)
    outputs = model(**full_ids)
    logits = outputs.logits  # [1, total_len, vocab_size]

    # Per-token log-probs
    # logits[t] predicts token[t+1], so:
    # log_prob of token at position p is at logits[p-1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_ids = full_ids["input_ids"][:, 1:]
    all_token_logprobs = log_probs.gather(
        2, token_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, total_len-1]

    # Response tokens only: first response token is at position prompt_len
    # Its log-prob is at index prompt_len-1 in all_token_logprobs
    response_logprobs = all_token_logprobs[:, prompt_len - 1:]
    n_response_tokens = response_logprobs.shape[1]
    total_logprob = response_logprobs.sum()

    return total_logprob, response_logprobs, n_response_tokens


def compute_trajectory_logprobs(
    model,
    tokenizer,
    trajectory,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    """
    Compute total log-prob of all assistant responses in a trajectory.

    Args:
        trajectory: RLMTrajectory with .messages attribute

    Returns:
        (total_logprob, list_of_per_turn_logprobs, total_response_tokens)
    """
    messages = trajectory.messages
    if not messages:
        zero = torch.tensor(0.0, device=device)
        return zero, [], 0

    total_logprob = torch.tensor(0.0, device=device, requires_grad=True)
    turn_logprobs = []
    total_tokens = 0

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        messages_before = messages[:i]
        turn_lp, per_token_lp, n_tokens = compute_turn_logprobs(
            model, tokenizer, messages_before, msg, device,
        )
        total_logprob = total_logprob + turn_lp
        turn_logprobs.append(per_token_lp)
        total_tokens += n_tokens

    return total_logprob, turn_logprobs, total_tokens


def compute_turn_kl_logits(
    policy_model,
    ref_model,
    tokenizer,
    messages_before: list[dict[str, str]],
    assistant_message: dict[str, str],
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute token-level KL divergence between policy and reference for one turn.

    Uses proper KL: D_KL(π_θ || π_ref) = Σ_a π_θ(a) * (log π_θ(a) - log π_ref(a))

    Returns:
        (kl_mean, policy_logprob, ref_logprob)
        kl_mean: scalar, mean KL per response token
        policy_logprob: total log-prob under policy (with gradients)
        ref_logprob: total log-prob under reference (no gradients)
    """
    context_text = _apply_chat_template(
        tokenizer, messages_before, add_generation_prompt=True,
    )
    full_messages = list(messages_before) + [assistant_message]
    full_text = _apply_chat_template(
        tokenizer, full_messages, add_generation_prompt=False,
    )

    context_ids = tokenizer(context_text, return_tensors="pt").to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)

    prompt_len = context_ids["input_ids"].shape[1]
    total_len = full_ids["input_ids"].shape[1]

    if total_len <= prompt_len:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero

    # Forward pass through both models
    policy_outputs = policy_model(**full_ids)
    with torch.no_grad():
        ref_outputs = ref_model(**full_ids)

    # Logits that predict response tokens
    # Token at position prompt_len is predicted by logits at prompt_len-1
    policy_logits = policy_outputs.logits[:, prompt_len - 1:total_len - 1, :]
    ref_logits = ref_outputs.logits[:, prompt_len - 1:total_len - 1, :]

    # Token-level KL: D_KL(π_θ || π_ref)
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    kl_mean = kl_per_token.mean()

    # Also compute log-probs of the actual generated tokens
    token_ids = full_ids["input_ids"][:, prompt_len:total_len]  # response token ids

    policy_token_lps = policy_log_probs.gather(
        2, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    ref_token_lps = ref_log_probs.gather(
        2, token_ids.unsqueeze(-1)
    ).squeeze(-1)

    return kl_mean, policy_token_lps.sum(), ref_token_lps.sum()
