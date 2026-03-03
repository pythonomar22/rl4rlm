"""
LLM query wrapper for RLM sub-calls.

All calls are in-process. No servers, no HTTP, no ports.
The same model handles both root generation and sub-calls.

Two implementations:
1. HFModel: uses transformers model.generate() directly
2. MockModel: returns fixed strings for testing
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from Qwen3 model output.
    Preserves the content after the thinking block."""
    # Remove <think>...</think> blocks (including empty ones)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()


class MockModel:
    """Mock model for testing the scaffold without GPU."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_idx = 0
        self.call_log: list[dict] = []

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a response given chat messages."""
        self.call_log.append({
            "messages": messages,
            "kwargs": kwargs,
            "time": time.time(),
        })

        if self.call_idx < len(self.responses):
            resp = self.responses[self.call_idx]
            self.call_idx += 1
            return resp

        # Default: echo back a placeholder
        return '```repl\nprint("mock response")\nFINAL("mock answer")\n```'

    def sub_query(self, prompt_str: str) -> str:
        """Sub-call interface: takes a plain string, returns a plain string.
        This is what gets injected into the REPL as `llm_query`."""
        self.call_log.append({
            "type": "sub_query",
            "prompt": prompt_str[:200],
            "time": time.time(),
        })

        if self.call_idx < len(self.responses):
            resp = self.responses[self.call_idx]
            self.call_idx += 1
            return resp

        return f"[mock sub-response to {len(prompt_str)} char prompt]"


class HFModel:
    """
    In-process HuggingFace transformers model.

    Loads model and tokenizer once, generates via model.generate().
    Same instance handles root generation AND sub-calls (llm_query).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        device: str = "cuda:0",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model {model_name} on {device}...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.generation_log: list[dict] = []

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s on {device}")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Generate from chat messages. Used for root RLM turns.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
        Returns:
            Generated text (assistant response only)
        """
        import torch

        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        # Apply chat template (disable thinking to save tokens)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback for models without enable_thinking
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][input_len:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = strip_think_tags(raw_response)
        elapsed = time.time() - t0

        self.generation_log.append({
            "type": "root",
            "input_tokens": input_len,
            "output_tokens": len(new_tokens),
            "time": elapsed,
            "raw_response": raw_response[:500],
        })

        logger.debug(f"Generated {len(new_tokens)} tokens in {elapsed:.2f}s")
        return response

    def sub_query(self, prompt_str: str) -> str:
        """
        Sub-call interface for llm_query() inside the REPL.

        Takes a plain string prompt, wraps it as a user message,
        generates a response. No system prompt — leaf calls are
        just regular LLM requests.
        """
        messages = [{"role": "user", "content": prompt_str}]

        import torch

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

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = strip_think_tags(raw_response)
        elapsed = time.time() - t0

        self.generation_log.append({
            "type": "sub_query",
            "input_tokens": input_len,
            "output_tokens": len(new_tokens),
            "time": elapsed,
        })

        logger.debug(f"Sub-query: {len(new_tokens)} tokens in {elapsed:.2f}s")
        return response

    def total_stats(self) -> dict:
        """Return total generation statistics."""
        root_calls = [g for g in self.generation_log if g["type"] == "root"]
        sub_calls = [g for g in self.generation_log if g["type"] == "sub_query"]
        return {
            "root_calls": len(root_calls),
            "sub_calls": len(sub_calls),
            "total_input_tokens": sum(g["input_tokens"] for g in self.generation_log),
            "total_output_tokens": sum(g["output_tokens"] for g in self.generation_log),
            "total_time": sum(g["time"] for g in self.generation_log),
        }
