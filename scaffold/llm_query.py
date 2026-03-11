"""
LLM query wrapper for RLM sub-calls.

Three implementations:
1. HFModel: uses transformers model.generate() directly (local GPU)
2. TinkerModel: uses Tinker API for remote generation (no GPU needed)
3. MockModel: returns fixed strings for testing
"""

from __future__ import annotations

import logging
import os
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


class TinkerModel:
    """
    Tinker API model for remote generation.

    Uses Tinker's sampling_client.sample() for both root generation
    and sub-calls. No local GPU needed — all computation runs on
    Tinker's distributed GPU cluster.

    Supports both base models and fine-tuned checkpoints.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        model_path: str | None = None,
        renderer_name: str = "qwen3_disable_thinking",
    ):
        from dotenv import load_dotenv
        load_dotenv()

        import tinker
        from tinker_cookbook import renderers, tokenizer_utils

        logger.info(f"Creating TinkerModel for {model_name}...")
        t0 = time.time()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_log: list[dict] = []
        self.capture_logprobs = False  # Enable for RL training
        self.last_logprobs: list[float] | None = None
        self.last_tokens: list[int] | None = None

        # Setup tokenizer and renderer
        self.tokenizer = tokenizer_utils.get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        self.stop_sequences = self.renderer.get_stop_sequences()

        # Create sampling client
        self.service_client = tinker.ServiceClient()
        if model_path and "/weights/state-" in model_path:
            # State checkpoint — must load via training client
            logger.info(f"Loading from state checkpoint: {model_path}")
            tc = self.service_client.create_training_client_from_state(model_path)
            self.sampling_client = tc.save_weights_and_get_sampling_client(
                name="eval-from-state"
            )
            logger.info(f"Loaded fine-tuned model from state: {model_path}")
        elif model_path:
            # Sampler weights checkpoint
            self.sampling_client = self.service_client.create_sampling_client(
                model_path=model_path
            )
            logger.info(f"Loaded fine-tuned model from {model_path}")
        else:
            # Base model
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=model_name
            )
            logger.info(f"Loaded base model {model_name}")

        self._tinker = tinker  # Keep reference for SamplingParams
        elapsed = time.time() - t0
        logger.info(f"TinkerModel ready in {elapsed:.1f}s")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Generate from chat messages. Used for root RLM turns.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
        Returns:
            Generated text (assistant response only)
        """
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        num_samples = kwargs.get("num_samples", 1)

        # Build prompt using renderer (handles chat template correctly)
        prompt = self.renderer.build_generation_prompt(messages)

        params = self._tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=self.stop_sequences,
        )

        t0 = time.time()
        result = self.sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=num_samples,
        ).result()
        elapsed = time.time() - t0

        # Parse response
        seq = result.sequences[0]
        msg, parse_ok = self.renderer.parse_response(seq.tokens)
        response = msg.get("content", "") if isinstance(msg, dict) else str(msg)

        # Strip think tags if present
        response = strip_think_tags(response)

        # Capture logprobs and tokens for RL training
        if self.capture_logprobs and hasattr(seq, 'logprobs') and seq.logprobs is not None:
            self.last_logprobs = list(seq.logprobs)
            self.last_tokens = list(seq.tokens)
        else:
            self.last_logprobs = None
            self.last_tokens = None

        self.generation_log.append({
            "type": "root",
            "input_tokens": prompt.length,
            "output_tokens": len(seq.tokens),
            "time": elapsed,
            "parse_ok": parse_ok,
        })

        logger.debug(f"Generated {len(seq.tokens)} tokens in {elapsed:.2f}s")
        return response

    def sub_query(self, prompt_str: str) -> str:
        """
        Sub-call interface for llm_query() inside the REPL.

        Takes a plain string prompt, wraps it as a user message,
        generates a response. No system prompt — leaf calls are
        just regular LLM requests.

        Uses a fixed low temperature for factual extraction (not the
        root generation temperature, which can be up to 1.3 during RL).
        """
        messages = [{"role": "user", "content": prompt_str}]
        prompt = self.renderer.build_generation_prompt(messages)

        # Sub-calls do factual extraction — always use low temperature
        # regardless of root generation temperature setting.
        # Root temp can be 0.6-1.3 during RL training; sub-calls should
        # never be that high or they hallucinate chunk extractions.
        sub_temp = getattr(self, 'sub_temperature', 0.3)

        params = self._tinker.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=sub_temp,
            stop=self.stop_sequences,
        )

        t0 = time.time()
        result = self.sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=1,
        ).result()
        elapsed = time.time() - t0

        seq = result.sequences[0]
        msg, parse_ok = self.renderer.parse_response(seq.tokens)
        response = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        response = strip_think_tags(response)

        self.generation_log.append({
            "type": "sub_query",
            "input_tokens": prompt.length,
            "output_tokens": len(seq.tokens),
            "time": elapsed,
        })

        logger.debug(f"Sub-query: {len(seq.tokens)} tokens in {elapsed:.2f}s")
        return response

    def reset_stats(self):
        """Reset generation stats. Call between trajectory collections."""
        self.generation_log = []
        self.last_logprobs = None
        self.last_tokens = None

    def refresh_sampling_client(self, model_path: str | None = None):
        """Refresh sampling client after training (gets updated weights)."""
        if model_path:
            self.sampling_client = self.service_client.create_sampling_client(
                model_path=model_path
            )
        else:
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=self.model_name
            )

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


class HybridTinkerModel:
    """
    Hybrid model: fine-tuned model for root calls, base model for sub-calls.

    Key insight: RL training on code generation degrades the model's
    question-answering ability for sub-calls. Using the base model for
    sub-calls preserves QA quality while benefiting from trained code generation.

    This matches the original RLM paper's approach (separate root/sub models).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        model_path: str | None = None,
        renderer_name: str = "qwen3_disable_thinking",
    ):
        from dotenv import load_dotenv
        load_dotenv()

        import tinker
        from tinker_cookbook import renderers, tokenizer_utils

        logger.info(f"Creating HybridTinkerModel for {model_name}...")
        logger.info(f"  Root model: {model_path or 'base'}")
        logger.info(f"  Sub-call model: base (untrained)")
        t0 = time.time()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_log: list[dict] = []
        self.capture_logprobs = False
        self.last_logprobs: list[float] | None = None
        self.last_tokens: list[int] | None = None

        # Setup tokenizer and renderer
        self.tokenizer = tokenizer_utils.get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        self.stop_sequences = self.renderer.get_stop_sequences()

        self.service_client = tinker.ServiceClient()
        self._tinker = tinker

        # Root model: fine-tuned checkpoint
        if model_path and "/weights/state-" in model_path:
            logger.info(f"Loading root model from state: {model_path}")
            tc = self.service_client.create_training_client_from_state(model_path)
            self.root_sampling_client = tc.save_weights_and_get_sampling_client(
                name="hybrid-root"
            )
        elif model_path:
            self.root_sampling_client = self.service_client.create_sampling_client(
                model_path=model_path
            )
        else:
            self.root_sampling_client = self.service_client.create_sampling_client(
                base_model=model_name
            )

        # Sub-call model: always base model (untrained QA ability)
        self.sub_sampling_client = self.service_client.create_sampling_client(
            base_model=model_name
        )
        logger.info(f"Sub-call model: base {model_name}")

        elapsed = time.time() - t0
        logger.info(f"HybridTinkerModel ready in {elapsed:.1f}s")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Root call: uses fine-tuned model for code generation."""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        num_samples = kwargs.get("num_samples", 1)

        prompt = self.renderer.build_generation_prompt(messages)
        params = self._tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=self.stop_sequences,
        )

        t0 = time.time()
        result = self.root_sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=num_samples,
        ).result()
        elapsed = time.time() - t0

        seq = result.sequences[0]
        msg, parse_ok = self.renderer.parse_response(seq.tokens)
        response = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        response = strip_think_tags(response)

        if self.capture_logprobs and hasattr(seq, 'logprobs') and seq.logprobs is not None:
            self.last_logprobs = list(seq.logprobs)
            self.last_tokens = list(seq.tokens)
        else:
            self.last_logprobs = None
            self.last_tokens = None

        self.generation_log.append({
            "type": "root",
            "input_tokens": prompt.length,
            "output_tokens": len(seq.tokens),
            "time": elapsed,
            "parse_ok": parse_ok,
        })
        return response

    def sub_query(self, prompt_str: str) -> str:
        """Sub-call: uses BASE model for question answering."""
        messages = [{"role": "user", "content": prompt_str}]
        prompt = self.renderer.build_generation_prompt(messages)

        # Sub-calls do factual extraction — always use low temperature
        # regardless of root generation temperature setting.
        # BUG FIX: was using self.temperature (0.7 in eval, up to 1.3 in RL)
        # which caused hallucinated sub-call responses.
        sub_temp = getattr(self, 'sub_temperature', 0.3)

        params = self._tinker.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=sub_temp,
            stop=self.stop_sequences,
        )

        t0 = time.time()
        result = self.sub_sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=1,
        ).result()
        elapsed = time.time() - t0

        seq = result.sequences[0]
        msg, parse_ok = self.renderer.parse_response(seq.tokens)
        response = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        response = strip_think_tags(response)

        self.generation_log.append({
            "type": "sub_query",
            "input_tokens": prompt.length,
            "output_tokens": len(seq.tokens),
            "time": elapsed,
        })
        return response

    def reset_stats(self):
        self.generation_log = []
        self.last_logprobs = None
        self.last_tokens = None

    def total_stats(self) -> dict:
        root_calls = [g for g in self.generation_log if g["type"] == "root"]
        sub_calls = [g for g in self.generation_log if g["type"] == "sub_query"]
        return {
            "root_calls": len(root_calls),
            "sub_calls": len(sub_calls),
            "total_input_tokens": sum(g["input_tokens"] for g in self.generation_log),
            "total_output_tokens": sum(g["output_tokens"] for g in self.generation_log),
            "total_time": sum(g["time"] for g in self.generation_log),
        }


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
