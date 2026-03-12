"""
Main RLM loop (Algorithm 1 from the paper).

The loop:
1. Init REPL with prompt as `context`
2. Build metadata (never full prompt) → send to LLM
3. LLM generates code in ```repl blocks
4. Execute code in REPL
5. Build metadata of stdout → append to history
6. Repeat until FINAL/FINAL_VAR or max iterations
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from scaffold.repl import (
    REPLState, REPLResult,
    init_repl, repl_execute, metadata, stdout_metadata,
)

logger = logging.getLogger(__name__)


@dataclass
class RLMTrajectory:
    """Complete record of an RLM execution for training/analysis."""
    prompt: str
    system_prompt: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    terminated: bool = False
    total_time: float = 0.0
    model_stats: dict | None = None
    messages: list[dict[str, str]] = field(default_factory=list)


def parse_repl_code(response: str) -> str | None:
    """
    Extract Python code from ```repl ... ``` blocks in model response.

    The model should wrap its code in:
        ```repl
        <python code>
        ```

    Also accepts ```python as fallback (common model behavior).
    Returns None if no code block found.
    """
    # Try ```repl first (preferred)
    patterns = [
        r"```repl\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

    # Last resort: if the entire response looks like Python code
    # (no markdown, starts with valid Python)
    stripped = response.strip()
    if stripped and not stripped.startswith("#") and (
        any(stripped.startswith(kw) for kw in
            ["import ", "from ", "for ", "if ", "while ", "def ",
             "class ", "print(", "context", "llm_query", "FINAL",
             "result", "answer", "chunks", "data"])
        or "=" in stripped.split("\n")[0]
    ):
        return stripped

    return None


def build_initial_message(state: REPLState, system_prompt: str) -> list[dict[str, str]]:
    """Build the initial message history for the LLM."""
    meta = metadata(state)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": meta},
    ]
    return messages


def rlm(
    prompt: str,
    model: Any,
    system_prompt: str,
    max_iterations: int = 10,
    stdout_max_chars: int = 1000,
    code_timeout: int | None = None,
    verbose: bool = True,
) -> RLMTrajectory:
    """
    Run the full RLM loop on a prompt.

    Args:
        prompt: The full input text (stored in REPL as `context`)
        model: Object with .generate(messages) and .sub_query(str) methods
        system_prompt: System prompt for the root LLM
        max_iterations: Max REPL turns before forced termination
        stdout_max_chars: Truncation limit for stdout metadata
        code_timeout: Timeout per REPL execution in seconds.
            If None, auto-scales: 60s base + 30s per 100K chars of prompt.
        verbose: Print detailed execution info

    Returns:
        RLMTrajectory with full execution record
    """
    t0 = time.time()
    trajectory = RLMTrajectory(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    # Auto-scale timeout with document length: 120s base + 60s per 100K chars
    # Tinker API sub-calls take 10-20s each; 54K doc with 3 chunks needs ~60-90s
    # Previous 90s base caused excessive timeouts in training
    if code_timeout is None:
        code_timeout = max(120, 120 + (len(prompt) // 100000) * 60)

    # 1. Init REPL with prompt as context, model's sub_query as llm_query
    state = init_repl(
        prompt=prompt,
        llm_query_fn=model.sub_query,
        timeout=code_timeout,
    )

    # 2. Build initial messages with metadata (never full prompt)
    messages = build_initial_message(state, system_prompt)

    if verbose:
        logger.info(f"RLM start: prompt={len(prompt)} chars, max_iter={max_iterations}")
        logger.info(f"Initial metadata:\n{messages[-1]['content'][:500]}")

    for iteration in range(max_iterations):
        iter_t0 = time.time()

        # 3. Generate code from LLM
        response = model.generate(messages)

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Turn {iteration + 1}: Model response ({len(response)} chars):")
            logger.info(f"RAW OUTPUT:\n{response}")

        # 4. Parse code from response
        code = parse_repl_code(response)

        turn_record = {
            "iteration": iteration + 1,
            "raw_response": response,
            "parsed_code": code,
            "time": 0.0,
        }

        # Capture per-turn logprobs for RL training (importance sampling)
        if hasattr(model, 'last_logprobs') and model.last_logprobs is not None:
            turn_record["logprobs"] = model.last_logprobs
            turn_record["tokens"] = model.last_tokens

        if code is None:
            # Check for gibberish (MoE routing failure) — abort early to save time
            if len(response) > 200:
                non_ascii = sum(1 for c in response if ord(c) > 127)
                if non_ascii / len(response) > 0.15:
                    error_msg = "Gibberish detected (MoE routing failure). Aborting."
                    turn_record["error"] = error_msg
                    trajectory.turns.append(turn_record)
                    if verbose:
                        logger.warning(f"Turn {iteration + 1}: Gibberish detected, aborting trajectory")
                    break

            # Model didn't produce parseable code
            error_msg = (
                "Could not parse Python code from your response. "
                "Please write code inside ```repl\\n...\\n``` blocks. "
                "Write ONLY Python code, no explanations."
            )
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Error: {error_msg}"})

            turn_record["error"] = error_msg
            trajectory.turns.append(turn_record)

            if verbose:
                logger.warning(f"Turn {iteration + 1}: No parseable code found")
            continue

        if verbose:
            logger.info(f"Parsed code:\n{code}")

        # 5. Execute in REPL
        result = repl_execute(state, code, timeout=code_timeout)

        turn_record["stdout"] = result.stdout
        turn_record["stderr"] = result.stderr
        turn_record["error"] = result.error
        turn_record["terminated"] = result.terminated
        turn_record["time"] = time.time() - iter_t0

        if verbose:
            if result.stdout:
                logger.info(f"STDOUT:\n{result.stdout[:2000]}")
            if result.error:
                logger.warning(f"ERROR:\n{result.error}")

        # 6. Build next message
        messages.append({"role": "assistant", "content": response})

        if result.terminated:
            trajectory.answer = result.answer
            trajectory.terminated = True
            turn_record["answer"] = result.answer
            trajectory.turns.append(turn_record)

            if verbose:
                logger.info(f"TERMINATED with answer: {str(result.answer)[:500]}")
            break

        if result.error:
            # Report error and updated state
            error_feedback = f"Error executing code:\n{result.error}\n\n"
            error_feedback += "Current state:\n" + metadata(state)
            messages.append({"role": "user", "content": error_feedback})
        else:
            # Report stdout metadata and updated state
            stdout_meta = stdout_metadata(result.stdout, max_chars=stdout_max_chars)
            state_meta = metadata(state)
            feedback = ""
            if stdout_meta and stdout_meta != "[No output]":
                feedback += f"Output:\n{stdout_meta}\n\n"
            feedback += f"State:\n{state_meta}"
            messages.append({"role": "user", "content": feedback})

        trajectory.turns.append(turn_record)
    else:
        # Hit max iterations without termination
        if verbose:
            logger.warning(f"Hit max iterations ({max_iterations}) without FINAL")
        trajectory.answer = None
        trajectory.terminated = False

    trajectory.total_time = time.time() - t0
    trajectory.messages = list(messages)

    # Collect model stats if available
    if hasattr(model, "total_stats"):
        trajectory.model_stats = model.total_stats()

    return trajectory


def trajectory_to_dict(traj: RLMTrajectory) -> dict:
    """Serialize a trajectory for JSON storage."""
    return {
        "prompt": traj.prompt[:500] + ("..." if len(traj.prompt) > 500 else ""),
        "prompt_length": len(traj.prompt),
        "system_prompt": traj.system_prompt[:200] + "...",
        "answer": traj.answer,
        "terminated": traj.terminated,
        "num_turns": len(traj.turns),
        "total_time": traj.total_time,
        "model_stats": traj.model_stats,
        "turns": traj.turns,
    }
