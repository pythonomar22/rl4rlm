"""
Persistent Python REPL for RLM execution.

The REPL holds the full prompt as `context` and provides:
- llm_query(prompt_str) -> str: recursive sub-call
- FINAL(answer) -> terminates with literal answer
- FINAL_VAR(var_name) -> terminates with value of REPL variable
- print() -> captured stdout

The full prompt NEVER enters the LLM's context window. It lives only here.
"""

from __future__ import annotations

import io
import signal
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable


class FinalAnswer(Exception):
    """Raised when FINAL() or FINAL_VAR() is called to terminate execution."""
    def __init__(self, answer: Any):
        self.answer = answer


@dataclass
class REPLResult:
    """Result of executing code in the REPL."""
    stdout: str
    stderr: str
    error: str | None  # Exception traceback if any
    terminated: bool   # True if FINAL/FINAL_VAR was called
    answer: Any = None # The final answer if terminated


@dataclass
class REPLState:
    """Persistent REPL state."""
    namespace: dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    terminated: bool = False
    answer: Any = None
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def context(self) -> str:
        """The full prompt stored in the REPL."""
        return self.namespace.get("context", "")


def _make_final(state: REPLState) -> Callable:
    """Create FINAL() function that terminates with a literal answer."""
    def final(answer: Any) -> None:
        state.terminated = True
        state.answer = str(answer)
        raise FinalAnswer(state.answer)
    return final


def _make_final_var(state: REPLState) -> Callable:
    """Create FINAL_VAR() function that terminates with a REPL variable's value."""
    def final_var(var_name: str) -> None:
        if var_name not in state.namespace:
            raise NameError(f"Variable '{var_name}' not found in REPL. "
                            f"Available: {[k for k in state.namespace if not k.startswith('_')]}")
        state.terminated = True
        state.answer = str(state.namespace[var_name])
        raise FinalAnswer(state.answer)
    return final_var


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("REPL execution timed out")


def init_repl(
    prompt: str,
    llm_query_fn: Callable[[str], str] | None = None,
    timeout: int = 30,
) -> REPLState:
    """
    Initialize REPL with prompt as `context` variable.

    Args:
        prompt: The full input prompt (stored as `context` in the REPL)
        llm_query_fn: The sub-call function. If None, a dummy is used.
        timeout: Max seconds per code execution
    """
    state = REPLState()

    # The full prompt lives ONLY in the REPL namespace
    state.namespace["context"] = prompt

    # Termination functions
    state.namespace["FINAL"] = _make_final(state)
    state.namespace["FINAL_VAR"] = _make_final_var(state)

    # Sub-call function
    if llm_query_fn is not None:
        state.namespace["llm_query"] = llm_query_fn
    else:
        def dummy_llm_query(prompt_str: str) -> str:
            return f"[llm_query not available] prompt was {len(prompt_str)} chars"
        state.namespace["llm_query"] = dummy_llm_query

    # Standard builtins available
    state.namespace["__builtins__"] = __builtins__

    # Store timeout config
    state.namespace["_timeout"] = timeout

    return state


def repl_execute(state: REPLState, code: str, timeout: int | None = None) -> REPLResult:
    """
    Execute code in the persistent REPL.

    Returns REPLResult with captured stdout/stderr and termination status.
    The namespace persists across calls (variables, imports, etc. survive).
    """
    if state.terminated:
        return REPLResult(
            stdout="", stderr="", error="REPL already terminated",
            terminated=True, answer=state.answer,
        )

    timeout = timeout or state.namespace.get("_timeout", 30)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Set up timeout (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    except (ValueError, AttributeError):
        # Not on main thread or not Unix
        pass

    error = None
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, state.namespace)
    except FinalAnswer:
        # Normal termination via FINAL() or FINAL_VAR()
        pass
    except TimeoutError as e:
        error = f"TimeoutError: {e}"
    except Exception:
        error = traceback.format_exc()
    finally:
        # Cancel alarm
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, AttributeError):
            pass

    state.turn += 1

    result = REPLResult(
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        error=error,
        terminated=state.terminated,
        answer=state.answer if state.terminated else None,
    )

    # Record in history
    state.history.append({
        "turn": state.turn,
        "code": code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
        "terminated": result.terminated,
    })

    return result


def metadata(state: REPLState) -> str:
    """
    Build metadata string about REPL state. This is what the LLM sees
    instead of the full prompt.

    Per paper: context type, total char count, chunk lengths,
    available functions, short prefix.
    """
    ctx = state.context
    ctx_len = len(ctx)

    # Short prefix (first 500 chars)
    prefix = ctx[:500]
    if len(ctx) > 500:
        prefix += f"\n... [{ctx_len - 500} more characters]"

    # Available user-defined variables (exclude builtins and internals)
    user_vars = [
        k for k in state.namespace
        if not k.startswith("_") and k not in (
            "context", "llm_query", "FINAL", "FINAL_VAR",
            "__builtins__", "print",
        )
    ]

    parts = [
        f"Context length: {ctx_len} characters",
        f"Context prefix:\n{prefix}",
        f"\nAvailable functions: llm_query(prompt_str), FINAL(answer), FINAL_VAR(variable_name)",
        f"Available variable: context (the full input, {ctx_len} chars)",
    ]

    if user_vars:
        var_summaries = []
        for v in user_vars[:10]:  # Cap at 10
            val = state.namespace[v]
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:200] + f"... [{len(val_str)} total chars]"
            var_summaries.append(f"  {v} = {val_str}")
        parts.append(f"User variables:\n" + "\n".join(var_summaries))

    if state.turn > 0:
        parts.append(f"Turn: {state.turn}")

    return "\n".join(parts)


def stdout_metadata(stdout: str, max_chars: int = 1000) -> str:
    """
    Build metadata string for stdout. Truncate to force use of REPL variables.

    Per paper: first N chars + total length.
    """
    if not stdout:
        return "[No output]"

    if len(stdout) <= max_chars:
        return stdout

    return stdout[:max_chars] + f"\n... [truncated, {len(stdout)} total characters. Use REPL variables to store and access full output.]"
