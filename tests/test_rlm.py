"""Tests for the RLM loop with mock model."""

import sys
import logging

sys.path.insert(0, ".")

from scaffold.llm_query import MockModel
from scaffold.rlm import rlm, parse_repl_code, trajectory_to_dict

logging.basicConfig(level=logging.INFO)

SIMPLE_SYSTEM_PROMPT = """You are an RLM. You have access to a Python REPL with:
- `context`: the full input text
- `llm_query(prompt)`: call a sub-LLM
- `FINAL(answer)`: return a literal answer
- `FINAL_VAR(var_name)`: return the value of a REPL variable

Write Python code inside ```repl blocks. No explanations."""


def test_parse_repl_code():
    """Code extraction from various formats."""
    # Standard ```repl block
    assert parse_repl_code('```repl\nprint("hi")\n```') == 'print("hi")'

    # ```python fallback
    assert parse_repl_code('```python\nx = 1\n```') == 'x = 1'

    # Generic ``` block
    assert parse_repl_code('```\ny = 2\n```') == 'y = 2'

    # Raw Python (no fences)
    assert parse_repl_code('context[:100]') is not None

    # No code at all
    assert parse_repl_code("I'll help you with that!") is None

    # Explanations around code
    code = parse_repl_code("Let me check:\n```repl\nprint(len(context))\n```\nDone.")
    assert code == "print(len(context))"


def test_single_turn_final():
    """Model produces FINAL() in one turn."""
    responses = [
        '```repl\nFINAL("hello world")\n```'
    ]
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="What is the greeting?",
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    assert traj.terminated
    assert traj.answer == "hello world"
    assert len(traj.turns) == 1


def test_multi_turn_with_context():
    """Model inspects context, then answers."""
    responses = [
        '```repl\nfirst_10 = context[:10]\nprint(first_10)\n```',
        '```repl\nFINAL(first_10)\n```',
    ]
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="ABCDEFGHIJ" + "X" * 1000,
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    assert traj.terminated
    assert traj.answer == "ABCDEFGHIJ"
    assert len(traj.turns) == 2


def test_final_var():
    """Model uses FINAL_VAR to return a computed variable."""
    responses = [
        '```repl\nresult = len(context)\n```',
        '```repl\nFINAL_VAR("result")\n```',
    ]
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="A" * 500,
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    assert traj.terminated
    assert traj.answer == "500"


def test_sub_calls():
    """Model uses llm_query for sub-calls."""
    responses = [
        # Root turn: chunk context and sub-call
        '```repl\nchunks = [context[i:i+100] for i in range(0, len(context), 100)]\nresults = []\nfor c in chunks:\n    r = llm_query(f"Summarize: {c}")\n    results.append(r)\nanswer = "\\n".join(results)\n```',
        # Root turn: return answer
        '```repl\nFINAL_VAR("answer")\n```',
    ]

    # Sub-call responses
    sub_responses = [
        "Summary of chunk 1",
        "Summary of chunk 2",
        "Summary of chunk 3",
    ]

    # Interleave: root responses first, then sub-calls happen during execution
    # MockModel returns responses in order, so root and sub-calls share the queue
    # Root call 0 → responses[0] (chunk code)
    # Sub-call 0 → responses[1] ("Summary of chunk 1")
    # Sub-call 1 → responses[2] ("Summary of chunk 2")
    # Sub-call 2 → responses[3] ("Summary of chunk 3")
    # Root call 1 → responses[4] (FINAL_VAR)
    all_responses = [
        responses[0],
        *sub_responses,
        responses[1],
    ]

    model = MockModel(responses=all_responses)

    traj = rlm(
        prompt="A" * 300,
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    assert traj.terminated
    assert "Summary of chunk 1" in traj.answer
    assert "Summary of chunk 3" in traj.answer


def test_error_recovery():
    """Model gets an error, then fixes it."""
    responses = [
        '```repl\nprint(undefined_var)\n```',  # Will error
        '```repl\nresult = "fixed"\nFINAL(result)\n```',  # Recovery
    ]
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="test",
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    assert traj.terminated
    assert traj.answer == "fixed"
    assert traj.turns[0]["error"] is not None  # First turn had error


def test_max_iterations():
    """Stops at max_iterations if never terminates."""
    responses = [
        '```repl\nprint("still going")\n```',
    ] * 5
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="test",
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=3,
        verbose=False,
    )

    assert not traj.terminated
    assert traj.answer is None
    assert len(traj.turns) == 3


def test_no_code_parsed():
    """Model outputs text without code blocks — gets error feedback."""
    responses = [
        "I'll help you with that! Let me think...",  # No code
        '```repl\nFINAL("ok")\n```',  # Fixed
    ]
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="test",
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    assert traj.terminated
    assert traj.answer == "ok"
    assert traj.turns[0].get("error") is not None  # Parse error


def test_trajectory_serialization():
    """Trajectory can be serialized to dict."""
    responses = ['```repl\nFINAL("done")\n```']
    model = MockModel(responses=responses)

    traj = rlm(
        prompt="A" * 1000,
        model=model,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        max_iterations=5,
        verbose=False,
    )

    d = trajectory_to_dict(traj)
    assert d["terminated"]
    assert d["answer"] == "done"
    assert d["num_turns"] == 1
    assert d["prompt_length"] == 1000


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
