"""Tests for the REPL environment."""

import sys
sys.path.insert(0, ".")

from scaffold.repl import init_repl, repl_execute, metadata, stdout_metadata


def test_basic_execution():
    """Code runs and stdout is captured."""
    state = init_repl("Hello world this is a test prompt")
    result = repl_execute(state, 'print("hello from repl")')
    assert result.stdout.strip() == "hello from repl"
    assert result.error is None
    assert not result.terminated


def test_context_available():
    """The full prompt is accessible as `context`."""
    prompt = "A" * 10000
    state = init_repl(prompt)
    result = repl_execute(state, "x = len(context)\nprint(x)")
    assert result.stdout.strip() == "10000"
    assert state.namespace["x"] == 10000


def test_context_slicing():
    """Model can slice context — the core RLM operation."""
    prompt = "needle" + "X" * 5000 + "haystack"
    state = init_repl(prompt)
    result = repl_execute(state, 'first_6 = context[:6]\nprint(first_6)')
    assert result.stdout.strip() == "needle"
    assert state.namespace["first_6"] == "needle"


def test_persistence_across_turns():
    """Variables survive across exec calls."""
    state = init_repl("test")
    repl_execute(state, "a = 1")
    repl_execute(state, "b = 2")
    result = repl_execute(state, "print(a + b)")
    assert result.stdout.strip() == "3"
    assert state.turn == 3


def test_final_terminates():
    """FINAL() sets answer and terminates."""
    state = init_repl("test prompt")
    result = repl_execute(state, 'FINAL("the answer is 42")')
    assert result.terminated
    assert result.answer == "the answer is 42"
    assert state.terminated


def test_final_var_terminates():
    """FINAL_VAR() reads a REPL variable and terminates."""
    state = init_repl("test prompt")
    repl_execute(state, 'my_answer = "computed result"')
    result = repl_execute(state, 'FINAL_VAR("my_answer")')
    assert result.terminated
    assert result.answer == "computed result"


def test_final_var_missing_variable():
    """FINAL_VAR() with nonexistent variable gives error, not termination."""
    state = init_repl("test")
    result = repl_execute(state, 'FINAL_VAR("nonexistent")')
    assert not result.terminated
    assert result.error is not None
    assert "nonexistent" in result.error


def test_already_terminated():
    """After termination, further execution returns early."""
    state = init_repl("test")
    repl_execute(state, 'FINAL("done")')
    result = repl_execute(state, 'print("this should not run")')
    assert result.terminated
    assert result.stdout == ""


def test_syntax_error():
    """Syntax errors are caught and reported."""
    state = init_repl("test")
    result = repl_execute(state, "def f(\n  broken")
    assert result.error is not None
    assert "SyntaxError" in result.error
    assert not result.terminated


def test_runtime_error():
    """Runtime errors are caught, REPL continues."""
    state = init_repl("test")
    result = repl_execute(state, "x = 1/0")
    assert result.error is not None
    assert "ZeroDivision" in result.error
    # REPL should still work after error
    result2 = repl_execute(state, "print('still alive')")
    assert result2.stdout.strip() == "still alive"
    assert result2.error is None


def test_llm_query_callable():
    """llm_query is available in the REPL."""
    def mock_llm(prompt_str: str) -> str:
        return f"answer to: {prompt_str[:20]}"

    state = init_repl("big context here", llm_query_fn=mock_llm)
    result = repl_execute(state, 'response = llm_query("What is 2+2?")\nprint(response)')
    assert "answer to: What is 2+2?" in result.stdout
    assert state.namespace["response"] == "answer to: What is 2+2?"


def test_imports_work():
    """Can import standard library modules."""
    state = init_repl("test")
    result = repl_execute(state, "import re\nm = re.search(r'\\d+', 'abc123def')\nprint(m.group())")
    assert result.stdout.strip() == "123"


def test_metadata_format():
    """Metadata includes length, prefix, functions."""
    prompt = "Hello world " * 100
    state = init_repl(prompt)
    m = metadata(state)
    assert "1200 characters" in m
    assert "Hello world" in m
    assert "llm_query" in m
    assert "FINAL" in m
    assert "context" in m


def test_stdout_metadata_truncation():
    """Long stdout gets truncated."""
    long_output = "x" * 5000
    m = stdout_metadata(long_output, max_chars=100)
    assert len(m) < 5000
    assert "truncated" in m
    assert "5000 total" in m


def test_multiline_code():
    """Multi-line code blocks work (loops, functions)."""
    state = init_repl("test")
    code = """
results = []
for i in range(5):
    results.append(i * 2)
print(results)
"""
    result = repl_execute(state, code)
    assert result.stdout.strip() == "[0, 2, 4, 6, 8]"


def test_llm_query_in_loop():
    """llm_query can be called in a loop — the core recursive pattern."""
    call_log = []

    def mock_llm(prompt_str: str) -> str:
        call_log.append(prompt_str)
        return f"processed chunk {len(call_log)}"

    state = init_repl("AAABBBCCC", llm_query_fn=mock_llm)
    code = """
chunks = [context[i:i+3] for i in range(0, len(context), 3)]
answers = []
for chunk in chunks:
    ans = llm_query(f"Process: {chunk}")
    answers.append(ans)
final_answer = " | ".join(answers)
print(final_answer)
"""
    result = repl_execute(state, code)
    assert len(call_log) == 3
    assert "processed chunk 1" in result.stdout
    assert "processed chunk 3" in result.stdout


def test_history_recorded():
    """Execution history is tracked."""
    state = init_repl("test")
    repl_execute(state, "x = 1")
    repl_execute(state, "y = 2")
    assert len(state.history) == 2
    assert state.history[0]["code"] == "x = 1"
    assert state.history[1]["code"] == "y = 2"


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
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
