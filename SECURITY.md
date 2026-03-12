# Security Considerations

## REPL Execution

The RLM scaffold executes model-generated Python code in a persistent REPL (`scaffold/repl.py`). This is by design — the model must write and execute code to process long contexts.

### Sandboxing

The REPL implements several safety measures:

- **Module whitelist:** Only approved modules can be imported: `re`, `math`, `statistics`, `collections`, `itertools`, `json`, `csv`, `pandas`, `numpy`, `sklearn`, `scipy`, `ast`, and a few others
- **Blocked builtins:** `exec`, `eval`, `compile` are removed from the execution namespace
- **No file system access:** `open`, `os`, `sys`, `subprocess` are not available
- **No network access:** `urllib`, `requests`, `socket` are not importable

### Known Limitations

- The REPL uses Python's `exec()` internally, which cannot be fully sandboxed in CPython
- A sufficiently adversarial model could potentially escape the sandbox via Python internals
- For production use, run the REPL in an isolated container with no network access

### Recommendations

1. Only use models you trust (the fine-tuned checkpoint or the base Qwen model)
2. Do not pass untrusted user input directly as `context` without additional sandboxing
3. For deployment, use Docker or similar containerization
4. Monitor REPL output for unexpected behavior

## API Keys

- Never commit API keys to the repository
- Use `.env` files (excluded via `.gitignore`) or environment variables
- The `.env.example` file provides a template

## Training Data

- Training data in `data/` is excluded from the repository via `.gitignore`
- SFT training data contains model-generated code trajectories, not user data
