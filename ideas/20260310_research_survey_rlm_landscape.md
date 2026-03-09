# Research Survey: RLM Landscape as of March 2026
**Date:** 2026-03-10

## Key Papers & Findings

### 1. Original RLM Paper (Zhang et al., arXiv:2512.24601)
- RLM-Qwen3-8B: +28.3% over base Qwen3-8B
- Approaches GPT-5 on three long-context tasks
- Training: SFT on trajectories from 480B teacher → RL left as future work
- **Our CS234 work showed RL (DPO/GRPO-v4) CAN work on top of SFT**

### 2. "Think, But Don't Overthink" (arXiv:2603.02615, March 2026)
**Critical finding for us:** Deeper recursion (depth=2) can HURT performance.
- Depth-1 RLMs improve accuracy on complex tasks
- Depth-2 causes "overthinking" — degrades performance AND exponentially inflates cost
  - 3.6s → 344.5s execution time
- Tested on DeepSeek v3.2, Kimi K2
- Benchmarks: S-NIAH, OOLONG
- **Implication for us:** Don't pursue multi-level recursion initially. Focus on depth-1 with better strategies.

### 3. Prime Intellect's RLM Work (primeintellect.ai/blog/rlm)
**Key details:**
- Models: GPT-5-mini (best), GLM 4.6, INTELLECT-3
- Benchmarks: DeepDive, math-python, Oolong, verbatim-copy
- **They did NOT train yet** — only ablation studies with API calls
- "The true potential of RLM and context folding will be unleashed after being trained via RL"
- **This means we could be FIRST to train a natively recursive model on a strong base!**

**RLMEnv implementation details:**
- Tools restricted to sub-LLMs only
- Parallelized sub-LLM calls via `llm_batch()`
- 8192-char REPL output limit
- 120s per-REPL timeout
- Answer diffusion through iterative dict updates
- Open-source in their `verifiers` repo

### 4. Official RLM Repo (github.com/alexzhang13/rlm)
- General plug-and-play inference library
- Supports various sandboxes
- No training code (only inference)

## Implications for Our Work

1. **We are in a unique position.** Prime Intellect hasn't trained their RLM yet. The original paper only did SFT. Our CS234 work showed RL works. We could be the first to:
   - Train a natively recursive model on a strong base (Qwen3.5-35B-A3B)
   - Release open weights with RL-optimized RLM capabilities

2. **Don't pursue deep recursion.** The reproduction study shows depth-2 hurts. Focus on depth-1 with better prompting and strategies.

3. **Consider `llm_batch()` parallelization.** Prime Intellect found this valuable for efficiency. Could add to our scaffold.

4. **OOLONG is the key benchmark.** Both the original paper and Prime Intellect use it. We should add it.

5. **Verbatim copy** is a good additional benchmark — tests precision of extraction.

## Sources
- Original paper: https://arxiv.org/abs/2512.24601
- Reproduction: https://arxiv.org/abs/2603.02615
- Prime Intellect: https://www.primeintellect.ai/blog/rlm
- Official repo: https://github.com/alexzhang13/rlm
