# TINKER.md — Tinker API Reference for RLM Training

Tinker is a training API by Thinking Machines Lab. You write a training loop locally (CPU only); Tinker executes forward/backward/optimization on distributed GPU clusters. LoRA-only fine-tuning.

Currently, the codebase is structured for single-GPU training and things. You must migrate everything over to use Tinker. You must do this in teh most natural and intuitive way possible, Tinker shoudl have endpoints and functionality for everything. If something doesn't seem in this doc, you must brows the docs yourself to find the answer directly to find the best way to do something in Tinker. 

## Where to Find Answers

**For any Tinker API question not covered here, browse the docs directly:**

- **Main docs:** https://tinker-docs.thinkingmachines.ai/
- **Cookbook (code examples):** https://github.com/thinking-machines-lab/tinker-cookbook
- **Console (manage runs):** https://console.thinkingmachines.ai/

The docs cover: Training and Sampling, Loss Functions, Saving and Loading, Downloading Weights, Publishing Weights, Async and Futures, Model Lineup, full API Reference (ServiceClient, TrainingClient, SamplingClient, RestClient, APIFuture, Parameters, Exceptions), OpenAI-compatible APIs, Rendering, LoRA Primer, Supervised Learning (basic, training loop, hyperparams, prompt distillation, sweep case study), Reinforcement Learning (basic, environments, training loop, hyperparams, sequence extension), Preferences (DPO guide, RLHF example), Evaluations, Completers, Under the Hood, and Development Tips.

**When stuck:** fetch the relevant doc page. The cookbook's `llms.txt` at `https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/llms.txt` is a single-file dump of all docs — useful for searching.

## Setup

```bash
pip install tinker transformers torch
# or: uv pip install tinker transformers torch

# Cookbook (has renderers, hyperparam utils, example recipes):
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook && pip install -e .
```

**API key** is stored in the project root `.env` file:
```
TINKER_API_KEY=<key>
```

Load it in scripts:
```python
from dotenv import load_dotenv
load_dotenv()  # reads .env from project root
# tinker client picks up TINKER_API_KEY from environment automatically
```

Or export directly: `export TINKER_API_KEY=$(grep TINKER_API_KEY .env | cut -d= -f2)`

## Core Concepts

### Everything is Remote and Non-Blocking

All API calls return `Future` objects immediately. The actual computation runs on Tinker's GPU cluster.

### Clock Cycles (~10s each)

Tinker's cluster runs in lock-step clock cycles. `forward_backward` and `optim_step` submitted together land on the **same** clock cycle. Always pipeline them:

```python
# CORRECT — same clock cycle, ~10s total
fwd_bwd_future = training_client.forward_backward(data, "cross_entropy")
optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=1e-4))
result = fwd_bwd_future.result()
optim_result = optim_future.result()

# WRONG — two clock cycles, ~20s total
result = training_client.forward_backward(data, "cross_entropy").result()  # wait
optim_result = training_client.optim_step(tinker.AdamParams(learning_rate=1e-4)).result()  # wait again
```

For maximum throughput, pipeline the training loop: submit batch N+1 before waiting for batch N. See docs on Async and Futures for details.

## Client Setup

```python
import tinker

service_client = tinker.ServiceClient()

# Create LoRA training client
training_client = service_client.create_lora_training_client(
    base_model="Qwen3.5-35B-A3B",  # our target model
    rank=32,              # LoRA rank (default 32)
    train_mlp=True,       # train MLP layers (recommended — attention-only underperforms)
    train_attn=True,      # train attention layers
    train_unembed=True,   # train unembedding layer
    seed=42,              # deterministic LoRA init
)

# Resume from checkpoint (weights only, optimizer resets)
training_client = service_client.create_training_client_from_state(
    "tinker://run-id/weights/checkpoint-001"
)

# Resume with optimizer state (Adam momentum preserved)
training_client = service_client.create_training_client_from_state_with_optimizer(
    "tinker://run-id/state/checkpoint-001"
)
```

## Rendering: Messages → Tokens

**Don't hand-tokenize.** Use Tinker's `Renderer` classes — they handle chat templates, per-token loss weights for SFT, stop sequences, and response parsing. They match HuggingFace's `apply_chat_template` output, which matters for OpenAI-compatible endpoint compatibility.

```python
from tinker_cookbook import renderers, tokenizer_utils

tokenizer = tokenizer_utils.get_tokenizer("Qwen3.5-35B-A3B")
renderer = renderers.get_renderer("qwen3", tokenizer)
# For thinking-disabled: renderers.get_renderer("qwen3_disable_thinking", tokenizer)
```

### For SFT: `build_supervised_example`

Returns tokens AND per-token weights (0 = prompt, 1 = completion). Only the **final** assistant message gets weight=1.

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": metadata_text},
    {"role": "assistant", "content": "```repl\nchunks = ...\n```"},
]
model_input, weights = renderer.build_supervised_example(messages)
# model_input → tinker.ModelInput (ready for forward_backward)
# weights → list of 0s and 1s per token
```

### For RL / Inference: `build_generation_prompt`

Builds a prompt for sampling (everything up to where assistant starts generating):

```python
prompt = renderer.build_generation_prompt(messages[:-1])  # remove last assistant msg
stop_sequences = renderer.get_stop_sequences()  # e.g., [151645] for <|im_end|>
```

### Parsing Sampled Output

```python
sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
# sampled_message = {"role": "assistant", "content": "..."}
```

## Training: forward_backward

### SFT (cross_entropy)

```python
import numpy as np

model_input, weights = renderer.build_supervised_example(messages)

datum = tinker.Datum(
    model_input=model_input,
    loss_fn_inputs={
        "weights": tinker.TensorData.from_numpy(np.array(weights, dtype=np.float32)),
    }
)

fwd_bwd = training_client.forward_backward([datum], "cross_entropy")
optim = training_client.optim_step(tinker.AdamParams(learning_rate=2e-4))
result = fwd_bwd.result()
# result.loss_fn_outputs[i]['logprobs'] — per-token logprobs
```

### RL: Built-in Loss Functions

| Loss | Use Case | Key Inputs |
|------|----------|------------|
| `cross_entropy` | SFT | `weights` (0/1) |
| `importance_sampling` | REINFORCE / GRPO | `target_tokens`, `logprobs` (from sampling), `advantages` |
| `ppo` | PPO | Same + `clip_eps` in `loss_fn_config` |
| `cispo` | CISPO | Similar to PPO |
| `dro` | Distributionally robust | Similar |

All losses are **token-level**. Tensors have shape `(N,)` where N = `model_input.length`.

```python
import torch

datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": tinker.TensorData.from_torch(torch.tensor(sampling_logprobs)),
        "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
    }
)
result = training_client.forward_backward([datum], "importance_sampling")
```

### Custom Loss Functions (DPO, etc.)

For losses not in the built-in set, use `forward_backward_custom`. It takes a Python function that receives `(data, logprobs)` and returns `(loss, metrics)`. Costs 1.5× FLOPs (extra forward pass).

```python
def dpo_loss(data, logprobs):
    chosen_lp = logprobs[0].sum()
    rejected_lp = logprobs[1].sum()
    ref_chosen = data[0].loss_fn_inputs["ref_logprobs"].sum()
    ref_rejected = data[1].loss_fn_inputs["ref_logprobs"].sum()
    beta = 0.1
    loss = -torch.nn.functional.logsigmoid(
        beta * ((chosen_lp - ref_chosen) - (rejected_lp - ref_rejected))
    )
    return loss, {"loss": loss.item()}

loss, metrics = training_client.forward_backward_custom([chosen_datum, rejected_datum], dpo_loss)
```

Tinker also has a full DPO recipe in the cookbook: `tinker_cookbook/recipes/preference/train_dpo.py`. See the DPO guide in docs for details on beta tuning, dataset setup, and evaluation.

## Sampling (Generation)

```python
# From training (transfers current weights)
sampling_client = training_client.save_weights_and_get_sampling_client(name="my-model")

# From saved weights
sampling_client = service_client.create_sampling_client(
    model_path="tinker://run-id/weights/checkpoint-001"
)

# Base model (no fine-tuning)
sampling_client = service_client.create_sampling_client(base_model="Qwen3.5-35B-A3B")
```

### Generate Completions

```python
prompt = renderer.build_generation_prompt(messages)
stop_sequences = renderer.get_stop_sequences()

params = tinker.SamplingParams(
    max_tokens=1024,
    temperature=0.8,
    stop=stop_sequences,
)

result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
for seq in result.sequences:
    msg, ok = renderer.parse_response(seq.tokens)
    print(msg["content"])
```

### K Completions for GRPO

```python
result = sampling_client.sample(
    prompt=prompt, sampling_params=params, num_samples=8,  # K=8
).result()
# result.sequences[i].tokens — token IDs
# result.sequences[i].logprobs — per-token logprobs (for importance sampling)
```

### Logprobs on Prompt Tokens

```python
result = sampling_client.sample(
    prompt=prompt, sampling_params=params, num_samples=1,
    include_prompt_logprobs=True, topk_prompt_logprobs=5,
).result()
# result.topk_prompt_logprobs — list of [(token_id, logprob), ...] per position
```

## Saving & Loading

```python
# Save weights (for sampling or resuming)
sampling_client = training_client.save_weights_and_get_sampling_client(name="ckpt-100")
# Weights at: tinker://run-id/weights/ckpt-100

# Save full state (weights + optimizer momentum)
training_client.save_state(name="state-100")

# Download weights locally
rest_client = service_client.create_rest_client()
url = rest_client.get_checkpoint_archive_url_from_tinker_path(sampling_client.model_path)
with open("checkpoint.tar.gz", "wb") as f:
    f.write(url.result())
```

## LoRA: What to Know

- Tinker is **LoRA only** — no full fine-tuning.
- Default rank = 32. For RL, small ranks work fine (even rank 8-16). For large SFT datasets, increase to 64-128.
- **LoRA needs much larger LR than full fine-tuning** — typically 20-100× larger. Use the cookbook utility:

```python
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr, get_lora_param_count
factor = get_lora_lr_over_full_finetune_lr("Qwen3.5-35B-A3B")
param_count = get_lora_param_count("Qwen3.5-35B-A3B", lora_rank=32)
```

- **Always train all layer types** (`train_mlp=True, train_attn=True, train_unembed=True`). Attention-only LoRA underperforms even at matched parameter count.
- Optimal LR does **not** depend on LoRA rank — same LR works across ranks.
- For RL: LoRA performs equivalently to full fine-tuning even at small ranks. RL requires very low capacity.

## RL Hyperparameters

- **Learning rate:** Use the LoRA LR scaling utility. This is the most important hyperparameter.
- **group_size (K):** Number of rollouts per prompt. We use K=8 for GRPO. Increase if you have few unique prompts.
- **batch_size:** Number of unique prompts per step. Scale LR proportionally to √batch_size.
- **num_substeps:** Optimizer updates per batch. Default 1 (on-policy). Try 2-4 with PPO if experimenting.
- **KL divergence:** Monitor `kl_sample_train_v1` / `kl_sample_train_v2`. Stable training has KL < 0.01.

### Sequence Extension (Multi-Turn RL)

**Critical for RLM training** — our RLM loop is inherently multi-turn (multiple REPL iterations).

The **extension property** means each successive observation contains the previous as a prefix. When it holds:
- Multiple turns merge into a single training Datum
- KV-cache reuse during sampling
- Compute scales O(T) instead of O(T²)

For Qwen3 with thinking:
- `strip_thinking_from_history=False` → extension holds → O(T) — **use this for RLM**
- `strip_thinking_from_history=True` (default) → extension breaks → O(T²)

Since our RLM trajectories are multi-turn (typically 3-8 REPL iterations), the extension property matters significantly. Use `strip_thinking_from_history=False` and check `renderer.has_extension_property`.

Consider **periodic compaction** for very long trajectories: keep thinking visible for N turns, then strip old thinking blocks. Breaks extension only every N turns instead of every turn.

## RL Environments (Cookbook Pattern)

The cookbook defines an `Env` interface for RL environments. For RLM training, we'll likely want to define our own environment where:
- `initial_observation` builds the metadata from a benchmark task
- `step` receives the model's code, executes it in the REPL, returns stdout/error metadata
- Reward is computed from the final answer

The key classes: `Env`, `EnvGroupBuilder` (creates K envs for group), `RLDataset` (dataset of env builders).

See `tinker_cookbook/rl/types.py` for interfaces and `tinker_cookbook/recipes/twenty_questions/` for an example multi-step environment. The cookbook's `rl/train.py` has the optimized training loop.

**Browse the RL docs for details:** https://tinker-docs.thinkingmachines.ai/rl

## DPO on Tinker

Tinker supports DPO natively via `forward_backward_custom` or the cookbook's DPO recipe.

Key settings:
- `dpo_beta=0.1` is a good default (controls preference learning strength)
- Use **lower LR** than SFT (1e-5 to 1e-6)
- Base model should be in-distribution with the preference data — do SFT first
- Metrics to watch: `dpo_loss`, `accuracy` (implicit reward model accuracy), `margin` (chosen vs rejected reward gap)

Cookbook recipe: `python -m tinker_cookbook.recipes.preference.train`

## Available Models

### Our Targets
| Model | Type | Active | Notes |
|-------|------|--------|-------|
| **`Qwen3.5-35B-A3B`** | **MoE** | **3B** | **Primary target** |
| `Qwen3.5-4B` | Dense | 4B | Cheap debugging |
| `Qwen3.5-27B` | Dense | 27B | Fallback |
| `Qwen3.5-397B-A17B` | MoE | 17B | Teacher for trajectory gen |

### Full Lineup (check docs for latest)
Qwen3.5: 4B, 27B, 35B-A3B (MoE), 397B-A17B (MoE)
Qwen3: 4B-Instruct, 8B-Base, 8B, 30B-A3B-Base (MoE), 30B-A3B (MoE), 30B-A3B-Instruct, 32B, 235B-A22B-Instruct (MoE)
Qwen3-VL: 30B-A3B-Instruct, 235B-A22B-Instruct
Llama: 3.2-1B, 3.2-3B, 3.1-8B, 3.1-8B-Instruct, 3.1-70B, 3.3-70B-Instruct

Use `service_client.get_server_capabilities().supported_models` for the live list.

**MoE models are more cost-effective** — you pay for active parameters, not total.

## OpenAI-Compatible Endpoint

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://api.thinkingmachines.ai/v1",
    api_key=os.environ["TINKER_API_KEY"],
)
response = client.completions.create(
    model="tinker://run-id/weights/ckpt-100",
    prompt="...", max_tokens=20, temperature=0.0,
)
```

**Important:** This endpoint uses HuggingFace chat templates to tokenize. If you trained with a non-HF-compatible renderer, your model won't work correctly here. The default renderers (`qwen3`, `llama3`) match HF behavior.

## Common Pitfalls

1. **Don't wait between forward_backward and optim_step** — submit together, wait after both
2. **Refresh sampling_client after training** — it uses weights from creation time, not live
3. **Use the renderer, not manual tokenization** — ensures HF compatibility and correct loss weights
4. **LoRA LR is NOT full-FT LR** — use 20-100× higher. Use the cookbook utility.
5. **forward_backward_custom costs 1.5×** — prefer built-in losses when possible
6. **Sampling is non-deterministic** even at temperature=0.0 (batching on their end)
7. **Online RL has serial dependency** — need completion → score → train. This is the bottleneck for RLM training on Tinker.