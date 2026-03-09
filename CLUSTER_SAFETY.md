# CLUSTER_SAFETY.md

This runs on a shared 8×H200 GPU server. Other researchers and services are running on GPUs 0–5. **You may only use GPUs 6 and 7. Do not interact with anything running on other GPUs.**

## Before ANY GPU Operation

```bash
nvidia-smi --id=6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

If either GPU shows memory > 1000 MiB or utilization > 5%, **that GPU is occupied. Do not use it.** If both are occupied, stop and tell the user.

## Every Script That Touches a GPU Must:

1. **Set `CUDA_VISIBLE_DEVICES` before anything else.** Only use free GPUs from the check above.
   - One GPU: `export CUDA_VISIBLE_DEVICES=7` (or 6)
   - Two GPUs: `export CUDA_VISIBLE_DEVICES=6,7` (only if both free)
   
   Set this in the environment, not just Python. If you forget, PyTorch claims GPU 0 and kills someone's job.

2. **Verify isolation:**
   ```python
   import torch; assert torch.cuda.device_count() <= 2, f"CUDA_VISIBLE_DEVICES not set"
   ```

3. **Clean up on exit.** Kill all processes you spawned. After your job finishes, verify GPUs are released:
   ```bash
   nvidia-smi --id=6,7 --query-gpu=index,memory.used --format=csv,noheader,nounits
   ```

## Never Do These Things

- **Never use GPUs 0–5.** Never interact with processes on them. Never `curl` their servers. They are not ours.
- **Never spawn mass parallel processes.** We crashed the cluster once with hundreds of concurrent requests. Keep everything sequential or with a small semaphore (max 4 concurrent).
- **Never start inference servers that listen on network ports.** No sglang, no vLLM server mode. Load models in-process with `transformers` or `vllm.LLM()`. Servers risk port conflicts and orphan processes.
- **Never make external API calls.** No OpenAI, no Fireworks, no Anthropic, no external anything. Zero dollars. Everything runs locally on our GPUs.
- **Never download models > 10GB without asking the user first.** Large downloads saturate shared network.