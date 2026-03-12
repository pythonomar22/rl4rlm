# Strategy Evaluation Final Results (2026-03-12)

## Setup
- Seed-offset 10000 for both base and V11-s5 (same tasks)
- Per-benchmark best strategy prompts validated on held-out set (seed 30000, N=10)
- Strategies: table_preserve, notebook_sequential, lookup_thorough, multi_hop_decompose, event_extract_python, code_debug_focused

## Results (14 benchmarks, identical seeds)

| Benchmark | Base+Strat | V11-s5+Strat | Delta |
|-----------|-----------|-------------|-------|
| code_debug | 40.0% | **58.9%** | +18.9pp |
| cross_doc_compare | **36.6%** | 28.8% | -7.8pp |
| dataframe_qa | **75.0%** | 74.0% | -1.0pp |
| doc_classify | 66.0% | **91.8%** | +25.8pp |
| event_counting | 40.1% | **53.4%** | +13.3pp |
| hard_multi_hop | 80.0% | **90.0%** | +10.0pp |
| hard_niah | 86.7% | **90.0%** | +3.3pp |
| key_value_retrieval | 41.7% | **50.8%** | +9.2pp |
| multi_hop_qa | 55.0% | **60.0%** | +5.0pp |
| multi_niah | **100.0%** | 95.0% | -5.0pp |
| niah | **65.0%** | 60.0% | -5.0pp |
| notebook_qa | 66.7% | **70.0%** | +3.3pp |
| oolong | **10.0%** | 0.0% | -10.0pp |
| verbatim_copy | 40.0% | 40.0% | 0.0pp |
| **Average** | **57.3%** | **61.6%** | **+4.3pp** |

V11-s5 wins: 8, loses: 5, ties: 1

## Key Findings

1. **V11-s5 +4.3pp with strategies** — third independent confirmation of training benefit
   - Cross-eval set 1 (seed 0): +2.1pp
   - Cross-eval set 2 (deterministic): +6.3pp
   - This set (seed 10000): +4.3pp
   - **Weighted average across 3 sets: ~+4.2pp**

2. **Strategies are NOT uniformly beneficial**
   - Some dramatically help: hard_multi_hop +40pp, code_debug +19pp
   - Some catastrophically hurt: verbatim_copy -60pp, multi_hop_qa -25pp
   - This is why "base+strategy" averages only 57.3% (below base's 60.9% on seed 0)

3. **Seed-offset 10000 tasks are harder** than seed 0
   - Base+strat 57.3% (seed 10000) vs Base 60.9% (seed 0) — 3.6pp gap
   - Verbatim copy: 100% (seed 0) → 40% (seed 10000) — some task instances much harder

4. **V11-s5 biggest gains with strategies**: doc_classify +25.8pp, code_debug +18.9pp, event +13.3pp
   These are exactly the task types in the training distribution.

## RS-SFT Collection Complete
- 960 trajectories collected (6 benchmarks × 20 tasks × 8 attempts)
- 547 correct (57.0%), 427 high-quality (score >= 0.9)
- Saved to data/sft/rs_sft_v1.jsonl
- Ready for SFT training if we want to try SFT→GRPO pipeline

## Checkpoint Download SOLVED
- `save_weights_for_sampler(name=...)` returns a `.path` attribute
- Use that path with `get_checkpoint_archive_url_from_tinker_path()`
- V11-s5 downloaded: 2.1 GB LoRA adapter (rank 32, safetensors format)
- Saved to checkpoints/v11_s5/ (adapter_config.json + adapter_model.safetensors)
