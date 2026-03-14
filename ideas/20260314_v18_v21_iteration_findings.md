# V18-V21 Iteration Findings: SFT Data Scaling Limits (2026-03-14)

## Summary
After confirming V17 RS-SFT as best model (71.6% avg), attempted 4 different approaches to beat it. All failed. V17 remains the best model.

## Bug Fix: Missing doc_classify + multi_niah Trajectories
`mine_eval_trajectories.py` only checked `task_result.get("score", 0)`, but:
- doc_classify uses `accuracy` field
- multi_niah uses `recall` field

**All V10-V17 models trained with ZERO mined doc_classify/multi_niah trajectories.** Fix added 858 + 798 = 1,656 new trajectories.

## Models Trained

### V18 RS-SFT (from scratch, 4,660 samples)
- Same config as V17 but with fixed mining (adds doc_classify + multi_niah)
- 3 epochs, lr 2e-5, 873 steps, 3.54h
- **Result: 69.7%** (-1.9pp vs V17)
- Wins: doc_classify +7.7pp, oolong +20pp, NIAH +5pp, verbatim_copy +10pp
- Losses: code_debug -20pp, hard_multi_hop -20pp, multi_hop_qa -10pp, event_counting -11.4pp, KV -10.4pp
- **Takeaway:** Adding 1,000+ new samples diluted reasoning-task capabilities

### V18 Epoch 2 Checkpoint (step 580)
- **Result: 66.5%** (-5.1pp vs V17)
- Doc-classify peaked at 94.9% (!) but KV retrieval collapsed to 41.7%, cross-doc to 31.6%
- **Takeaway:** Earlier checkpoints not better — different tradeoff profile, lower average

### V19 (V17 + 1 epoch V18 data, lr 5e-6)
- Fine-tune V17 with V18's full dataset at low LR, single epoch
- **Result: 69.5%** (-2.1pp vs V17)
- Wins: multi_niah +1.7pp, dataframe_qa +4pp, notebook_qa +13.3pp, verbatim_copy +20pp, KV +1.3pp, oolong +10pp
- Losses: NIAH -5pp, hard_niah -3.4pp, code_debug -13.3pp, multi_hop -10pp, hard_multi_hop -20pp, event_counting -15.8pp, cross_doc -8.7pp
- **Takeaway:** Even low-LR fine-tuning damages V17's reasoning capabilities

### V20 (reasoning-boosted data, from scratch, 4,277 samples)
- Boosted reasoning tasks (hard_multi_hop 223→300, cross_doc 240→300, oolong 90→150)
- Reduced easy tasks (doc_classify 400→200, multi_niah 400→200)
- 3 epochs, lr 2e-5, 801 steps, 3.24h
- **Result: 69.4%** (-2.2pp vs V17)
- Multi-hop QA collapsed to 45% (below base!), hard_niah back to base 83.3%
- But: notebook_qa 90% (NEW BEST), KV 88.9% (NEW BEST), verbatim_copy 100%
- **Takeaway:** Data rebalancing doesn't help; the issue isn't proportions

### V21 (V17 + targeted 1,732 samples, lr 3e-6)
- Focused on V17's weak areas: doc_classify, notebook_qa, verbatim_copy, oolong
- Added small amounts of V17 strengths to prevent forgetting
- 1 epoch, very low LR
- **Result: 64.8%** (-6.8pp vs V17)
- Cross-doc collapsed to 18.5% (worst ever), KV to 63.9%, multi_niah to 88.1%
- **Takeaway:** Even minimal targeted fine-tuning causes catastrophic forgetting on other tasks

## Key Insights

1. **SFT creates unavoidable benchmark tradeoffs.** Every model that gains on some benchmarks loses on others. The tradeoff frontier may be hard to push via data alone.

2. **V17's balance is remarkably hard to reproduce.** Its specific mix of 3,644 samples at 3 epochs with the V17-era mining (missing doc_classify/multi_niah) apparently created an ideal balance.

3. **Reasoning tasks are fragile.** code_debug, multi_hop_qa, hard_multi_hop consistently degrade when training data changes. These may require specific trajectory patterns.

4. **The oracle at 77.7% shows significant untapped potential.** Different models excel at different tasks. An ensemble or routing approach could capture this.

5. **Forgetting is severe even at very low learning rates.** V21 used lr 3e-6 (10x lower than V17's training) for only 108 steps and still lost 6.8pp average.

## What Would Actually Help?
- DPO/RLHF to avoid negative transfer (tried but Tinker API format issue blocked it)
- Per-task model routing at inference time
- Much larger trajectory collection (V16 collection still running: 40 tasks × 8 × 14 benchmarks)
- Training with explicit multi-task loss weighting
- Longer context training to address oolong/external benchmarks
