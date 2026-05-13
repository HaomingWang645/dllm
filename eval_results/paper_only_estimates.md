# Paper-Only H100-Hour Estimates for Replicating Training of 7 Multimodal Diffusion LMs

Generated 2026-05-13. Source for each line is the arXiv paper itself (abstract page + HTML + PDF). When the paper does not disclose hardware/wall-clock, the estimate uses the Chinchilla-style training-FLOPs proxy `6 * N_params * N_tokens` and an H100 effective bf16 throughput of 500 TFLOPS (~50% of 989 TFLOPS theoretical). All such cases are flagged "FLOPs-based estimate" with low confidence.

Conversion rules used:
- A100 (80GB) -> H100 (80GB) bf16: H100-hours = A100-hours * 0.55
- V100 -> H100: H100-hours = V100-hours * 0.30
- L40S -> H100: H100-hours = L40S-hours * 0.35 (L40S ~362 bf16 TFLOPS vs H100 ~989)
- H100: reported directly

---

### VidLaDA (arXiv:2601.17868)
- arxiv: https://arxiv.org/abs/2601.17868
- Training data: 3-stage video curriculum (Appendix D.1.2 / E.1.2):
  - Stage 1 short-clip pretrain: 1.8M samples, 32 frames, 8K context
  - Stage 2 temporal warm-up: 500K samples, 64 frames, 16K context
  - Stage 3 long-form expansion: 500K samples, 64 frames, 16K context, 2-30 min videos
- Model size: 8B parameters (initialized from LLaDA-V)
- Training compute (per paper): GPU type, count, steps, and wall-clock are not disclosed in accessible Appendix text. Only per-stage data scale and LRs (2e-6 / 1e-5) are given.
- **H100-hour estimate to replicate: ~1,000-3,000 H100-hours (FLOPs-based estimate)**
  - FLOPs proxy: ~2.8M video samples * avg ~12K visual+text tokens/sample ~= 34B tokens. 6 * 8e9 * 34e9 = 1.63e21 FLOPs. /(500e12 * 3600) ~= 906 H100-hours for a single pass. Allowing 1.5-3 effective epoch-equivalents across the 3 curriculum stages -> 1.4K-2.7K H100-hours.
- Method: FLOPs estimate from disclosed sample counts; sequence length assumed from frames * tokens-per-frame heuristic for SigLIP/LLaDA-V (~256-512 tokens per frame).
- Confidence: low
- Caveats: hardware/wall-clock missing; video sequence length per sample is heuristic; multi-stage finetune from LLaDA-V means much of the base-model cost is excluded.

---

### Sparse-LaViDa (arXiv:2512.14008)
- arxiv: https://arxiv.org/abs/2512.14008
- Training data: ~20M text-image pairs (LAION-2B, COYO-700M, BLIP3o-60k, ShareGPT4o-Image, plus image understanding/editing sets), 100k SFT steps
- Model size: 10.4B parameters (initialized from LaViDa-O)
- Training compute (per paper): "64 NVIDIA H100 GPUs" * "5 days in total" = 64 * 5 * 24 = **7,680 H100-hours**. Authors explicitly note this is "15% of the LaViDa-O training budget".
- **H100-hour estimate to replicate: 7,680 H100-hours** (post-training only; full from-scratch replication including LaViDa-O base would be ~7,680 / 0.15 ~= 51,200 H100-hours).
- Method: direct from disclosed hardware * wall-clock.
- Confidence: high (for the post-training stage that is the paper's actual contribution).
- Caveats: assumes you start from LaViDa-O weights; replicating from scratch would require the LaViDa-O pretraining (paper estimates 6.66x the disclosed budget).

---

### MMaDA-Parallel (arXiv:2511.09611)
- arxiv: https://arxiv.org/abs/2511.09611
- Training data: 150K thinking-aware image editing+generation pairs (100K edit + 50K gen). SFT: 30K steps at global batch 768. ParaRL: 10K steps on a 15K-example subset, s=3 trajectory points per step.
- Model size: built on MMaDA-MixCoT (LLaDA-8B text backbone + MagVIT-v2 image tokenizer) -> effectively ~8B trainable parameters.
- Training compute (per paper): "32 NVIDIA A100 GPUs" for SFT (30K steps) + ParaRL (10K steps). Wall-clock not given.
- **H100-hour estimate to replicate: ~1,800-3,500 H100-hours**
  - Step-time proxy: 30K SFT steps at batch 768 + 10K RL steps (RL is 3-5x slower per step due to rollout sampling) on an 8B masked-diffusion model on 32 A100s ~ typical 2-4 s/step SFT, 8-15 s/step RL. SFT: 30K * 3 s = 90Ks = 25 h wall; RL: 10K * 10 s = 100Ks = 27.8 h wall. Total ~53 h * 32 GPU = 1,696 A100-hours -> ~933 H100-hours at low end. Padding for setup, eval, and slower diffusion steps: 3,300-6,300 A100-hours = ~1,800-3,500 H100-hours.
- Method: indirect; step-time heuristic for 8B masked diffusion on A100. No wall-clock disclosed.
- Confidence: low-medium
- Caveats: real number depends heavily on per-step cost of MaskGiT-style diffusion with image-token rollouts; could easily 2x in either direction.

---

### Unified Diffusion VLA (arXiv:2511.01718)
- arxiv: https://arxiv.org/abs/2511.01718
- Training data: Stage 1 = large-scale video post-training on Emu3 backbone (data/size not disclosed in paper); Stage 2 = joint image+action SFT on robot datasets (CALVIN, LIBERO, SimplerEnv, plus real-world).
- Model size: Emu3 backbone (~8B; not stated in the paper, taken from Emu3 release).
- Training compute (per paper): Only the real-world fine-tune compute is disclosed: "4 NVIDIA H100 GPUs for 24 hours, a total of 9000 steps" (Appendix D). Stage 1 and main Stage 2 (CALVIN/LIBERO/SimplerEnv) compute are not disclosed.
- **H100-hour estimate to replicate:**
  - Real-world fine-tune only: 4 * 24 = **96 H100-hours** (high confidence, directly stated).
  - Full pipeline (stage 1 video post-training + stage 2 on CALVIN/LIBERO/SimplerEnv + real-world): **~5,000-20,000 H100-hours (FLOPs/heuristic estimate)**. Stage 1 video post-training on an 8B model typically uses tens of B-tokens; if we assume 50B tokens, 6 * 8e9 * 50e9 = 2.4e21 FLOPs -> ~1,330 H100-hours. Stage 2 across three robot benchmarks (CALVIN has ~24K demos, LIBERO ~6K, SimplerEnv eval-only) likely needs 1-3 days on 16-32 H100s each = ~1,000-2,500 H100-hours. Real-world: +96 H100-hours.
- Method: real-world stage is direct; full pipeline is FLOPs+heuristic.
- Confidence: high (real-world only); low (full pipeline).
- Caveats: paper does not disclose stage 1 video data size or any stage 2 main-benchmark compute. Replication needs the CALVIN simulator + LIBERO + SimplerEnv; main numbers depend on those sims. Range is wide.

---

### dVLA (arXiv:2509.25681)
- arxiv: https://arxiv.org/abs/2509.25681
- Training data: LIBERO (4 task suites, ~6K demos total) + real-world data; "same training pipeline as MMaDA".
- Model size: initialized from MMaDA-8B -> ~8B parameters (not explicitly stated by the paper).
- Training compute (per paper): No GPU type, count, batch, LR, or wall-clock disclosed. Only the statement that the MMaDA training pipeline is reused.
- **H100-hour estimate to replicate:**
  - dVLA fine-tune only (LIBERO + real-world): **~200-600 H100-hours (FLOPs/heuristic)**. LIBERO has ~6K demos * ~250 frames * ~512 tokens/frame ~= 770M tokens; with ~3-5 epochs that is ~2.3-3.8B tokens. 6 * 8e9 * 3e9 = 1.44e20 FLOPs -> ~80 H100-hours minimum, with overhead 200-600.
  - If "replicating training" includes the MMaDA-8B base: MMaDA paper reports 64 A100 * 700K steps at batch 1280. Estimating step time ~2.5 s on A100 80GB for 8B diffusion: 700K * 2.5 = 1.75M s = 486 h wall -> 64 * 486 = 31,100 A100-hours -> ~**17,100 H100-hours** for the base, plus 200-600 H100-hours for dVLA fine-tune = **~17,300-17,700 H100-hours total**.
- Method: FLOPs estimate for finetune; MMaDA wall-clock estimate from step count and step-time proxy.
- Confidence: low
- Caveats: no public code/weights; "same training pipeline as MMaDA" gives the prior but exact step count for dVLA finetune is not disclosed. LIBERO simulator required for evaluation.

---

### LLaDA-VLA (arXiv:2509.06932)
- arxiv: https://arxiv.org/abs/2509.06932
- Training data: CALVIN ABC->D (~24K demos, ~1.5M frames) and SimplerEnv data. 3 epochs.
- Model size: LLaDA-V backbone, ~8B parameters (LLaDA-8B + SigLIP-2 vision encoder); paper does not state parameter count explicitly.
- Training compute (per paper): Only fine-tune hyperparameters disclosed: "3 epochs, learning rate 2e-5, batch size 128". No GPU type, count, or wall-clock.
- **H100-hour estimate to replicate: ~200-800 H100-hours (FLOPs-based estimate)**
  - Tokens: ~1.5M frames * ~512 tokens/frame ~= 770M visual tokens; * 3 epochs ~= 2.3B tokens. 6 * 8e9 * 2.3e9 = 1.1e20 FLOPs -> ~61 H100-hours minimum at 50% MFU. With diffusion overhead (multiple denoising passes per sample), batched VLA training overhead, and realistic ~30-40% MFU for an 8B diffusion VLA: ~200-800 H100-hours is a defensible range.
- Method: FLOPs estimate from disclosed epoch+data heuristic.
- Confidence: low
- Caveats: no public code/weights; hardware not stated; CALVIN simulator required for the headline benchmark. SimplerEnv adds further sim infra. Range is wide because token count per CALVIN demo is heuristic.

---

### UniDisc (arXiv:2503.20853)
- arxiv: https://arxiv.org/abs/2503.20853
- Training data: 250M image/caption pairs at 256x256 (DataComp1B + PixelProse + JourneyDB + Cambrian-10M) for pretrain; 30M pairs at 512x512 for fine-tune. Smaller scaling experiments: 11B tokens (20% text + 80% image) on 30M images.
- Model size: 1.4B (largest); also 115M and 340M non-embedding for scaling.
- Training compute (per paper): Only the smaller-scale scaling runs disclose hardware: "300 L40S GPU hours" at batch 512, LR 3e-4. The 1.4B model training duration / GPU count is not disclosed; acknowledgements thank CMU FLAME cluster, LambdaLabs, and Google TPU Research Cloud.
- **H100-hour estimate to replicate:**
  - Smaller scaling models (115M / 340M): 300 L40S-hours * 0.35 = **~105 H100-hours per scaling point** (high confidence).
  - 1.4B full pretrain + finetune: **~500-1,500 H100-hours (FLOPs-based estimate)**. Pretrain: 250M samples * ~280 tokens/sample (32 text + 256 image tokens at 256x256) ~= 70B tokens; 6 * 1.4e9 * 70e9 = 5.88e20 FLOPs -> ~327 H100-hours. Finetune at 512x512: 30M samples * ~1056 tokens/sample (32 text + 1024 image tokens) ~= 31.7B tokens; 6 * 1.4e9 * 31.7e9 = 2.66e20 FLOPs -> ~148 H100-hours. Subtotal ~475 H100-hours; padding for multi-epoch + overhead: 500-1,500.
  - The paper itself notes "13.2x more compute is required for UniDisc to achieve the same overall loss as AR", which provides a lower-bound sanity check.
- Method: scaling runs are direct (L40S-hours); 1.4B is FLOPs-based.
- Confidence: medium for scaling runs; low for 1.4B.
- Caveats: 1.4B compute not disclosed in paper; assumed single-epoch pretrain. Released weights target generation, not VQA, so replicating for VQA needs additional task-specific finetuning beyond the budget above.

---

## Summary Table

| Model | H100-hours (estimate) | Confidence |
|---|---|---|
| VidLaDA | ~1,000-3,000 (FLOPs) | low |
| Sparse-LaViDa | 7,680 (post-training only); ~51,200 if including LaViDa-O base | high (post-training); low (full) |
| MMaDA-Parallel | ~1,800-3,500 | low-medium |
| Unified Diffusion VLA | 96 (real-world only); ~5,000-20,000 (full pipeline, FLOPs) | high (real-world); low (full) |
| dVLA | ~200-600 (finetune only); ~17,300-17,700 (incl. MMaDA-8B base) | low |
| LLaDA-VLA | ~200-800 (FLOPs) | low |
| UniDisc | ~105 per scaling run (115M/340M); ~500-1,500 for 1.4B (FLOPs) | medium (scaling); low (1.4B) |

Totals (full-replication, best central estimate, summing the larger pipeline scope per row):
- Lower bound: 7,680 + 1,800 + 5,000 + 17,300 + 200 + 500 + 1,000 = ~33,480 H100-hours
- Upper bound: 51,200 + 3,500 + 20,000 + 17,700 + 800 + 1,500 + 3,000 = ~97,700 H100-hours

Rough order-of-magnitude budget to replicate all 7 from disclosed training settings: **30K - 100K H100-hours**, dominated by Sparse-LaViDa (if replicating LaViDa-O base), Unified Diffusion VLA (if replicating Stage 1 video post-training), and dVLA (if replicating MMaDA-8B base). The four pure-finetune scopes (MMaDA-Parallel, LLaDA-VLA, UniDisc-1.4B, dVLA-finetune-only, Unified-VLA real-world-only) collectively are only ~3K-8K H100-hours.
