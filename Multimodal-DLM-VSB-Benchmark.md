# Multimodal Diffusion Language Models on ViewSpatial-Bench

Evaluated 9 open-weights multimodal diffusion language models (dLLMs) from the [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) multimodal section on the full ViewSpatial-Bench (5712 single-image MCQ questions, 4-way A/B/C/D, 5 spatial reasoning task types). DiffusionVL is included as the reference model from the previous eval pass.

All models run BF16 (or FP32 for Muddit/FUDOKI) on a single H100 (NVL or PCIe). Generation length is tuned per model to emit ~1 letter (typical: `gen_length=8, steps=4`); we record per-task accuracy, mean latency per question, and peak VRAM.

## Overall accuracy

| Model | n | Overall acc | Mean lat (s/Q) | Peak VRAM (GB) | Total wall | Status |
|---|---:|---:|---:|---:|---:|---|
| DiffusionVL-Qwen2.5VL-7B | 5712 | **36.01%** | 0.242 | 18.84 | 28.6 min | ✓ done |
| LLaDA-V | 5712 | **31.06%** | 1.521 | 32.80 | 144.8 min | ✓ done |
| LaViDa-LLaDa | 5712 | **34.96%** | 0.513 | 20.50 | 48.8 min | ✓ done |
| MMaDA-8B-MixCoT | 5712 | **27.33%** | 0.329 | 20.44 | 31.3 min | ✓ done |
| Dimple-7B | 5712 | **37.83%** | 0.094 | 16.93 | 8.9 min | ✓ done |
| Muddit-1B | — | — | — | — | — | ⏳ pending |
| ReDiff | 5712 | **26.31%** | 2.258 | 32.82 | 222.7 min | ✓ done |
| FUDOKI | 5712 | **27.08%** | 0.661 | 13.73 | 62.9 min | ✓ done |
| Lumina-DiMOO | 5712 | **34.45%** | 1.344 | 31.68 | 128.0 min | ✓ done |
| LaViDa-O | 5712 | **24.16%** | 2.263 | 23.02 | 215.5 min | ✓ done |

## Per-task accuracy

| Model | Cam-RelDir | Cam-ObjOri | Pers-RelDir | Pers-ObjOri | Pers-SimRelDir | Overall |
|---|---:|---:|---:|---:|---:|---:|
| DiffusionVL-Qwen2.5VL-7B | 44.3% | 34.4% | 35.6% | 36.2% | 24.3% | **36.0%** |
| LLaDA-V | 34.4% | 25.1% | 30.0% | 39.4% | 24.3% | **31.1%** |
| LaViDa-LLaDa | 32.4% | 43.1% | 34.1% | 38.3% | 29.5% | **35.0%** |
| MMaDA-8B-MixCoT | 28.5% | 32.4% | 32.5% | 21.7% | 22.0% | **27.3%** |
| Dimple-7B | 41.3% | 31.0% | 33.8% | 48.0% | 32.2% | **37.8%** |
| Muddit-1B | — | — | — | — | — | — |
| ReDiff | 33.6% | 21.7% | 23.8% | 28.1% | 19.2% | **26.3%** |
| FUDOKI | 28.5% | 27.8% | 32.7% | 17.7% | 28.3% | **27.1%** |
| Lumina-DiMOO | 33.4% | 12.0% | 37.8% | 59.9% | 30.8% | **34.5%** |
| LaViDa-O | 22.9% | 23.0% | 29.5% | 22.0% | 25.2% | **24.2%** |

Task abbreviations:
- **Cam-RelDir**: Camera perspective - Relative Direction
- **Cam-ObjOri**: Camera perspective - Object View Orientation
- **Pers-RelDir**: Person perspective - Relative Direction
- **Pers-ObjOri**: Person perspective - Object View Orientation
- **Pers-SimRelDir**: Person perspective - Scene Simulation Relative Direction

## Latency & memory detail

| Model | n | Mean | Median | p95 | Max | Weights (GB) | Peak alloc | Peak reserved | Setup (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffusionVL-Qwen2.5VL-7B | 5712 | 0.242s | 0.209s | 0.391s | 0.85s | 16.63 | 18.84 | 20.39 | 4.9 |
| LLaDA-V | 5712 | 1.521s | 1.629s | 1.932s | 3.83s | 17.15 | 32.80 | 38.19 | 6.8 |
| LaViDa-LLaDa | 5712 | 0.513s | 0.511s | 0.584s | 0.98s | 16.88 | 20.50 | 35.58 | 6.9 |
| MMaDA-8B-MixCoT | 5712 | 0.329s | 0.326s | 0.363s | 0.92s | 16.55 | 20.44 | 44.00 | 4.5 |
| Dimple-7B | 5712 | 0.094s | 0.093s | 0.101s | 1.33s | 16.69 | 16.93 | 17.65 | 5.0 |
| Muddit-1B | — | — | — | — | — | — | — | — | — |
| ReDiff | 5712 | 2.258s | 2.215s | 3.323s | 5.16s | 17.15 | 32.82 | 40.30 | 6.3 |
| FUDOKI | 5712 | 0.661s | 0.653s | 0.704s | 1.21s | 10.04 | 13.73 | 14.54 | 4.5 |
| Lumina-DiMOO | 5712 | 1.344s | 1.323s | 1.414s | 1.78s | 16.75 | 31.68 | 83.24 | 4.8 |
| LaViDa-O | 5712 | 2.263s | 1.638s | 7.626s | 19.04s | 22.19 | 23.02 | 23.68 | 6.1 |

## Model sources

| Model | HF weights | Adapter | Env |
|---|---|---|---|
| DiffusionVL-Qwen2.5VL-7B | `hustvl/DiffusionVL-Qwen2.5VL-7B` | [adapters/diffusionvl.py](adapters/diffusionvl.py) | `~/miniconda3/envs/diffusionvl` |
| LLaDA-V | `GSAI-ML/LLaDA-V` | [adapters/llada_v.py](adapters/llada_v.py) | `~/miniconda3/envs/dlm_llada-v` |
| LaViDa-LLaDa | `jacklishufan/lavida-llada-v1.0-instruct` | [adapters/lavida.py](adapters/lavida.py) | `~/miniconda3/envs/dlm_lavida` |
| MMaDA-8B-MixCoT | `Gen-Verse/MMaDA-8B-MixCoT` | [adapters/mmada.py](adapters/mmada.py) | `~/miniconda3/envs/dlm_mmada` |
| Dimple-7B | `rp-yu/Dimple-7B` | [adapters/dimple.py](adapters/dimple.py) | `~/miniconda3/envs/dlm_dimple` |
| Muddit-1B | `MeissonFlow/Muddit` | [adapters/muddit.py](adapters/muddit.py) | `~/miniconda3/envs/dlm_muddit` |
| ReDiff | `jiyatai/ReDiff` | [adapters/rediff.py](adapters/rediff.py) | `~/miniconda3/envs/dlm_rediff` |
| FUDOKI | `LucasJinWang/FUDOKI` | [adapters/fudoki.py](adapters/fudoki.py) | `~/miniconda3/envs/dlm_fudoki` |
| Lumina-DiMOO | `Alpha-VLLM/Lumina-DiMOO` | [adapters/lumina_dimoo.py](adapters/lumina_dimoo.py) | `~/miniconda3/envs/dlm_lumina-dimoo` |
| LaViDa-O | `jacklishufan/LaViDa-O-v1.0` | [adapters/lavida_o.py](adapters/lavida_o.py) | `~/miniconda3/envs/dlm_lavida-o` |

## Paper-only models — H100-hour estimates to replicate

See [eval_results/paper_only_estimates.md](eval_results/paper_only_estimates.md) for full details. Summary:

| Model | H100-hours (estimate) | Confidence |
|---|---|---|
| VidLaDA | ~1,000-3,000 (FLOPs) | low |
| Sparse-LaViDa | 7,680 (post-training only); ~51,200 if including LaViDa-O base | high (post-training); low (full) |
| MMaDA-Parallel | ~1,800-3,500 | low-medium |
| Unified Diffusion VLA | 96 (real-world only); ~5,000-20,000 (full pipeline, FLOPs) | high (real-world); low (full) |
| dVLA | ~200-600 (finetune only); ~17,300-17,700 (incl. MMaDA-8B base) | low |
| LLaDA-VLA | ~200-800 (FLOPs) | low |
| UniDisc | ~105 per scaling run (115M/340M); ~500-1,500 for 1.4B (FLOPs) | medium (scaling); low (1.4B) |
