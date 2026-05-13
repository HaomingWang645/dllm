# DiffusionVL-Qwen2.5VL-7B

Local notes for running [`hustvl/DiffusionVL-Qwen2.5VL-7B`](https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B) on this machine.

- **Paper:** [DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models](https://arxiv.org/abs/2512.15713) (Zeng et al., 2025)
- **Code:** https://github.com/hustvl/DiffusionVL
- **HF model:** https://huggingface.co/hustvl/DiffusionVL-Qwen2.5VL-7B
- **License:** Apache 2.0

## What it is

A diffusion vision-language model (dVLM) translated from the autoregressive `Qwen2.5-VL-7B-Instruct`. Generates tokens in parallel via a masked-diffusion process with block decoding instead of left-to-right AR decoding.

- ~8B parameters, BF16
- Reported ~2.0× inference speedup over prior dVLMs
- Trained with <5% of typical dVLM data (738K vs 16.5M samples)
- Conversational image-text-to-text

## Verified hardware

| Item              | Value                                              |
|-------------------|----------------------------------------------------|
| GPU               | NVIDIA H100 NVL (95 GB) — 1 GPU used               |
| VRAM (BF16, eager)| **16.58 GB** allocated                             |
| Disk for weights  | ~16 GB (4 safetensors shards)                      |
| Model load (cached)| 4.4 s                                             |
| 128-tok gen, 8 steps | 3.72 s                                          |

Fits comfortably on a single 24 GB+ GPU.

## Environment

Conda env: `diffusionvl` at `/home/haoming/miniconda3/envs/diffusionvl/`

Minimal deps (per the official `docs/INFERENCE.md`):

```
torch==2.6.0          # +cu124 wheel
torchvision==0.21.0
transformers==4.55.0
accelerate==1.10.1
pillow==10.4.0
requests==2.32.5
safetensors
hf_transfer
psutil
regex tokenizers sentencepiece protobuf
```

Recreate from scratch:

```bash
conda create -n diffusionvl python=3.10 -y
PY=/home/haoming/miniconda3/envs/diffusionvl/bin/python
$PY -m pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install transformers==4.55.0 accelerate==1.10.1 \
    pillow==10.4.0 requests==2.32.5 safetensors hf_transfer psutil \
    regex tokenizers sentencepiece protobuf
```

> **Gotcha on this machine:** always export `PYTHONNOUSERSITE=1` when invoking the env's Python — `~/.local/lib/python3.10/site-packages` otherwise shadows the env and causes `ModuleNotFoundError: No module named 'regex'` and similar.

## Quick start

See [`test_diffusionvl.py`](test_diffusionvl.py).

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
    /home/haoming/miniconda3/envs/diffusionvl/bin/python test_diffusionvl.py
```

Inline:

```python
import requests, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "hustvl/DiffusionVL-Qwen2.5VL-7B"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."},
]}]
text = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

output_ids = model.generate(
    inputs=inputs["input_ids"],
    images=inputs.get("pixel_values"),
    image_grid_thws=inputs.get("image_grid_thw"),
    gen_length=128,
    steps=8,
    temperature=0.0,
    remasking_strategy="low_confidence_static",
)
print(processor.decode(output_ids[0], skip_special_tokens=True))
```

## Generation knobs

Diffusion-specific kwargs passed to `model.generate(...)`:

| Argument              | Meaning                                                   | Typical value             |
|-----------------------|-----------------------------------------------------------|---------------------------|
| `gen_length`          | Max output length in tokens                                | 128 – 1024                |
| `steps`               | Number of diffusion denoising steps                        | 4 – 16 (fewer = faster)   |
| `temperature`         | Sampling temperature; 0.0 = greedy unmask                  | 0.0 for deterministic     |
| `remasking_strategy`  | Which tokens to re-mask each step                          | `"low_confidence_static"` |

`steps` × `block_size` directly controls the speed/quality tradeoff. The headline 2× speedup vs prior dVLMs comes from block decoding with KV-cache reuse.

## Benchmark results

Evaluated on a single NVIDIA H100 NVL (95 GB), BF16, `gen_length=8`, `steps=4`, `temperature=0.0`, `remasking_strategy="low_confidence_static"`, greedy unmask. Answer extraction = first A/B/C/D token in the decoded output. Script: [`eval_diffusionvl.py`](eval_diffusionvl.py). Raw JSON: [`eval_results/`](eval_results/).

### Latency & memory (overall)

| Bench | n | Total wall | Mean latency | Median | p95 | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViewSpatial-Bench (single image) | 5712 | 28.6 min | **0.242 s/Q** | 0.209 s | 0.391 s | **18.8 GB** | 20.4 GB |
| MindCube tinybench (native multi-image) | 1050 | 7.1 min | 0.323 s/Q | 0.242 s | 0.936 s | 23.3 GB | 32.1 GB |
| MindCube tinybench (composite 1-image) | 1050 | 7.3 min | **0.313 s/Q** | 0.216 s | 0.949 s | 23.4 GB | 32.8 GB |

Resident weights: **16.58 GB**. Peak VRAM rises with image count (vision encoder activations).

### ViewSpatial-Bench — per task accuracy

5 712 questions, 4-way MCQ, all 5 task types, single image per question.

| Task | n | Accuracy |
|---|---:|---:|
| Camera perspective — Relative Direction | 1773 | **44.28 %** |
| Person perspective — Object View Orientation | 996 | 36.24 % |
| Person perspective — Relative Direction | 842 | 35.63 % |
| Camera perspective — Object View Orientation | 996 | 34.44 % |
| Person perspective — Scene Simulation Relative Direction | 1105 | 24.25 % |
| **Overall** | **5712** | **36.01 %** |

Random baseline = 25 % (4-way MCQ). Parse-failure rate: 0.0 % across all tasks.

### MindCube (tinybench) — per task accuracy

1 050 questions, 4-way MCQ. Each question carries 2–4 images; all 1050 entries are multi-image.

**Native multi-image input** — feed each image as a separate `<image>` placeholder:

| Task type | n | Accuracy | Parse fail |
|---|---:|---:|---:|
| 0_frame | 140 | 2.86 % | 116/140 |
| 1_frame | 140 | 6.43 % | 118/140 |
| 2_frame | 149 | 1.34 % | 127/149 |
| 3_frame | 145 | 6.90 % | 115/145 |
| general | 26 | 0.00 % | 26/26 |
| int_1 | 13 | 23.08 % | 9/13 |
| int_2 | 125 | 4.00 % | 117/125 |
| int_3 | 112 | 2.68 % | 96/112 |
| three_view | 200 | 2.50 % | 189/200 |
| **Overall** | **1050** | **3.90 %** | **913/1050 (87 %)** |

**Composite-image workaround** — tile the N images into a single grid (`2×1`, `3×1`, or `2×2`) and prepend a layout-describing sentence, so the model sees one image:

| Task type | n | Accuracy |
|---|---:|---:|
| general | 26 | **50.00 %** |
| int_2 | 125 | 62.40 % |
| int_3 | 112 | 47.32 % |
| 2_frame | 149 | 41.61 % |
| 0_frame | 140 | 39.29 % |
| 1_frame | 140 | 39.29 % |
| three_view | 200 | 33.50 % |
| int_1 | 13 | 30.77 % |
| 3_frame | 145 | 29.66 % |
| **Overall** | **1050** | **40.95 %** |

Parse-failure rate: 0.0 % across all composite tasks.

## Modality support — important caveat

Despite Qwen2.5-VL's native multi-image / video support, **this DiffusionVL checkpoint cannot ingest multiple `<image>` placeholders correctly**:

- With ≥ 3 image placeholders the model degenerates to outputting `'\n'` regardless of `gen_length` / `steps`.
- 2-image inputs sometimes work, sometimes not — unstable.
- This is consistent with the model's `processing_diffusionvl_qwen2_5_vl.py`, whose docstring marks video as *"Currently not fully supported"*, and `config.json` overriding `video_token_id` to `null`. Training data was LLaVA-Pretrain + LLaVA-NeXT, both image-only.

**Recommended workaround when feeding multi-view scenes:** tile views into a single composite image and describe the layout in text. On MindCube this lifts overall accuracy from 3.9 % to 41.0 %.

## Smoke test output

Input image: Qwen demo (woman + dog on beach), `gen_length=128`, `steps=8`, `temperature=0.0`.

> In the image, a woman and her dog are enjoying a moment of connection on a sandy beach. The woman, dressed in a blue plaid shirt and black shorts, is seated on the sand with her legs crossed. She is holding the dog's paw, which is adorned in a red collar. The dog, a light brown breed, is sitting beside her, its attention focused on the woman. The ocean waves roll gently in the background, providing a serene backdrop to this intimate scene.

## Related variants

| Model                              | Base                  |
|------------------------------------|-----------------------|
| `hustvl/DiffusionVL-Qwen2.5VL-3B`  | Qwen2.5-VL-3B         |
| `hustvl/DiffusionVL-Qwen2.5VL-7B`  | Qwen2.5-VL-7B *(this)*|
| `hustvl/DiffusionVL-Qwen2.5-7B`    | Qwen2.5-7B (text-only)|

## Citation

```bibtex
@misc{zeng2025diffusionvltranslatingautoregressivemodels,
    title  = {DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models},
    author = {Lunbin Zeng and Jingfeng Yao and Bencheng Liao and Hongyuan Tao and Wenyu Liu and Xinggang Wang},
    year   = {2025},
    eprint = {2512.15713},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url = {https://arxiv.org/abs/2512.15713},
}
```
