"""Aggregate per-model VSB results into a single markdown report."""
import json
from pathlib import Path

RESULTS = Path("/home/haoming/dllm/eval_results")
PER_MODEL = RESULTS / "per_model"
OUT_MD = Path("/home/haoming/dllm/Multimodal-DLM-VSB-Benchmark.md")

# Mapping: filename stem -> display info
RUN_ORDER = [
    ("diffusionvl", "DiffusionVL-Qwen2.5VL-7B", "hustvl/DiffusionVL-Qwen2.5VL-7B"),
    ("llada_v", "LLaDA-V", "GSAI-ML/LLaDA-V"),
    ("lavida", "LaViDa-LLaDa", "jacklishufan/lavida-llada-v1.0-instruct"),
    ("mmada", "MMaDA-8B-MixCoT", "Gen-Verse/MMaDA-8B-MixCoT"),
    ("dimple", "Dimple-7B", "rp-yu/Dimple-7B"),
    ("muddit", "Muddit-1B", "MeissonFlow/Muddit"),
    ("rediff", "ReDiff", "jiyatai/ReDiff"),
    ("fudoki", "FUDOKI", "LucasJinWang/FUDOKI"),
    ("lumina_dimoo", "Lumina-DiMOO", "Alpha-VLLM/Lumina-DiMOO"),
    ("lavida_o", "LaViDa-O", "jacklishufan/LaViDa-O-v1.0"),
]


def load(stem):
    p = PER_MODEL / f"{stem}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


VSB_TASKS = [
    "Camera perspective - Relative Direction",
    "Camera perspective - Object View Orientation",
    "Person perspective - Relative Direction",
    "Person perspective - Object View Orientation",
    "Person perspective - Scene Simulation Relative Direction",
]
TASK_SHORT = {
    "Camera perspective - Relative Direction": "Cam-RelDir",
    "Camera perspective - Object View Orientation": "Cam-ObjOri",
    "Person perspective - Relative Direction": "Pers-RelDir",
    "Person perspective - Object View Orientation": "Pers-ObjOri",
    "Person perspective - Scene Simulation Relative Direction": "Pers-SimRelDir",
}


def main():
    rows = []
    for stem, display, hf_id in RUN_ORDER:
        d = load(stem)
        rows.append((stem, display, hf_id, d))

    md = []
    md.append("# Multimodal Diffusion Language Models on ViewSpatial-Bench\n")
    md.append("Evaluated 9 open-weights multimodal diffusion language models (dLLMs) from the [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) multimodal section on the full ViewSpatial-Bench (5712 single-image MCQ questions, 4-way A/B/C/D, 5 spatial reasoning task types). DiffusionVL is included as the reference model from the previous eval pass.\n")
    md.append("All models run BF16 (or FP32 for Muddit/FUDOKI) on a single H100 (NVL or PCIe). Generation length is tuned per model to emit ~1 letter (typical: `gen_length=8, steps=4`); we record per-task accuracy, mean latency per question, and peak VRAM.\n")

    # Overall headline table
    md.append("## Overall accuracy\n")
    md.append("| Model | n | Overall acc | Mean lat (s/Q) | Peak VRAM (GB) | Total wall | Status |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for stem, display, hf_id, d in rows:
        if d is None:
            md.append(f"| {display} | — | — | — | — | — | ⏳ pending |")
            continue
        md.append(
            f"| {display} | {d['n']} | **{d['overall_accuracy']*100:.2f}%** | "
            f"{d['latency_s']['mean']:.3f} | "
            f"{d['gpu_memory_gb']['peak_allocated']:.2f} | "
            f"{d['latency_s']['total_minutes']:.1f} min | ✓ done |"
        )
    md.append("")

    # Per-task table
    md.append("## Per-task accuracy\n")
    md.append("| Model | " + " | ".join(TASK_SHORT[t] for t in VSB_TASKS) + " | Overall |")
    md.append("|---|" + "---:|" * (len(VSB_TASKS) + 1))
    for stem, display, hf_id, d in rows:
        if d is None:
            md.append(f"| {display} |" + " — |" * (len(VSB_TASKS) + 1))
            continue
        cells = []
        for t in VSB_TASKS:
            pt = d["per_task"].get(t, {})
            acc = pt.get("accuracy", 0.0)
            n = pt.get("n", 0)
            cells.append(f"{acc*100:.1f}%")
        cells.append(f"**{d['overall_accuracy']*100:.1f}%**")
        md.append(f"| {display} | " + " | ".join(cells) + " |")
    md.append("\nTask abbreviations:")
    for full, short in TASK_SHORT.items():
        md.append(f"- **{short}**: {full}")
    md.append("")

    # Latency + memory detail
    md.append("## Latency & memory detail\n")
    md.append("| Model | n | Mean | Median | p95 | Max | Weights (GB) | Peak alloc | Peak reserved | Setup (s) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stem, display, hf_id, d in rows:
        if d is None:
            md.append(f"| {display} |" + " — |" * 9)
            continue
        L = d["latency_s"]
        G = d["gpu_memory_gb"]
        md.append(
            f"| {display} | {d['n']} | {L['mean']:.3f}s | {L['median']:.3f}s | "
            f"{L['p95']:.3f}s | {L['max']:.2f}s | {G['weights']:.2f} | "
            f"{G['peak_allocated']:.2f} | {G['peak_reserved']:.2f} | "
            f"{d.get('setup_seconds', 0):.1f} |"
        )
    md.append("")

    # Footer with HF ids + adapter files
    md.append("## Model sources\n")
    md.append("| Model | HF weights | Adapter | Env |")
    md.append("|---|---|---|---|")
    for stem, display, hf_id, d in rows:
        adapter_path = f"adapters/{stem}.py"
        env_path = f"~/miniconda3/envs/dlm_{stem.replace('_','-')}" if stem != "diffusionvl" else "~/miniconda3/envs/diffusionvl"
        md.append(f"| {display} | `{hf_id}` | [{adapter_path}]({adapter_path}) | `{env_path}` |")
    md.append("")

    # Include paper-only estimates as appendix
    md.append("## Paper-only models — H100-hour estimates to replicate\n")
    md.append("See [eval_results/paper_only_estimates.md](eval_results/paper_only_estimates.md) for full details. Summary:\n")
    est_p = RESULTS / "paper_only_estimates.md"
    if est_p.exists():
        # Try to grab the summary table at the end
        with open(est_p) as f:
            content = f.read()
        # Find the last table in the file
        lines = content.splitlines()
        # Pull last contiguous table block
        last_table = []
        in_table = False
        for line in lines:
            if line.startswith("|"):
                last_table.append(line)
                in_table = True
            else:
                if in_table:
                    pass  # keep going to grab the final one; reset
                in_table = False
        # Heuristic: extract final table block
        block, current = [], []
        for line in lines:
            if line.startswith("|"):
                current.append(line)
            else:
                if current:
                    block = current
                    current = []
        if current:
            block = current
        if block:
            md.extend(block)
        else:
            md.append(f"(see {est_p.name})")
    md.append("")

    out = "\n".join(md)
    OUT_MD.write_text(out)
    print(f"Wrote {OUT_MD}")
    print(out[:1500])


if __name__ == "__main__":
    main()
