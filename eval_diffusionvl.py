"""
Evaluate hustvl/DiffusionVL-Qwen2.5VL-7B on ViewSpatial-Bench and MindCube.

Reports per-task accuracy, latency stats, and peak GPU memory.

Usage:
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \\
        /home/haoming/miniconda3/envs/diffusionvl/bin/python eval_diffusionvl.py \\
            --bench {vsb|mindcube|both} \\
            [--per-task-limit N] [--out PATH]
"""
import argparse, json, os, re, time, gc, sys, traceback
from collections import defaultdict
from statistics import mean, median
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "hustvl/DiffusionVL-Qwen2.5VL-7B"
VSB_ROOT = Path("/home/haoming/dllm/benchmarks/ViewSpatial-Bench")
MC_ROOT = Path("/home/haoming/dllm/benchmarks/MindCube")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_vsb():
    """ViewSpatial-Bench: list of dicts with question_type, image_path, question, answer, choices."""
    with open(VSB_ROOT / "ViewSpatial-Bench.json") as f:
        data = json.load(f)
    items = []
    for it in data:
        # answer like "A. right" -> letter "A"
        m = re.match(r"\s*([A-D])\b", it["answer"])
        gt = m.group(1) if m else None
        # resolve image path: stored as 'ViewSpatial-Bench/scannetv2_val/...' -> strip prefix
        rel = it["image_path"][0]
        if rel.startswith("ViewSpatial-Bench/"):
            rel = rel[len("ViewSpatial-Bench/"):]
        img_path = VSB_ROOT / rel
        items.append({
            "task": it["question_type"],
            "question": it["question"],
            "choices": it["choices"],            # already "A. ...\nB. ...\nC. ...\nD. ..."
            "images": [str(img_path)],
            "gt": gt,
        })
    return items


def load_mindcube():
    """MindCube tinybench: question already inlines choices; gt_answer is single letter."""
    items = []
    with open(MC_ROOT / "data" / "raw" / "MindCube_tinybench.jsonl") as f:
        for line in f:
            it = json.loads(line)
            task = it["type"] if isinstance(it["type"], str) else f"int_{it['type']}"
            imgs = [str(MC_ROOT / "data" / p) for p in it["images"]]
            items.append({
                "task": task,
                "question": it["question"],     # contains "A. ... B. ... C. ... D. ..." inline
                "choices": None,
                "images": imgs,
                "gt": it["gt_answer"].strip().upper()[:1] if it["gt_answer"] else None,
            })
    return items


# --------------------------------------------------------------------------- #
# Prompt + answer extraction
# --------------------------------------------------------------------------- #
LETTER_RE = re.compile(r"\b([A-D])\b")


def tile_images(images):
    """Tile N PIL images into one composite. Returns ([composite], prefix_text)."""
    n = len(images)
    if n == 2:
        cols, rows = 2, 1
    elif n == 3:
        cols, rows = 3, 1
    elif n == 4:
        cols, rows = 2, 2
    else:
        cols = min(4, n)
        rows = (n + cols - 1) // cols
    # use first image's size as cell
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(images):
        r, c = i // cols, i % cols
        if im.size != (w, h):
            im = im.resize((w, h))
        canvas.paste(im, (c * w, r * h))
    layout = "left to right" if rows == 1 else f"a {rows}x{cols} grid (row-major: top-left=image 1, ...)"
    prefix = f"The composite image contains {n} sub-images arranged {layout}. "
    return [canvas], prefix


def build_prompt(processor, question, choices, n_images):
    """Build chat prompt with N image placeholders followed by the question."""
    text_blob = question if choices is None else f"{question}\n{choices}"
    text_blob += "\nAnswer with the letter only."
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": text_blob}]
    messages = [{"role": "user", "content": content}]
    return processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def extract_letter(text):
    """Pull first A/B/C/D from the generated text."""
    if not text:
        return None
    t = text.strip()
    # Strict: starts with the letter
    if t and t[0].upper() in "ABCD":
        return t[0].upper()
    m = LETTER_RE.search(t.upper())
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Main eval loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["vsb", "mindcube", "both"], default="both")
    ap.add_argument("--per-task-limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--gen-length", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default="/home/haoming/dllm/eval_results/results.json")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--composite", action="store_true",
                    help="For multi-image questions, tile into a single grid image.")
    args = ap.parse_args()

    print(f"[setup] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[setup] loading {MODEL_ID}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    ).eval()
    print(f"[setup] loaded in {time.time()-t0:.1f}s; "
          f"vram alloc={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    benches = []
    if args.bench in ("vsb", "both"):
        benches.append(("ViewSpatial-Bench", load_vsb()))
    if args.bench in ("mindcube", "both"):
        benches.append(("MindCube-tinybench", load_mindcube()))

    # Optional per-task subsample (deterministic: take first N per task)
    if args.per_task_limit > 0:
        for name, items in benches:
            counts = defaultdict(int); kept = []
            for it in items:
                if counts[it["task"]] < args.per_task_limit:
                    kept.append(it); counts[it["task"]] += 1
            items[:] = kept

    results = {"model": MODEL_ID, "config": vars(args), "benchmarks": {}}
    torch.cuda.reset_peak_memory_stats()

    for bench_name, items in benches:
        print(f"\n[{bench_name}] {len(items)} questions", flush=True)
        per_task_correct = defaultdict(int)
        per_task_total = defaultdict(int)
        per_task_parse_fail = defaultdict(int)
        latencies = []
        bench_start = time.time()

        for i, it in enumerate(items):
            try:
                images = [Image.open(p).convert("RGB") for p in it["images"]]
                if args.composite and len(images) > 1:
                    images, prefix = tile_images(images)
                    question_for_prompt = prefix + it["question"]
                else:
                    question_for_prompt = it["question"]
                text = build_prompt(processor, question_for_prompt, it["choices"], len(images))
                inputs = processor(text=[text], images=images,
                                   return_tensors="pt", padding=True)
                inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}

                t = time.time()
                with torch.no_grad():
                    out_ids = model.generate(
                        inputs=inputs["input_ids"],
                        images=inputs.get("pixel_values"),
                        image_grid_thws=inputs.get("image_grid_thw"),
                        gen_length=args.gen_length,
                        steps=args.steps,
                        temperature=0.0,
                        remasking_strategy="low_confidence_static",
                    )
                latencies.append(time.time() - t)
                # model.generate() returns ONLY the generated tokens for this model
                gen_text = processor.decode(out_ids[0], skip_special_tokens=True)
                pred = extract_letter(gen_text)
            except Exception as e:
                pred, gen_text = None, f"<error: {e!r}>"

            task = it["task"]
            per_task_total[task] += 1
            if pred is None:
                per_task_parse_fail[task] += 1
            elif pred == it["gt"]:
                per_task_correct[task] += 1

            if (i + 1) % args.log_every == 0 or i == len(items) - 1:
                done = i + 1; tot = len(items)
                acc = sum(per_task_correct.values()) / max(1, sum(per_task_total.values()))
                eta = (time.time() - bench_start) / done * (tot - done)
                print(f"  [{bench_name}] {done}/{tot}  acc={acc:.3f}  "
                      f"mean_lat={mean(latencies):.3f}s  ETA={eta/60:.1f}min", flush=True)

        bench_time = time.time() - bench_start
        per_task = {}
        for t in sorted(per_task_total):
            n = per_task_total[t]
            per_task[t] = {
                "n": n,
                "correct": per_task_correct[t],
                "accuracy": per_task_correct[t] / n if n else 0.0,
                "parse_fail": per_task_parse_fail[t],
            }
        overall_correct = sum(v["correct"] for v in per_task.values())
        overall_n = sum(v["n"] for v in per_task.values())
        results["benchmarks"][bench_name] = {
            "n": overall_n,
            "overall_accuracy": overall_correct / overall_n if overall_n else 0.0,
            "per_task": per_task,
            "latency_s": {
                "mean": mean(latencies) if latencies else 0.0,
                "median": median(latencies) if latencies else 0.0,
                "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0,
                "min": min(latencies) if latencies else 0.0,
                "max": max(latencies) if latencies else 0.0,
                "total_minutes": bench_time / 60.0,
            },
        }

    results["gpu_memory"] = {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {args.out}", flush=True)
    print(json.dumps({k: v if k != "benchmarks" else
                      {bn: {"overall_accuracy": bv["overall_accuracy"], "n": bv["n"]}
                       for bn, bv in v.items()}
                      for k, v in results.items() if k in ("benchmarks", "gpu_memory")},
                     indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
