"""
Generic ViewSpatial-Bench evaluator that loads a model adapter.

Usage:
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=N \\
        <env-python> eval_vsb.py --adapter <name> [--limit N] [--out PATH]

Each adapter lives in adapters/<name>.py and exports a class `Adapter` with:
    - setup(self) -> None                   # loads model+processor onto current device
    - infer(self, image_paths, question, choices) -> str   # returns generated text
    - name (class attr): str
"""
import argparse, importlib, json, os, re, sys, time, traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch
from PIL import Image

VSB_ROOT = Path("/home/haoming/dllm/benchmarks/ViewSpatial-Bench")
LETTER_RE = re.compile(r"\b([A-D])\b")


def load_vsb():
    with open(VSB_ROOT / "ViewSpatial-Bench.json") as f:
        data = json.load(f)
    items = []
    for it in data:
        m = re.match(r"\s*([A-D])\b", it["answer"])
        gt = m.group(1) if m else None
        rel = it["image_path"][0]
        if rel.startswith("ViewSpatial-Bench/"):
            rel = rel[len("ViewSpatial-Bench/"):]
        items.append({
            "task": it["question_type"],
            "question": it["question"],
            "choices": it["choices"],
            "image": str(VSB_ROOT / rel),
            "gt": gt,
        })
    return items


def extract_letter(text):
    if not text:
        return None
    t = text.strip()
    if t and t[0].upper() in "ABCD":
        return t[0].upper()
    m = LETTER_RE.search(t.upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter module name under adapters/")
    ap.add_argument("--limit", type=int, default=0, help="0 = full bench, else first N")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    mod = importlib.import_module(f"adapters.{args.adapter}")
    adapter = mod.Adapter()
    name = getattr(adapter, "name", args.adapter)

    print(f"[setup] adapter={name} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    t0 = time.time()
    adapter.setup()
    setup_time = time.time() - t0
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[setup] loaded in {setup_time:.1f}s; weights VRAM={weights_gb:.2f} GB", flush=True)

    items = load_vsb()
    if args.limit:
        items = items[:args.limit]
    print(f"[run] {len(items)} questions", flush=True)

    per_task_correct = defaultdict(int)
    per_task_total = defaultdict(int)
    per_task_parse_fail = defaultdict(int)
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    bench_start = time.time()

    for i, it in enumerate(items):
        try:
            t = time.time()
            out_text = adapter.infer([it["image"]], it["question"], it["choices"])
            torch.cuda.synchronize()
            latencies.append(time.time() - t)
            pred = extract_letter(out_text)
        except Exception as e:
            pred, out_text = None, f"<error: {e!r}>"
            if i < 3:
                traceback.print_exc()

        task = it["task"]
        per_task_total[task] += 1
        if pred is None:
            per_task_parse_fail[task] += 1
        elif pred == it["gt"]:
            per_task_correct[task] += 1

        if (i + 1) % args.log_every == 0 or i == len(items) - 1:
            done = i + 1
            tot = len(items)
            acc = sum(per_task_correct.values()) / max(1, sum(per_task_total.values()))
            eta = (time.time() - bench_start) / done * (tot - done)
            print(f"  [{name}] {done}/{tot}  acc={acc:.3f}  "
                  f"mean_lat={mean(latencies):.3f}s  ETA={eta/60:.1f}min", flush=True)

    bench_time = time.time() - bench_start
    per_task = {}
    for t in sorted(per_task_total):
        n = per_task_total[t]
        per_task[t] = {
            "n": n, "correct": per_task_correct[t],
            "accuracy": per_task_correct[t] / n if n else 0.0,
            "parse_fail": per_task_parse_fail[t],
        }
    oc = sum(v["correct"] for v in per_task.values())
    on = sum(v["n"] for v in per_task.values())

    result = {
        "model": name,
        "adapter": args.adapter,
        "n": on,
        "overall_accuracy": oc / on if on else 0.0,
        "per_task": per_task,
        "latency_s": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0.0,
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "total_minutes": bench_time / 60.0,
        },
        "gpu_memory_gb": {
            "weights": weights_gb,
            "peak_allocated": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved": torch.cuda.max_memory_reserved() / 1e9,
        },
        "setup_seconds": setup_time,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {args.out}", flush=True)
    print(f"  overall_acc={result['overall_accuracy']:.4f}  "
          f"mean_lat={result['latency_s']['mean']:.3f}s  "
          f"peak_vram={result['gpu_memory_gb']['peak_allocated']:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
