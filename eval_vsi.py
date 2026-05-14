"""
Generic VSI-Bench evaluator with video frame sampling and MCQ + numeric scoring.

Usage:
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=N \\
        <env-python> eval_vsi.py --adapter <name> --num-frames 32 [--limit N] --out PATH

Each adapter must implement:
    - setup(self) -> None
    - infer_video(self, video_path, question, options) -> str
        options is None for free-form numeric tasks, or list[str] like ["A. ...", "B. ..."]
        for MCQ.  Should return the generated text (we parse letter or number out of it).
"""
import argparse, importlib, json, os, re, sys, time, traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch

VSI_ROOT = Path("/home/haoming/dllm/benchmarks/VSI-Bench")
TEST_JSONL = VSI_ROOT / "test.jsonl"

LETTER_RE = re.compile(r"\b([A-E])\b")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")

# Word-form numbers — VLMs frequently emit "three" instead of "3".
_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
    "half": 0.5, "quarter": 0.25,
    "no": 0, "none": 0,
}
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_NUM) + r")\b", re.IGNORECASE,
)


# ---------------------------------------------------------------- data load
def load_vsi():
    items = []
    with open(TEST_JSONL) as f:
        for line in f:
            it = json.loads(line)
            ds = it["dataset"]
            scene = it["scene_name"]
            video_path = VSI_ROOT / ds / f"{scene}.mp4"
            items.append({
                "id": it["id"],
                "task": it["question_type"],
                "question": it["question"],
                "options": it["options"],          # None for free-form
                "gt": it["ground_truth"],
                "video": str(video_path),
                "dataset": ds,
                "scene": scene,
            })
    return items


# ---------------------------------------------------------------- scoring
def parse_letter(text):
    """Pull first A-E letter from generated text."""
    if not text:
        return None
    t = text.strip()
    if t and t[0].upper() in "ABCDE":
        return t[0].upper()
    m = LETTER_RE.search(t.upper())
    return m.group(1) if m else None


def parse_number(text):
    """Pull first number from generated text. Handles both digit and word forms."""
    if not text:
        return None
    # Prefer digit form if present.
    m = NUMBER_RE.search(text)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass
    # Fall back to single word-form number.
    m2 = _WORD_NUM_RE.search(text)
    if m2:
        return float(_WORD_NUM[m2.group(1).lower()])
    return None


def mra_score(pred, gt):
    """Mean Relative Accuracy: 1 if |p-gt|/gt < threshold else 0, averaged over thresholds.

    Standard VSI-Bench MRA thresholds: 0.05 .. 0.95 step 0.05 (19 thresholds).
    """
    if pred is None:
        return 0.0
    try:
        gt = float(gt)
    except Exception:
        return 0.0
    if gt == 0:
        return 1.0 if abs(pred) < 1e-6 else 0.0
    rel_err = abs(pred - gt) / abs(gt)
    thresholds = [0.05 * i for i in range(1, 20)]   # 0.05 .. 0.95
    hits = sum(1 for thr in thresholds if rel_err < thr)
    return hits / len(thresholds)


MCQ_TASKS = {
    "object_rel_distance",
    "obj_appearance_order",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "route_planning",
}
NUMERIC_TASKS = {
    "object_size_estimation",
    "object_abs_distance",
    "object_counting",
    "room_size_estimation",
}


def score(item, gen_text):
    """Return (score in [0,1], pred_str, parse_failed_bool)."""
    task = item["task"]
    if task in MCQ_TASKS:
        pred = parse_letter(gen_text)
        if pred is None:
            return 0.0, None, True
        return (1.0 if pred == str(item["gt"]).strip().upper()[:1] else 0.0), pred, False
    elif task in NUMERIC_TASKS:
        pred = parse_number(gen_text)
        if pred is None:
            return 0.0, None, True
        return mra_score(pred, item["gt"]), str(pred), False
    else:
        return 0.0, None, True


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = full")
    ap.add_argument("--start", type=int, default=0,
                    help="Index of first question to evaluate (inclusive). Used with --end "
                         "to split a run across multiple GPUs.")
    ap.add_argument("--end", type=int, default=0,
                    help="Index of last question (exclusive). 0 = end of dataset.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-preds", action="store_true",
                    help="also save per-question predictions to <out>.preds.jsonl")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    mod = importlib.import_module(f"adapters.{args.adapter}")
    adapter = mod.Adapter()
    if hasattr(adapter, "num_frames"):
        adapter.num_frames = args.num_frames
    name = getattr(adapter, "name", args.adapter)

    print(f"[setup] adapter={name} num_frames={args.num_frames} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    t0 = time.time()
    adapter.setup()
    setup_time = time.time() - t0
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[setup] loaded in {setup_time:.1f}s; weights VRAM={weights_gb:.2f} GB", flush=True)

    items = load_vsi()
    # Drop items whose video doesn't exist (rare; flag if many).
    items = [it for it in items if os.path.exists(it["video"])]
    # Optional slicing for parallel splits.
    if args.start or args.end:
        end = args.end if args.end > 0 else len(items)
        items = items[args.start:end]
    if args.limit:
        items = items[:args.limit]
    print(f"[run] {len(items)} questions", flush=True)

    per_task_correct_sum = defaultdict(float)
    per_task_total = defaultdict(int)
    per_task_parse_fail = defaultdict(int)
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    bench_start = time.time()

    preds_f = None
    if args.save_preds:
        preds_f = open(args.out + ".preds.jsonl", "w")

    for i, it in enumerate(items):
        try:
            t = time.time()
            out_text = adapter.infer_video(it["video"], it["question"], it["options"])
            torch.cuda.synchronize()
            latencies.append(time.time() - t)
        except Exception as e:
            out_text = f"<error: {e!r}>"
            if i < 3:
                traceback.print_exc()
            latencies.append(0.0)

        s, pred, parse_fail = score(it, out_text)
        per_task_total[it["task"]] += 1
        per_task_correct_sum[it["task"]] += s
        if parse_fail:
            per_task_parse_fail[it["task"]] += 1

        if preds_f:
            preds_f.write(json.dumps({
                "id": it["id"], "task": it["task"], "gt": it["gt"],
                "pred": pred, "raw": out_text[:300], "score": s,
            }) + "\n")

        if (i + 1) % args.log_every == 0 or i == len(items) - 1:
            done = i + 1; tot = len(items)
            acc = sum(per_task_correct_sum.values()) / max(1, sum(per_task_total.values()))
            eta = (time.time() - bench_start) / done * (tot - done)
            print(f"  [{name}] {done}/{tot}  score={acc:.3f}  "
                  f"mean_lat={mean(latencies):.2f}s  ETA={eta/60:.1f}min", flush=True)

    if preds_f:
        preds_f.close()

    bench_time = time.time() - bench_start
    per_task = {}
    for t in sorted(per_task_total):
        n = per_task_total[t]
        per_task[t] = {
            "n": n,
            "score_sum": per_task_correct_sum[t],
            "score": per_task_correct_sum[t] / n if n else 0.0,
            "parse_fail": per_task_parse_fail[t],
            "metric": "accuracy" if t in MCQ_TASKS else "MRA",
        }
    overall_n = sum(v["n"] for v in per_task.values())
    overall_score = sum(v["score_sum"] for v in per_task.values()) / max(1, overall_n)
    mcq_n = sum(v["n"] for k, v in per_task.items() if k in MCQ_TASKS)
    mcq_score = (
        sum(v["score_sum"] for k, v in per_task.items() if k in MCQ_TASKS) / mcq_n
        if mcq_n else 0.0
    )
    num_n = sum(v["n"] for k, v in per_task.items() if k in NUMERIC_TASKS)
    num_score = (
        sum(v["score_sum"] for k, v in per_task.items() if k in NUMERIC_TASKS) / num_n
        if num_n else 0.0
    )

    result = {
        "model": name,
        "adapter": args.adapter,
        "num_frames": args.num_frames,
        "n": overall_n,
        "overall_score": overall_score,
        "mcq_accuracy": mcq_score,
        "numeric_mra": num_score,
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
    print(
        f"  overall={overall_score:.4f}  mcq_acc={mcq_score:.4f}  "
        f"num_mra={num_score:.4f}  mean_lat={result['latency_s']['mean']:.2f}s  "
        f"peak_vram={result['gpu_memory_gb']['peak_allocated']:.2f}GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
