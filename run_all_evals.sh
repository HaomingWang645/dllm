#!/usr/bin/env bash
# Launch full VSB evals for all models that have a working adapter, one per GPU.
# Usage: bash run_all_evals.sh
#
# Each line: ADAPTER ENV_PYTHON GPU_INDEX
# Will skip rows where the adapter file or env-python doesn't exist.

set -u
RESULTS=/home/haoming/dllm/eval_results/per_model
LOGS=/home/haoming/dllm/eval_results/logs
mkdir -p "$RESULTS" "$LOGS"

declare -a CONFIGS=(
  # adapter        env-python                                                  gpu
  "diffusionvl    /home/haoming/miniconda3/envs/diffusionvl/bin/python         0"
  "llada_v        /home/haoming/miniconda3/envs/dlm_llada-v/bin/python         1"
  "lavida         /home/haoming/miniconda3/envs/dlm_lavida/bin/python          3"
  "mmada          /home/haoming/miniconda3/envs/dlm_mmada/bin/python           4"
  "dimple         /home/haoming/miniconda3/envs/dlm_dimple/bin/python          5"
  "muddit         /home/haoming/miniconda3/envs/dlm_muddit/bin/python          6"
  "rediff         /home/haoming/miniconda3/envs/dlm_rediff/bin/python          7"
)

pids=()
for cfg in "${CONFIGS[@]}"; do
  read -r ADAPTER PYBIN GPU <<<"$cfg"
  ADAPTER_FILE="/home/haoming/dllm/adapters/${ADAPTER}.py"
  if [[ ! -f "$ADAPTER_FILE" ]]; then
    echo "[skip] no adapter: $ADAPTER_FILE"
    continue
  fi
  if [[ ! -x "$PYBIN" ]]; then
    echo "[skip] no env python: $PYBIN"
    continue
  fi
  OUT="$RESULTS/${ADAPTER}.json"
  LOG="$LOGS/${ADAPTER}.log"
  echo "[launch] $ADAPTER on GPU $GPU -> $OUT (log: $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 \
    "$PYBIN" /home/haoming/dllm/eval_vsb.py \
      --adapter "$ADAPTER" --out "$OUT" --log-every 200 \
      > "$LOG" 2>&1 &
  pids+=($!)
done

echo "[wait] PIDs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    echo "[error] process $p exited non-zero"
    fail=1
  fi
done

echo
echo "[done] fail=$fail"
ls -la "$RESULTS"/ | grep -v smoke
