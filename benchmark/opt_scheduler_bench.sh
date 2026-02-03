#!/bin/bash
# =============================================================================
# Scheduler Optimization Benchmark Suite
# Compares baseline (main) vs optimization branches under queue pressure
#
# Usage:
#   ./benchmark/opt_scheduler_bench.sh [--model MODEL] [--runs N] [--rate RATE]
#
# Requirements:
#   - sglang installed (pip install sglang[all])
#   - GPU with enough VRAM for the chosen model
#   - HF_HOME set if model cache is in a custom location
# =============================================================================
set -euo pipefail

# --- Defaults ---
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT=30000
PYTHON="${PYTHON:-python3}"
NUM_RUNS="${NUM_RUNS:-10}"
REQUEST_RATE="${REQUEST_RATE:-32}"
NUM_CLIENTS="${NUM_CLIENTS:-400}"
MAX_PARALLEL="${MAX_PARALLEL:-256}"
REQUEST_LENGTH="${REQUEST_LENGTH:-1024}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-128}"
NUM_ROUNDS="${NUM_ROUNDS:-5}"
SEED=42
RESULTS_DIR="${RESULTS_DIR:-/root/bench_results}"
SGLANG_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --runs) NUM_RUNS="$2"; shift 2 ;;
    --rate) REQUEST_RATE="$2"; shift 2 ;;
    --clients) NUM_CLIENTS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --request-length) REQUEST_LENGTH="$2"; shift 2 ;;
    --output-length) OUTPUT_LENGTH="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

BRANCHES=(
  "main"
  "origin/opt/scheduler-set-construction"
  "origin/opt/consolidate-any-passes"
)
TAGS=(
  "baseline"
  "opt1_set_cache"
  "opt2_any_consolidate"
)

BENCH_SCRIPT="$SGLANG_DIR/benchmark/hicache/bench_multiturn.py"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

kill_server() {
  log "Stopping server..."
  kill "$(pgrep -f 'sglang.launch_server')" 2>/dev/null || true
  sleep 5
  kill -9 "$(pgrep -f 'sglang')" 2>/dev/null || true
  sleep 3
}

start_server() {
  local tag=$1
  log "Starting server for $tag..."

  export PYTHONPATH="$SGLANG_DIR/python"
  nohup "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" --port "$PORT" \
    > "$RESULTS_DIR/server_${tag}.log" 2>&1 &

  for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
      log "Server ready (${i}x2s)"
      return 0
    fi
    sleep 2
  done
  log "ERROR: Server failed to start"
  tail -30 "$RESULTS_DIR/server_${tag}.log"
  return 1
}

run_bench() {
  local tag=$1 run=$2
  local out="$RESULTS_DIR/${tag}_run${run}.out"
  local jsonl="$RESULTS_DIR/${tag}.jsonl"

  log "Benchmark $tag run $run/$NUM_RUNS"
  export PYTHONPATH="$SGLANG_DIR/python"

  "$PYTHON" "$BENCH_SCRIPT" \
    --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --num-clients "$NUM_CLIENTS" --max-parallel "$MAX_PARALLEL" \
    --request-length "$REQUEST_LENGTH" --output-length "$OUTPUT_LENGTH" \
    --num-rounds "$NUM_ROUNDS" --distribution uniform \
    --request-rate "$REQUEST_RATE" --disable-auto-run --seed "$SEED" \
    --log-file "$jsonl" --tag "${tag}_run${run}" \
    > "$out" 2>&1

  grep -E "TTFT|latency|throughput|Throughput|Cache" "$out" | sed 's/^/    /'
}

# =============================================================================
log "=== Scheduler Optimization Benchmark ==="
log "Model:    $MODEL"
log "Rate:     $REQUEST_RATE req/s"
log "Clients:  $NUM_CLIENTS (max_parallel=$MAX_PARALLEL)"
log "Prompt:   $REQUEST_LENGTH tokens, Output: $OUTPUT_LENGTH tokens"
log "Runs:     $NUM_RUNS per branch (+ 1 warmup)"
log "Results:  $RESULTS_DIR"
log "========================================="

for idx in "${!BRANCHES[@]}"; do
  branch="${BRANCHES[$idx]}"
  tag="${TAGS[$idx]}"

  log ""
  log "=== $tag ($branch) ==="
  cd "$SGLANG_DIR"
  git checkout --detach "$branch" 2>&1 || git checkout "$branch" 2>&1

  kill_server
  start_server "$tag" || { log "SKIPPING $tag"; continue; }

  # Warmup (discarded)
  log "Warmup..."
  run_bench "${tag}_warmup" 0

  # Actual runs
  for run in $(seq 1 "$NUM_RUNS"); do
    run_bench "$tag" "$run"
  done

  kill_server
done

# =============================================================================
# Summary: compute mean/std from JSONL
# =============================================================================
log ""
log "=== SUMMARY ==="

"$PYTHON" -c "
import json, sys, os, math

results_dir = '$RESULTS_DIR'
tags = ${TAGS[@]/#/\"}
tags = ['baseline', 'opt1_set_cache', 'opt2_any_consolidate']
metrics = ['average_ttft', 'p90_ttft', 'p99_ttft', 'average_latency', 'p90_latency',
           'p99_latency', 'input_token_throughput', 'output_token_throughput', 'throughput']

for tag in tags:
    jsonl = os.path.join(results_dir, f'{tag}.jsonl')
    if not os.path.exists(jsonl):
        print(f'  {tag}: NO DATA')
        continue
    runs = []
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            # skip warmup
            if 'warmup' not in d.get('tag', ''):
                runs.append(d['summary'])
    if not runs:
        print(f'  {tag}: NO RUNS')
        continue
    print(f'\n--- {tag} ({len(runs)} runs) ---')
    for m in metrics:
        vals = [r[m] for r in runs if m in r]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals)) if len(vals) > 1 else 0
        print(f'  {m:30s}: {mean:10.3f} +/- {std:7.3f}')
" 2>&1 || log "Summary script failed, check JSONL files manually"

log ""
log "=== DONE ($(date)) ==="
