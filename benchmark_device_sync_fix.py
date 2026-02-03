#!/usr/bin/env python3
"""
Benchmark to measure device sync fix impact in FlashInfer prefill.

Usage:
    1. python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000
    2. python benchmark_device_sync_fix.py
    3. Compare results between main branch and fix branch.
"""
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def one(url, model, prompt_tokens, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "x " * prompt_tokens}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=payload)
    dt = time.perf_counter() - t0
    return dt, r.json()["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--prompt-tokens", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()

    # warmup
    for _ in range(5):
        one(args.url, args.model, 10, 20)

    t0 = time.perf_counter()
    lats, toks = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(one, args.url, args.model, args.prompt_tokens, args.max_tokens)
            for _ in range(args.requests)
        ]
        for f in as_completed(futs):
            dt, tk = f.result()
            lats.append(dt)
            toks += tk
    total = time.perf_counter() - t0

    lats.sort()
    p99 = lats[int(0.99 * len(lats))]
    print(
        f"total={total:.2f}s  req/s={args.requests/total:.2f}  "
        f"tok/s={toks/total:.1f}  p99={p99*1000:.1f}ms"
    )


if __name__ == "__main__":
    main()
