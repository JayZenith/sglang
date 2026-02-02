#!/usr/bin/env python3
"""
Benchmark script to measure the impact of the device sync fix in FlashInfer prefill.

Usage:
    1. Start the server:
       python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000

    2. Run this benchmark:
       python benchmark_device_sync_fix.py

    3. Compare results between main branch and fix/remove-flashinfer-device-sync branch.

Expected improvements with fix:
    - ~8% throughput improvement under concurrent load
    - ~34% P99 latency improvement
"""

import requests
import time
import statistics
import concurrent.futures
import argparse


def send_request(url, model, prompt_tokens, max_tokens):
    prompt = "x " * prompt_tokens
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    start = time.perf_counter()
    resp = requests.post(f"{url}/v1/chat/completions", json=data)
    elapsed = time.perf_counter() - start
    result = resp.json()
    tokens_generated = result['usage']['completion_tokens']
    return elapsed, tokens_generated


def benchmark_concurrent(url, model, num_requests, prompt_tokens, max_tokens, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(send_request, url, model, prompt_tokens, max_tokens)
            for _ in range(num_requests)
        ]
        start = time.perf_counter()
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    tokens = sum(r[1] for r in results)
    return {
        'total_time': total_time,
        'avg_latency': statistics.mean(latencies),
        'p50_latency': statistics.median(latencies),
        'p99_latency': sorted(latencies)[int(0.99 * len(latencies))],
        'throughput_req_s': num_requests / total_time,
        'throughput_tok_s': tokens / total_time,
        'total_tokens': tokens
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark device sync fix")
    parser.add_argument("--url", default="http://127.0.0.1:30000", help="Server URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--prompt-tokens", type=int, default=50, help="Prompt token count")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("DEVICE SYNC FIX BENCHMARK")
    print("=" * 60)
    print(f"Server: {args.url}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}, Requests: {args.requests}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(5):
        send_request(args.url, args.model, 10, 20)

    # Single request latency
    print("\n--- Single Request Latency ---")
    single_latencies = []
    for i in range(5):
        latency, tokens = send_request(args.url, args.model, args.prompt_tokens, args.max_tokens)
        single_latencies.append(latency)
        print(f"  Request {i+1}: {latency*1000:.1f}ms, {tokens} tokens")
    print(f"  Average: {statistics.mean(single_latencies)*1000:.1f}ms")

    # Concurrent load
    print(f"\n--- Concurrent Load ({args.workers} workers, {args.requests} requests) ---")
    result = benchmark_concurrent(
        args.url, args.model, args.requests,
        args.prompt_tokens, args.max_tokens, args.workers
    )
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Avg latency: {result['avg_latency']*1000:.1f}ms")
    print(f"  P50 latency: {result['p50_latency']*1000:.1f}ms")
    print(f"  P99 latency: {result['p99_latency']*1000:.1f}ms")
    print(f"  Throughput: {result['throughput_req_s']:.2f} req/s, {result['throughput_tok_s']:.1f} tok/s")
    print(f"  Total tokens: {result['total_tokens']}")

    print("\n" + "=" * 60)
    print("KEY METRICS TO COMPARE:")
    print(f"  Throughput: {result['throughput_req_s']:.2f} req/s")
    print(f"  P99 Latency: {result['p99_latency']*1000:.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
