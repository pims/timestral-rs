#!/usr/bin/env python3
"""Measure HTTP round trips and queue/inference times against a running server."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--history-length", type=int, default=129)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.horizon, args.history_length) < 1:
        parser.error("Counts, concurrency, and lengths must be positive")
    payload = json.dumps({
        "history": [100 + 5 * math.sin(i / 10) for i in range(args.history_length)],
        "horizon": args.horizon,
    }).encode()

    def call(_):
        request = Request(args.url.rstrip("/") + "/v1/forecast", data=payload,
                          headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
                status = response.status
        except HTTPError as error:
            with error:
                result = json.load(error)
                status = error.code
        return status, (time.perf_counter() - started) * 1000, result

    # Warm server caches before the measured run; fail clearly on an invalid setup.
    try:
        status, _, result = call(None)
    except (URLError, TimeoutError) as error:
        parser.exit(1, f"Could not reach forecast server: {error}\n")
    if status != 200:
        parser.exit(1, f"Warmup failed ({status}): {result}\n")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, range(args.requests)))
    elapsed = time.perf_counter() - started
    successful = [(latency, body) for status, latency, body in results if status == 200]
    metrics = {"round_trip": [latency for latency, _ in successful]}
    for name in ("queue", "inference"):
        metrics[name] = [body["timing_ms"][name] for _, body in successful]
    print(json.dumps({
        "requests": args.requests, "concurrency": args.concurrency,
        "statuses": dict(Counter(status for status, _, _ in results)),
        "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(successful) / elapsed,
        "successful_latency_ms": {
            name: {"p50": percentile(values, .5), "p95": percentile(values, .95)}
            for name, values in metrics.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
