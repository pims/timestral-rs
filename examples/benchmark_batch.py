"""Measure one native batch or equivalent individual calls, excluding model loading.

Run each mode in a fresh process to measure its own peak resident memory.
The default workload is 16 independent targets, 97 observations, 65 forecast steps.
Use --request to measure a JSON request accepted by the batch_forecast example.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import re
import resource
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mode", choices=("batch", "individual"), required=True)
    parser.add_argument("--device", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("Use at least two repetitions to include a warmup")
    if args.request:
        data = json.loads(args.request.read_text())
    else:
        members = []
        for index in range(16):
            history = [index * 10 + 8 * math.sin(step / (4 + index)) for step in range(97)]
            members.append({"history": history})
        data = {"series": members, "horizon": 65}
    with tempfile.TemporaryDirectory(prefix="timestral-benchmark-") as directory:
        directory = Path(directory)
        source, output = directory / "request.json", directory / "forecast.npy"
        source.write_text(json.dumps(data, allow_nan=False))
        command = [str(args.binary.resolve()), str(args.checkpoint.resolve()), str(source),
                   str(output), args.device, args.mode, str(args.repetitions)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    timing = re.search(r"median_ms=([0-9.]+)", result.stderr)
    if timing is None:
        raise RuntimeError(f"Missing native timing: {result.stderr}")
    milliseconds = float(timing.group(1))
    peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Darwin reports bytes; Linux reports KiB. This is native process RSS,
    # including the loaded checkpoint, and excludes GPU-resident allocations.
    peak_bytes = peak_rss if platform.system() == "Darwin" else peak_rss * 1024
    print(json.dumps({
        "checkpoint": args.checkpoint.name, "device": args.device, "mode": args.mode,
        "machine": platform.machine(), "system": platform.platform(),
        "series": len(data["series"]), "history": len(data["series"][0]["history"]),
        "horizon": data["horizon"], "repetitions": args.repetitions,
        "median_ms": milliseconds,
        "series_per_second": len(data["series"]) * 1000 / milliseconds,
        "peak_process_rss_bytes": peak_bytes,
    }, indent=2))


if __name__ == "__main__":
    main()
