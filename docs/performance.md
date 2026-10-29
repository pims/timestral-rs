# Performance and profiling

Run all commands from the repository root. Python commands use the existing parent project environment (`../.venv`).

## Benchmark Python against Rust

```sh
cargo build --release --features metal --example benchmark
../.venv/bin/python scripts/benchmark.py
```

Defaults match the included example: 129 history observations, horizon 96, F32,
and the same symmetric averaging/clipping options in both runtimes. Each model
loads once, runs its first forecast, performs five further warmups, then measures
30 complete forecasts. The table separates loading, first forecast, warm median,
and warm p90 latency. Every timed result is synchronized and copied to CPU;
model loading, CSV parsing, output file writing and parity checks are excluded
from forecast latency. Device initialization is recorded separately in JSON.

The runner compares Rust/CPU, Rust/Metal, Python/CPU, and attempts Python/MPS.
Unsupported/unavailable backends are reported with their errors, without silently
switching devices or changing the official implementation. CPU thread settings
default to four for both runtimes, including Candle and Rayon worker pools.
These are implementation/backend comparisons, not measurements of language overhead.

The official Python forecaster converts statistics to float64 on its active
device. On MPS this raises a float64 unsupported-dtype error, recorded as
`python/mps: unsupported`. The Rust port keeps these statistics on CPU and runs
the neural network on Metal. Use `--python-devices cpu` to omit the MPS attempt;
the benchmark will still compare Python/CPU with Rust/CPU and Rust/Metal.

The existing forecast example's single timing measures the first forecast after
loading; compare it with the benchmark's `First ms`, not its warm median.
Rust benchmark binaries must be built with `--release`.

Use `--history measurements.csv --horizon 96 --threads 4 --iterations 50` for a
different workload, or `--options settings.json` to change forecast options in
both implementations. `--rust-devices cpu --python-devices cpu` restricts the
comparison to CPU. The same Metal-enabled binary supports both Rust devices.

Results, raw latency samples, workload hashes and output-parity results are saved
to `output/benchmark.json` (override with `--output`). Successful implementations
are checked against Python/CPU with the same scale-aware tolerance as the forecast
comparison. Keep the input, options, thread count and background load consistent
when comparing runs.

## Single-variable attention experiment

For one row with no variable-attention mask, each attention head has one key and
`softmax([score]) = [1]`. The default path caches the V part of each variable
attention projection, reordered from the checkpoint's `[head_dim, heads]` rows
to the output's `[heads, head_dim]` order. Inference then computes V, sigmoid
gating and the output projection, omitting Q/K projection, scores, softmax and
attention mixing. Both CPU and Metal use this specialization.

Explicit variable masks and multi-row inputs use the general path, preserving
fully masked queries, group isolation and covariate behavior. Diagnostic tracing
still supplies Q/K for Python comparisons; they are computed only when tracing
is requested and do not contribute to the fast path's output.

The feature `full-variate-attention` disables the specialization. These matched
builds retain the same RMSNorm selection, isolating this experiment:

```sh
mkdir -p target/benchmarks
cargo build --release --features metal,full-variate-attention --example benchmark
cp target/release/examples/benchmark target/benchmarks/benchmark-full-variate
cargo build --release --features metal --example benchmark
cp target/release/examples/benchmark target/benchmarks/benchmark-singleton

../.venv/bin/python scripts/benchmark.py --rust-binary target/benchmarks/benchmark-full-variate \
  --rust-devices metal --python-devices cpu --output output/full-variate.json
../.venv/bin/python scripts/benchmark.py --rust-binary target/benchmarks/benchmark-singleton \
  --rust-devices metal --python-devices cpu --output output/singleton.json
```

The benchmark JSON records `singleton_variate_attention` for each Rust result.
The existing transformer, network and complete forecast comparison commands
exercise the specialization automatically for unmasked single-row inputs.

An initial local CPU comparison (six alternating pairs, 100 measured forecasts
per binary per pair, four threads, 129 history steps and horizon 96) measured
aggregate warm medians of 8.14 ms for the general path and 7.70 ms for the
specialization, with identical outputs for that workload. Raw samples are saved
locally in `output/singleton-cpu-ab.json`. Metal correctness and performance
must be checked from a terminal with GPU access using the comparison commands.

## RMSNorm fusion experiment

Metal uses Candle's fused RMSNorm by default. CPU retains the original reduction:
the fused CPU path did not improve our short-history latency and pushed one
long-horizon tail value just outside the existing comparison tolerance. Tolerances
were not changed. The feature `unfused-rmsnorm` selects the original Metal path
for controlled comparisons; the benchmark JSON records the active `rmsnorm` mode.

Build and keep separate optimized executables:

```sh
mkdir -p target/benchmarks
cargo build --release --features metal,unfused-rmsnorm --example benchmark
cp target/release/examples/benchmark target/benchmarks/benchmark-unfused
cargo build --release --features metal --example benchmark
cp target/release/examples/benchmark target/benchmarks/benchmark-fused
```

Measure each on the same workload, with Python CPU as a reference:

```sh
../.venv/bin/python scripts/benchmark.py --rust-binary target/benchmarks/benchmark-unfused \
  --rust-devices metal --python-devices cpu --output output/rmsnorm-unfused.json
../.venv/bin/python scripts/benchmark.py --rust-binary target/benchmarks/benchmark-fused \
  --rust-devices metal --python-devices cpu --output output/rmsnorm-fused.json
```

Repeat in reversed order to check for thermal/background-load effects. Validate
the fused Metal path with `compare_transformer.py` and `compare_forecast.py` using
the usual Metal-built examples before treating a timing improvement as accepted.

## Profile with Samply

Install with `cargo +stable install --locked samply`. The local `profiling`
profile inherits release optimizations and keeps full debug information without
stripping, so Samply can resolve source locations and inlined Rust functions.

```sh
cargo +stable build --profile profiling --features metal --example benchmark

CANDLE_NUM_THREADS=4 RAYON_NUM_THREADS=4 samply record \
  target/profiling/examples/benchmark \
  models/Timestral-5M examples/history.csv 96 10 1000 cpu /tmp/timestral-profile.npy \
  > /tmp/timestral-profile-timings.json
```

The positional arguments request a 96-step forecast, ten warmups and 1,000
measured forecasts. The model loads once; the many repeated forecasts make the
steady-state work dominate the recording. The profiler still records startup
and warmup, so select the later repeated-work portion in the timeline when
examining steady-state stacks. The NumPy file is only the final forecast output,
not the profile. Samply stores the profile locally and opens its browser viewer.
Benchmark JSON output is redirected to the separate timings file to avoid printing
1,000 latency samples in the terminal.
Keep Samply running while using the viewer's local symbol server.

Start with the CPU run to inspect computation hot spots. Use the call tree/flame
graph to find matrix multiplication, allocations, copies, and normalization.
On macOS, examine worker threads as well as the main thread. Change `cpu` to
`metal` to inspect CPU-side GPU submission, copies and waits; Samply does not
provide a breakdown of GPU kernel execution. Use the unprofiled release benchmark
for latency comparisons, since sampling adds overhead.

## HTTP benchmark

With the server running, measure HTTP latency and throughput (standard-library
Python client, no additional packages):

```sh
python3 scripts/benchmark_http.py --requests 100 --concurrency 1
python3 scripts/benchmark_http.py --requests 100 --concurrency 8
```

The report separates successful request round-trip, queue and inference p50/p95
times, reports throughput, and counts all HTTP statuses including overloads.
Round-trip timing includes client/network overhead; use the existing native
benchmark for controlled Python-versus-Rust inference comparisons.
