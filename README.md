# timestral-rs

This folder implements original-unit forecasting for a single target series with
the official 5M and 28M checkpoints. It handles missing observations, causal normalization,
all configured transformer layers, 21 ordered quantiles, advancing-frame refinement,
long-horizon block decoding, symmetric averaging, and clipping. CPU is the default;
Metal runs the neural network on Apple GPUs. Statistics and forecast orchestration
run on CPU. Batching, multivariate forecast assembly and covariates are not yet
supported by the public forecaster, although the transformer supports variable attention.
The existing Python demo in the parent directory is the reference implementation.

```text
src/config.rs       loading settings and 5M/28M architecture validation
src/loading.rs      strict local safetensors loading and tensor validation
src/layers.rs       residual MLP for patch embedding and raw output projection
src/model.rs        context-only univariate forecast block and diagnostics
src/scaler.rs       causal normalization, inverse scaling and frame refinement
src/masks.rs        context/forecast masks and positions for a single target row
src/transformer.rs  scheduled time/variable attention stack and final RMSNorm
src/forecaster.rs   univariate API, missing data, block decoding and clampsym
src/head.rs         quantile projection and non-crossing quantile transformation
src/main.rs         checkpoint loading and real patch projection smoke check
examples/
  residual_block.rs synthetic patch-embedding shape demonstration
  first_block.rs    execute one block and export intermediate tensors
  transformer.rs    execute all configured layers and export intermediate tensors
  quantile_head.rs  execute the head or transform explicit raw logits
  network.rs        transformer plus head, returning normalized quantiles
  forecast.rs       forecast a history CSV without Python at runtime
  forecast_block.rs export original-unit single-block stages
```

## HTTP server

Run from this directory with the downloaded 5M checkpoint:

```sh
cargo +stable run --release --features server --bin server -- models/Timestral-5M
```

For Apple GPU inference:

```sh
cargo +stable run --release --features server,metal --bin server -- models/Timestral-5M --device metal
```

The same binary supports either size. To serve 28M instead, download its checkpoint
once and change the checkpoint argument:

```sh
../.venv/bin/python scripts/download_model.py --model 28M
cargo +stable run --release --features server,metal --bin server -- models/Timestral-28M --device metal
```

For CPU, use `--features server` and omit `--device metal`. One model is loaded per
server process, selected by the checkpoint configuration rather than its directory
name. `/v1/model`, forecast responses and startup logs identify the loaded size;
the request body stays the same. To compare both concurrently, start a second
process with a different `--bind` port.

| Checkpoint | Parameters | Width | Heads | Layers | F32 weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M | 5,208,960 | 192 | 3 | 8 | ~21 MB |
| 28M | 27,821,184 | 384 | 6 | 12 | ~111 MB |

The server loads and warms the model once before listening on `127.0.0.1:8080`.
Startup logs report checkpoint load time (including device synchronization),
warmup time, and total startup time separately. Total startup also includes device
initialization and checkpoint hashing.
It uses the native Rust forecaster; Python is not required at runtime. From a
second terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/forecast \
  -H 'Content-Type: application/json' \
  -d '{"history":[100,102,null,101],"horizon":24,"quantiles":[0.1,0.5,0.9]}'
```

`null` represents a missing observation. At least one observation must be present.
`quantiles` is optional and defaults to `[0.1,0.5,0.9]`; supported levels are
listed by `/v1/model`. Results are in the original units, using the forecaster's
default symmetric averaging and nonnegative clipping:

```json
{
  "request_id": "0000000000000001",
  "model": "mistralai/Timestral-5M",
  "horizon": 2,
  "quantiles": {"0.1": [99.0, 98.0], "0.5": [101.0, 102.0], "0.9": [103.0, 106.0]},
  "timing_ms": {"queue": 0.02, "inference": 7.5}
}
```

The numbers above illustrate the response format. Each selected quantile contains
exactly `horizon` values. Queue timing begins after validation; inference timing
includes copying GPU results to CPU and selecting quantiles. Neither includes HTTP
transport or JSON serialization. Forecast responses also carry `X-Request-ID`.

Plot the daily-sales example (28 history days and a 7-day forecast) with its
median and p10–p90 band. With the server running, from this directory:

```sh
../.venv/bin/python scripts/plot_forecast.py
open output/sales-forecast.png
```

The existing project Python environment includes Matplotlib. Use `--request
path/to/request.json` for your own series, `--url` for another server, or
`--response path/to/response.json` to plot a saved response without a server.
The default request is `examples/sales_request.json`; the plotting script requests
p10, p50 and p90. Missing observations appear as gaps in the history line.

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Process liveness |
| `GET /readyz` | Worker readiness and current queued job count |
| `GET /v1/model` | Device, model dimensions, quantile levels, checkpoint/config SHA-256, optimization flags and configured limits |
| `POST /v1/forecast` | Forecast one target series |

One dedicated thread owns the model and runs forecasts serially. HTTP handlers
stay asynchronous. The queue holds up to 32 waiting jobs, with admission limited
before parsing to one active request plus queue capacity. Overload returns `503`
with `Retry-After: 1`. Error responses use
`{"error":{"code":"...","message":"...","request_id":"..."}}`.
Malformed JSON returns `400`, missing JSON content type `415`, invalid input `422`,
oversized bodies `413`, request deadlines `504`, and inference failures `500`.
Forecast request/status/timing logs go to stderr without logging history values.

| Option | Default |
| --- | --- |
| `--bind` | `127.0.0.1:8080` |
| `--device` | `cpu` (`metal` requires the Metal feature) |
| `--queue-capacity` | `32` (allowed: 1–4096) |
| `--timeout-ms` | `30000` |
| `--max-history` | `32768` (cannot exceed model context) |
| `--max-horizon` | `1024` |
| `--max-body-bytes` | `1048576` |

The request deadline includes body parsing, queueing and inference. An already
running forecast finishes even if its HTTP request expires or disconnects;
expired/canceled queued jobs are skipped. Ctrl-C or SIGTERM stops new admissions,
drains accepted requests, then joins the worker. Shutdown waits for running
inference. There is no authentication or TLS in this initial server; put those at
a reverse proxy before making it available beyond localhost.

Run the HTTP contract/concurrency tests and the real-checkpoint parity test:

```sh
cargo +stable test --features server --test http
cargo +stable test --features server --test http real_model_matches_direct_forecast -- --ignored
cargo +stable test --features server --test http real_28m_model_matches_direct_forecast -- --ignored
```

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

## Try the scaffold

Requires a current stable Rust/Cargo and access to crates.io for the initial
dependency download. `rust-toolchain.toml` selects stable for this folder.

```sh
cd timestral-candle
cargo run --example residual_block

# Apple GPU (requires a supported Mac and Metal build tools)
cargo run --features metal --example residual_block
```

CPU is the default. With `--features metal`, device creation must succeed; there
is no automatic fallback. The residual-block example uses synthetic weights.

## Forecast a series

Once the checkpoint is downloaded, forecasting itself needs only Rust/Candle:

```sh
cargo run --release --features metal --example forecast -- \
  models/Timestral-5M examples/history.csv 96 forecast.csv
```

The included history is synthetic. Replace it with a headerless CSV containing one
numeric value per line, ordered oldest to newest at a regular interval. `nan` or
other non-finite observations are missing. The horizon counts observations. Output
contains a `step` column and all 21 quantile columns, including `q0.5` for the median.
The timing includes input preparation and both symmetric passes, excludes model
loading, and synchronizes the output back from Metal. Omit `--features metal` for CPU.

An optional fifth argument is a JSON options file; omitted fields retain defaults:

```json
{
  "max_context": 1024,
  "per_shot_horizon_patches": 32,
  "quantile_real_cap_k": 10000,
  "extreme_value_threshold": 1e30,
  "symmetric_averaging": true,
  "clip_nonnegative": true
}
```

`max_context` defaults to the trained 32,768-step cap and must be between 1 and that
cap. Horizons beyond one decode block feed refined medians back as context; the
oldest values are dropped as needed. The quantile cap uses observed context bounds
plus/minus `quantile_real_cap_k * final_context_scale`; zero disables it.

The Rust API is `TimestralForecaster::new(model).forecast(&history, horizon)`.
`forecast_with_options` controls the settings above; `forecast_masked` additionally
accepts an observation mask. Results have shape `[1, 1, horizon, 21]` on the selected
device, with `median()` and `quantile(level)` accessors.

### Validate complete forecasts

```sh
cargo build --features metal --example forecast --example forecast_block
../.venv/bin/python scripts/compare_forecast.py models/Timestral-5M
```

This checks seven single-block cases across 11 stages, including exact Rust mask
and position agreement, then 17 full forecasts against the official Python API.
Cases cover constant/missing/short histories, explicit masks, caps, symmetry and
nonnegative clipping, forced multi-block feedback, context truncation, and a
1,057-step horizon. It also checks CSV I/O and rejection of empty histories.

Normalized stages use elementwise `rtol=1e-4`, `atol=2e-4`. Full forecasts use
`rtol=1e-4` and an absolute floor of `2e-4 * max(1, history_std)` in original units.
The script reports maximum and median-channel errors; long-horizon tail quantiles
can have much larger absolute errors because their magnitudes grow during frame
refinement. The raw block checks precede the forecaster's final caps and symmetry.

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

## Load either checkpoint

From this folder, use the existing parent Python environment for the one-time
download (about 21 MB for 5M or 111 MB for 28M). Optionally pass
`--revision <commit>` to pin it. The old `download_5m.py` command still defaults to 5M.

```sh
../.venv/bin/python scripts/download_model.py --model 5M
cargo run --features metal -- models/Timestral-5M patch-metal.npy

../.venv/bin/python scripts/download_model.py --model 28M
cargo run --features metal -- models/Timestral-28M patch-28m.npy
```

The Rust loader reads local files only. It checks the configuration, every tensor
name, shape and F32 dtype, and the expected parameter count for the selected size. All tensors
load onto the chosen device. It constructs the patch projection, learned forecast
query, complete transformer, and monotone quantile head.
The smoke check feeds deterministic `[1, 3, 64]` features through the real patch
projection and optionally saves its `[1, 3, width]` output as a NumPy file.
`loading::load_model` accepts either supported architecture; `loading::load_5m`
remains available and continues to reject larger models.

Compare that output with the original Python layer:

```sh
../.venv/bin/python scripts/compare_patch_embedding.py models/Timestral-5M patch-metal.npy
../.venv/bin/python scripts/compare_patch_embedding.py models/Timestral-28M patch-28m.npy
```

For CPU, omit `--features metal`. Without Python at runtime, you can also copy
`config.json` and `model.safetensors` from the official checkpoint into a local
directory and pass its path directly to the Rust executable.

The integration check uses seeded synthetic weights from the official Python
model for both sizes, verifies patch projection parity, and checks rejection of missing,
extra, wrong-shape, wrong-dtype tensors and unsupported architectures:

```sh
cargo build
../.venv/bin/python tests/check_loading.py target/debug/timestral-rs
```

Synthetic-checkpoint validation does not establish parity with released weights.

All checkpoint-driven diagnostic examples, comparison scripts and the native
benchmark accept either size. For example, after `cargo build --examples`:

```sh
../.venv/bin/python scripts/compare_network.py models/Timestral-28M
../.venv/bin/python scripts/compare_forecast.py models/Timestral-28M
```

The full 1,024-step single-block diagnostic bypasses the public forecast's output
caps and can generate enormous tail quantiles. For that case, the comparison checks
normalized network outputs end to end, then checks denormalization and advancing
refinement against Python using identical upstream inputs. This isolates nonlinear
amplification of F32 differences without widening tolerances. It still reports the
original end-to-end error. Complete public forecasts, including the 1,057-step
multi-block case, retain their end-to-end comparisons.

Use `scripts/benchmark.py --checkpoint models/Timestral-28M ...` to benchmark 28M;
the report records the selected model identity and configuration hash.

## Compare the first transformer block

Build the example for your chosen backend, then let the Python comparison script
generate test inputs and invoke it. The script uses the released weights and
compares 19 intermediate tensors with the official implementation:

```sh
cargo build --features metal --example first_block
../.venv/bin/python scripts/compare_first_block.py models/Timestral-5M
```

For CPU validation, build with `cargo build --example first_block` instead. Each
build replaces the same example executable, so run the comparison after building
the desired backend. You can select a different executable using `--binary`.

Cases cover embedded context plus learned forecast tokens, missing leading
patches, negative and large positions, fully masked queries, zero input, a single
token, and an 11-token random-input stress case. A changed future token also
checks that its value cannot affect earlier outputs. Python verifies the explicit
reference calculation against the original block's split context/horizon path.

The comparison uses `rtol=1e-4` and `atol=2e-4`, prints measured maximum absolute
errors at each stage, and fails on a missing stage or tolerance violation. The
absolute floor accounts for accumulated F32 differences and near-zero cancellation
in the trained MLP on random inputs; override `--rtol` / `--atol` if needed.
RoPE sine/cosine tables have a separate, tighter `1e-6` relative/absolute check.
They are calculated on CPU in F32 and uploaded to the selected device: Metal's
trigonometric evaluation showed excessive error for large position angles.
Attention, projections, normalization, and the MLP still execute on Metal when
enabled. Table preparation currently adds a CPU/GPU transfer; caching tables is
a future optimization.

The block deliberately materializes attention matrices and trace intermediates
for debugging. It accepts an explicit mask and positions. This is not a long-context
performance benchmark.

## Compare the full transformer

```sh
cargo build --features metal --example transformer
../.venv/bin/python scripts/compare_transformer.py models/Timestral-5M
```

For CPU, build with `cargo build --example transformer` instead. This executes
all eight layers, including time then variable attention at indices 3 and 7,
and the final RMSNorm. The regular forward path avoids retaining diagnostic
traces. Both regular and traced execution are checked by the example.

The comparison checks 158 tensors per case. Cases include a single row, multiple
variables, independent groups in one packed batch, shifted positions, fully
masked queries, and target/past-only/known-future covariate masks. Additional
invariants check that changing a peer affects its group, changing another group
does not leak across groups, changing future tokens cannot affect earlier ones,
and changing targets cannot affect covariate rows.

The comparison keeps elementwise `rtol=1e-4`, `atol=2e-4` at the final normalized
output and the `1e-6` RoPE checks. For intermediate vectors across eight layers,
it bounds the RMS error by `atol + rtol * max(1, reference_RMS)` along the final
dimension. This accounts for accumulated F32 roundoff and cancellation in large
intermediate activations. It prints maximum absolute layer-output errors too.
The original first-block comparison retains its elementwise stage checks.

Python constructs masks and positions for these transformer-only fixtures. The
univariate forecaster constructs its own masks and positions in Rust.
Attention still materializes square
matrices, so these examples do not establish long-context memory/performance.

## Compare the quantile head and normalized network

```sh
cargo build --features metal --example quantile_head --example network
../.venv/bin/python scripts/compare_quantile_head.py models/Timestral-5M
../.venv/bin/python scripts/compare_network.py models/Timestral-5M
```

Omit `--features metal` to build the CPU examples. The head comparison checks
the raw projection, stable softplus, probability-gap-scaled positive increments,
and ordered quantiles against the official head. It covers eight cases including
a full 32-patch decode block, zero/random inputs, large finite logits, tiny
softplus tails, and saturation near F32's maximum. Every case checks quantile
ordering and exact preservation of the raw median. The extreme saturation case
reports error relative to tensor scale as well as absolute error.

The network comparison uses the same ten masked fixtures as the transformer
checks. It selects forecast tokens after the encoder split and checks the entire
transformer-to-head result with elementwise `rtol=1e-4`, `atol=2e-4`. Outputs are
`[rows, forecast_patches * 32, 21]` and remain in normalized space; for the
covariate-window path these include all decoder patches before horizon selection.
These diagnostic examples deliberately omit original-unit scaling/refinement;
the public forecaster applies both.

`TimestralModel::normalized_quantiles` provides this path directly for embedded
tokens with explicit masks, positions and an encoder split. The configured
`quantile_width_factor` is read from the checkpoint (3.0 for the released 5M
model), with the official 2.0 default when the setting is absent.

### Older Rust installations

An `edition2024` manifest error means the active Cargo is too old for a dependency.
Update stable rather than installing a newer nightly. If rustup reports that
`rust-std` for `wasm32-wasi` is unavailable, remove that retired target from the
existing stable toolchain first:

```sh
rustup target remove wasm32-wasi --toolchain stable
rustup update stable
cargo +stable --version
cargo +stable run --features metal --example residual_block
```

This project does not need a WASI target. For separate projects targeting WASI
preview 1, the replacement target is `wasm32-wasip1`. Updating only stable avoids
the unrelated nightly toolchain's retired-target issue. If an explicit directory
override still selects nightly, use `cargo +stable` as above.

## Data flow: smallest checkpoint

Timestral-5M has 8 transformer layers, width 192, and 32-step patches.
For one series with 96 history steps and a 64-step horizon:

```text
history [1, 96] + observation mask
  -> causal normalization
  -> patch features [1, 3, 64] (32 values + 32 missing indicators)
  -> residual patch projection [1, 3, 192]
  -> append 2 learned forecast tokens [1, 5, 192]
  -> 8 transformer layers, following the checkpoint's attention schedule
  -> final RMSNorm; select forecast tokens [1, 2, 192]
  -> residual quantile head [1, 2, 32 * 21]
  -> ordered quantiles [1, 64, 21]
  -> denormalization and advancing-frame refinement
```

This is the path without covariates. The covariate path needs additional tokens
and role-aware masks. Full forecasting also includes symmetric averaging,
clipping, and block-by-block median feedback for long horizons.

## Remaining work

1. Validate complete univariate forecasts on Metal with the comparison above.
2. Extend public forecast assembly to batches, multiple targets and covariates.
3. Benchmark and optimize attention memory, table caching, and allocation reuse.

Keep running statistics in CPU f64 initially; run the neural network in f32.
Validate QKV layout, partial RoPE, mask semantics (including fully masked queries),
and `gate * silu(up)` ordering explicitly. Avoid assuming a generic transformer
block matches Timestral.

References: [Timestral source](https://github.com/mistralai/timestral),
[5M checkpoint](https://huggingface.co/mistralai/Timestral-5M),
[Candle](https://github.com/huggingface/candle).
