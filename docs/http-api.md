# HTTP API reference

Run all commands from the repository root. Python commands use the existing parent project environment (`../.venv`).

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

See [development and validation](development.md#http-tests) for API tests,
[performance](performance.md#http-benchmark) for HTTP benchmarks, and
[forecasting and plotting](forecasting.md#plot-an-http-forecast) for visualization.
