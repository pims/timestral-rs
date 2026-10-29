# timestral-rs

Native Rust/Candle time-series forecasting with Timestral-5M and Timestral-28M,
CPU and Apple Metal support, and an HTTP API.

## Download a model

From the repository root, create a Python environment for the download helper:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install huggingface_hub
.venv/bin/python scripts/download_model.py --model 5M
```

This downloads the checkpoint to `models/Timestral-5M` (about 21 MB).
For the larger 28M model (about 111 MB):

```sh
.venv/bin/python scripts/download_model.py --model 28M
```

Models are stored locally under `models/` and excluded from Git. Python is only
needed for the download helper; inference runs entirely in Rust.

## Run the HTTP API

From the repository root, with a current stable Rust toolchain and a
[downloaded checkpoint](docs/models.md#load-either-checkpoint):

```sh
cargo +stable run --release --features server --bin server -- models/Timestral-5M
```

For Apple GPU inference:

```sh
cargo +stable run --release --features server,metal --bin server -- models/Timestral-5M --device metal
```

To use 28M, replace the checkpoint path with `models/Timestral-28M`.
The server loads and warms the model once, prints startup timings, and listens on
`http://127.0.0.1:8080`. Python is not required at runtime.

From another terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/forecast \
  -H 'Content-Type: application/json' \
  -d '{"history":[100,102,null,101],"horizon":24,"quantiles":[0.1,0.5,0.9]}'
```

History is ordered oldest to newest at a regular interval; `null` marks a missing
observation. The response contains 24 forecast values for each selected quantile
in the original units, plus queue and inference timings. `0.5` is the median;
`0.1` and `0.9` give the p10–p90 forecast band.

See the [HTTP API reference](docs/http-api.md) for endpoints, options and limits.
The [documentation index](docs/README.md) covers model downloads, plotting,
validation, benchmarks, profiling and architecture.
