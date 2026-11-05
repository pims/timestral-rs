# timestral-rs

Native Rust inference and forecasting for Timestral-5M and Timestral-28M,
implemented with Candle. Supports CPU and Apple Metal.

## Supported forecasting

- One target series with missing observations.
- Original-unit forecasts with 21 quantiles.
- Numeric past-only and known-future covariates, including binary indicators.
- Context cropping, block decoding, symmetric averaging, and nonnegative clipping.

Covariate forecasts currently support horizons up to 256 steps, with bounded
attention allocations. Categorical inputs and multiple targets remain future work.

## Checkpoints

Obtain the official `mistralai/Timestral-5M` or `mistralai/Timestral-28M`
checkpoint. Place its `config.json` and `model.safetensors` together in a local
directory such as `models/Timestral-28M`. Model weights are not included here.

## Run a forecast

The example history contains one numeric observation per line. For CPU:

```sh
cargo run --release --example forecast -- \
  models/Timestral-28M examples/history.csv 31 forecast.csv
```

For Apple Metal, add `--features metal` before `--example forecast`.

## Library API

Load a checkpoint with `loading::load_model(path, &device)` and construct a
`forecaster::TimestralForecaster`. Use `forecast` for a target series or
`forecast_with_covariates` for numeric covariate inputs. Covariate input types
and validation live in `covariates`; forecast settings live in `ForecastOptions`.
Missing covariate values are represented by `f32::NAN`.

## Development

```sh
cargo test --locked
cargo check --all-targets --all-features --locked
```

The `metal`, `unfused-rmsnorm`, and `full-variate-attention` features control
device support and diagnostic execution paths. Model diagnostic examples and
configuration fixtures are included.

Upstream implementation: https://github.com/mistralai/timestral
