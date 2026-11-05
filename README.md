# timestral-rs

Native Rust inference and forecasting for Timestral-5M and Timestral-28M,
implemented with Candle. Supports CPU and Apple Metal.

## Supported forecasting

- Single targets and batches of independent targets with missing observations.
- Original-unit forecasts with 21 quantiles.
- Numeric and string/null categorical past-only and known-future covariates.
- Context cropping, block decoding, symmetric averaging, and nonnegative clipping.

Covariate forecasts currently support horizons up to 256 steps, with bounded
attention allocations. Joint forecasting of related targets remains future work.
Categorical integration is validated on CPU and Metal for both checkpoints.

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
`forecast_with_covariates` for numeric or categorical covariate inputs. Covariate input types
and validation live in `covariates`; forecast settings live in `ForecastOptions`.
Missing numeric covariate values are represented by `f32::NAN`. Use
`Covariates::categorical_past` or `Covariates::categorical_known`
(`KnownCategoricalCovariate`) for `Option<String>` labels. Labels are encoded once
against historical target values using the target mask and effective context limit;
future targets and predictions never enter fitting statistics. Labels are exact
UTF-8 identities, bounded to 256 bytes and 1,024 distinct non-null labels per row
across history/future. Null forms its own category; unseen labels use the fitted
target mean. Validation applies these bounds to direct library callers too.

### Independent batches

```rust,ignore
use timestral_rs::batch::BatchSeries;

let series = [BatchSeries::new(&first_history), BatchSeries::new(&second_history)];
let result = forecaster.forecast_batch(&series, 65)?;
// result.quantiles: [2, 1, 65, 21]; result.median(): [2, 1, 65].
```

Members remain in input order and share a history length, horizon, forecast options,
and counts of past and known covariate rows. Counts combine numeric and categorical
rows within each role; names and labels are local to each member. Set
`BatchSeries.observation_mask` and `.covariates` independently for each target.
Historical covariates and masks must match that member's history length; known
future covariates must contain exactly `horizon` values. Scaling, category fitting,
and attention remain isolated between members.

Use leading `f32::NAN` values to align histories of different lengths, with matching
padding in historical covariates and masks. Patch padding is automatic and remains
unobserved. Context cropping and forecast settings work as for individual calls.
Covariate batches retain the 256-step horizon limit.

`forecast_batch_with_options` also accepts `BatchLimits`. Defaults allow 64 members,
16 covariates per member, 131,072 input values, 131,072 output values, and 8,388,608
attention score elements. Input values count all target/covariate history values
before cropping plus known future values. Output values count every quantile.
Attention is bounded across the whole batch before model tensors are allocated;
these bounds do not represent a total process memory limit.

## Development

```sh
cargo test --locked
cargo check --all-targets --all-features --locked
```

The `metal`, `unfused-rmsnorm`, and `full-variate-attention` features control
device support and diagnostic execution paths. Model diagnostic examples and
configuration fixtures are included.

## Python reference environment

Build the Python dependency image once, from this repository root:

```sh
docker build --platform linux/arm64 -t timestral-python -f scripts/parity/Dockerfile .
```

The image contains Python 3.12, CPU PyTorch 2.14.1, NumPy 2.5.4, safetensors 0.8.0,
and the pinned official Timestral reference. It contains no Rust toolchain or
project files. Mount the repository and run Python commands directly:

```sh
docker run --rm --platform linux/arm64 -v "$PWD:/work" timestral-python \
  python3 scripts/test_reference_fixtures.py
```

This runs four Python reference tests covering 23 encoding cases, 28 exact sums,
22 normalization cases, and the pinned source/version checks. It needs no Rust
executable or weights. To regenerate fixtures, run
`python3 scripts/generate_categorical_fixtures.py` in the same container.

To run a different Python test suite, replace the Python command. No wrapper or
image rebuild is needed when changing mounted scripts. The ARM64 platform matches
the reference normalization fixtures; x86 hosts need ARM64 emulation.

## Covariate parity

The model repository owns `examples/covariate_forecast.rs`,
`scripts/compare_covariates.py`, and the fixture generator. Complete Rust/Python
forecast comparisons require a Rust probe executable on the same OS as the Python
comparison process. This is separate from the Python-only dependency image.

```sh
cargo build --release --features metal --example covariate_forecast --locked
python3 scripts/compare_covariates.py /path/to/Timestral-5M --device metal
python3 scripts/compare_covariates.py /path/to/Timestral-28M --device metal
```

The comparison accepts `--binary` for an explicit probe path; otherwise Cargo
metadata locates the build output. Each model must report
**32 complete-forecast cases passed**.

## Native batch validation

Native checkpoint tests also run without Python. Repeat for each checkpoint;
replace `checkpoint_metal` with `checkpoint_cpu` to test CPU:

```sh
TIMESTRAL_CHECKPOINT="$PWD/models/Timestral-5M" cargo test --features metal --test batch_forecast checkpoint_metal --locked -- --ignored
TIMESTRAL_CHECKPOINT="$PWD/models/Timestral-28M" cargo test --features metal --test batch_forecast checkpoint_metal --locked -- --ignored
```

Measure latency, throughput, and peak process RSS in separate fresh processes:

```sh
python3 examples/benchmark_batch.py models/Timestral-5M --binary target/release/examples/batch_forecast --mode batch
python3 examples/benchmark_batch.py models/Timestral-5M --binary target/release/examples/batch_forecast --mode individual
```

The default workload has 16 targets, 97 observations, and a 65-step horizon. Timing
excludes model loading and includes one warmup before 11 measured repetitions.
Peak RSS includes the loaded model and warmup; it excludes GPU-resident memory.
Use `--request request.json` for another workload or `--device metal` for GPU timing.

Planned model capabilities and parity work are tracked in the
[model improvements backlog](docs/model-improvements.md).

Upstream implementation: https://github.com/mistralai/timestral
