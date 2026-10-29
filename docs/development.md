# Development and validation

Run all commands from the repository root. Python commands use the existing parent project environment (`../.venv`).

## Try the scaffold

Requires a current stable Rust/Cargo and access to crates.io for the initial
dependency download. `rust-toolchain.toml` selects stable for this folder.

```sh
cargo run --example residual_block

# Apple GPU (requires a supported Mac and Metal build tools)
cargo run --features metal --example residual_block
```

CPU is the default. With `--features metal`, device creation must succeed; there
is no automatic fallback. The residual-block example uses synthetic weights.

## Validate complete forecasts

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

## HTTP tests

Run the HTTP contract/concurrency tests and the real-checkpoint parity test:

```sh
cargo +stable test --features server --test http
cargo +stable test --features server --test http real_model_matches_direct_forecast -- --ignored
cargo +stable test --features server --test http real_28m_model_matches_direct_forecast -- --ignored
```

## Older Rust installations

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
