# MODEL-001: Batch independent series

- **Status:** Done
- **Created:** 2026-11-05
- **Updated:** 2026-11-05
- **Reference:** [Python `a534bdbd`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)

**Need:** Forecast many unrelated series efficiently using one loaded model.

`forecast_batch` accepts independent single-target series and returns quantiles
shaped `(B, 1, horizon, Q)` in input order. Members share history length, horizon,
options, and past/known covariate row counts. Masks and category fitting are local
to each member. Shorter histories require explicit leading missing values.

## Acceptance criteria

- [x] Define a public batch input/output API with explicit ordering, shapes,
      observation masks, padding rules, and covariate alignment.
- [x] Independent series remain isolated: changing one series or its covariates
      cannot affect another series' forecast or encoding statistics.
- [x] Batched results match the pinned Python reference and agree with equivalent
      individual Rust forecasts within the documented numerical tolerances.
- [x] Cover batch size one, multiple series, distinct missing-value patterns,
      numeric/categorical covariates, cropping, and multiple decode blocks.
- [x] Bound total input/output slots and packed attention allocations before
      allocating model tensors.
- [x] Validate both checkpoints on CPU and Metal. Measure throughput, latency,
      and peak memory against individual calls before claiming a speedup.

## Validation

CPU: both checkpoints pass seven Python batch comparisons, native checkpoint
tests, and the existing 32 covariate comparisons. Isolation and reordering require
exact equality; numerical tolerances and commands are in the
[batch validation instructions](../README.md#native-batch-validation).

Metal: both checkpoints pass the native batch test and all seven Python batch
comparisons, confirmed by the contributor on 2026-11-05. Performance measurements
below cover CPU only.

### CPU measurements — 2026-11-05

ARM64 macOS, release build, 16 targets, 97 observations, 65 forecast steps,
default options, no covariates. Median of 11 repetitions after one warmup;
latency excludes model loading. Peak process RSS includes the loaded model.

| Checkpoint | Mode | Latency (ms) | Series/s | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| 5M | Batch | 27.728 | 577.0 | 54.3 |
| 5M | Individual | 116.569 | 137.3 | 52.2 |
| 28M | Batch | 102.219 | 156.5 | 233.3 |
| 28M | Individual | 416.767 | 38.4 | 226.3 |

These measurements describe this workload and machine. GPU memory and Metal
performance have not been measured.

[Back to backlog](model-improvements.md)
