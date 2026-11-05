# MODEL-001: Batch independent series

- **Status:** Backlog
- **Created:** 2026-11-05
- **Updated:** 2026-11-05
- **Reference:** [Python `a534bdbd`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)

**Need:** Forecast many unrelated series efficiently using one loaded model.

Python accepts contexts shaped `(B, T)` or `(B, V, T)`. Rust's public forecaster
currently processes one target series per call. Multiple independent series must
currently be submitted as separate calls.

## Acceptance criteria

- [ ] Define a public batch input/output API with explicit ordering, shapes,
      observation masks, padding rules, and covariate alignment.
- [ ] Independent series remain isolated: changing one series or its covariates
      cannot affect another series' forecast or encoding statistics.
- [ ] Batched results match the pinned Python reference and agree with equivalent
      individual Rust forecasts within the documented numerical tolerances.
- [ ] Cover batch size one, multiple series, distinct missing-value patterns,
      numeric/categorical covariates, cropping, and multiple decode blocks.
- [ ] Bound total input/output slots and packed attention allocations before
      allocating model tensors.
- [ ] Validate both checkpoints on CPU and Metal. Measure throughput, latency,
      and peak memory against individual calls before claiming a speedup.

[Back to backlog](model-improvements.md)
