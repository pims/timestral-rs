# MODEL-003: Extend covariate forecast horizons

- **Status:** Backlog
- **Created:** 2026-11-05
- **Updated:** 2026-11-05
- **Reference:** [Python `a534bdbd`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)

**Need:** Support forecasts beyond 256 steps when the caller has enough context,
future inputs, and resources.

Rust already decodes multiple blocks, but covariate forecasts are capped at 256
steps. A previous 1,057-step experiment exceeded the reference tolerance in 28M
tails. This item requires numerical investigation and validation before changing
the supported limit.

## Acceptance criteria

- [ ] Reproduce and explain the existing long-horizon discrepancy with the pinned
      source, checkpoints, options, and per-stage errors recorded.
- [ ] Cover horizons above 256, including native block boundaries and 1,057 steps,
      plus forced smaller blocks and context cropping.
- [ ] Verify categorical mappings are fitted once at the original cutoff and
      never refitted against predicted targets.
- [ ] Compare normalized outputs, medians, and transformed tails separately using
      justified tolerances; do not widen tolerances merely to pass cases.
- [ ] Keep attention/input/output limits effective as contexts and horizons grow,
      and measure memory and latency for representative workloads.
- [ ] Document the validated library limit for each supported backend.

[Back to backlog](model-improvements.md)
