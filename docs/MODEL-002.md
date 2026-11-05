# MODEL-002: Forecast related targets jointly

- **Status:** Backlog
- **Created:** 2026-11-05
- **Updated:** 2026-11-05
- **Reference:** [Python `a534bdbd`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)
- **Dependency:** [MODEL-004](MODEL-004.md) for full categorical compatibility.

**Need:** Allow related target series to condition on each other and return a
forecast for every target.

Python's `V` dimension supports jointly forecast target variables. Rust currently
supports one target plus exogenous inputs, which are not forecast outputs.

## Acceptance criteria

- [ ] Define multiple-target inputs and stable output mapping for every target's
      quantiles, preserving the existing single-target API.
- [ ] Assemble target roles, masks, normalization, clipping eligibility, and
      refinement independently for each target while allowing the reference's
      target-to-target attention.
- [ ] Feed each target's predicted median into subsequent decode blocks while
      advancing supplied known-future inputs and masking past-only inputs.
- [ ] Match Python for multiple targets alone and with numeric past/known
      covariates, including signed targets, missing values, and cropped contexts.
- [ ] Define categorical behavior explicitly. Python uses supplied numeric codes
      directly when there is no single target to encode against. Do not invent
      an aggregate target mean or silently coerce string labels into numbers.
- [ ] Validate isolation between independent batch groups, resource bounds, and
      both checkpoints on CPU and Metal.

An initial slice can cover joint targets with numeric covariates. Full categorical
compatibility depends on [MODEL-004](MODEL-004.md). [MODEL-001](MODEL-001.md)'s group-isolation work can be reused.

[Back to backlog](model-improvements.md)
