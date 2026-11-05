# Model improvements backlog

Created: **2026-11-05** (America/Los_Angeles)  
Last updated: **2026-11-05**

Track missing features relative to the
[Python reference (`a534bdbd`)](https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff).

Rust supports one target with numeric or categorical covariates on CPU and Metal
(float32). Covariate forecasts are limited to 256 steps.

## Improvements

- [ ] [MODEL-001: Batch independent series](MODEL-001.md)
- [ ] [MODEL-002: Forecast related targets jointly](MODEL-002.md)
- [ ] [MODEL-003: Extend covariate forecast horizons](MODEL-003.md)
- [ ] [MODEL-004: Accept numeric categorical codes](MODEL-004.md)

## Updating status

Keep an item unchecked (`[ ]`) until its acceptance criteria are complete. Mark
it checked (`[x]`) when done, and record implementation and validation evidence
in its document. Keep each item's status and updated date current; use Backlog,
In progress, Done, or Parked.

Preserve item IDs, reference source hashes, and documented numerical tolerances.
Keep validation tooling self-contained in this repository.

Related guidance: [README](../README.md),
[Python fixture provenance](../tests/fixtures/README.md).
