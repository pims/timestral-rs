# MODEL-004: Accept numeric categorical codes

- **Status:** Backlog
- **Created:** 2026-11-05
- **Updated:** 2026-11-05
- **Reference:** [Python `a534bdbd`](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)
- **Related:** [MODEL-002](MODEL-002.md) for multi-target behavior.

**Need:** Accept existing categorical datasets without requiring callers to
convert their category identities to strings.

Python accepts finite numeric codes with NaN for missing labels and preserves
code identities in float64 before encoding. Rust accepts `Option<String>` labels.

## Acceptance criteria

- [ ] Add a typed library representation for numeric category identities without
      changing the meaning of existing string/null inputs.
- [ ] Preserve distinct codes beyond float32's exact-integer range; do not cast
      category identities to float32 before fitting the mapping.
- [ ] Define missing, non-finite, mixed string/numeric, and identity-equality rules
      explicitly. Retain label/cardinality bounds appropriate to each input type.
- [ ] Match Python encodings and complete forecasts for seen, unseen, missing,
      rare, and large numeric codes, including cropped fitting windows.
- [ ] Cover the multi-target raw-code behavior described in [MODEL-002](MODEL-002.md) when that
      API is implemented.

[Back to backlog](model-improvements.md)
