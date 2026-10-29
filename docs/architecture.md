# Architecture and implementation

This folder implements original-unit forecasting for a single target series with
the official 5M and 28M checkpoints. It handles missing observations, causal normalization,
all configured transformer layers, 21 ordered quantiles, advancing-frame refinement,
long-horizon block decoding, symmetric averaging, and clipping. CPU is the default;
Metal runs the neural network on Apple GPUs. Statistics and forecast orchestration
run on CPU. Batching, multivariate forecast assembly and covariates are not yet
supported by the public forecaster, although the transformer supports variable attention.
The existing Python demo in the parent directory is the reference implementation.

```text
src/config.rs       loading settings and 5M/28M architecture validation
src/loading.rs      strict local safetensors loading and tensor validation
src/layers.rs       residual MLP for patch embedding and raw output projection
src/model.rs        context-only univariate forecast block and diagnostics
src/scaler.rs       causal normalization, inverse scaling and frame refinement
src/masks.rs        context/forecast masks and positions for a single target row
src/transformer.rs  scheduled time/variable attention stack and final RMSNorm
src/forecaster.rs   univariate API, missing data, block decoding and clampsym
src/head.rs         quantile projection and non-crossing quantile transformation
src/main.rs         checkpoint loading and real patch projection smoke check
examples/
  residual_block.rs synthetic patch-embedding shape demonstration
  first_block.rs    execute one block and export intermediate tensors
  transformer.rs    execute all configured layers and export intermediate tensors
  quantile_head.rs  execute the head or transform explicit raw logits
  network.rs        transformer plus head, returning normalized quantiles
  forecast.rs       forecast a history CSV without Python at runtime
  forecast_block.rs export original-unit single-block stages
```

## Data flow: smallest checkpoint

Timestral-5M has 8 transformer layers, width 192, and 32-step patches.
For one series with 96 history steps and a 64-step horizon:

```text
history [1, 96] + observation mask
  -> causal normalization
  -> patch features [1, 3, 64] (32 values + 32 missing indicators)
  -> residual patch projection [1, 3, 192]
  -> append 2 learned forecast tokens [1, 5, 192]
  -> 8 transformer layers, following the checkpoint's attention schedule
  -> final RMSNorm; select forecast tokens [1, 2, 192]
  -> residual quantile head [1, 2, 32 * 21]
  -> ordered quantiles [1, 64, 21]
  -> denormalization and advancing-frame refinement
```

This is the path without covariates. The covariate path needs additional tokens
and role-aware masks. Full forecasting also includes symmetric averaging,
clipping, and block-by-block median feedback for long horizons.

## Remaining work

1. Validate complete univariate forecasts on Metal with the [forecast comparison](development.md#validate-complete-forecasts).
2. Extend public forecast assembly to batches, multiple targets and covariates.
3. Benchmark and optimize attention memory, table caching, and allocation reuse.

Keep running statistics in CPU f64 initially; run the neural network in f32.
Validate QKV layout, partial RoPE, mask semantics (including fully masked queries),
and `gate * silu(up)` ordering explicitly. Avoid assuming a generic transformer
block matches Timestral.

References: [Timestral source](https://github.com/mistralai/timestral),
[5M checkpoint](https://huggingface.co/mistralai/Timestral-5M),
[Candle](https://github.com/huggingface/candle).
