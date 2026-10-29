# Forecasting and plotting

Run all commands from the repository root. Python commands use the existing parent project environment (`../.venv`).

Once the checkpoint is downloaded, forecasting itself needs only Rust/Candle:

```sh
cargo run --release --features metal --example forecast -- \
  models/Timestral-5M examples/history.csv 96 forecast.csv
```

The included history is synthetic. Replace it with a headerless CSV containing one
numeric value per line, ordered oldest to newest at a regular interval. `nan` or
other non-finite observations are missing. The horizon counts observations. Output
contains a `step` column and all 21 quantile columns, including `q0.5` for the median.
The timing includes input preparation and both symmetric passes, excludes model
loading, and synchronizes the output back from Metal. Omit `--features metal` for CPU.

An optional fifth argument is a JSON options file; omitted fields retain defaults:

```json
{
  "max_context": 1024,
  "per_shot_horizon_patches": 32,
  "quantile_real_cap_k": 10000,
  "extreme_value_threshold": 1e30,
  "symmetric_averaging": true,
  "clip_nonnegative": true
}
```

`max_context` defaults to the trained 32,768-step cap and must be between 1 and that
cap. Horizons beyond one decode block feed refined medians back as context; the
oldest values are dropped as needed. The quantile cap uses observed context bounds
plus/minus `quantile_real_cap_k * final_context_scale`; zero disables it.

The Rust API is `TimestralForecaster::new(model).forecast(&history, horizon)`.
`forecast_with_options` controls the settings above; `forecast_masked` additionally
accepts an observation mask. Results have shape `[1, 1, horizon, 21]` on the selected
device, with `median()` and `quantile(level)` accessors.

## Plot an HTTP forecast

Plot the daily-sales example (28 history days and a 7-day forecast) with its
median and p10–p90 band. With the server running, from this directory:

```sh
../.venv/bin/python scripts/plot_forecast.py
open output/sales-forecast.png
```

The existing project Python environment includes Matplotlib. Use `--request
path/to/request.json` for your own series, `--url` for another server, or
`--response path/to/response.json` to plot a saved response without a server.
The default request is `examples/sales_request.json`; the plotting script requests
p10, p50 and p90. Missing observations appear as gaps in the history line.

See [development and validation](development.md#validate-complete-forecasts) for Python parity checks.
