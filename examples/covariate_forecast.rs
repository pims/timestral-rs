mod support;
use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use support::{numeric, CovariatesInput};
use timestral_rs::{
    forecaster::{ForecastOptions, TimestralForecaster},
    loading::load_model,
};

#[derive(Deserialize)]
struct Input {
    history: Vec<Option<f32>>,
    horizon: usize,
    #[serde(default)]
    covariates: CovariatesInput,
}

fn main() -> Result<()> {
    let mut raw = std::env::args_os().skip(1);
    let mut positional = Vec::new();
    let mut backend = String::from("cpu");
    while let Some(arg) = raw.next() {
        if arg == "--device" {
            backend = raw
                .next()
                .context("Missing --device cpu|metal")?
                .into_string()
                .map_err(|_| anyhow::anyhow!("Invalid device name"))?;
        } else {
            positional.push(arg);
        }
    }
    let mut args = positional.into_iter();
    let checkpoint = args.next().context(
        "Usage: covariate_forecast <checkpoint> <request.json> <output.npy> [options.json] [trace.safetensors] [--device cpu|metal]",
    )?;
    let input: Input =
        serde_json::from_slice(&std::fs::read(args.next().context("Missing request")?)?)?;
    let output = args.next().context("Missing output")?;
    let options: ForecastOptions = match args.next() {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => Default::default(),
    };
    let trace_path = args.next();
    let history = numeric(input.history)?;
    let covariates = input.covariates.into_covariates()?;
    covariates.validate(history.len(), input.horizon, 16, 131072)?;
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)?
            }
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("Rebuild with --features metal to use --device metal")
            }
        }
        _ => anyhow::bail!("Expected --device cpu or metal"),
    };
    eprintln!("forecast_device={backend}");
    let forecaster = TimestralForecaster::new(load_model(checkpoint, &device)?);
    if let Some(path) = trace_path {
        let trace =
            forecaster.covariate_block_trace(&history, input.horizon, &covariates, &options)?;
        candle_core::safetensors::save(
            &trace
                .into_iter()
                .collect::<std::collections::HashMap<_, _>>(),
            path,
        )?;
    }
    forecaster
        .forecast_with_covariates(&history, None, input.horizon, &covariates, &options)?
        .quantiles
        .to_device(&Device::Cpu)?
        .write_npy(output)?;
    Ok(())
}
