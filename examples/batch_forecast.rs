//! Forecast a JSON batch, or the same members individually, with one loaded model.
mod support;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::time::Instant;
use support::{numeric, CovariatesInput};
use timestral_rs::{
    batch::{BatchLimits, BatchSeries},
    forecaster::{ForecastOptions, TimestralForecaster},
    loading::load_model,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    history: Vec<Option<f32>>,
    observation_mask: Option<Vec<bool>>,
    #[serde(default)]
    covariates: CovariatesInput,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    series: Vec<Member>,
    horizon: usize,
    #[serde(default)]
    options: ForecastOptions,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let checkpoint = args.next().context(
        "Usage: batch_forecast <checkpoint> <request.json> <output.npy> [cpu|metal] [batch|individual] [repetitions]",
    )?;
    let request: Request =
        serde_json::from_slice(&std::fs::read(args.next().context("Missing request")?)?)?;
    let output = args.next().context("Missing output")?;
    let backend = args.next().unwrap_or_else(|| "cpu".into());
    let mode = args.next().unwrap_or_else(|| "batch".into());
    let repetitions: usize = args.next().unwrap_or_else(|| "1".into()).parse()?;
    anyhow::ensure!(repetitions > 0, "Repetitions must be positive");
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)?
            }
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("Rebuild with --features metal")
            }
        }
        _ => anyhow::bail!("Expected cpu or metal"),
    };
    let forecaster = TimestralForecaster::new(load_model(checkpoint, &device)?);
    let members = request
        .series
        .into_iter()
        .map(|member| {
            Ok((
                numeric(member.history)?,
                member.observation_mask,
                member.covariates.into_covariates()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let series: Vec<_> = members
        .iter()
        .map(|(history, mask, covariates)| BatchSeries {
            history,
            observation_mask: mask.as_deref(),
            covariates: Some(covariates),
        })
        .collect();
    let run = || -> Result<Tensor> {
        Ok(match mode.as_str() {
            "batch" => {
                forecaster
                    .forecast_batch_with_options(
                        &series,
                        request.horizon,
                        &request.options,
                        &BatchLimits::default(),
                    )?
                    .quantiles
            }
            "individual" => {
                let forecasts = members
                    .iter()
                    .map(|(history, mask, covariates)| {
                        Ok(forecaster
                            .forecast_with_covariates(
                                history,
                                mask.as_deref(),
                                request.horizon,
                                covariates,
                                &request.options,
                            )?
                            .quantiles)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&forecasts, 0)?
            }
            _ => anyhow::bail!("Expected batch or individual"),
        }
        .to_device(&Device::Cpu)?)
    };
    // Warm up benchmark runs. The CPU copy synchronizes device work.
    if repetitions > 1 {
        run()?;
    }
    let mut elapsed = Vec::with_capacity(repetitions);
    let mut forecast = None;
    for _ in 0..repetitions {
        let started = Instant::now();
        forecast = Some(run()?);
        elapsed.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    elapsed.sort_by(f64::total_cmp);
    eprintln!(
        "forecast_device={backend} mode={mode} repetitions={repetitions} median_ms={:.3}",
        elapsed[repetitions / 2]
    );
    forecast
        .context("No forecast produced")?
        .write_npy(output)?;
    Ok(())
}
