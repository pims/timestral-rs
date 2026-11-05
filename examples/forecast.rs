//! Original-unit univariate forecasts, with CSV for people and safetensors for checks.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use timestral_rs::{
    forecaster::{ForecastOptions, TimestralForecaster},
    loading::load_model,
};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args.next().context("Usage: forecast <checkpoint> <history.csv|inputs.safetensors> <horizon> <output.csv|output.safetensors> [options.json]")?;
    let input = std::path::PathBuf::from(args.next().context("Missing history input")?);
    let horizon: usize = args
        .next()
        .context("Missing horizon")?
        .to_str()
        .context("Invalid horizon")?
        .parse()?;
    let output = std::path::PathBuf::from(args.next().context("Missing output path")?);
    let options: ForecastOptions = match args.next() {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => ForecastOptions::default(),
    };
    ensure!(args.next().is_none(), "Too many arguments");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let forecaster = TimestralForecaster::new(load_model(checkpoint, &device)?);
    let (history, mask) = if input.extension().is_some_and(|ext| ext == "safetensors") {
        let tensors = candle_core::safetensors::load(&input, &Device::Cpu)?;
        let history = tensors
            .get("history")
            .context("Missing history tensor")?
            .to_vec1::<f32>()?;
        let mask = tensors
            .get("observed")
            .map(|m| {
                m.to_vec1::<u8>()
                    .map(|v| v.iter().map(|v| *v != 0).collect::<Vec<_>>())
            })
            .transpose()?;
        (history, mask)
    } else {
        let text = std::fs::read_to_string(&input)?;
        let history = text
            .lines()
            .enumerate()
            .map(|(i, line)| {
                ensure!(
                    !line.trim().is_empty() && !line.contains(','),
                    "Expected one value on CSV line {}",
                    i + 1
                );
                line.trim()
                    .parse::<f32>()
                    .with_context(|| format!("Invalid value on line {}", i + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        (history, None)
    };
    let started = std::time::Instant::now();
    let forecast = forecaster.forecast_masked(&history, mask.as_deref(), horizon, &options)?;
    let values = forecast
        .quantiles
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let elapsed = started.elapsed();
    if output.extension().is_some_and(|ext| ext == "safetensors") {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("quantiles", forecast.quantiles.to_device(&Device::Cpu)?);
        candle_core::safetensors::save(&tensors, &output)?;
    } else {
        let mut csv = String::from("step");
        for level in &forecast.quantile_levels {
            csv.push_str(&format!(",q{level}"));
        }
        csv.push('\n');
        for (i, step) in values
            .chunks_exact(forecast.quantile_levels.len())
            .enumerate()
        {
            csv.push_str(&i.to_string());
            for value in step {
                csv.push_str(&format!(",{value}"));
            }
            csv.push('\n');
        }
        std::fs::write(&output, csv)?;
    }
    println!(
        "Forecast on {device:?}: {} history steps -> {horizon} steps, 21 quantiles ({:.1} ms)",
        history.len(),
        elapsed.as_secs_f64() * 1000.0
    );
    println!("Saved {}", output.display());
    Ok(())
}
