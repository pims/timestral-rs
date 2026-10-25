//! Persistent-process release benchmark; emits JSON and one untimed NumPy result.
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use serde_json::json;
use std::time::Instant;
use timestral_rs::{
    forecaster::{ForecastOptions, TimestralForecaster},
    loading::load_model,
};

fn main() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "Build this benchmark with --release"
    );
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args.next().context("Usage: benchmark <checkpoint> <history.csv> <horizon> <warmups> <iterations> <cpu|metal> <output.npy> [options.json]")?;
    let history_path = args.next().context("Missing history CSV")?;
    let horizon: usize = args
        .next()
        .context("Missing horizon")?
        .to_str()
        .context("Invalid horizon")?
        .parse()?;
    let warmups: usize = args
        .next()
        .context("Missing warmup count")?
        .to_str()
        .context("Invalid warmups")?
        .parse()?;
    let iterations: usize = args
        .next()
        .context("Missing iteration count")?
        .to_str()
        .context("Invalid iterations")?
        .parse()?;
    let backend = args.next().context("Missing device")?;
    let backend = backend.to_str().context("Invalid device")?;
    let output = args.next().context("Missing NumPy output path")?;
    let options: ForecastOptions = match args.next() {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => ForecastOptions::default(),
    };
    ensure!(
        args.next().is_none() && horizon > 0 && iterations > 0,
        "Invalid benchmark arguments"
    );
    let history = std::fs::read_to_string(history_path)?
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
                .with_context(|| format!("Invalid value on CSV line {}", i + 1))
        })
        .collect::<Result<Vec<_>>>()?;

    let started = Instant::now();
    let device = match backend {
        "cpu" => Device::Cpu,
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)?
            }
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("Rebuild with --features metal to benchmark the GPU")
            }
        }
        _ => anyhow::bail!("Expected cpu or metal device"),
    };
    device.synchronize()?;
    let device_init_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let forecaster = TimestralForecaster::new(load_model(checkpoint, &device)?);
    device.synchronize()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;

    // Inputs begin on CPU and results finish as CPU F32 values in both runtimes.
    // No model load, file I/O, benchmark validation, or JSON formatting is timed here.
    let predict = || -> Result<(f64, Vec<f32>)> {
        device.synchronize()?;
        let started = Instant::now();
        let forecast = forecaster.forecast_with_options(&history, horizon, &options)?;
        let values = forecast
            .quantiles
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        device.synchronize()?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok((ms, values))
    };
    let (first_forecast_ms, first_values) = predict()?;
    ensure!(
        first_values.iter().all(|v| v.is_finite()),
        "Non-finite forecast"
    );
    for _ in 0..warmups {
        std::hint::black_box(predict()?);
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut last = first_values;
    for _ in 0..iterations {
        let (ms, values) = predict()?;
        samples.push(ms);
        last = values;
    }
    let q = forecaster.model.config.quantile_knots.len();
    Tensor::from_vec(last, (horizon, q), &Device::Cpu)?.write_npy(output)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "engine": "rust", "device": backend, "device_detail": format!("{device:?}"),
            "status": "ok", "build": "release", "history_steps": history.len(), "horizon": horizon,
            "rmsnorm": if backend == "metal" && !cfg!(feature = "unfused-rmsnorm") { "fused" } else { "unfused" },
            "singleton_variate_attention": !cfg!(feature = "full-variate-attention"),
            "options": options, "device_init_ms": device_init_ms, "load_ms": load_ms,
            "first_forecast_ms": first_forecast_ms, "warm_samples_ms": samples,
            "package_version": env!("CARGO_PKG_VERSION"),
        }))?
    );
    Ok(())
}
