//! Checkpoint tests: set TIMESTRAL_CHECKPOINT and explicitly run ignored tests.
use anyhow::Result;
use candle_core::Device;
use timestral_rs::{
    batch::{BatchLimits, BatchSeries},
    covariates::{Covariates, KnownCategoricalCovariate},
    forecaster::{ForecastOptions, TimestralForecaster},
    loading::load_model,
};

fn check_batch(device: Device) -> Result<()> {
    let checkpoint = std::env::var("TIMESTRAL_CHECKPOINT")?;
    let forecaster = TimestralForecaster::new(load_model(checkpoint, &device)?);
    let histories: Vec<Vec<f32>> = (0..2)
        .map(|member| {
            (0..97)
                .map(|step| member as f32 * 60.0 - 30.0 + 8.0 * (step as f32 / 4.0).sin())
                .collect()
        })
        .collect();
    let masks: Vec<Vec<bool>> = (0..2)
        .map(|member| {
            (0..97)
                .map(|step| step >= 32 * member && step % 13 != 0)
                .collect()
        })
        .collect();
    let labels: Vec<_> = (0..162)
        .map(|step| Some(format!("day-{}", step % 7)))
        .collect();
    let mut covariates = Covariates::default();
    covariates.categorical_known.insert(
        "weekday".into(),
        KnownCategoricalCovariate {
            history: labels[..97].to_vec(),
            future: labels[97..].to_vec(),
        },
    );
    let options = ForecastOptions {
        max_context: Some(65),
        per_shot_horizon_patches: 1,
        ..Default::default()
    };
    for with_covariates in [false, true] {
        let series: Vec<_> = histories
            .iter()
            .zip(&masks)
            .map(|(history, mask)| BatchSeries {
                history,
                observation_mask: Some(mask),
                covariates: with_covariates.then_some(&covariates),
            })
            .collect();
        let batch = forecaster.forecast_batch_with_options(
            &series,
            65,
            &options,
            &BatchLimits::default(),
        )?;
        assert_eq!(batch.quantiles.dims(), [2, 1, 65, 21]);
        assert_eq!(batch.median()?.dims(), [2, 1, 65]);
        let actual = batch
            .quantiles
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, member) in series.iter().enumerate() {
            let individual = forecaster
                .forecast_batch_with_options(&[*member], 65, &options, &BatchLimits::default())?
                .quantiles
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let observed: Vec<_> = member.history[32..]
                .iter()
                .zip(&masks[index][32..])
                .filter_map(|(value, mask)| mask.then_some(*value as f64))
                .collect();
            let mean = observed.iter().sum::<f64>() / observed.len() as f64;
            let unit = (observed
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (observed.len() - 1) as f64)
                .sqrt()
                .max(1.0);
            let absolute_tolerance = if device.is_cpu() { 2e-4 } else { 1e-3 } * unit;
            for (position, (&value, &expected)) in actual[index * 65 * 21..(index + 1) * 65 * 21]
                .iter()
                .zip(&individual)
                .enumerate()
            {
                let atol = if position % 21 == 10 {
                    2e-4 * unit
                } else {
                    absolute_tolerance
                };
                assert!(value.is_finite() && (value as f64 - expected as f64).abs() <= atol + 1e-4 * (expected as f64).abs(),
                    "covariates={with_covariates} member={index} position={position}: {value} != {expected}");
            }
        }
        let reversed: Vec<_> = series.iter().rev().copied().collect();
        let reordered = forecaster
            .forecast_batch_with_options(&reversed, 65, &options, &BatchLimits::default())?
            .quantiles
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(&actual[..65 * 21], &reordered[65 * 21..]);
        assert_eq!(&actual[65 * 21..], &reordered[..65 * 21]);
    }
    Ok(())
}

#[test]
#[ignore = "requires TIMESTRAL_CHECKPOINT pointing to real weights"]
fn checkpoint_cpu_batch_matches_individual_forecasts() -> Result<()> {
    check_batch(Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires TIMESTRAL_CHECKPOINT and an accessible Metal device"]
fn checkpoint_metal_batch_matches_individual_forecasts() -> Result<()> {
    check_batch(Device::new_metal(0)?)
}
