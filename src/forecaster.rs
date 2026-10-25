//! Complete context-only forecasting for a single target series.
use anyhow::{ensure, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::{model::TimestralModel, scaler};

pub struct Forecast {
    /// [1, 1, horizon, quantile_levels], on the model's selected device.
    pub quantiles: Tensor,
    pub quantile_levels: Vec<f64>,
}

impl Forecast {
    pub fn quantile(&self, level: f64) -> Result<Tensor> {
        let index = self
            .quantile_levels
            .iter()
            .position(|q| *q == level)
            .ok_or_else(|| anyhow::anyhow!("Unsupported quantile level: {level}"))?;
        Ok(self.quantiles.narrow(3, index, 1)?.squeeze(3)?)
    }

    pub fn median(&self) -> Result<Tensor> {
        self.quantile(0.5)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForecastOptions {
    pub max_context: Option<usize>,
    pub per_shot_horizon_patches: usize,
    pub quantile_real_cap_k: f32,
    pub extreme_value_threshold: f32,
    pub symmetric_averaging: bool,
    pub clip_nonnegative: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            max_context: None,
            per_shot_horizon_patches: 32,
            quantile_real_cap_k: 1e4,
            extreme_value_threshold: 1e30,
            symmetric_averaging: true,
            clip_nonnegative: true,
        }
    }
}

pub struct TimestralForecaster {
    pub model: TimestralModel,
}

impl TimestralForecaster {
    pub fn new(model: TimestralModel) -> Self {
        Self { model }
    }

    /// Forecast one regularly sampled target. Non-finite values mark missing data.
    pub fn forecast(&self, history: &[f32], horizon: usize) -> Result<Forecast> {
        self.forecast_with_options(history, horizon, &ForecastOptions::default())
    }

    pub fn forecast_with_options(
        &self,
        history: &[f32],
        horizon: usize,
        options: &ForecastOptions,
    ) -> Result<Forecast> {
        self.forecast_masked(history, None, horizon, options)
    }

    pub fn forecast_masked(
        &self,
        history: &[f32],
        observation_mask: Option<&[bool]>,
        horizon: usize,
        options: &ForecastOptions,
    ) -> Result<Forecast> {
        let c = &self.model.config;
        ensure!(
            !history.is_empty() && horizon > 0,
            "History and horizon must be nonempty/positive"
        );
        horizon
            .checked_add(c.patch_size - 1)
            .and_then(|h| h.checked_mul(c.quantile_knots.len()))
            .ok_or_else(|| anyhow::anyhow!("Forecast size overflow"))?;
        ensure!(
            observation_mask.is_none_or(|mask| mask.len() == history.len()),
            "Observation mask length must match history"
        );
        let max_context = options.max_context.unwrap_or(c.context_length);
        ensure!(
            (1..=c.context_length).contains(&max_context),
            "max_context must be in 1..={}",
            c.context_length
        );
        ensure!(
            options.per_shot_horizon_patches > 0,
            "per_shot_horizon_patches must be positive"
        );
        ensure!(
            options.quantile_real_cap_k.is_finite() && options.quantile_real_cap_k >= 0.0,
            "Invalid quantile cap"
        );
        ensure!(
            options.extreme_value_threshold.is_finite() && options.extreme_value_threshold > 0.0,
            "Invalid extreme value threshold"
        );
        let observed: Vec<bool> = history
            .iter()
            .enumerate()
            .map(|(i, x)| x.is_finite() && observation_mask.is_none_or(|mask| mask[i]))
            .collect();
        let nonnegative = observed.iter().any(|v| *v)
            && history
                .iter()
                .zip(&observed)
                .all(|(x, obs)| !obs || *x >= 0.0);
        let (context, mask) = prepare_history(history, &observed, max_context, c.patch_size);
        let mut output = self.decode(&context, &mask, horizon, max_context, options)?;
        if options.symmetric_averaging {
            ensure!(
                c.quantile_knots
                    .iter()
                    .zip(c.quantile_knots.iter().rev())
                    .all(|(a, b)| (a + b - 1.0).abs() < 1e-9),
                "Symmetric averaging requires symmetric quantile levels"
            );
            let mirrored: Vec<f32> = context.iter().map(|x| -*x).collect();
            let mirror = self.decode(&mirrored, &mask, horizon, max_context, options)?;
            let q = c.quantile_knots.len();
            for (values, reflected) in output.chunks_exact_mut(q).zip(mirror.chunks_exact(q)) {
                for i in 0..q {
                    values[i] = 0.5 * (values[i] + -reflected[q - 1 - i]);
                }
            }
        }
        if options.clip_nonnegative && nonnegative {
            for value in &mut output {
                *value = value.max(0.0);
            }
        }
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "Forecast produced non-finite values"
        );
        Ok(Forecast {
            quantiles: Tensor::from_vec(
                output,
                (1, 1, horizon, c.quantile_knots.len()),
                self.model.forecast_query.device(),
            )?,
            quantile_levels: c.quantile_knots.clone(),
        })
    }

    fn decode(
        &self,
        initial: &[f32],
        initial_mask: &[bool],
        horizon: usize,
        max_context: usize,
        options: &ForecastOptions,
    ) -> Result<Vec<f32>> {
        let c = &self.model.config;
        let q = c.quantile_knots.len();
        let mut context = initial.to_vec();
        let mut mask = initial_mask.to_vec();
        let mut remaining = horizon.div_ceil(c.patch_size);
        let mut output = Vec::new();
        while remaining > 0 {
            let patches = remaining
                .min(options.per_shot_horizon_patches)
                .min(c.block_size);
            let anchor = self
                .model
                .predict_univariate(&context, &mask, patches, None)?;
            let refined = scaler::refine_block(&anchor, &context, &mask, c)?;
            remaining -= patches;
            if remaining > 0 {
                context.extend(refined.chunks_exact(q).map(|step| {
                    let value = step[10];
                    if value.is_finite() && value.abs() < options.extreme_value_threshold {
                        value
                    } else {
                        0.0
                    }
                }));
                mask.resize(context.len(), true);
                if context.len() > max_context {
                    let keep = c
                        .patch_size
                        .max((max_context / c.patch_size) * c.patch_size);
                    let remove = context.len() - keep;
                    context.drain(..remove);
                    mask.drain(..remove);
                }
            }
            output.extend(refined);
        }
        output.truncate(horizon * q);
        if options.quantile_real_cap_k > 0.0 {
            let stats = scaler::scale_context(
                initial,
                initial_mask,
                c.patch_size,
                c.minimum_scale,
                c.use_asinh,
            )?;
            let span = *stats.scale.last().expect("prepared context") * options.quantile_real_cap_k;
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for (&x, &obs) in initial.iter().zip(initial_mask) {
                if obs {
                    low = low.min(x);
                    high = high.max(x);
                }
            }
            if low == f32::INFINITY {
                low = 0.0;
                high = 0.0;
            }
            for value in &mut output {
                if !value.is_nan() {
                    *value = value.clamp(low - span, high + span);
                }
            }
        }
        ensure!(
            !output.iter().any(|x| x.is_nan()),
            "Forecast produced NaN values; input may be degenerate"
        );
        Ok(output)
    }
}

/// Forward-fill before truncation, retaining the observation mask, then left-pad.
fn prepare_history(
    history: &[f32],
    observed: &[bool],
    max_context: usize,
    patch_size: usize,
) -> (Vec<f32>, Vec<bool>) {
    let mut last = 0.0;
    let filled: Vec<f32> = history
        .iter()
        .zip(observed)
        .map(|(&x, &obs)| {
            if obs {
                last = x;
            }
            last
        })
        .collect();
    let start = history.len().saturating_sub(max_context);
    let kept = history.len() - start;
    let padding = (patch_size - kept % patch_size) % patch_size;
    let mut context = vec![0.0; padding];
    context.extend_from_slice(&filled[start..]);
    let mut mask = vec![false; padding];
    mask.extend_from_slice(&observed[start..]);
    (context, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_fill_precedes_context_crop_and_padding_stays_missing() {
        let history = [2.0, f32::NAN, f32::NAN, f32::NAN, 5.0];
        let observed = [true, false, false, false, true];
        let (values, mask) = prepare_history(&history, &observed, 3, 4);
        assert_eq!(values, [0.0, 2.0, 2.0, 5.0]);
        assert_eq!(mask, [false, false, false, true]);
    }
}
