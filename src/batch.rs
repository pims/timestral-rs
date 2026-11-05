//! Independent single-target series, evaluated together with group-local attention.
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Module, Tensor};

use crate::{
    covariate_forecast::{prepare_rows, scale_known, Row},
    covariates::{CovariateAttention, Covariates, MAX_COVARIATE_HORIZON},
    forecaster::{Forecast, ForecastOptions, TimestralForecaster},
    masks::{covariate_masks, univariate_masks, RowRole},
    scaler,
};

/// One batch member. Histories have a common length; masks and covariate arrays
/// align to that member's history. NaN marks missing numeric values. Covariate
/// names are local to each member, and category statistics are never shared.
#[derive(Clone, Copy, Debug)]
pub struct BatchSeries<'a> {
    pub history: &'a [f32],
    pub observation_mask: Option<&'a [bool]>,
    pub covariates: Option<&'a Covariates>,
}

impl<'a> BatchSeries<'a> {
    /// Construct a target without an explicit observation mask or covariates.
    pub fn new(history: &'a [f32]) -> Self {
        Self {
            history,
            observation_mask: None,
            covariates: None,
        }
    }
}

/// Aggregate bounds, checked before preparing rows or allocating model tensors.
/// Attention elements bound score matrices, not total resident process memory.
#[derive(Clone, Debug)]
pub struct BatchLimits {
    pub max_series: usize,
    pub max_covariates_per_series: usize,
    pub max_input_values: usize,
    pub max_output_values: usize,
    pub max_attention_elements: usize,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_series: 64,
            max_covariates_per_series: 16,
            max_input_values: 131_072,
            max_output_values: 131_072,
            max_attention_elements: crate::covariates::DEFAULT_ATTENTION_ELEMENTS,
        }
    }
}

struct Layout {
    context: usize,
    rows_per_group: usize,
    covariates: bool,
}

impl TimestralForecaster {
    /// Forecast independent series in input order. Returns [batch, 1, horizon, Q].
    /// All members share history length, horizon, options, and numeric/categorical
    /// past/known row counts combined by role. Left padding to patch boundaries
    /// is automatic. For unequal histories, supply explicit leading NaNs and
    /// align every historical covariate/mask to the same length.
    pub fn forecast_batch(&self, series: &[BatchSeries<'_>], horizon: usize) -> Result<Forecast> {
        self.forecast_batch_with_options(
            series,
            horizon,
            &ForecastOptions::default(),
            &BatchLimits::default(),
        )
    }

    /// Forecast with shared options and aggregate allocation bounds. Covariate
    /// batches must satisfy both attention limits in `options` and `limits`.
    pub fn forecast_batch_with_options(
        &self,
        series: &[BatchSeries<'_>],
        horizon: usize,
        options: &ForecastOptions,
        limits: &BatchLimits,
    ) -> Result<Forecast> {
        let layout = Self::validate_batch(&self.model.config, series, horizon, options, limits)?;
        let empty = Covariates::default();
        if series.len() == 1 {
            let member = &series[0];
            return self.forecast_with_covariates(
                member.history,
                member.observation_mask,
                horizon,
                member.covariates.unwrap_or(&empty),
                options,
            );
        }
        let c = &self.model.config;
        let mut rows = Vec::new();
        let mut nonnegative = Vec::with_capacity(series.len());
        for member in series {
            let observed: Vec<bool> = member
                .history
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    value.is_finite() && member.observation_mask.is_none_or(|mask| mask[i])
                })
                .collect();
            nonnegative.push(
                observed.iter().any(|value| *value)
                    && member
                        .history
                        .iter()
                        .zip(&observed)
                        .all(|(value, observed)| !observed || *value >= 0.0),
            );
            rows.extend(prepare_rows(
                member.history,
                &observed,
                member.covariates.unwrap_or(&empty),
                horizon,
                layout.context,
                c.patch_size,
            )?);
        }
        let mut output = self.decode_batch(&rows, horizon, options, &layout)?;
        if options.symmetric_averaging {
            let mut mirrored = rows.clone();
            for row in &mut mirrored {
                for value in row.values.iter_mut().chain(&mut row.future) {
                    *value = -*value;
                }
            }
            let reflected = self.decode_batch(&mirrored, horizon, options, &layout)?;
            let q = c.quantile_knots.len();
            for (values, mirror) in output.iter_mut().zip(reflected) {
                for (step, other) in values.chunks_exact_mut(q).zip(mirror.chunks_exact(q)) {
                    for (value, reflected) in step.iter_mut().zip(other.iter().rev()) {
                        *value = 0.5 * (*value + -*reflected);
                    }
                }
            }
        }
        for (values, eligible) in output.iter_mut().zip(nonnegative) {
            if options.clip_nonnegative && eligible {
                for value in values {
                    *value = value.max(0.0);
                }
            }
        }
        ensure!(
            output.iter().flatten().all(|value| value.is_finite()),
            "Batch forecast produced non-finite values"
        );
        Ok(Forecast {
            quantiles: Tensor::from_vec(
                output.into_iter().flatten().collect::<Vec<_>>(),
                (series.len(), 1, horizon, c.quantile_knots.len()),
                self.model.forecast_query.device(),
            )?,
            quantile_levels: c.quantile_knots.clone(),
        })
    }

    fn validate_batch(
        c: &crate::config::TimestralConfig,
        series: &[BatchSeries<'_>],
        horizon: usize,
        options: &ForecastOptions,
        limits: &BatchLimits,
    ) -> Result<Layout> {
        ensure!(
            !series.is_empty() && series.len() <= limits.max_series,
            "Batch must contain 1..=max_series members"
        );
        let history = series[0].history.len();
        ensure!(
            history > 0 && horizon > 0,
            "History and horizon must be positive"
        );
        let context = options.max_context.unwrap_or(c.context_length);
        ensure!(
            (1..=c.context_length).contains(&context) && options.per_shot_horizon_patches > 0,
            "Invalid context or block options"
        );
        ensure!(
            options.quantile_real_cap_k.is_finite()
                && options.quantile_real_cap_k >= 0.0
                && options.extreme_value_threshold.is_finite()
                && options.extreme_value_threshold > 0.0,
            "Invalid forecast options"
        );
        if options.symmetric_averaging {
            ensure!(
                c.quantile_knots
                    .iter()
                    .zip(c.quantile_knots.iter().rev())
                    .all(|(low, high)| (low + high - 1.0).abs() < 1e-9),
                "Symmetric quantile levels required"
            );
        }
        let counts = |member: &BatchSeries<'_>| {
            member
                .covariates
                .map_or((0, 0), |rows| (rows.past_count(), rows.known_count()))
        };
        let (past, known) = counts(&series[0]);
        let rows_per_group = past
            .checked_add(known)
            .and_then(|count| count.checked_add(1))
            .context("Row count overflow")?;
        let covariates = rows_per_group > 1;
        if covariates {
            ensure!(
                horizon <= MAX_COVARIATE_HORIZON,
                "Covariate forecasts currently support at most 256 steps"
            );
            ensure!(
                c.inference_settings
                    .get("covariate_enabled")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(true),
                "Checkpoint does not support covariates"
            );
        }
        let input = rows_per_group
            .checked_mul(history)
            .and_then(|count| count.checked_add(known.checked_mul(horizon)?))
            .and_then(|count| count.checked_mul(series.len()))
            .context("Input size overflow")?;
        ensure!(
            input <= limits.max_input_values,
            "Batch exceeds max_input_values"
        );
        let output = series
            .len()
            .checked_mul(horizon)
            .and_then(|count| count.checked_mul(c.quantile_knots.len()))
            .context("Output size overflow")?;
        ensure!(
            output <= limits.max_output_values,
            "Batch exceeds max_output_values"
        );
        // The existing two-stream estimate also conservatively bounds the shorter
        // context-only path. Independent variable attention is linear in groups.
        let attention = CovariateAttention {
            patch_size: c.patch_size,
            heads: c.num_heads,
            block_size: c.block_size.min(options.per_shot_horizon_patches),
        }
        .elements(rows_per_group, history, horizon, context)?
        .checked_mul(series.len())
        .context("Attention size overflow")?;
        ensure!(
            attention <= limits.max_attention_elements,
            "Batch exceeds max_attention_elements"
        );
        if covariates {
            ensure!(
                attention <= options.max_covariate_attention_elements,
                "Batch exceeds max_covariate_attention_elements"
            );
        }
        for (index, member) in series.iter().enumerate() {
            ensure!(
                member.history.len() == history,
                "Batch member {index}: history lengths must match"
            );
            ensure!(
                member
                    .observation_mask
                    .is_none_or(|mask| mask.len() == history),
                "Batch member {index}: observation mask length must match history"
            );
            ensure!(
                counts(member) == (past, known),
                "Batch member {index}: past/known covariate row counts must match"
            );
            if let Some(rows) = member.covariates {
                rows.validate(
                    history,
                    horizon,
                    limits.max_covariates_per_series,
                    usize::MAX,
                )
                .with_context(|| format!("Batch member {index}"))?;
            }
        }
        Ok(Layout {
            context,
            rows_per_group,
            covariates,
        })
    }

    fn decode_batch(
        &self,
        initial: &[Row],
        horizon: usize,
        options: &ForecastOptions,
        layout: &Layout,
    ) -> Result<Vec<Vec<f32>>> {
        let c = &self.model.config;
        let q = c.quantile_knots.len();
        let groups = initial.len() / layout.rows_per_group;
        let mut output = vec![Vec::new(); groups];
        let mut rows = initial.to_vec();
        let mut remaining = horizon.div_ceil(c.patch_size);
        let mut offset = 0;
        while remaining > 0 {
            let block = remaining
                .min(c.block_size)
                .min(options.per_shot_horizon_patches);
            let steps = block * c.patch_size;
            let anchors = self.predict_batch_block(&rows, offset, block, layout)?;
            remaining -= block;
            for (group, anchor) in anchors.into_iter().enumerate() {
                let start = group * layout.rows_per_group;
                let refined =
                    scaler::refine_block(&anchor, &rows[start].values, &rows[start].observed, c)?;
                if remaining > 0 {
                    for row in &mut rows[start..start + layout.rows_per_group] {
                        match row.role {
                            RowRole::Target => {
                                row.values.extend(refined.chunks_exact(q).map(|step| {
                                    if step[10].is_finite()
                                        && step[10].abs() < options.extreme_value_threshold
                                    {
                                        step[10]
                                    } else {
                                        0.0
                                    }
                                }));
                                row.observed.resize(row.values.len(), true);
                            }
                            RowRole::Past => {
                                row.values.resize(row.values.len() + steps, 0.0);
                                row.observed.resize(row.values.len(), false);
                            }
                            RowRole::Known => {
                                row.values
                                    .extend_from_slice(&row.future[offset..offset + steps]);
                                row.observed.extend_from_slice(
                                    &row.future_observed[offset..offset + steps],
                                );
                            }
                        }
                        if row.values.len() > layout.context {
                            let keep = c
                                .patch_size
                                .max(layout.context / c.patch_size * c.patch_size);
                            let remove = row.values.len() - keep;
                            row.values.drain(..remove);
                            row.observed.drain(..remove);
                        }
                    }
                }
                output[group].extend(refined);
            }
            offset += steps;
        }
        for (group, values) in output.iter_mut().enumerate() {
            values.truncate(horizon * q);
            if options.quantile_real_cap_k > 0.0 {
                let target = &initial[group * layout.rows_per_group];
                let stats = scaler::scale_context(
                    &target.values,
                    &target.observed,
                    c.patch_size,
                    c.minimum_scale,
                    c.use_asinh,
                )?;
                let span = stats.scale.last().context("Missing target scale")?
                    * options.quantile_real_cap_k;
                let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
                for (&value, &observed) in target.values.iter().zip(&target.observed) {
                    if observed {
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
                if low == f32::INFINITY {
                    low = 0.0;
                    high = 0.0;
                }
                for value in values {
                    if !value.is_nan() {
                        *value = value.clamp(low - span, high + span);
                    }
                }
            }
        }
        Ok(output)
    }

    fn predict_batch_block(
        &self,
        rows: &[Row],
        offset: usize,
        block: usize,
        layout: &Layout,
    ) -> Result<Vec<Vec<f32>>> {
        let c = &self.model.config;
        let device = self.model.forecast_query.device();
        let groups = rows.len() / layout.rows_per_group;
        let context = rows[0].values.len();
        let steps = block * c.patch_size;
        let window_patches = context / c.patch_size + block;
        let encoder_patches = if layout.covariates {
            window_patches
        } else {
            context / c.patch_size
        };
        let query_patches = if layout.covariates {
            window_patches
        } else {
            block
        };
        let tokens = encoder_patches + query_patches;
        let mut scaled = Vec::with_capacity(rows.len());
        let mut observed = Vec::with_capacity(rows.len());
        for row in rows {
            let mut values = row.values.clone();
            let mut mask = row.observed.clone();
            if layout.covariates {
                values.extend_from_slice(&row.future[offset..offset + steps]);
                mask.extend_from_slice(&row.future_observed[offset..offset + steps]);
            }
            scaled.push(if row.role == RowRole::Known {
                scale_known(&values, &mask, c.minimum_scale as f32, c.use_asinh)
            } else {
                scaler::scale_context(&values, &mask, c.patch_size, c.minimum_scale, c.use_asinh)?
            });
            observed.push(mask);
        }
        let mut positions = Vec::new();
        let mut time = Vec::new();
        let mut variable = Vec::new();
        if layout.covariates {
            for (masks, group) in observed
                .chunks_exact(layout.rows_per_group)
                .zip(rows.chunks_exact(layout.rows_per_group))
            {
                let roles: Vec<_> = group.iter().map(|row| row.role).collect();
                let masks = covariate_masks(
                    masks,
                    &roles,
                    c.patch_size,
                    context / c.patch_size,
                    c.block_size,
                    c.attend_missing_patches,
                )?;
                positions.extend(masks.positions);
                time.extend(masks.time);
                variable.extend(masks.variable);
            }
        } else {
            for mask in &observed {
                let masks = univariate_masks(mask, c.patch_size, block, c.attend_missing_patches)?;
                positions.extend(masks.positions);
                time.extend(masks.allowed);
            }
        }
        let mut features = Vec::new();
        for (stats, mask) in scaled.iter().zip(&observed) {
            for (values, mask) in stats
                .values
                .chunks_exact(c.patch_size)
                .zip(mask.chunks_exact(c.patch_size))
            {
                features.extend_from_slice(values);
                features.extend(
                    mask.iter()
                        .map(|observed| if *observed { 0.0 } else { 1.0 }),
                );
            }
        }
        let features = Tensor::from_vec(
            features,
            (rows.len(), encoder_patches, c.patch_size * 2),
            device,
        )?;
        let embedded = self.model.patch_projection.forward(&features)?;
        let queries = self
            .model
            .forecast_query
            .reshape((1, 1, c.d_model))?
            .broadcast_as((rows.len(), query_patches, c.d_model))?
            .contiguous()?;
        let inputs = Tensor::cat(&[&embedded, &queries], 1)?;
        let positions = Tensor::from_vec(positions, (rows.len(), tokens), device)?;
        let time = Tensor::from_vec(time, (rows.len(), 1, tokens, tokens), device)?;
        let variable = if layout.covariates {
            Some(Tensor::from_vec(
                variable,
                (
                    groups * tokens,
                    1,
                    layout.rows_per_group,
                    layout.rows_per_group,
                ),
                device,
            )?)
        } else {
            None
        };
        let normalized = self.model.normalized_quantiles_grouped(
            &inputs,
            encoder_patches,
            &positions,
            (&time, variable.as_ref()),
            layout.rows_per_group,
        )?;
        let forecast_offset = if layout.covariates { context } else { 0 };
        let targets: Result<Vec<_>, _> = (0..groups)
            .map(|group| {
                normalized
                    .narrow(0, group * layout.rows_per_group, 1)?
                    .narrow(1, forecast_offset, steps)
            })
            .collect();
        let normalized = Tensor::cat(&targets?, 0)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        normalized
            .chunks_exact(steps * c.quantile_knots.len())
            .enumerate()
            .map(|(group, values)| {
                let stats = &scaled[group * layout.rows_per_group];
                Ok(scaler::denormalize(
                    values,
                    stats.loc[context - 1],
                    stats.scale[context - 1],
                    c.use_asinh,
                    c.quantile_knots.len(),
                ))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(series: &[BatchSeries<'_>], limits: &BatchLimits) -> Result<Layout> {
        let config = serde_json::from_str(include_str!("../tests/fixtures/config-5m.json"))?;
        TimestralForecaster::validate_batch(
            &config,
            series,
            65,
            &ForecastOptions::default(),
            limits,
        )
    }

    #[test]
    fn aggregate_limits_reject_individually_small_members() {
        let history = [1.0; 97];
        let series = [BatchSeries::new(&history); 2];
        let defaults = BatchLimits::default();
        assert!(validate(&series, &defaults).is_ok());
        let bounds = [
            (
                BatchLimits {
                    max_series: 1,
                    ..defaults.clone()
                },
                "max_series",
            ),
            (
                BatchLimits {
                    max_input_values: 193,
                    ..defaults.clone()
                },
                "max_input_values",
            ),
            (
                BatchLimits {
                    max_output_values: 2729,
                    ..defaults.clone()
                },
                "max_output_values",
            ),
            (
                BatchLimits {
                    max_attention_elements: 630,
                    ..defaults
                },
                "max_attention_elements",
            ),
        ];
        for (limits, message) in bounds {
            assert!(validate(&series[..1], &limits).is_ok());
            let error = validate(&series, &limits)
                .err()
                .expect("must reject oversized batch");
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn malformed_members_are_rejected_before_tensor_preparation() {
        let history = [1.0; 97];
        let mask = [true; 96];
        let limits = BatchLimits::default();
        assert!(validate(&[], &limits).is_err());
        let short = [BatchSeries::new(&history), BatchSeries::new(&history[..96])];
        assert!(validate(&short, &limits)
            .err()
            .unwrap()
            .to_string()
            .contains("history lengths"));
        let masked = [BatchSeries {
            observation_mask: Some(&mask),
            ..BatchSeries::new(&history)
        }];
        assert!(validate(&masked, &limits)
            .err()
            .unwrap()
            .to_string()
            .contains("mask length"));
        let mut covariates = Covariates::default();
        covariates.past.insert("rain".into(), history.to_vec());
        let mismatched = [
            BatchSeries::new(&history),
            BatchSeries {
                covariates: Some(&covariates),
                ..BatchSeries::new(&history)
            },
        ];
        assert!(validate(&mismatched, &limits)
            .err()
            .unwrap()
            .to_string()
            .contains("row counts"));
        covariates.past.insert("rain".into(), vec![1.0]);
        let invalid = [BatchSeries {
            covariates: Some(&covariates),
            ..BatchSeries::new(&history)
        }];
        assert!(validate(&invalid, &limits).is_err());
    }

    #[test]
    fn known_future_values_count_toward_the_aggregate_input_limit() {
        let history = [1.0; 97];
        let mut covariates = Covariates::default();
        covariates.known.insert(
            "trend".into(),
            crate::covariates::KnownCovariate {
                history: history.to_vec(),
                future: vec![2.0; 65],
            },
        );
        let series = [BatchSeries {
            covariates: Some(&covariates),
            ..BatchSeries::new(&history)
        }; 2];
        // Each member contributes 97 target + 97 covariate history + 65 future values.
        let limits = BatchLimits {
            max_input_values: 517,
            ..Default::default()
        };
        assert!(validate(&series[..1], &limits).is_ok());
        assert!(validate(&series, &limits)
            .err()
            .unwrap()
            .to_string()
            .contains("max_input_values"));
    }
}
