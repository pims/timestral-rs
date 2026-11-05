//! One target plus numeric and categorical covariates. Preparation/statistics run on CPU; the
//! existing Candle transformer runs on the model's selected device.
use crate::{
    categorical::encode_categorical,
    covariates::{CovariateAttention, Covariates},
    forecaster::{prepare_history, Forecast, ForecastOptions, TimestralForecaster},
    masks::{covariate_masks, RowRole},
    model::TimestralModel,
    scaler,
};
use anyhow::{ensure, Result};
use candle_core::{Device, Module, Tensor};
use std::collections::BTreeMap;

#[derive(Clone)]
pub(crate) struct Row {
    pub(crate) role: RowRole,
    pub(crate) values: Vec<f32>,
    pub(crate) observed: Vec<bool>,
    pub(crate) future: Vec<f32>,
    pub(crate) future_observed: Vec<bool>,
}

impl TimestralForecaster {
    /// Diagnostic first-block stages for Python parity checks. Does not retain
    /// intermediate tensors in the normal forecast path.
    pub fn covariate_block_trace(
        &self,
        history: &[f32],
        horizon: usize,
        covariates: &Covariates,
        options: &ForecastOptions,
    ) -> Result<BTreeMap<String, Tensor>> {
        covariates.validate(history.len(), horizon, usize::MAX, usize::MAX)?;
        let c = &self.model.config;
        ensure!(
            !history.is_empty() && horizon > 0 && !covariates.is_empty(),
            "Nonempty history, horizon, and covariates required"
        );
        let max_context = options.max_context.unwrap_or(c.context_length);
        ensure!(
            (1..=c.context_length).contains(&max_context) && options.per_shot_horizon_patches > 0,
            "Invalid trace options"
        );
        let layout = CovariateAttention {
            patch_size: c.patch_size,
            heads: c.num_heads,
            block_size: c.block_size.min(options.per_shot_horizon_patches),
        };
        ensure!(
            layout.elements(
                1 + covariates.past_count() + covariates.known_count(),
                history.len(),
                horizon,
                max_context
            )? <= options.max_covariate_attention_elements,
            "Trace attention limit exceeded"
        );
        let observed: Vec<bool> = history.iter().map(|v| v.is_finite()).collect();
        let rows = prepare_rows(
            history,
            &observed,
            covariates,
            horizon,
            max_context,
            c.patch_size,
        )?;
        let block = horizon
            .div_ceil(c.patch_size)
            .min(c.block_size)
            .min(options.per_shot_horizon_patches);
        let mut trace = BTreeMap::new();
        // Export the uncropped encodings as well as model stages so parity checks
        // catch fitting-window and missing-label errors hidden by later padding.
        for (name, labels) in &covariates.categorical_past {
            let encoded = encode_categorical(history, Some(&observed), labels, None, max_context)?;
            trace.insert(
                format!("categorical_past:{name}:history"),
                Tensor::from_vec(encoded.history, history.len(), &Device::Cpu)?,
            );
        }
        for (name, labels) in &covariates.categorical_known {
            let encoded = encode_categorical(
                history,
                Some(&observed),
                &labels.history,
                Some(&labels.future),
                max_context,
            )?;
            trace.insert(
                format!("categorical_known:{name}:history"),
                Tensor::from_vec(encoded.history, history.len(), &Device::Cpu)?,
            );
            trace.insert(
                format!("categorical_known:{name}:future"),
                Tensor::from_vec(encoded.future.expect("known labels"), horizon, &Device::Cpu)?,
            );
        }

        let anchor = self
            .model
            .predict_covariate_block(&rows, 0, block, Some(&mut trace))?;
        let refined = scaler::refine_block(&anchor, &rows[0].values, &rows[0].observed, c)?;
        trace.insert(
            "anchor_quantiles".into(),
            Tensor::from_vec(
                anchor,
                (1, block * c.patch_size, c.quantile_knots.len()),
                &Device::Cpu,
            )?,
        );
        trace.insert(
            "refined_quantiles".into(),
            Tensor::from_vec(
                refined,
                (1, block * c.patch_size, c.quantile_knots.len()),
                &Device::Cpu,
            )?,
        );
        Ok(trace)
    }

    /// Forecast one target with aligned numeric or categorical exogenous rows. Missing values
    /// are NaN; categorical rows use string/null labels. Known rows supply
    /// exactly `horizon` future values. Categories are fit once using the
    /// masked historical target and effective `max_context`. Only target
    /// quantiles are returned, with the same shape as `forecast_masked`.
    pub fn forecast_with_covariates(
        &self,
        history: &[f32],
        observation_mask: Option<&[bool]>,
        horizon: usize,
        covariates: &Covariates,
        options: &ForecastOptions,
    ) -> Result<Forecast> {
        if covariates.is_empty() {
            return self.forecast_masked(history, observation_mask, horizon, options);
        }
        covariates.validate(history.len(), horizon, usize::MAX, usize::MAX)?;
        let c = &self.model.config;
        ensure!(
            c.inference_settings
                .get("covariate_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            "Checkpoint does not support covariates"
        );
        ensure!(
            !history.is_empty() && horizon > 0,
            "History and horizon must be nonempty/positive"
        );
        ensure!(
            horizon <= crate::covariates::MAX_COVARIATE_HORIZON,
            "Covariate forecasts currently support at most 256 steps"
        );
        ensure!(
            observation_mask.is_none_or(|m| m.len() == history.len()),
            "Observation mask length must match history"
        );
        let max_context = options.max_context.unwrap_or(c.context_length);
        ensure!(
            (1..=c.context_length).contains(&max_context) && options.per_shot_horizon_patches > 0,
            "Invalid context or block options"
        );
        ensure!(
            options.quantile_real_cap_k.is_finite()
                && options.quantile_real_cap_k >= 0.0
                && options.extreme_value_threshold.is_finite()
                && options.extreme_value_threshold > 0.0,
            "Invalid forecast options"
        );
        let count = 1usize
            .checked_add(covariates.past_count())
            .and_then(|n| n.checked_add(covariates.known_count()))
            .ok_or_else(|| anyhow::anyhow!("Row count overflow"))?;
        let layout = CovariateAttention {
            patch_size: c.patch_size,
            heads: c.num_heads,
            block_size: c.block_size.min(options.per_shot_horizon_patches),
        };
        ensure!(
            layout.elements(count, history.len(), horizon, max_context)?
                <= options.max_covariate_attention_elements,
            "Covariate request exceeds max_covariate_attention_elements"
        );
        let padded_horizon = horizon
            .checked_add(c.patch_size - 1)
            .map(|n| n / c.patch_size * c.patch_size)
            .ok_or_else(|| anyhow::anyhow!("Horizon overflow"))?;
        ensure!(
            padded_horizon.checked_mul(c.quantile_knots.len()).is_some(),
            "Forecast size overflow"
        );
        let observed: Vec<bool> = history
            .iter()
            .enumerate()
            .map(|(i, v)| v.is_finite() && observation_mask.is_none_or(|m| m[i]))
            .collect();
        let nonnegative = observed.iter().any(|v| *v)
            && history
                .iter()
                .zip(&observed)
                .all(|(v, obs)| !obs || *v >= 0.0);
        let rows = prepare_rows(
            history,
            &observed,
            covariates,
            horizon,
            max_context,
            c.patch_size,
        )?;
        let mut output = self.decode_covariates(&rows, horizon, max_context, options)?;
        let q = c.quantile_knots.len();
        if options.symmetric_averaging {
            ensure!(
                c.quantile_knots
                    .iter()
                    .zip(c.quantile_knots.iter().rev())
                    .all(|(a, b)| (a + b - 1.0).abs() < 1e-9),
                "Symmetric quantile levels required"
            );
            let mut mirrored = rows.clone();
            for row in &mut mirrored {
                for value in row.values.iter_mut().chain(&mut row.future) {
                    *value = -*value;
                }
            }
            let mirror = self.decode_covariates(&mirrored, horizon, max_context, options)?;
            for (values, reflected) in output.chunks_exact_mut(q).zip(mirror.chunks_exact(q)) {
                for i in 0..q {
                    values[i] = 0.5 * (values[i] + -reflected[q - 1 - i]);
                }
            }
        }
        if options.clip_nonnegative && nonnegative {
            for v in &mut output {
                *v = v.max(0.0);
            }
        }
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "Covariate forecast produced non-finite values"
        );
        Ok(Forecast {
            quantiles: Tensor::from_vec(
                output,
                (1, 1, horizon, q),
                self.model.forecast_query.device(),
            )?,
            quantile_levels: c.quantile_knots.clone(),
        })
    }

    fn decode_covariates(
        &self,
        initial: &[Row],
        horizon: usize,
        max_context: usize,
        options: &ForecastOptions,
    ) -> Result<Vec<f32>> {
        let c = &self.model.config;
        let q = c.quantile_knots.len();
        let mut rows = initial.to_vec();
        let (mut remaining, mut offset) = (horizon.div_ceil(c.patch_size), 0);
        let mut output = Vec::new();
        while remaining > 0 {
            let block = remaining
                .min(c.block_size)
                .min(options.per_shot_horizon_patches);
            let steps = block * c.patch_size;
            let anchor = self
                .model
                .predict_covariate_block(&rows, offset, block, None)?;
            let refined = scaler::refine_block(&anchor, &rows[0].values, &rows[0].observed, c)?;
            remaining -= block;
            if remaining > 0 {
                for row in &mut rows {
                    match row.role {
                        RowRole::Target => {
                            row.values.extend(refined.chunks_exact(q).map(|s| {
                                if s[10].is_finite()
                                    && s[10].abs() < options.extreme_value_threshold
                                {
                                    s[10]
                                } else {
                                    0.0
                                }
                            }));
                            row.observed.resize(row.values.len(), true);
                        }
                        RowRole::Known => {
                            row.values
                                .extend_from_slice(&row.future[offset..offset + steps]);
                            row.observed
                                .extend_from_slice(&row.future_observed[offset..offset + steps]);
                        }
                        RowRole::Past => {
                            row.values.resize(row.values.len() + steps, 0.0);
                            row.observed.resize(row.values.len(), false);
                        }
                    }
                    if row.values.len() > max_context {
                        let keep = c.patch_size.max(max_context / c.patch_size * c.patch_size);
                        let remove = row.values.len() - keep;
                        row.values.drain(..remove);
                        row.observed.drain(..remove);
                    }
                }
            }
            offset += steps;
            output.extend(refined);
        }
        output.truncate(horizon * q);
        if options.quantile_real_cap_k > 0.0 {
            let target = &initial[0];
            let stats = scaler::scale_context(
                &target.values,
                &target.observed,
                c.patch_size,
                c.minimum_scale,
                c.use_asinh,
            )?;
            let span = stats.scale.last().unwrap() * options.quantile_real_cap_k;
            let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
            for (&v, &obs) in target.values.iter().zip(&target.observed) {
                if obs {
                    low = low.min(v);
                    high = high.max(v);
                }
            }
            if low == f32::INFINITY {
                low = 0.0;
                high = 0.0;
            }
            for v in &mut output {
                if !v.is_nan() {
                    *v = v.clamp(low - span, high + span);
                }
            }
        }
        ensure!(
            !output.iter().any(|v| v.is_nan()),
            "Covariate forecast produced NaN values"
        );
        Ok(output)
    }
}

impl TimestralModel {
    fn predict_covariate_block(
        &self,
        rows: &[Row],
        offset: usize,
        horizon_patches: usize,
        trace: Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let r = rows.len();
        let context = rows[0].values.len();
        let horizon = horizon_patches * c.patch_size;
        let n = context / c.patch_size + horizon_patches;
        let mut observed = Vec::with_capacity(r);
        let mut scaled = Vec::with_capacity(r);
        for row in rows {
            let mut values = row.values.clone();
            values.extend_from_slice(&row.future[offset..offset + horizon]);
            let mut mask = row.observed.clone();
            mask.extend_from_slice(&row.future_observed[offset..offset + horizon]);
            let stats = if row.role == RowRole::Known {
                scale_known(&values, &mask, c.minimum_scale as f32, c.use_asinh)
            } else {
                scaler::scale_context(&values, &mask, c.patch_size, c.minimum_scale, c.use_asinh)?
            };
            observed.push(mask);
            scaled.push(stats);
        }
        let roles: Vec<_> = rows.iter().map(|r| r.role).collect();
        let masks = covariate_masks(
            &observed,
            &roles,
            c.patch_size,
            context / c.patch_size,
            c.block_size,
            c.attend_missing_patches,
        )?;
        let mut features = Vec::with_capacity(r * n * c.patch_size * 2);
        for (stats, mask) in scaled.iter().zip(&observed) {
            for (values, obs) in stats
                .values
                .chunks_exact(c.patch_size)
                .zip(mask.chunks_exact(c.patch_size))
            {
                features.extend_from_slice(values);
                features.extend(obs.iter().map(|v| if *v { 0.0 } else { 1.0 }));
            }
        }
        let device = self.forecast_query.device();
        let features = Tensor::from_vec(features, (r, n, 2 * c.patch_size), device)?;
        let embedded = self.patch_projection.forward(&features)?;
        let queries = self
            .forecast_query
            .reshape((1, 1, c.d_model))?
            .broadcast_as((r, n, c.d_model))?
            .contiguous()?;
        let tokens = Tensor::cat(&[&embedded, &queries], 1)?;
        let positions = Tensor::from_vec(masks.positions, (r, 2 * n), device)?;
        let time = Tensor::from_vec(masks.time, (r, 1, 2 * n, 2 * n), device)?;
        let variable = Tensor::from_vec(masks.variable, (2 * n, 1, r, r), device)?;
        let normalized = self
            .normalized_quantiles(&tokens, n, &positions, &time, Some(&variable))?
            .narrow(0, 0, 1)?
            .narrow(1, context, horizon)?
            .to_device(&Device::Cpu)?;
        if let Some(trace) = trace {
            for (name, tensor) in [
                ("positions", positions),
                ("time_mask", time),
                ("variable_mask", variable),
                ("patch_features", features),
                ("embedded", embedded),
                ("normalized_quantiles", normalized.clone()),
            ] {
                trace.insert(name.into(), tensor.to_device(&Device::Cpu)?);
            }
            for (name, values) in [
                (
                    "scaled",
                    scaled
                        .iter()
                        .flat_map(|s| s.values.iter().copied())
                        .collect(),
                ),
                (
                    "loc",
                    scaled.iter().flat_map(|s| s.loc.iter().copied()).collect(),
                ),
                (
                    "scale",
                    scaled
                        .iter()
                        .flat_map(|s| s.scale.iter().copied())
                        .collect(),
                ),
            ] {
                trace.insert(
                    name.into(),
                    Tensor::from_vec(values, (r, n * c.patch_size), &Device::Cpu)?,
                );
            }
            trace.insert(
                "key_valid".into(),
                Tensor::from_vec(masks.key_valid, (r, n), &Device::Cpu)?,
            );
        }
        let normalized = normalized.flatten_all()?.to_vec1::<f32>()?;
        Ok(scaler::denormalize(
            &normalized,
            scaled[0].loc[context - 1],
            scaled[0].scale[context - 1],
            c.use_asinh,
            c.quantile_knots.len(),
        ))
    }
}

/// Reference instance_norm_stats uses F32 population variance over the complete
/// observed window. Binary known covariates keep loc=0, scale=1 and skip asinh.
pub(crate) fn scale_known(
    values: &[f32],
    observed: &[bool],
    minimum: f32,
    asinh: bool,
) -> scaler::ScaledContext {
    let count = observed.iter().filter(|v| **v).count();
    let binary = count > 0
        && values
            .iter()
            .zip(observed)
            .all(|(v, obs)| !obs || *v == 0.0 || *v == 1.0);
    let loc = if binary {
        0.0
    } else {
        reference_f32_sum(values.len(), |i| if observed[i] { values[i] } else { 0.0 })
            / count.max(1) as f32
    };
    let scale = if binary {
        1.0
    } else {
        (reference_f32_sum(values.len(), |i| {
            if observed[i] {
                (values[i] - loc).powi(2)
            } else {
                0.0
            }
        }) / count.max(1) as f32)
            .sqrt()
            .max(minimum)
    };
    let scaled = values
        .iter()
        .zip(observed)
        .map(|(v, obs)| {
            if !obs {
                0.0
            } else {
                let v = (*v - loc) / scale.max(f32::EPSILON);
                if asinh && !binary {
                    v.asinh()
                } else {
                    v
                }
            }
        })
        .collect();
    scaler::ScaledContext {
        values: scaled,
        loc: vec![loc; values.len()],
        scale: vec![scale; values.len()],
    }
}

/// Fixed F32 cascade reduction matching the pinned Python CPU reference's
/// four-lane contiguous sum. Keep masked slots (zero weights) in their original
/// positions: filtering them changes rounding. In nearly constant encoded rows,
/// a one-ULP mean difference can otherwise dominate the normalized features.
/// Order follows PyTorch v2.14.1 aten/src/ATen/native/cpu/SumKernel.cpp.
/// Use this same portable order on every Rust backend; do not depend on the host SIMD width.
fn reference_f32_sum(length: usize, value: impl Fn(usize) -> f32) -> f32 {
    let lanes = if length >= 4 { 4 } else { 1 };
    let vectors = length / lanes;
    let groups = vectors / 4;
    let power = ((usize::BITS - groups.saturating_sub(1).leading_zeros()) / 4).max(4);
    let step = 1usize << power;
    let mut sums = [[[0.0f32; 4]; 4]; 4];
    for group in 0..groups {
        for (partial, accumulators) in sums[0].iter_mut().enumerate() {
            for (lane, accumulator) in accumulators.iter_mut().take(lanes).enumerate() {
                *accumulator += value((group * 4 + partial) * lanes + lane);
            }
        }
        let completed = group + 1;
        if completed % step == 0 {
            for level in 1..4 {
                let (lower_levels, higher_levels) = sums.split_at_mut(level);
                let previous = &mut lower_levels[level - 1];
                let current = &mut higher_levels[0];
                for (destination, source) in current.iter_mut().zip(previous) {
                    for (accumulator, value) in destination.iter_mut().zip(source).take(lanes) {
                        *accumulator += *value;
                        *value = 0.0;
                    }
                }
                if completed & ((step - 1) << (level as u32 * power)) != 0 {
                    break;
                }
            }
        }
    }
    // Preserve the reference's level, partial, then lane accumulation order.
    let (first_level, higher_levels) = sums.split_at_mut(1);
    for level in higher_levels {
        for (destination, source) in first_level[0].iter_mut().zip(level) {
            for (accumulator, value) in destination.iter_mut().zip(source).take(lanes) {
                *accumulator += *value;
            }
        }
    }
    for vector in groups * 4..vectors {
        for (lane, accumulator) in sums[0][0].iter_mut().take(lanes).enumerate() {
            *accumulator += value(vector * lanes + lane);
        }
    }
    let (first_partial, remaining_partials) = sums[0].split_at_mut(1);
    for partial in remaining_partials {
        for (accumulator, value) in first_partial[0].iter_mut().zip(partial).take(lanes) {
            *accumulator += *value;
        }
    }
    let mut total = 0.0f32;
    for i in vectors * lanes..length {
        total += value(i);
    }
    for &accumulator in sums[0][0].iter().take(lanes) {
        total += accumulator;
    }
    total
}

pub(crate) fn prepare_rows(
    history: &[f32],
    observed: &[bool],
    covariates: &Covariates,
    horizon: usize,
    max_context: usize,
    patch_size: usize,
) -> Result<Vec<Row>> {
    let padded_horizon = horizon
        .checked_add(patch_size - 1)
        .map(|n| n / patch_size * patch_size)
        .ok_or_else(|| anyhow::anyhow!("Horizon overflow"))?;
    let (values, mask) = prepare_history(history, observed, max_context, patch_size);
    let mut rows = vec![Row {
        role: RowRole::Target,
        values,
        observed: mask,
        future: vec![0.0; padded_horizon],
        future_observed: vec![false; padded_horizon],
    }];
    let start = history.len().saturating_sub(max_context);
    let padding = rows[0].values.len() - (history.len() - start);
    let prepare = |values: &[f32], future: Option<&[f32]>, role| -> Result<Row> {
        ensure!(
            values.len() == history.len() && values.iter().all(|v| !v.is_infinite()),
            "Invalid covariate history"
        );
        let mut r = Row {
            role,
            values: vec![0.0; padding],
            observed: vec![false; padding],
            future: vec![0.0; padded_horizon],
            future_observed: vec![false; padded_horizon],
        };
        r.values.extend(
            values[start..]
                .iter()
                .map(|v| if v.is_finite() { *v } else { 0.0 }),
        );
        r.observed
            .extend(values[start..].iter().map(|v| v.is_finite()));
        if let Some(future) = future {
            ensure!(
                future.len() == horizon && future.iter().all(|v| !v.is_infinite()),
                "Invalid known covariate future"
            );
            for (i, v) in future.iter().enumerate() {
                r.future[i] = if v.is_finite() { *v } else { 0.0 };
                r.future_observed[i] = v.is_finite();
            }
        }
        Ok(r)
    };
    for values in covariates.past.values() {
        rows.push(prepare(values, None, RowRole::Past)?);
    }
    // Match the reference packing: target, numeric past, categorical past,
    // numeric known, categorical known. Encoding is done once before decoding,
    // using the original target mask and context limit for every future block.
    for labels in covariates.categorical_past.values() {
        let encoded = encode_categorical(history, Some(observed), labels, None, max_context)?;
        rows.push(prepare(&encoded.history, None, RowRole::Past)?);
    }
    for values in covariates.known.values() {
        rows.push(prepare(
            &values.history,
            Some(&values.future),
            RowRole::Known,
        )?);
    }
    for labels in covariates.categorical_known.values() {
        let encoded = encode_categorical(
            history,
            Some(observed),
            &labels.history,
            Some(&labels.future),
            max_context,
        )?;
        rows.push(prepare(
            &encoded.history,
            encoded.future.as_deref(),
            RowRole::Known,
        )?);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_normalization_matches_python_for_nearly_constant_encoded_rows() {
        #[derive(serde::Deserialize)]
        struct Case {
            length: usize,
            constant: bool,
            mean: f32,
            scale: f32,
            sum: f32,
            normalized_prefix: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct Fixtures {
            cases: Vec<Case>,
        }
        let fixture: Fixtures = serde_json::from_str(include_str!(
            "../tests/fixtures/categorical-known-normalization.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let values: Vec<f32> = (0..case.length)
                .map(|i| {
                    (104.45187f64
                        + if case.constant {
                            0.0
                        } else {
                            (i % 17) as f64 * 0.001
                        }) as f32
                })
                .collect();
            let observed: Vec<bool> = (0..case.length).map(|i| i % 7 != 0).collect();
            let sum = reference_f32_sum(case.length, |i| if observed[i] { values[i] } else { 0.0 });
            assert_eq!(
                sum, case.sum,
                "sum for {} constant={}",
                case.length, case.constant
            );
            let actual = scale_known(&values, &observed, 1e-6, true);
            assert_eq!(
                actual.loc[0], case.mean,
                "mean for {} constant={}",
                case.length, case.constant
            );
            assert_eq!(
                actual.scale[0], case.scale,
                "scale for {} constant={}",
                case.length, case.constant
            );
            for (actual, expected) in actual.values.iter().zip(&case.normalized_prefix) {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "normalized {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn known_normalization_preserves_binary_values_and_uses_population_variance() {
        let binary = scale_known(
            &[0.0, 0.0, 1.0, 0.0],
            &[true, true, true, false],
            1e-6,
            true,
        );
        assert_eq!(binary.values, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(binary.loc, vec![0.0; 4]);
        assert_eq!(binary.scale, vec![1.0; 4]);
        let numeric = scale_known(
            &[1.0, 2.0, 0.0, 3.0],
            &[true, true, false, true],
            1e-6,
            true,
        );
        assert_eq!(numeric.loc, vec![2.0; 4]);
        assert!((numeric.scale[0] - (2.0f32 / 3.0).sqrt()).abs() < 1e-7);
        assert_eq!(numeric.values[2], 0.0);
        let reflected = scale_known(&[0.0, -1.0], &[true, true], 1e-6, true);
        assert_eq!(reflected.loc, [-0.5, -0.5]);
        assert_eq!(reflected.scale, [0.5, 0.5]);
        assert!((reflected.values[0] - 1.0f32.asinh()).abs() < 1e-7);
    }
}
