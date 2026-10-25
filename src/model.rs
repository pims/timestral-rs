use crate::config::TimestralConfig;
use crate::head::QuantileHead;
use crate::layers::ResidualBlock;
use crate::transformer::Transformer;
use crate::{masks::univariate_masks, scaler};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use std::collections::BTreeMap;

/// Loaded network with a complete context-only block path for a single target.
pub struct TimestralModel {
    pub config: TimestralConfig,
    pub patch_projection: ResidualBlock,
    pub forecast_query: Tensor,
    pub head: QuantileHead,
    pub transformer: Transformer,
    pub tensor_count: usize,
    pub parameter_count: usize,
    /// Retains checkpoint tensors for inspection and layer diagnostics.
    pub weights: VarBuilder<'static>,
}

impl TimestralModel {
    /// Execute the normalized network on already embedded context and forecast
    /// tokens. Caller supplies masks/positions and the encoder split point.
    /// This also supports the covariate path, where forecast tokens span the window.
    pub fn normalized_quantiles(
        &self,
        tokens: &Tensor,
        encoder_patches: usize,
        positions: &Tensor,
        time_mask: &Tensor,
        variable_mask: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        let (_, patches, _) = tokens.dims3()?;
        anyhow::ensure!(
            encoder_patches > 0 && encoder_patches < patches,
            "Encoder split must leave nonempty context and forecast tokens"
        );
        let hidden = self
            .transformer
            .forward(tokens, positions, time_mask, variable_mask)?;
        let decoder = hidden
            .narrow(1, encoder_patches, patches - encoder_patches)?
            .contiguous()?;
        self.head.forward(&decoder)
    }

    /// Context-only single-block primitive for one target row. Context must be
    /// prepared, finite F32 [1, time], with a U8 [1, time] observation mask.
    /// Returns anchor-denormalized [1, forecast_steps, 21]; the forecaster refines it.
    pub fn predict_block(
        &self,
        context: &Tensor,
        observed: &Tensor,
        horizon_patches: usize,
    ) -> anyhow::Result<Tensor> {
        let (rows, steps) = context.dims2()?;
        ensure!(
            rows == 1
                && context.dtype() == DType::F32
                && observed.dims() == [1, steps]
                && observed.dtype() == DType::U8,
            "Expected F32 [1, time] context and matching U8 mask"
        );
        let values = context
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mask: Vec<bool> = observed
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?
            .iter()
            .map(|v| *v != 0)
            .collect();
        let raw = self.predict_univariate(&values, &mask, horizon_patches, None)?;
        Ok(Tensor::from_vec(
            raw,
            (
                1,
                horizon_patches * self.config.patch_size,
                self.config.quantile_knots.len(),
            ),
            self.forecast_query.device(),
        )?)
    }

    pub fn predict_block_trace(
        &self,
        context: &[f32],
        observed: &[bool],
        horizon_patches: usize,
    ) -> Result<BTreeMap<String, Tensor>> {
        let mut trace = BTreeMap::new();
        let anchor =
            self.predict_univariate(context, observed, horizon_patches, Some(&mut trace))?;
        let c = &self.config;
        let refined = scaler::refine_block(&anchor, context, observed, c)?;
        trace.insert(
            "refined_quantiles".into(),
            Tensor::from_vec(
                refined,
                (1, horizon_patches * c.patch_size, c.quantile_knots.len()),
                &Device::Cpu,
            )?,
        );
        Ok(trace)
    }

    pub(crate) fn predict_univariate(
        &self,
        context: &[f32],
        observed: &[bool],
        horizon_patches: usize,
        trace: Option<&mut BTreeMap<String, Tensor>>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        ensure!(
            (1..=c.block_size).contains(&horizon_patches),
            "Block horizon must be in 1..={}",
            c.block_size
        );
        let stats = scaler::scale_context(
            context,
            observed,
            c.patch_size,
            c.minimum_scale,
            c.use_asinh,
        )?;
        let masks = univariate_masks(
            observed,
            c.patch_size,
            horizon_patches,
            c.attend_missing_patches,
        )?;
        let device = self.forecast_query.device();
        let context_patches = context.len() / c.patch_size;
        let patches = context_patches + horizon_patches;
        let values = Tensor::from_vec(
            stats.values.clone(),
            (1, context_patches, c.patch_size),
            device,
        )?;
        let missing: Vec<f32> = observed
            .iter()
            .map(|v| if *v { 0.0 } else { 1.0 })
            .collect();
        let missing = Tensor::from_vec(missing, (1, context_patches, c.patch_size), device)?;
        let features = Tensor::cat(&[&values, &missing], 2)?;
        let embedded = self.patch_projection.forward(&features)?;
        let queries = self
            .forecast_query
            .reshape((1, 1, c.d_model))?
            .broadcast_as((1, horizon_patches, c.d_model))?
            .contiguous()?;
        let tokens = Tensor::cat(&[&embedded, &queries], 1)?;
        let positions = Tensor::from_vec(masks.positions, (1, patches), device)?;
        let time_mask = Tensor::from_vec(masks.allowed, (1, 1, patches, patches), device)?;
        let normalized =
            self.normalized_quantiles(&tokens, context_patches, &positions, &time_mask, None)?;
        let normalized_values = normalized
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let loc = *stats.loc.last().expect("validated context");
        let scale = *stats.scale.last().expect("validated context");
        let anchor = scaler::denormalize(
            &normalized_values,
            loc,
            scale,
            c.use_asinh,
            c.quantile_knots.len(),
        );
        if let Some(trace) = trace {
            trace.insert(
                "scaled".into(),
                Tensor::from_vec(stats.values, (1, context.len()), &Device::Cpu)?,
            );
            trace.insert(
                "loc".into(),
                Tensor::from_vec(stats.loc, (1, context.len()), &Device::Cpu)?,
            );
            trace.insert(
                "scale".into(),
                Tensor::from_vec(stats.scale, (1, context.len()), &Device::Cpu)?,
            );
            trace.insert(
                "key_valid".into(),
                Tensor::from_vec(masks.key_valid, (1, context_patches), &Device::Cpu)?,
            );
            for (name, value) in [
                ("positions", positions),
                ("time_mask", time_mask),
                ("patch_features", features),
                ("embedded", embedded),
                ("normalized_quantiles", normalized),
            ] {
                trace.insert(name.into(), value.to_device(&Device::Cpu)?);
            }
            trace.insert(
                "anchor_quantiles".into(),
                Tensor::from_vec(
                    anchor.clone(),
                    (1, horizon_patches * c.patch_size, c.quantile_knots.len()),
                    &Device::Cpu,
                )?,
            );
        }
        Ok(anchor)
    }
}
