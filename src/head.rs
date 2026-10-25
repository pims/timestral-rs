//! Residual quantile projection and non-crossing quantiles in normalized space.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use candle_core::{DType, Module, Tensor, D};
use candle_nn::VarBuilder;

use crate::{config::TimestralConfig, layers::ResidualBlock};

pub type HeadTrace = BTreeMap<String, Tensor>;

pub struct QuantileHead {
    projection: ResidualBlock,
    width: usize,
    patch_size: usize,
    num_quantiles: usize,
    median: usize,
    width_weights: Tensor,
}

impl QuantileHead {
    /// `vb` is rooted at `head`, preserving the original parameter names.
    pub fn load(config: &TimestralConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let levels = &config.quantile_knots;
        let median = levels
            .iter()
            .position(|q| *q == 0.5)
            .expect("validated median");
        // Compute probability gaps in f64, cast to f32, then multiply by the
        // factor in f32, matching the Python operation order.
        let weights: Vec<f32> = (0..levels.len())
            .map(|i| {
                let gap = if i < median {
                    levels[i + 1] - levels[i]
                } else if i > median {
                    levels[i] - levels[i - 1]
                } else {
                    0.0
                };
                (config.quantile_width_factor as f32) * (gap as f32)
            })
            .collect();
        Ok(Self {
            projection: ResidualBlock::load(
                config.d_model,
                4 * config.d_model,
                config.patch_size * levels.len(),
                false,
                vb.pp("projection"),
            )?,
            width: config.d_model,
            patch_size: config.patch_size,
            num_quantiles: levels.len(),
            median,
            width_weights: Tensor::from_vec(weights, levels.len(), vb.device())?,
        })
    }

    /// F32 decoder tokens [rows, patches, width] -> [rows, patches * 32, 21].
    /// Values are normalized quantiles; no original-unit scaling is applied here.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let raw = self.project(tokens)?;
        let (rows, patches, _, _) = raw.dims4()?;
        self.transform(&raw, None)?
            .reshape((rows, patches * self.patch_size, self.num_quantiles))
            .map_err(Into::into)
    }

    pub fn forward_trace(&self, tokens: &Tensor) -> Result<HeadTrace> {
        let raw = self.project(tokens)?;
        let (rows, patches, _, _) = raw.dims4()?;
        let mut trace = self.monotone_trace(&raw)?;
        let quantiles =
            trace["ordered"].reshape((rows, patches * self.patch_size, self.num_quantiles))?;
        trace.insert("quantiles".into(), quantiles);
        Ok(trace)
    }

    /// Diagnostic entry point for testing extreme raw logits without the MLP.
    /// Shape is preserved and the last axis must contain all quantile channels.
    pub fn monotone_trace(&self, raw: &Tensor) -> Result<HeadTrace> {
        let mut trace = HeadTrace::new();
        self.transform(raw, Some(&mut trace))?;
        Ok(trace)
    }

    fn project(&self, tokens: &Tensor) -> Result<Tensor> {
        let (rows, patches, width) = tokens.dims3()?;
        ensure!(
            rows > 0 && patches > 0 && width == self.width && tokens.dtype() == DType::F32,
            "Expected nonempty F32 decoder tokens [rows, patches, {}]",
            self.width
        );
        Ok(self.projection.forward(&tokens.contiguous()?)?.reshape((
            rows,
            patches,
            self.patch_size,
            self.num_quantiles,
        ))?)
    }

    fn transform(&self, raw: &Tensor, trace: Option<&mut HeadTrace>) -> Result<Tensor> {
        ensure!(
            raw.rank() > 0
                && raw.dim(D::Minus1)? == self.num_quantiles
                && raw.dtype() == DType::F32,
            "Expected F32 raw quantiles with final dimension {}",
            self.num_quantiles
        );
        let positive = softplus(raw)?;
        let increments = positive.broadcast_mul(&self.width_weights)?;
        let median = raw.narrow(D::Minus1, self.median, 1)?;
        let last = raw.rank() - 1;
        let down = increments
            .narrow(last, 0, self.median)?
            .contiguous()?
            .flip(&[last])?
            .cumsum(last)?
            .contiguous()?
            .flip(&[last])?;
        let up = increments
            .narrow(last, self.median + 1, self.num_quantiles - self.median - 1)?
            .contiguous()?
            .cumsum(last)?;
        let lower = median.broadcast_sub(&down)?;
        let upper = median.broadcast_add(&up)?;
        let ordered = Tensor::cat(&[&lower, &median, &upper], last)?.clamp(f32::MIN, f32::MAX)?;
        if let Some(trace) = trace {
            trace.insert("raw".into(), raw.clone());
            trace.insert("softplus".into(), positive);
            trace.insert("increments".into(), increments);
            trace.insert("ordered".into(), ordered.clone());
        }
        Ok(ordered)
    }
}

fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    // max(x, 0) + log1p(exp(-abs(x))) avoids overflow. Candle has no log1p
    // primitive; compensate for rounding in 1+z so tiny negative tails survive.
    // All heavy projections and this transformation stay on the selected device.
    let z = x.abs()?.neg()?.exp()?;
    let u = z.affine(1.0, 1.0)?;
    let correction = z.sub(&u.affine(1.0, -1.0)?)?.div(&u)?;
    x.maximum(0f32)?.add(&u.log()?.add(&correction)?)
}
