//! F32 time/variable attention and the complete checkpoint-scheduled transformer.
//!
//! Attention matrices are materialized for correctness/debugging. Long-context
//! memory optimizations and forecast mask/position construction are separate work.
use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear_b, Linear, VarBuilder};

use crate::config::TimestralConfig;

pub type BlockTrace = BTreeMap<String, Tensor>;

struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(width: usize, eps: f64, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            scale: vb.get(width, "scale")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(not(feature = "unfused-rmsnorm"))]
        if x.device().is_metal() {
            // Fuse GPU dispatches. Keep the reference CPU reduction: Candle's
            // fused CPU path changed a long-horizon tail beyond our tolerance
            // and did not improve the measured short-history CPU latency.
            // contiguous() is a no-op for an already contiguous input.
            return candle_nn::ops::rms_norm(&x.contiguous()?, &self.scale, self.eps as f32);
        }
        let inverse_rms = x
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.eps)?
            .sqrt()?
            .recip()?;
        x.broadcast_mul(&inverse_rms)?.broadcast_mul(&self.scale)
    }
}

struct RotaryEmbedding {
    inverse_frequencies: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &TimestralConfig) -> candle_core::Result<Self> {
        let head_dim = config.d_model / config.num_heads;
        let rotary_dim = (head_dim as f64 * config.rope_partial_factor) as usize;
        let base = config.rope_theta as f32;
        let mut frequencies = vec![0f32; head_dim / 2];
        for (i, frequency) in frequencies.iter_mut().take(rotary_dim / 2).enumerate() {
            *frequency = 1.0 / base.powf((2 * i) as f32 / rotary_dim as f32);
        }
        // Python pads a half-head frequency table with zeros, then repeats it.
        // rotate_half pairs two head halves, NOT adjacent coordinates.
        frequencies.extend_from_within(..);
        Ok(Self {
            inverse_frequencies: Tensor::from_vec(frequencies, (1, 1, head_dim), &Device::Cpu)?,
        })
    }

    fn tables(&self, positions: &Tensor, target: &Device) -> candle_core::Result<(Tensor, Tensor)> {
        // CPU F32 table preparation avoids Metal trig errors at large angles.
        // Calculate once per stack, then share the uploaded tables across layers.
        let angles = positions
            .to_device(&Device::Cpu)?
            .unsqueeze(2)?
            .broadcast_mul(&self.inverse_frequencies)?;
        Ok((
            angles.cos()?.to_device(target)?,
            angles.sin()?.to_device(target)?,
        ))
    }
}

struct Attention {
    qkv: Linear,
    singleton_value: Option<Linear>,
    output_projection: Linear,
    gate: Option<Linear>,
    is_time: bool,
    width: usize,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(
        config: &TimestralConfig,
        is_time: bool,
        vb: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let d = config.d_model;
        let gate = if is_time {
            &config.time_attn_gate
        } else {
            &config.variate_attn_gate
        };
        let qkv = linear_b(d, 3 * d, config.attn_bias, vb.pp("wQKV"))?;
        let singleton_value = if !is_time && !cfg!(feature = "full-variate-attention") {
            Some(value_only_projection(
                &qkv,
                d / config.num_heads,
                config.num_heads,
            )?)
        } else {
            None
        };
        Ok(Self {
            qkv,
            singleton_value,
            output_projection: linear_b(d, d, config.attn_bias, vb.pp("wO"))?,
            gate: if gate == "sdpa_sigmoid" {
                Some(linear_b(d, d, config.attn_bias, vb.pp("_gate"))?)
            } else {
                None
            },
            is_time,
            width: d,
            heads: config.num_heads,
            head_dim: d / config.num_heads,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        allowed: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        mut trace: Option<&mut BlockTrace>,
    ) -> Result<Tensor> {
        // With one row and no variable mask, every attention head has one key:
        // softmax([score]) = [1], so the attention output is exactly its V.
        // Explicit masks (including fully masked queries) retain the general path.
        if !self.is_time && x.dim(0)? == 1 && allowed.is_none() {
            if let Some(value) = &self.singleton_value {
                return self.forward_singleton(x, value, trace);
            }
        }
        // Variable attention treats patches as batches and rows as the sequence.
        // RoPE is applied only to time attention.
        let inputs = if self.is_time {
            x.clone()
        } else {
            x.transpose(0, 1)?.contiguous()?
        };
        let (batch, sequence, _) = inputs.dims3()?;
        let fused = self.qkv.forward(&inputs.contiguous()?)?.reshape((
            batch,
            sequence,
            3,
            self.head_dim,
            self.heads,
        ))?;
        let split = |i| -> candle_core::Result<Tensor> {
            fused
                .narrow(2, i, 1)?
                .squeeze(2)?
                .transpose(2, 3)?
                .contiguous()
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        record(&mut trace, "q", &q);
        record(&mut trace, "k", &k);
        record(&mut trace, "v", &v);
        let (query, key) = if self.is_time {
            let qr = rotate(&q, cos, sin)?;
            let kr = rotate(&k, cos, sin)?;
            record(&mut trace, "q_rotated", &qr);
            record(&mut trace, "k_rotated", &kr);
            (qr, kr)
        } else {
            (q, k)
        };
        let query = query.transpose(1, 2)?.contiguous()?;
        let key = key.transpose(1, 2)?.contiguous()?;
        let value = v.transpose(1, 2)?.contiguous()?;
        let scores = query
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let probabilities = match allowed {
            Some(allowed) => {
                ensure!(
                    allowed.dims() == [batch, 1, sequence, sequence]
                        && allowed.dtype() == DType::U8,
                    "Expected U8 {} mask [{batch}, 1, {sequence}, {sequence}]",
                    if self.is_time { "time" } else { "variable" }
                );
                let mask = allowed.broadcast_as(scores.shape())?.contiguous()?;
                // Fully masked queries produce zero attention, matching SDPA.
                let blocked = Tensor::full(f32::MIN, scores.shape(), scores.device())?;
                candle_nn::ops::softmax_last_dim(&mask.where_cond(&scores, &blocked)?)?
                    .mul(&mask.to_dtype(DType::F32)?)?
            }
            None => {
                ensure!(!self.is_time, "Time attention requires an explicit mask");
                candle_nn::ops::softmax_last_dim(&scores)?
            }
        };
        let attention = probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, sequence, self.width))?;
        let gate = match &self.gate {
            Some(layer) => candle_nn::ops::sigmoid(&layer.forward(&inputs)?)?,
            None => Tensor::ones(attention.shape(), DType::F32, inputs.device())?,
        };
        let gated = attention.mul(&gate)?;
        record(&mut trace, "attention", &attention);
        record(&mut trace, "gate", &gate);
        record(&mut trace, "gated", &gated);
        // Flatten heads in their attention output order, then restore packed rows.
        // Python applies wO AFTER restoring [rows, patches, width].
        let restored = if self.is_time {
            gated
        } else {
            gated.transpose(0, 1)?.contiguous()?
        };
        let projected = self.output_projection.forward(&restored)?;
        record(&mut trace, "attn_projected", &projected);
        Ok(projected)
    }

    fn forward_singleton(
        &self,
        x: &Tensor,
        value: &Linear,
        trace: Option<&mut BlockTrace>,
    ) -> Result<Tensor> {
        let (_, patches, _) = x.dims3()?;
        // Cached V weights already use [head, head_dim] output ordering.
        // Working directly in packed [1, patches, D] also avoids row transposes.
        let attention = value.forward(&x.contiguous()?)?;
        let gate = match &self.gate {
            Some(layer) => candle_nn::ops::sigmoid(&layer.forward(x)?)?,
            None => Tensor::ones(attention.shape(), DType::F32, x.device())?,
        };
        let gated = attention.mul(&gate)?;
        let projected = self.output_projection.forward(&gated)?;
        if let Some(trace) = trace {
            // Q/K do not contribute to the optimized output. Compute them only
            // for the existing diagnostic trace schema and Python comparisons.
            let fused = self.qkv.forward(&x.contiguous()?)?.reshape((
                patches,
                1,
                3,
                self.head_dim,
                self.heads,
            ))?;
            for (name, i) in [("q", 0), ("k", 1)] {
                trace.insert(
                    name.into(),
                    fused
                        .narrow(2, i, 1)?
                        .squeeze(2)?
                        .transpose(2, 3)?
                        .contiguous()?,
                );
            }
            trace.insert(
                "v".into(),
                attention.reshape((patches, 1, self.heads, self.head_dim))?,
            );
            for (name, tensor) in [("attention", attention), ("gate", gate), ("gated", gated)] {
                trace.insert(name.into(), tensor.reshape((patches, 1, self.width))?);
            }
            trace.insert("attn_projected".into(), projected.clone());
        }
        Ok(projected)
    }
}

/// The checkpoint's V rows are [head_dim, heads]. Reorder once at load time so
/// the cached projection emits the same flattened [heads, head_dim] as attention.
fn value_only_projection(
    qkv: &Linear,
    head_dim: usize,
    heads: usize,
) -> candle_core::Result<Linear> {
    let width = head_dim * heads;
    let weight = qkv
        .weight()
        .narrow(0, 2 * width, width)?
        .reshape((head_dim, heads, width))?
        .transpose(0, 1)?
        .contiguous()?
        .reshape((width, width))?;
    let bias = qkv
        .bias()
        .map(|bias| {
            bias.narrow(0, 2 * width, width)?
                .reshape((head_dim, heads))?
                .transpose(0, 1)?
                .contiguous()?
                .reshape(width)
        })
        .transpose()?;
    Ok(Linear::new(weight, bias))
}

/// One scheduled layer: attention sublayers in order, followed by a shared MLP.
struct TransformerBlock {
    attentions: Vec<(RmsNorm, Attention)>,
    mlp_norm: RmsNorm,
    w12: Linear,
    w3: Linear,
    mlp_hidden: usize,
}

impl TransformerBlock {
    fn load(config: &TimestralConfig, index: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let kind = config
            .block_schedule
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("Invalid layer index: {index}"))?;
        let mut attentions = Vec::new();
        for (j, axis) in kind.split('_').enumerate() {
            let norm = RmsNorm::load(
                config.d_model,
                config.norm_eps,
                vb.pp(format!("attn_norms.{j}")),
            )?;
            let attention = Attention::load(config, axis == "time", vb.pp(format!("attns.{j}")))?;
            attentions.push((norm, attention));
        }
        Ok(Self {
            attentions,
            mlp_norm: RmsNorm::load(config.d_model, config.norm_eps, vb.pp("mlp_norm"))?,
            w12: linear_b(
                config.d_model,
                2 * config.mlp_hidden_dim,
                config.mlp_bias,
                vb.pp("mlp.w12"),
            )?,
            w3: linear_b(
                config.mlp_hidden_dim,
                config.d_model,
                config.mlp_bias,
                vb.pp("mlp.w3"),
            )?,
            mlp_hidden: config.mlp_hidden_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        time_mask: &Tensor,
        variable_mask: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        mut trace: Option<&mut BlockTrace>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for (j, (norm, attention)) in self.attentions.iter().enumerate() {
            let normalized = norm.forward(&x)?;
            let mut subtrace = trace.as_ref().map(|_| BlockTrace::new());
            record(&mut subtrace.as_mut(), "attn_norm", &normalized);
            let mask = if attention.is_time {
                Some(time_mask)
            } else {
                variable_mask
            };
            let projected = attention.forward(&normalized, mask, cos, sin, subtrace.as_mut())?;
            x = x.add(&projected)?;
            record(&mut subtrace.as_mut(), "attn_residual", &x);
            if let (Some(destination), Some(subtrace)) = (trace.as_deref_mut(), subtrace) {
                destination.extend(
                    subtrace
                        .into_iter()
                        .map(|(name, value)| (format!("attn{j}.{name}"), value)),
                );
            }
        }
        let normalized = self.mlp_norm.forward(&x)?;
        let expanded = self.w12.forward(&normalized)?;
        let gate = expanded.narrow(D::Minus1, 0, self.mlp_hidden)?;
        let up = expanded.narrow(D::Minus1, self.mlp_hidden, self.mlp_hidden)?;
        let hidden = gate.mul(&up.silu()?)?;
        let mlp_output = self.w3.forward(&hidden)?;
        let output = x.add(&mlp_output)?;
        for (name, value) in [
            ("mlp_norm", &normalized),
            ("mlp_gate", &gate),
            ("mlp_up", &up),
            ("mlp_hidden", &hidden),
            ("mlp_output", &mlp_output),
            ("output", &output),
        ] {
            record(&mut trace, name, value);
        }
        Ok(output)
    }
}

/// All checkpoint-scheduled layers and the final RMSNorm.
///
/// Inputs are packed F32 [rows, patches, width]. The caller supplies explicit
/// time masks [rows, 1, patches, patches], variable masks [patches, 1, rows, rows],
/// and F32 integer-valued positions [rows, patches]. Masks are U8, 1 means attend.
/// A missing variable mask permits every row to attend every other row; callers
/// packing independent series must supply their group-isolation mask.
pub struct Transformer {
    blocks: Vec<TransformerBlock>,
    rope: RotaryEmbedding,
    final_norm: RmsNorm,
    width: usize,
}

impl Transformer {
    /// `vb` is rooted at `transformer`.
    pub fn load(config: &TimestralConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let blocks = (0..config.block_schedule.len())
            .map(|i| TransformerBlock::load(config, i, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            rope: RotaryEmbedding::new(config)?,
            final_norm: RmsNorm::load(config.d_model, config.norm_eps, vb.pp("out_norm"))?,
            width: config.d_model,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        time_mask: &Tensor,
        variable_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.run(x, positions, time_mask, variable_mask, None)
    }

    pub fn forward_trace(
        &self,
        x: &Tensor,
        positions: &Tensor,
        time_mask: &Tensor,
        variable_mask: Option<&Tensor>,
    ) -> Result<BlockTrace> {
        let mut trace = BlockTrace::new();
        self.run(x, positions, time_mask, variable_mask, Some(&mut trace))?;
        Ok(trace)
    }

    fn run(
        &self,
        x: &Tensor,
        positions: &Tensor,
        time_mask: &Tensor,
        variable_mask: Option<&Tensor>,
        mut trace: Option<&mut BlockTrace>,
    ) -> Result<Tensor> {
        validate_inputs(x, positions, self.width)?;
        let (cos, sin) = self.rope.tables(positions, x.device())?;
        record(&mut trace, "cos", &cos);
        record(&mut trace, "sin", &sin);
        let mut x = x.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            let mut block_trace = trace.as_ref().map(|_| BlockTrace::new());
            x = block.forward(
                &x,
                time_mask,
                variable_mask,
                &cos,
                &sin,
                block_trace.as_mut(),
            )?;
            if let (Some(destination), Some(block_trace)) = (trace.as_deref_mut(), block_trace) {
                destination.extend(
                    block_trace
                        .into_iter()
                        .map(|(name, value)| (format!("layers.{i}.{name}"), value)),
                );
            }
        }
        let output = self.final_norm.forward(&x)?;
        record(&mut trace, "output", &output);
        Ok(output)
    }
}

/// Compatibility wrapper for the original first-block diagnostic example.
pub struct TimeTransformerBlock {
    block: TransformerBlock,
    rope: RotaryEmbedding,
    width: usize,
}

impl TimeTransformerBlock {
    pub fn load(config: &TimestralConfig, index: usize, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.block_schedule.get(index).map(String::as_str) == Some("time"),
            "TimeTransformerBlock requires a time-only layer at index {index}"
        );
        Ok(Self {
            block: TransformerBlock::load(config, index, vb)?,
            rope: RotaryEmbedding::new(config)?,
            width: config.d_model,
        })
    }

    pub fn forward_trace(
        &self,
        x: &Tensor,
        positions: &Tensor,
        allowed: &Tensor,
    ) -> Result<BlockTrace> {
        validate_inputs(x, positions, self.width)?;
        let (cos, sin) = self.rope.tables(positions, x.device())?;
        let mut trace = BlockTrace::new();
        self.block
            .forward(x, allowed, None, &cos, &sin, Some(&mut trace))?;
        let mut trace: BlockTrace = trace
            .into_iter()
            .map(|(name, value)| {
                (
                    name.strip_prefix("attn0.").unwrap_or(&name).to_owned(),
                    value,
                )
            })
            .collect();
        trace.insert("cos".into(), cos);
        trace.insert("sin".into(), sin);
        Ok(trace)
    }
}

fn validate_inputs(x: &Tensor, positions: &Tensor, width: usize) -> Result<()> {
    let (rows, tokens, actual_width) = x.dims3()?;
    ensure!(
        rows > 0 && tokens > 0 && actual_width == width && x.dtype() == DType::F32,
        "Expected nonempty F32 [rows, tokens, {width}] input"
    );
    ensure!(
        positions.dims() == [rows, tokens] && positions.dtype() == DType::F32,
        "Expected F32 positions [rows, tokens]"
    );
    Ok(())
}

fn record(trace: &mut Option<&mut BlockTrace>, name: &str, value: &Tensor) {
    if let Some(trace) = trace.as_deref_mut() {
        trace.insert(name.into(), value.clone());
    }
}

fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let left = x.narrow(D::Minus1, 0, half)?;
    let right = x.narrow(D::Minus1, half, half)?.neg()?;
    let rotated_half = Tensor::cat(&[&right, &left], D::Minus1)?;
    x.broadcast_mul(&cos.unsqueeze(2)?)?
        .add(&rotated_half.broadcast_mul(&sin.unsqueeze(2)?)?)
}

#[cfg(test)]
mod singleton_tests {
    use super::*;

    fn qkv(bias: bool) -> Linear {
        let weight = Tensor::arange(0f32, 108f32, &Device::Cpu)
            .unwrap()
            .reshape((18, 6))
            .unwrap();
        let bias = bias.then(|| Tensor::arange(0f32, 18f32, &Device::Cpu).unwrap());
        Linear::new(weight, bias)
    }

    fn input() -> Tensor {
        Tensor::arange(1f32, 13f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 6))
            .unwrap()
    }

    #[test]
    fn value_cache_preserves_interleaved_heads_and_bias() {
        for bias in [false, true] {
            let qkv = qkv(bias);
            let cached = value_only_projection(&qkv, 2, 3).unwrap();
            let projected = qkv.forward(&input()).unwrap();
            let expected = projected
                .narrow(2, 12, 6)
                .unwrap()
                .reshape((2, 1, 2, 3))
                .unwrap()
                .transpose(2, 3)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((1, 2, 6))
                .unwrap();
            let actual = cached.forward(&input()).unwrap();
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn explicit_singleton_masks_preserve_fully_masked_query_behavior() {
        let qkv = qkv(true);
        let cached = value_only_projection(&qkv, 2, 3).unwrap();
        let expected = cached
            .forward(&input())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let attention = Attention {
            qkv,
            singleton_value: Some(cached),
            output_projection: Linear::new(Tensor::eye(6, DType::F32, &Device::Cpu).unwrap(), None),
            gate: None,
            is_time: false,
            width: 6,
            heads: 3,
            head_dim: 2,
        };
        let tables = Tensor::zeros((1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let fast = attention
            .forward(&input(), None, &tables, &tables, None)
            .unwrap();
        assert_eq!(
            fast.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected
        );
        let allowed = Tensor::ones((2, 1, 1, 1), DType::U8, &Device::Cpu).unwrap();
        let general = attention
            .forward(&input(), Some(&allowed), &tables, &tables, None)
            .unwrap();
        assert_eq!(
            general.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected
        );
        let blocked = Tensor::zeros((2, 1, 1, 1), DType::U8, &Device::Cpu).unwrap();
        let output = attention
            .forward(&input(), Some(&blocked), &tables, &tables, None)
            .unwrap();
        assert!(output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == 0.0));
        let mut trace = BlockTrace::new();
        let traced = attention
            .forward(&input(), None, &tables, &tables, Some(&mut trace))
            .unwrap();
        assert_eq!(
            traced.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected
        );
        assert_eq!(trace["v"].dims(), [2, 1, 3, 2]);
        assert!(trace.contains_key("q") && trace.contains_key("k"));
    }
}
