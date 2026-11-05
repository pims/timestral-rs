//! Strict local loading. Network access is handled by scripts/download_model.py.
use crate::head::QuantileHead;
use crate::transformer::Transformer;
use crate::{
    config::{ModelVariant, TimestralConfig},
    layers::ResidualBlock,
    model::TimestralModel,
};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

/// Compatibility entry point that continues to require the 5M architecture.
pub fn load_5m(directory: impl AsRef<Path>, device: &Device) -> Result<TimestralModel> {
    load_checkpoint(directory.as_ref(), device, Some(ModelVariant::FiveM))
}

/// Load either supported checkpoint and construct the embedding, transformer, and head.
pub fn load_model(directory: impl AsRef<Path>, device: &Device) -> Result<TimestralModel> {
    load_checkpoint(directory.as_ref(), device, None)
}

fn load_checkpoint(
    directory: &Path,
    device: &Device,
    required: Option<ModelVariant>,
) -> Result<TimestralModel> {
    let config_path = directory.join("config.json");
    let config: TimestralConfig = serde_json::from_slice(
        &std::fs::read(&config_path).with_context(|| format!("Read {}", config_path.display()))?,
    )
    .context("Parse Timestral configuration")?;
    config.validate()?;
    let variant = config.variant()?;
    if let Some(required) = required {
        ensure!(
            variant == required,
            "Expected {}, got {}",
            required.name(),
            variant.name()
        );
    }
    let path = directory.join("model.safetensors");
    let tensors = candle_core::safetensors::load(&path, device)
        .with_context(|| format!("Load {}", path.display()))?;
    validate_tensors(&config, &tensors)?;
    let tensor_count = tensors.len();
    let parameter_count: usize = tensors.values().map(Tensor::elem_count).sum();
    ensure!(
        parameter_count == variant.parameter_count(),
        "Expected {} parameters for {}, got {parameter_count}",
        variant.parameter_count(),
        variant.name()
    );
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let patch_projection = ResidualBlock::load(
        2 * config.patch_size,
        4 * config.d_model,
        config.d_model,
        true,
        vb.pp("representation.patch_proj"),
    )?;
    let forecast_query = vb.get(config.d_model, "representation.forecast_query")?;
    let head = QuantileHead::load(&config, vb.pp("head"))?;
    let transformer = Transformer::load(&config, vb.pp("transformer"))?;
    Ok(TimestralModel {
        config,
        patch_projection,
        forecast_query,
        head,
        transformer,
        weights: vb,
        tensor_count,
        parameter_count,
    })
}

// Mirrors the official model's state_dict, including every scheduled attention axis.
fn expected_shapes(c: &TimestralConfig) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    let d = c.d_model;
    let mut linear = |name: String, input, output, bias| {
        shapes.insert(format!("{name}.weight"), vec![output, input]);
        if bias {
            shapes.insert(format!("{name}.bias"), vec![output]);
        }
    };
    for (prefix, input, output, bias) in [
        ("representation.patch_proj", 2 * c.patch_size, d, true),
        (
            "head.projection",
            d,
            c.patch_size * c.quantile_knots.len(),
            false,
        ),
    ] {
        linear(format!("{prefix}.hidden_layer"), input, 4 * d, bias);
        linear(format!("{prefix}.output_layer"), 4 * d, output, bias);
        linear(format!("{prefix}.residual_layer"), input, output, bias);
    }
    for (i, kind) in c.block_schedule.iter().enumerate() {
        let prefix = format!("transformer.layers.{i}");
        for (j, axis) in kind.split('_').enumerate() {
            let attn = format!("{prefix}.attns.{j}");
            linear(format!("{attn}.wQKV"), d, 3 * d, c.attn_bias);
            linear(format!("{attn}.wO"), d, d, c.attn_bias);
            let gate = if axis == "time" {
                &c.time_attn_gate
            } else {
                &c.variate_attn_gate
            };
            if gate == "sdpa_sigmoid" {
                linear(format!("{attn}._gate"), d, d, c.attn_bias);
            }
        }
        linear(
            format!("{prefix}.mlp.w12"),
            d,
            2 * c.mlp_hidden_dim,
            c.mlp_bias,
        );
        linear(format!("{prefix}.mlp.w3"), c.mlp_hidden_dim, d, c.mlp_bias);
    }
    shapes.insert("representation.forecast_query".into(), vec![d]);
    shapes.insert("transformer.out_norm.scale".into(), vec![d]);
    for (i, kind) in c.block_schedule.iter().enumerate() {
        shapes.insert(format!("transformer.layers.{i}.mlp_norm.scale"), vec![d]);
        for j in 0..kind.split('_').count() {
            shapes.insert(
                format!("transformer.layers.{i}.attn_norms.{j}.scale"),
                vec![d],
            );
        }
    }
    shapes
}

fn validate_tensors(c: &TimestralConfig, tensors: &HashMap<String, Tensor>) -> Result<()> {
    let expected = expected_shapes(c);
    for (name, shape) in &expected {
        let tensor = tensors
            .get(name)
            .with_context(|| format!("Missing checkpoint tensor: {name}"))?;
        ensure!(
            tensor.dims() == shape,
            "Shape mismatch for {name}: expected {shape:?}, got {:?}",
            tensor.dims()
        );
        ensure!(
            tensor.dtype() == DType::F32,
            "Expected F32 checkpoint tensor: {name}"
        );
    }
    let mut extra: Vec<_> = tensors
        .keys()
        .filter(|n| !expected.contains_key(*n))
        .collect();
    extra.sort();
    ensure!(extra.is_empty(), "Unexpected checkpoint tensors: {extra:?}");
    Ok(())
}
