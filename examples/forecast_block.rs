//! Export normalization, mask, network and scale-refinement stages for one row.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use timestral_rs::loading::load_model;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args.next().context("Usage: forecast_block <checkpoint> <inputs.safetensors> <horizon-patches> <trace.safetensors>")?;
    let input = args.next().context("Missing input")?;
    let patches: usize = args
        .next()
        .context("Missing horizon patches")?
        .to_str()
        .context("Invalid horizon")?
        .parse()?;
    let output = args.next().context("Missing output")?;
    ensure!(args.next().is_none(), "Too many arguments");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let model = load_model(checkpoint, &device)?;
    let tensors = candle_core::safetensors::load(input, &Device::Cpu)?;
    let context = tensors
        .get("context")
        .context("Missing context")?
        .to_vec1::<f32>()?;
    let mask: Vec<bool> = tensors
        .get("observed")
        .context("Missing observed")?
        .to_vec1::<u8>()?
        .iter()
        .map(|v| *v != 0)
        .collect();
    let trace = model.predict_block_trace(&context, &mask, patches)?;
    let cpu = trace
        .iter()
        .map(|(name, value)| Ok((name.clone(), value.to_device(&Device::Cpu)?)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    candle_core::safetensors::save(&cpu, output)?;
    println!(
        "Original-unit block on {device:?}: {patches} patches, {} stages",
        trace.len()
    );
    Ok(())
}
