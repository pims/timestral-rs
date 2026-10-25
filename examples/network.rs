//! Transformer + head on embedded inputs; outputs remain in normalized space.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use timestral_rs::loading::load_model;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args
        .next()
        .context("Usage: network <checkpoint> <inputs.safetensors> <trace.safetensors>")?;
    let input = args.next().context("Missing input file")?;
    let output = args.next().context("Missing output file")?;
    ensure!(args.next().is_none(), "Too many arguments");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let model = load_model(checkpoint, &device)?;
    let inputs = candle_core::safetensors::load(input, &device)?;
    let get = |name| {
        inputs
            .get(name)
            .with_context(|| format!("Missing input: {name}"))
    };
    let split = get("encoder_patches")?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    ensure!(split.len() == 1, "Expected one encoder split value");
    let split = split[0] as usize;
    let x = get("x")?;
    let (_, patches, _) = x.dims3()?;
    ensure!(split > 0 && split < patches, "Invalid encoder split");
    let variable_mask = inputs.get("variable_allowed");
    let hidden =
        model
            .transformer
            .forward(x, get("positions")?, get("time_allowed")?, variable_mask)?;
    let decoder = hidden.narrow(1, split, patches - split)?.contiguous()?;
    let mut trace = model.head.forward_trace(&decoder)?;
    trace.insert("hidden".into(), hidden);
    trace.insert("decoder".into(), decoder);
    trace.insert(
        "direct_quantiles".into(),
        model.normalized_quantiles(
            x,
            split,
            get("positions")?,
            get("time_allowed")?,
            variable_mask,
        )?,
    );
    let cpu = trace
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor.to_device(&Device::Cpu)?)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    candle_core::safetensors::save(&cpu, output)?;
    println!(
        "Normalized network on {device:?}: {:?}",
        trace["quantiles"].dims()
    );
    Ok(())
}
