//! Run the complete transformer on explicit input tensors and export its trace.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use timestral_rs::loading::load_model;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args
        .next()
        .context("Usage: transformer <checkpoint> <inputs.safetensors> <trace.safetensors>")?;
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
    let variable_mask = inputs.get("variable_allowed");
    let mut trace = model.transformer.forward_trace(
        get("x")?,
        get("positions")?,
        get("time_allowed")?,
        variable_mask,
    )?;
    // Exercise the regular execution path as well as diagnostic tracing.
    let direct = model.transformer.forward(
        get("x")?,
        get("positions")?,
        get("time_allowed")?,
        variable_mask,
    )?;
    trace.insert("direct_output".into(), direct);
    let cpu = trace
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor.to_device(&Device::Cpu)?)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    candle_core::safetensors::save(&cpu, output)?;
    println!(
        "8-layer transformer on {device:?}: {:?}, {} trace tensors",
        trace["output"].dims(),
        trace.len()
    );
    Ok(())
}
