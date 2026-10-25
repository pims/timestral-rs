//! Run the head on decoder tokens, or its monotone transform on raw logits.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use timestral_rs::loading::load_model;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args
        .next()
        .context("Usage: quantile_head <checkpoint> <inputs.safetensors> <trace.safetensors>")?;
    let input = args.next().context("Missing input file")?;
    let output = args.next().context("Missing output file")?;
    ensure!(args.next().is_none(), "Too many arguments");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let model = load_model(checkpoint, &device)?;
    let inputs = candle_core::safetensors::load(input, &device)?;
    let mut trace = match (inputs.get("tokens"), inputs.get("raw")) {
        (Some(tokens), None) => {
            let mut trace = model.head.forward_trace(tokens)?;
            trace.insert("direct_quantiles".into(), model.head.forward(tokens)?);
            trace
        }
        (None, Some(raw)) => model.head.monotone_trace(raw)?,
        _ => anyhow::bail!("Provide exactly one of tokens or raw"),
    };
    for tensor in trace.values_mut() {
        *tensor = tensor.to_device(&Device::Cpu)?;
    }
    let cpu: std::collections::HashMap<_, _> = trace.into_iter().collect();
    candle_core::safetensors::save(&cpu, output)?;
    println!("Quantile head on {device:?}");
    Ok(())
}
