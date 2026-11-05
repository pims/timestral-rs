//! Execute and save intermediate tensors for the first time-only block.
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use timestral_rs::{loading::load_model, transformer::TimeTransformerBlock};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint = args
        .next()
        .context("Usage: first_block <checkpoint> <inputs.safetensors> <trace.safetensors>")?;
    let input_path = args.next().context("Missing input file")?;
    let output_path = args.next().context("Missing output file")?;
    ensure!(args.next().is_none(), "Too many arguments");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let model = load_model(checkpoint, &device)?;
    let block =
        TimeTransformerBlock::load(&model.config, 0, model.weights.pp("transformer.layers.0"))?;
    let inputs = candle_core::safetensors::load(input_path, &device)?;
    let get = |name| {
        inputs
            .get(name)
            .with_context(|| format!("Missing input: {name}"))
    };
    let trace = block.forward_trace(get("x")?, get("positions")?, get("allowed")?)?;
    // Explicit readback synchronizes GPU execution before writing the comparison.
    let cpu = trace
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor.to_device(&Device::Cpu)?)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    candle_core::safetensors::save(&cpu, output_path)?;
    println!(
        "First block on {device:?}: {:?}, {} trace tensors",
        trace["output"].dims(),
        trace.len()
    );
    Ok(())
}
