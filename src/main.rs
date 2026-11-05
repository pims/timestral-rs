use anyhow::{ensure, Context, Result};
use candle_core::{Device, Module, Tensor};
use timestral_rs::loading::load_model;

fn main() -> Result<()> {
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .context("Usage: timestral-rs <5M or 28M checkpoint directory> [patch-output.npy]")?;
    let output = args.next();
    ensure!(args.next().is_none(), "Too many arguments");
    let model = load_model(path, &device)?;
    println!("Device: {device:?}");
    println!(
        "Loaded {}: {} tensors, {} parameters",
        model.config.variant()?.name(),
        model.tensor_count,
        model.parameter_count
    );
    println!(
        "Layers: {}, width: {}, heads: {}",
        model.config.block_schedule.len(),
        model.config.d_model,
        model.config.num_heads
    );
    // Deterministic features for comparison with Python; not normalized history.
    let data: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) / 32.0).collect();
    let values = Tensor::from_vec(data, (1, 3, 32), &device)?;
    let missing = Tensor::zeros((1, 3, 32), candle_core::DType::F32, &device)?;
    let features = Tensor::cat(&[&values, &missing], 2)?;
    let embedding = model.patch_projection.forward(&features)?;
    let flat = embedding.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        flat.iter().all(|x| x.is_finite()),
        "Non-finite patch embeddings"
    );
    println!(
        "Checkpoint patch embeddings: {:?}; first 4: {:?}",
        embedding.dims(),
        &flat[..4]
    );
    if let Some(path) = output {
        embedding.to_device(&Device::Cpu)?.write_npy(path)?;
    }
    println!("Original-unit univariate forecasting is available via the forecast example.");
    Ok(())
}
