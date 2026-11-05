//! Tiny demonstration with synthetic weights, not a Timestral forecast.
use candle_core::{Device, Module, Tensor};
use candle_nn::{VarBuilder, VarMap};
use timestral_rs::layers::ResidualBlock;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    // VarMap initializes demonstration weights instead of reading a checkpoint.
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, candle_core::DType::F32, &device);
    let block = ResidualBlock::load(64, 768, 192, true, vb.pp("patch_proj"))?;

    // Three patches, each with 32 values plus 32 missing-value indicators.
    let features = Tensor::zeros((1, 3, 64), candle_core::DType::F32, &device)?;
    let embedded = block.forward(&features)?;
    println!("Device: {device:?}");
    println!("Patch features: {:?}", features.dims());
    println!("Embedded patches: {:?}", embedded.dims());
    Ok(())
}
