use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder};

/// A small, complete example shared by patch embedding and the quantile head:
/// output(silu(hidden(x))) + residual(x).
pub struct ResidualBlock {
    hidden: Linear,
    output: Linear,
    residual: Linear,
}

impl ResidualBlock {
    pub fn load(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        use_bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            hidden: linear_b(input_dim, hidden_dim, use_bias, vb.pp("hidden_layer"))?,
            output: linear_b(hidden_dim, output_dim, use_bias, vb.pp("output_layer"))?,
            residual: linear_b(input_dim, output_dim, use_bias, vb.pp("residual_layer"))?,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.hidden.forward(x)?.silu()?;
        let main = self.output.forward(&h)?;
        let skip = self.residual.forward(x)?;
        main.add(&skip)
    }
}

// Time/variable attention and the scheduled transformer stack are in transformer.rs.
