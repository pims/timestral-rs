use anyhow::{ensure, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelVariant {
    FiveM,
    TwentyEightM,
}

impl ModelVariant {
    pub fn name(self) -> &'static str {
        match self {
            Self::FiveM => "Timestral-5M",
            Self::TwentyEightM => "Timestral-28M",
        }
    }

    pub fn repo_id(self) -> &'static str {
        match self {
            Self::FiveM => "mistralai/Timestral-5M",
            Self::TwentyEightM => "mistralai/Timestral-28M",
        }
    }

    pub fn parameter_count(self) -> usize {
        match self {
            Self::FiveM => 5_208_960,
            Self::TwentyEightM => 27_821_184,
        }
    }
}

/// Loading-related fields; remaining inference settings are preserved verbatim.
#[derive(Debug, Deserialize)]
pub struct TimestralConfig {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub quantile_knots: Vec<f64>,
    pub block_size: usize,
    pub context_length: usize,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_gate")]
    pub time_attn_gate: String,
    #[serde(default = "default_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_partial_factor")]
    pub rope_partial_factor: f64,
    #[serde(default = "default_quantile_width_factor")]
    pub quantile_width_factor: f64,
    #[serde(default = "default_minimum_scale")]
    pub minimum_scale: f64,
    #[serde(default = "default_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "default_true")]
    pub attend_missing_patches: bool,
    #[serde(flatten)]
    pub inference_settings: BTreeMap<String, serde_json::Value>,
}

fn default_gate() -> String {
    "sdpa_sigmoid".into()
}

fn default_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rope_partial_factor() -> f64 {
    0.5
}
fn default_quantile_width_factor() -> f64 {
    2.0
}
fn default_minimum_scale() -> f64 {
    1e-6
}
fn default_true() -> bool {
    true
}

impl TimestralConfig {
    /// Identify a supported architecture from configuration, never the directory name.
    pub fn variant(&self) -> Result<ModelVariant> {
        let variant = match (
            self.d_model,
            self.num_heads,
            self.mlp_hidden_dim,
            self.block_schedule.len(),
            self.patch_size,
        ) {
            (192, 3, 512, 8, 32) => ModelVariant::FiveM,
            (384, 6, 1024, 12, 32) => ModelVariant::TwentyEightM,
            _ => anyhow::bail!(
                "Unsupported Timestral architecture: expected Timestral-5M \
                 (width 192, 3 heads, MLP 512, 8 layers) or Timestral-28M \
                 (width 384, 6 heads, MLP 1024, 12 layers), with patch size 32"
            ),
        };
        Ok(variant)
    }

    /// Compatibility validation for callers that explicitly require the 5M model.
    pub fn validate_5m(&self) -> Result<()> {
        self.validate()?;
        ensure!(
            self.variant()? == ModelVariant::FiveM,
            "Expected Timestral-5M"
        );
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.minimum_scale.is_finite()
                && self.minimum_scale > 0.0
                && self.minimum_scale <= f32::MAX as f64,
            "Invalid minimum scale"
        );
        ensure!(
            !self.stabilize_with_global,
            "Advancing refinement requires stabilize_with_global=false"
        );
        ensure!(
            self.quantile_width_factor.is_finite()
                && self.quantile_width_factor > 0.0
                && self.quantile_width_factor <= f32::MAX as f64,
            "Invalid quantile width factor"
        );
        self.variant()?;
        ensure!(
            self.num_heads > 0 && self.d_model.is_multiple_of(self.num_heads),
            "Invalid head count"
        );
        ensure!(
            self.norm_eps.is_finite() && self.norm_eps > 0.0,
            "Invalid normalization epsilon"
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "Invalid RoPE base"
        );
        ensure!(
            self.rope_partial_factor.is_finite()
                && self.rope_partial_factor > 0.0
                && self.rope_partial_factor <= 1.0,
            "Invalid partial RoPE factor"
        );
        let rope_dim = ((self.d_model / self.num_heads) as f64 * self.rope_partial_factor) as usize;
        ensure!(
            rope_dim > 0 && rope_dim.is_multiple_of(2),
            "RoPE dimension must be positive and even"
        );
        ensure!(
            self.mlp_hidden_dim > 0 && self.mlp_hidden_dim <= 4096,
            "Invalid MLP width"
        );
        ensure!(
            self.block_size > 0 && self.context_length > 0,
            "Invalid context/block size"
        );
        ensure!(
            self.quantile_knots.len() == 21
                && self.quantile_knots[10] == 0.5
                && self
                    .quantile_knots
                    .iter()
                    .all(|q| q.is_finite() && *q > 0.0 && *q < 1.0)
                && self.quantile_knots.windows(2).all(|q| q[0] < q[1]),
            "Expected 21 increasing quantile levels with median at index 10"
        );
        for kind in &self.block_schedule {
            ensure!(
                matches!(
                    kind.as_str(),
                    "time" | "variate" | "time_variate" | "variate_time"
                ),
                "Unknown layer kind: {kind}"
            );
        }
        for gate in [&self.time_attn_gate, &self.variate_attn_gate] {
            ensure!(
                matches!(gate.as_str(), "none" | "sdpa_sigmoid"),
                "Unknown attention gate: {gate}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(source: &str) -> TimestralConfig {
        serde_json::from_str(source).unwrap()
    }

    #[test]
    fn both_released_configs_are_recognized_and_5m_validation_stays_strict() {
        let small = config(include_str!("../tests/fixtures/config-5m.json"));
        let larger = config(include_str!("../tests/fixtures/config-28m.json"));
        small.validate().unwrap();
        small.validate_5m().unwrap();
        larger.validate().unwrap();
        assert!(larger.validate_5m().is_err());
        assert_eq!(small.variant().unwrap(), ModelVariant::FiveM);
        assert_eq!(larger.variant().unwrap(), ModelVariant::TwentyEightM);
    }

    #[test]
    fn reject_mixed_or_unsupported_architectures_and_invalid_inference_settings() {
        let source: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/config-28m.json")).unwrap();
        for (field, value) in [
            ("d_model", serde_json::json!(192)),
            ("num_heads", serde_json::json!(3)),
            ("mlp_hidden_dim", serde_json::json!(512)),
            ("patch_size", serde_json::json!(64)),
            ("block_schedule", serde_json::json!(vec!["time"; 8])),
            ("quantile_width_factor", serde_json::json!(0)),
            ("norm_eps", serde_json::json!(-1)),
        ] {
            let mut changed = source.clone();
            changed[field] = value;
            let changed: TimestralConfig = serde_json::from_value(changed).unwrap();
            assert!(changed.validate().is_err(), "Accepted invalid {field}");
        }
    }
}
