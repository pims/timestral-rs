//! Context-only mask/position construction for one target row.
use anyhow::{ensure, Result};

pub struct UnivariateMasks {
    pub key_valid: Vec<u8>,
    pub positions: Vec<f32>,
    pub allowed: Vec<u8>,
}

pub fn univariate_masks(
    observed: &[bool],
    patch_size: usize,
    horizon_patches: usize,
    attend_missing: bool,
) -> Result<UnivariateMasks> {
    ensure!(
        patch_size > 0
            && !observed.is_empty()
            && observed.len().is_multiple_of(patch_size)
            && horizon_patches > 0,
        "Expected nonempty patch-aligned context and positive horizon"
    );
    let obs_patches: Vec<bool> = observed
        .chunks_exact(patch_size)
        .map(|p| p.iter().any(|v| *v))
        .collect();
    let first_observed = obs_patches.iter().position(|v| *v).unwrap_or(0);
    let mut started = false;
    let key_valid: Vec<u8> = obs_patches
        .iter()
        .map(|obs| {
            started |= obs;
            u8::from(*obs || (attend_missing && started))
        })
        .collect();
    let context = obs_patches.len();
    let total = context
        .checked_add(horizon_patches)
        .ok_or_else(|| anyhow::anyhow!("Token count overflow"))?;
    let size = total
        .checked_mul(total)
        .ok_or_else(|| anyhow::anyhow!("Mask size overflow"))?;
    let mut allowed = vec![0; size];
    for query in 0..total {
        for key in 0..total {
            let attend = if query < context {
                key == query || (key <= query && key_valid[key] != 0)
            } else if key < context {
                key_valid[key] != 0
            } else {
                key <= query
            };
            allowed[query * total + key] = u8::from(attend);
        }
    }
    Ok(UnivariateMasks {
        key_valid,
        positions: (0..total)
            .map(|i| (i as isize - first_observed as isize) as f32)
            .collect(),
        allowed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_patches_and_forecast_causality_follow_reference_rules() {
        let mut observed = vec![false; 128];
        observed[64] = true;
        let masks = univariate_masks(&observed, 32, 2, true).unwrap();
        assert_eq!(masks.key_valid, [0, 0, 1, 1]);
        assert_eq!(masks.positions, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let attend = |q: usize, k: usize| masks.allowed[q * 6 + k] != 0;
        assert!(attend(0, 0)); // A missing patch can still read itself.
        assert!(!attend(3, 4)); // History cannot read forecast tokens.
        assert!(attend(4, 2) && attend(4, 3) && attend(4, 4));
        assert!(!attend(4, 0) && !attend(4, 1) && !attend(4, 5));
        assert_eq!(
            univariate_masks(&observed, 32, 2, false).unwrap().key_valid,
            [0, 0, 1, 0]
        );
    }
}
