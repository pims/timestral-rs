//! Context-only mask/position construction for one target row.
use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowRole {
    Target,
    Past,
    Known,
}

pub(crate) struct CovariateMasks {
    pub positions: Vec<f32>,
    pub key_valid: Vec<u8>,
    pub time: Vec<u8>,
    pub variable: Vec<u8>,
}

/// One group, packed [Enc(window); Dec(window)]. Matches the reference's
/// two-stream masks and directional exogeneity firewall.
pub(crate) fn covariate_masks(
    observed: &[Vec<bool>],
    roles: &[RowRole],
    patch_size: usize,
    context_patches: usize,
    block_size: usize,
    attend_missing: bool,
) -> Result<CovariateMasks> {
    let rows = observed.len();
    ensure!(
        rows > 0 && roles.len() == rows && patch_size > 0 && block_size > 0,
        "Invalid mask rows"
    );
    let steps = observed[0].len();
    ensure!(
        steps.is_multiple_of(patch_size) && observed.iter().all(|r| r.len() == steps),
        "Unaligned mask rows"
    );
    let n = steps / patch_size;
    ensure!(
        context_patches > 0 && context_patches < n,
        "Invalid context split"
    );
    let tokens = n
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("Mask size overflow"))?;
    let obs: Vec<Vec<bool>> = observed
        .iter()
        .map(|r| {
            r.chunks_exact(patch_size)
                .map(|p| p.iter().any(|v| *v))
                .collect()
        })
        .collect();
    let first = (0..n).find(|p| obs.iter().any(|r| r[*p])).unwrap_or(0);
    let key_valid: Vec<u8> = obs
        .iter()
        .flat_map(|r| {
            let mut started = false;
            r.iter().map(move |&v| {
                started |= v;
                u8::from(v || (started && attend_missing))
            })
        })
        .collect();
    let time_len = rows
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(tokens))
        .ok_or_else(|| anyhow::anyhow!("Mask size overflow"))?;
    let mut time = vec![0; time_len];
    for (row, role) in roles.iter().enumerate() {
        for query in 0..tokens {
            for key in 0..tokens {
                let p = query % n;
                let k = key % n;
                let allowed = match (query < n, key < n) {
                    (true, true) => {
                        p == k
                            || ((role == &RowRole::Known || k <= p) && key_valid[row * n + k] != 0)
                    }
                    (true, false) => false,
                    (false, true) => {
                        p >= context_patches
                            && k < context_patches
                                + block_size * ((p - context_patches) / block_size)
                            && key_valid[row * n + k] != 0
                    }
                    (false, false) => {
                        p == k
                            || (p >= context_patches
                                && k >= context_patches
                                && k <= p
                                && (p - context_patches) / block_size
                                    == (k - context_patches) / block_size)
                    }
                };
                time[(row * tokens + query) * tokens + key] = u8::from(allowed);
            }
        }
    }
    let var_len = tokens
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(rows))
        .ok_or_else(|| anyhow::anyhow!("Mask size overflow"))?;
    let mut variable = vec![0; var_len];
    for slot in 0..tokens {
        for (query, qr) in roles.iter().enumerate() {
            for (key, kr) in roles.iter().enumerate() {
                let firewall = (qr != &RowRole::Target && kr == &RowRole::Target)
                    || (qr == &RowRole::Known && kr != &RowRole::Known);
                let valid = slot >= n || key_valid[key * n + slot] != 0;
                variable[(slot * rows + query) * rows + key] =
                    u8::from(query == key || (valid && !firewall));
            }
        }
    }
    let positions = (0..rows)
        .flat_map(|_| (0..tokens).map(|p| ((p % n) as isize - first as isize) as f32))
        .collect();
    Ok(CovariateMasks {
        positions,
        key_valid,
        time,
        variable,
    })
}

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
    fn covariate_masks_preserve_exogenous_roles_and_known_future_visibility() {
        let roles = [RowRole::Target, RowRole::Past, RowRole::Known];
        let observed = vec![
            vec![false, false, true, true, false, false, false, false],
            vec![true, true, true, true, false, false, false, false],
            vec![false, false, false, false, true, true, true, true],
        ];
        let m = covariate_masks(&observed, &roles, 2, 2, 2, true).unwrap();
        let time = |r: usize, q: usize, k: usize| m.time[(r * 8 + q) * 8 + k] != 0;
        let variable = |s: usize, q: usize, k: usize| m.variable[(s * 3 + q) * 3 + k] != 0;
        assert!(!time(0, 0, 3)); // A target's historical encoder cannot read its horizon.
        assert!(time(2, 0, 3)); // Known rows are bidirectional over observed keys.
        assert!(!time(0, 6, 2)); // Target decoder cannot read future encoder slots.
        assert!(time(0, 6, 1)); // It can read valid history keys.
        assert!(!time(0, 6, 7)); // Decoder attention is causal.
        assert!(time(0, 7, 6));
        assert!(variable(2, 0, 2)); // Target reads available known future.
        for s in 0..8 {
            assert!(!variable(s, 1, 0)); // Past covariates cannot read targets.
            assert!(!variable(s, 2, 0));
            assert!(!variable(s, 2, 1)); // Known covariates cannot read causal rows.
            assert!(variable(s, 2, 2)); // Diagonal remains open even for missing rows.
        }
        assert!(!variable(0, 0, 2)); // Unknown encoder keys remain masked.
        assert_eq!(&m.positions[..8], &[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    }

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
