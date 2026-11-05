//! CPU normalization and advancing-frame refinement. Statistics use f64;
//! affine/asinh and inverse operations follow the reference's F32 boundaries.
use anyhow::{ensure, Result};

pub struct ScaledContext {
    pub values: Vec<f32>,
    pub loc: Vec<f32>,
    pub scale: Vec<f32>,
}

pub fn scale_context(
    values: &[f32],
    observed: &[bool],
    patch_size: usize,
    minimum_scale: f64,
    use_asinh: bool,
) -> Result<ScaledContext> {
    ensure!(
        patch_size > 0
            && !values.is_empty()
            && values.len() == observed.len()
            && values.len().is_multiple_of(patch_size),
        "Expected matching, nonempty patch-aligned context and mask"
    );
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "Prepared context must contain finite values"
    );
    let mut loc = vec![0f32; values.len()];
    let mut scale = vec![0f32; values.len()];
    let (mut count, mut sum, mut m2, mut previous_mean) = (0f64, 0f64, 0f64, 0f64);
    for (patch, (samples, mask)) in values
        .chunks_exact(patch_size)
        .zip(observed.chunks_exact(patch_size))
        .enumerate()
    {
        for (&sample, &obs) in samples.iter().zip(mask) {
            let x = sample as f64;
            let weight = if obs { 1.0 } else { 0.0 };
            count += weight;
            sum += x * weight;
            let mean = sum / count.max(1.0);
            m2 += (x - previous_mean) * (x - mean) * weight;
            previous_mean = mean;
        }
        let std = (m2 / (count - 1.0).max(1.0)).sqrt();
        ensure!(!std.is_nan(), "Invalid causal variance");
        let start = patch * patch_size;
        loc[start..start + patch_size].fill(previous_mean as f32);
        scale[start..start + patch_size].fill(std.max(minimum_scale) as f32);
    }
    let scaled = values
        .iter()
        .zip(observed)
        .zip(loc.iter().zip(&scale))
        .map(|((&x, &obs), (&mean, &std))| {
            if !obs {
                0.0
            } else {
                let normalized = (x - mean) / std.max(f32::EPSILON);
                if use_asinh {
                    normalized.asinh()
                } else {
                    normalized
                }
            }
        })
        .collect();
    Ok(ScaledContext {
        values: scaled,
        loc,
        scale,
    })
}

/// Replace +/-infinity with the nearest finite value of the same quantile vector.
/// Preserve NaNs so callers can report instability instead of hiding it.
pub fn clamp_nonfinite(values: &mut [f32], quantiles: usize) {
    for row in values.chunks_exact_mut(quantiles) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &value in row.iter().filter(|v| v.is_finite()) {
            lo = lo.min(value);
            hi = hi.max(value);
        }
        for value in row {
            if *value == f32::INFINITY {
                *value = if hi.is_finite() { hi } else { 0.0 };
            } else if *value == f32::NEG_INFINITY {
                *value = if lo.is_finite() { lo } else { 0.0 };
            }
        }
    }
}

pub fn denormalize(
    normalized: &[f32],
    loc: f32,
    scale: f32,
    use_asinh: bool,
    quantiles: usize,
) -> Vec<f32> {
    let mut output: Vec<f32> = normalized
        .iter()
        .map(|&value| {
            let inverted = if use_asinh { value.sinh() } else { value };
            inverted * scale + loc
        })
        .collect();
    clamp_nonfinite(&mut output, quantiles);
    output
}

#[derive(Clone, Copy, Default)]
struct Moments {
    count: f64,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn combine(self, other: Self) -> Self {
        let count = self.count + other.count;
        if count == 0.0 {
            return Self::default();
        }
        let delta = other.mean - self.mean;
        Self {
            count,
            mean: self.mean + delta * other.count / count,
            m2: self.m2 + other.m2 + delta * delta * self.count * other.count / count,
        }
    }

    fn from_samples(samples: &[f64]) -> Self {
        let count = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / count.max(1.0);
        let m2 = samples.iter().map(|x| (x - mean).powi(2)).sum();
        Self { count, mean, m2 }
    }
}

/// Reproduce the inclusive parallel prefix scan used by the reference refinement.
fn context_moments(
    context: &[f32],
    observed: &[bool],
    patch_size: usize,
    horizon_patches: usize,
) -> Moments {
    let mut patches: Vec<Moments> = context
        .chunks_exact(patch_size)
        .zip(observed.chunks_exact(patch_size))
        .map(|(values, mask)| {
            let samples: Vec<f64> = values
                .iter()
                .zip(mask)
                .filter(|(_, obs)| **obs)
                .map(|(&x, _)| x as f64)
                .collect();
            Moments::from_samples(&samples)
        })
        .collect();
    let last_context = patches.len() - 1;
    patches.resize(patches.len() + horizon_patches, Moments::default());
    let mut distance = 1;
    while distance < patches.len() {
        let previous = patches.clone();
        for i in 0..patches.len() {
            let left = if i >= distance {
                previous[i - distance]
            } else {
                Moments::default()
            };
            patches[i] = left.combine(previous[i]);
        }
        distance *= 2;
    }
    patches[last_context]
}

/// Refine an anchor-denormalized block using earlier predicted medians.
/// Returned quantiles stay in original measurement units.
pub fn refine_block(
    anchor: &[f32],
    context: &[f32],
    observed: &[bool],
    config: &crate::config::TimestralConfig,
) -> Result<Vec<f32>> {
    let patch_size = config.patch_size;
    let quantiles = config.quantile_knots.len();
    let median_index = 10;
    let minimum_scale = config.minimum_scale;
    let use_asinh = config.use_asinh;
    ensure!(
        quantiles > 0
            && median_index < quantiles
            && patch_size > 0
            && !anchor.is_empty()
            && anchor.len().is_multiple_of(patch_size * quantiles),
        "Invalid forecast block shape"
    );
    let horizon_patches = anchor.len() / (patch_size * quantiles);
    if horizon_patches == 1 {
        return Ok(anchor.to_vec());
    }
    let stats = scale_context(context, observed, patch_size, minimum_scale, use_asinh)?;
    let loc0 = *stats.loc.last().expect("nonempty context");
    let scale0 = *stats.scale.last().expect("nonempty context");
    let mut state = context_moments(context, observed, patch_size, horizon_patches);
    let offset = state.mean;
    let clip = (f32::MAX as f64).sqrt() / 1e3;
    let mut output = anchor.to_vec();
    for patch in 0..horizon_patches {
        let variance = (state.m2 / (state.count - 1.0).max(1.0)).max(0.0);
        let frame_scale = variance.sqrt().max(minimum_scale);
        let frame_loc = state.mean;
        let loc = if patch == 0 { loc0 } else { frame_loc as f32 };
        let scale = if patch == 0 {
            scale0
        } else {
            frame_scale as f32
        };
        let mut folded = Vec::with_capacity(patch_size);
        for step in patch * patch_size..(patch + 1) * patch_size {
            let median = anchor[step * quantiles + median_index] as f64;
            let sinh_median = (median - loc0 as f64) / scale0 as f64;
            folded
                .push((sinh_median * frame_scale + frame_loc - offset).clamp(-clip, clip) + offset);
            for q in 0..quantiles {
                let i = step * quantiles + q;
                let sinh_norm = (anchor[i] - loc0) / scale0;
                output[i] = sinh_norm * scale + loc;
            }
        }
        state = state.combine(Moments::from_samples(&folded));
    }
    clamp_nonfinite(&mut output, quantiles);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_does_not_read_future_patches_or_missing_outliers() {
        let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let observed = vec![true; 64];
        let before = scale_context(&values, &observed, 32, 1e-6, true).unwrap();
        let mut changed = values.clone();
        changed[32..].fill(1e6);
        let after = scale_context(&changed, &observed, 32, 1e-6, true).unwrap();
        assert_eq!(before.values[..32], after.values[..32]);
        assert_eq!(before.loc[..32], after.loc[..32]);
        assert_eq!(before.scale[..32], after.scale[..32]);
        let mut observed = observed;
        observed[40] = false;
        let ordinary = scale_context(&values, &observed, 32, 1e-6, true).unwrap();
        let mut outlier = values;
        outlier[40] = 1e30;
        let missing_outlier = scale_context(&outlier, &observed, 32, 1e-6, true).unwrap();
        assert_eq!(ordinary.values, missing_outlier.values);
        assert_eq!(ordinary.loc, missing_outlier.loc);
        assert_eq!(ordinary.scale, missing_outlier.scale);
    }

    #[test]
    fn infinities_use_finite_neighbors_and_nans_remain_visible() {
        let mut values = [f32::NEG_INFINITY, -3.0, 1.0, f32::INFINITY, f32::NAN];
        clamp_nonfinite(&mut values, 5);
        assert_eq!(values[..4], [-3.0, -3.0, 1.0, 1.0]);
        assert!(values[4].is_nan());
        let mut all_infinite = [f32::NEG_INFINITY, f32::INFINITY];
        clamp_nonfinite(&mut all_infinite, 2);
        assert_eq!(all_infinite, [0.0, 0.0]);
    }
}
