//! Request-local categorical target encoding, matching the official Timestral
//! preprocessing. Fit once at the original forecast cutoff, then pass these
//! numeric rows through the existing covariate decoder (including mirrored passes).
use anyhow::{ensure, Result};
use std::collections::BTreeMap;

/// Labels are exact UTF-8 identities, including empty strings and whitespace.
/// Null is a separate category. These bounds also apply to library callers.
pub const MAX_CATEGORY_LABEL_BYTES: usize = 256;
/// Distinct non-null labels per row, across history and supplied future labels.
pub const MAX_CATEGORIES_PER_COVARIATE: usize = 1024;

#[derive(Clone, Debug)]
pub struct EncodedCategorical {
    /// Full aligned history, before the numeric forecaster crops or pads it.
    /// Missing labels outside the target fitting window remain NaN.
    pub history: Vec<f32>,
    pub future: Option<Vec<f32>>,
}

/// Encode string/null labels against historical target values only.
///
/// The fitting target is forward-filled from its first observed value, then
/// restricted to its last `max_context` positions. Explicitly masked and
/// non-finite target values are missing. Each category receives
/// `(target_mean + category_sum) / (1 + category_count)`, accumulated in f64
/// and returned in f32. Null labels form their own bucket; unseen labels fall
/// back to the target mean. If no target is observed, the reference mean is 0.
///
/// Encoding all supplied future labels here freezes the mapping across decode
/// blocks: neither future targets nor model predictions participate in fitting.
/// No label normalization, trimming, or interpretation is performed.
pub fn encode_categorical(
    history: &[f32],
    observation_mask: Option<&[bool]>,
    past: &[Option<String>],
    future: Option<&[Option<String>]>,
    max_context: usize,
) -> Result<EncodedCategorical> {
    ensure!(
        !history.is_empty(),
        "Categorical encoding requires nonempty target history"
    );
    ensure!(
        max_context > 0,
        "Categorical encoding max_context must be positive"
    );
    ensure!(
        past.len() == history.len(),
        "Categorical history length must match target history"
    );
    ensure!(
        observation_mask.is_none_or(|m| m.len() == history.len()),
        "Observation mask length must match history"
    );

    let mut statistics = category_statistics(past, future)?;

    let target = encoding_target(history, observation_mask, max_context);
    let mut count = 0usize;
    for (&value, label) in target.iter().zip(past) {
        if value.is_finite() {
            let stats = statistics
                .get_mut(&label.as_deref())
                .expect("validated label");
            stats.0 += f64::from(value);
            stats.1 += 1;
            count += 1;
        }
    }
    // NumPy sums the full target array after replacing excluded positions with
    // zero. Keep those positions: filtering them changes pairwise grouping.
    // Category sums above deliberately remain sequential, matching bincount.
    let mean = if count > 0 {
        numpy_target_sum(&target) / count as f64
    } else {
        0.0
    };
    let lookup = |label: &Option<String>| {
        let (sum, count) = statistics[&label.as_deref()];
        ((mean + sum) / (1.0 + count as f64)) as f32
    };
    let encoded_history = past
        .iter()
        .zip(&target)
        .map(|(label, value)| {
            if label.is_none() && !value.is_finite() {
                f32::NAN
            } else {
                lookup(label)
            }
        })
        .collect();
    Ok(EncodedCategorical {
        history: encoded_history,
        future: future.map(|labels| labels.iter().map(lookup).collect()),
    })
}

/// Float64 contiguous pairwise reduction used by the pinned NumPy reference.
/// Eight accumulators per block of at most 128 values; larger blocks split on
/// multiples of eight. Non-finite fitting targets contribute zero, as in
/// `np.where(observed, target, 0.0).sum()`. Category bincounts use a different order.
/// Source: NumPy 2.5.4, numpy/_core/src/umath/loops_utils.h.src (DOUBLE_pairwise_sum).
fn numpy_target_sum(target: &[f32]) -> f64 {
    fn pairwise(target: &[f32]) -> f64 {
        let value = |i: usize| {
            if target[i].is_finite() {
                f64::from(target[i])
            } else {
                0.0
            }
        };
        if target.len() < 8 {
            return (0..target.len()).fold(-0.0, |sum, i| sum + value(i));
        }
        if target.len() > 128 {
            let midpoint = (target.len() / 2) / 8 * 8;
            return pairwise(&target[..midpoint]) + pairwise(&target[midpoint..]);
        }
        let mut lanes: [f64; 8] = std::array::from_fn(value);
        let aligned = target.len() / 8 * 8;
        for i in (8..aligned).step_by(8) {
            for lane in 0..8 {
                lanes[lane] += value(i + lane);
            }
        }
        let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        (aligned..target.len()).fold(sum, |sum, i| sum + value(i))
    }
    0.0 + pairwise(target)
}

/// Check label lengths and distinct-category bounds without fitting target statistics.
/// Used during admission as well as by the encoder for direct library calls.
pub fn validate_categorical_labels(
    past: &[Option<String>],
    future: Option<&[Option<String>]>,
) -> Result<()> {
    category_statistics(past, future).map(|_| ())
}

fn category_statistics<'a>(
    past: &'a [Option<String>],
    future: Option<&'a [Option<String>]>,
) -> Result<BTreeMap<Option<&'a str>, (f64, usize)>> {
    // Validate before building encoded vectors or computing statistics. Borrow
    // labels so fitting does not duplicate the request's strings.
    let mut statistics: BTreeMap<Option<&str>, (f64, usize)> = BTreeMap::new();
    for label in past.iter().chain(future.into_iter().flatten()) {
        if let Some(label) = label {
            ensure!(
                label.len() <= MAX_CATEGORY_LABEL_BYTES,
                "Category labels must be at most 256 UTF-8 bytes"
            );
        }
        statistics.entry(label.as_deref()).or_default();
        let count = statistics.len() - usize::from(statistics.contains_key(&None));
        ensure!(
            count <= MAX_CATEGORIES_PER_COVARIATE,
            "Distinct category labels exceed 1024 per covariate"
        );
    }

    Ok(statistics)
}

fn encoding_target(
    history: &[f32],
    observation_mask: Option<&[bool]>,
    max_context: usize,
) -> Vec<f32> {
    let start = history.len().saturating_sub(max_context);
    let mut last = f32::NAN;
    history
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            if value.is_finite() && observation_mask.is_none_or(|m| m[i]) {
                last = value;
            }
            if i >= start {
                last
            } else {
                f32::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixtures {
        reference_commit: String,
        source_sha256: BTreeMap<String, String>,
        cases: Vec<Case>,
        reduction_pattern: Vec<Option<f32>>,
        mean_reductions: Vec<ReductionCase>,
    }

    #[derive(Deserialize)]
    struct ReductionCase {
        length: usize,
        expected_sum_bits: String,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        history: Vec<Option<f32>>,
        observation_mask: Option<Vec<bool>>,
        past: Vec<Option<String>>,
        future: Option<Vec<Option<String>>>,
        max_context: usize,
        encoding_target: Vec<Option<f32>>,
        encoded_past: Vec<Option<f32>>,
        encoded_future: Option<Vec<Option<f32>>>,
    }

    fn compare(name: &str, actual: &[f32], expected: &[Option<f32>]) {
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (i, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            match expected {
                None => assert!(actual.is_nan(), "{name}[{i}]: expected NaN, got {actual}"),
                Some(expected) => assert!(
                    (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                    "{name}[{i}]: {actual} != {expected}"
                ),
            }
        }
    }

    #[test]
    fn matches_pinned_python_preprocessing() {
        let fixtures: Fixtures =
            serde_json::from_str(include_str!("../tests/fixtures/categorical-encoding.json"))
                .unwrap();
        assert_eq!(
            fixtures.reference_commit,
            "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
        );
        assert_eq!(fixtures.source_sha256.len(), 2);
        assert!(fixtures.source_sha256.values().all(|hash| hash.len() == 64));
        assert_eq!(fixtures.cases.len(), 23);
        assert_eq!(fixtures.mean_reductions.len(), 28);
        for case in fixtures.mean_reductions {
            let target: Vec<f32> = (0..case.length)
                .map(|i| {
                    fixtures.reduction_pattern[i % fixtures.reduction_pattern.len()]
                        .unwrap_or(f32::NAN)
                })
                .collect();
            assert_eq!(
                numpy_target_sum(&target),
                f64::from_bits(u64::from_str_radix(&case.expected_sum_bits, 16).unwrap()),
                "NumPy float64 sum for length {}",
                case.length
            );
        }
        for case in fixtures.cases {
            let history: Vec<f32> = case.history.iter().map(|v| v.unwrap_or(f32::NAN)).collect();
            let mask = case.observation_mask.as_deref();
            compare(
                &case.name,
                &encoding_target(&history, mask, case.max_context),
                &case.encoding_target,
            );
            let actual = encode_categorical(
                &history,
                mask,
                &case.past,
                case.future.as_deref(),
                case.max_context,
            )
            .unwrap();
            compare(&case.name, &actual.history, &case.encoded_past);
            assert_eq!(actual.future.is_some(), case.encoded_future.is_some());
            if let (Some(actual), Some(expected)) = (actual.future, case.encoded_future) {
                compare(&case.name, &actual, &expected);
            }
        }
    }

    fn labels(values: &[Option<&str>]) -> Vec<Option<String>> {
        values.iter().map(|v| v.map(str::to_owned)).collect()
    }

    #[test]
    fn renaming_reflection_and_request_isolation() {
        let past = labels(&[Some("a"), Some("a"), None, Some("b")]);
        let future = labels(&[Some("b"), None, Some("new")]);
        let target = [1.0, 3.0, -5.0, 7.0];
        let original = encode_categorical(&target, None, &past, Some(&future), 4).unwrap();
        let renamed_past = labels(&[Some("z"), Some("z"), None, Some("x")]);
        let renamed_future = labels(&[Some("x"), None, Some("unseen")]);
        let renamed =
            encode_categorical(&target, None, &renamed_past, Some(&renamed_future), 4).unwrap();
        assert_eq!(original.history, renamed.history);
        assert_eq!(original.future, renamed.future);
        let mirrored =
            encode_categorical(&target.map(|v| -v), None, &past, Some(&future), 4).unwrap();
        for (a, b) in original.history.iter().zip(mirrored.history) {
            assert_eq!(*a, -b);
        }
        for (a, b) in original
            .future
            .as_ref()
            .unwrap()
            .iter()
            .zip(mirrored.future.unwrap())
        {
            assert_eq!(*a, -b);
        }
        // A different request must not mutate a mapping used by this request.
        encode_categorical(&[100.0; 4], None, &past, Some(&future), 4).unwrap();
        let repeated = encode_categorical(&target, None, &past, Some(&future), 4).unwrap();
        assert_eq!(original.history, repeated.history);
        assert_eq!(original.future, repeated.future);
    }

    #[test]
    fn future_labels_cannot_change_fitted_statistics() {
        let past = labels(&[Some("a"), Some("b"), Some("a")]);
        let short = labels(&[Some("a"), Some("new")]);
        let mut long = short.clone();
        long.extend(labels(&[Some("future_only"), None, Some("b")]));
        let first = encode_categorical(&[1.0, 2.0, 3.0], None, &past, Some(&short), 3).unwrap();
        let second = encode_categorical(&[1.0, 2.0, 3.0], None, &past, Some(&long), 3).unwrap();
        assert_eq!(first.history, second.history);
        assert_eq!(first.future.unwrap(), second.future.unwrap()[..2]);
    }

    #[test]
    fn rejects_misalignment_and_resource_violations() {
        let past = labels(&[Some("a")]);
        assert!(encode_categorical(&[], None, &[], None, 1).is_err());
        assert!(encode_categorical(&[1.0], None, &past, None, 0).is_err());
        assert!(encode_categorical(&[1.0, 2.0], None, &past, None, 2).is_err());
        assert!(encode_categorical(&[1.0], Some(&[]), &past, None, 1).is_err());
        let oversized = vec![Some("é".repeat(129))];
        assert!(encode_categorical(&[1.0], None, &oversized, None, 1).is_err());
        assert!(encode_categorical(&[1.0], None, &past, Some(&oversized), 1).is_err());
        let boundary = vec![Some("é".repeat(128))];
        assert!(encode_categorical(&[1.0], None, &boundary, None, 1).is_ok());
        let many: Vec<_> = (0..MAX_CATEGORIES_PER_COVARIATE)
            .map(|i| Some(i.to_string()))
            .collect();
        let target = vec![1.0; many.len()];
        assert!(encode_categorical(&target, None, &many, None, many.len()).is_ok());
        let extra = labels(&[Some("extra")]);
        assert!(encode_categorical(&target, None, &many, Some(&extra), many.len()).is_err());
        // The missing bucket is separate from the non-null category bound.
        assert!(encode_categorical(&target, None, &many, Some(&[None]), many.len()).is_ok());
    }
}
