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

    fn labels(values: &[&str]) -> Vec<Option<String>> {
        values.iter().map(|value| Some((*value).into())).collect()
    }

    #[test]
    fn categories_use_smoothed_target_means() {
        let history = [10.0, 20.0, 60.0];
        let categories = labels(&["a", "a", "b"]);

        let encoded = encode_categorical(&history, None, &categories, None, 3).unwrap();

        // Target mean is 30. Category a: (30 + 10 + 20) / 3 = 20.
        // Category b: (30 + 60) / 2 = 45.
        assert_eq!(encoded.history, [20.0, 20.0, 45.0]);
        assert_eq!(encoded.future, None);
    }

    #[test]
    fn unseen_category_uses_target_mean() {
        let history = [10.0, 30.0];
        let past = labels(&["a", "b"]);
        let future = labels(&["unseen"]);

        let encoded = encode_categorical(&history, None, &past, Some(&future), 2).unwrap();

        assert_eq!(encoded.future.unwrap(), [20.0]);
    }

    #[test]
    fn missing_labels_form_their_own_category() {
        let history = [10.0, 30.0];
        let past = vec![None, Some("observed".into())];
        let future = vec![None];

        let encoded = encode_categorical(&history, None, &past, Some(&future), 2).unwrap();

        assert_eq!(encoded.history, [15.0, 25.0]);
        assert_eq!(encoded.future.unwrap(), [15.0]);
    }

    #[test]
    fn values_outside_context_do_not_affect_category_means() {
        let history = [1_000.0, 10.0, 30.0];
        let past = labels(&["old", "a", "b"]);
        let future = labels(&["old"]);

        let encoded = encode_categorical(&history, None, &past, Some(&future), 2).unwrap();

        // "old" occurs outside the fitting window, so it uses the fallback.
        assert_eq!(encoded.future.unwrap(), [20.0]);
    }

    #[test]
    fn target_is_forward_filled_before_context_is_cropped() {
        let history = [10.0, f32::NAN, 30.0];
        let past = labels(&["a", "a", "b"]);
        let future = labels(&["unseen"]);

        let encoded = encode_categorical(&history, None, &past, Some(&future), 2).unwrap();

        // The fitting window contains the filled values [10, 30].
        assert_eq!(encoded.future.unwrap(), [20.0]);
    }

    #[test]
    fn leading_missing_targets_do_not_affect_category_means() {
        let history = [f32::NAN, 10.0, 30.0];
        let past = labels(&["before_start", "a", "b"]);
        let future = labels(&["unseen"]);

        let encoded = encode_categorical(&history, None, &past, Some(&future), 3).unwrap();

        assert_eq!(encoded.future.unwrap(), [20.0]);
    }

    #[test]
    fn masked_target_values_do_not_affect_category_means() {
        let history = [10.0, 1_000.0, 30.0];
        let observed = [true, false, true];
        let past = labels(&["a", "a", "b"]);
        let future = labels(&["unseen"]);

        let encoded =
            encode_categorical(&history, Some(&observed), &past, Some(&future), 3).unwrap();

        // The masked value is forward-filled, giving [10, 10, 30].
        let expected_mean = 50.0 / 3.0;
        assert_eq!(encoded.future.unwrap(), [expected_mean]);
    }

    #[test]
    fn renaming_categories_preserves_encodings() {
        let history = [1.0, 3.0, -5.0, 7.0];
        let past = vec![Some("a".into()), Some("a".into()), None, Some("b".into())];
        let future = vec![Some("b".into()), None, Some("unseen".into())];
        let renamed_past = vec![Some("z".into()), Some("z".into()), None, Some("x".into())];
        let renamed_future = vec![Some("x".into()), None, Some("new".into())];

        let original = encode_categorical(&history, None, &past, Some(&future), 4).unwrap();
        let renamed =
            encode_categorical(&history, None, &renamed_past, Some(&renamed_future), 4).unwrap();

        assert_eq!(original.history, renamed.history);
        assert_eq!(original.future, renamed.future);
    }

    #[test]
    fn negating_targets_negates_encodings() {
        let history = [1.0, 3.0, -5.0, 7.0];
        let past = vec![Some("a".into()), Some("a".into()), None, Some("b".into())];
        let future = vec![Some("b".into()), None, Some("unseen".into())];
        let negated_history = history.map(|value| -value);

        let original = encode_categorical(&history, None, &past, Some(&future), 4).unwrap();
        let negated = encode_categorical(&negated_history, None, &past, Some(&future), 4).unwrap();

        let expected_history: Vec<f32> = original.history.iter().map(|value| -value).collect();
        let expected_future: Vec<f32> = original
            .future
            .unwrap()
            .iter()
            .map(|value| -value)
            .collect();
        assert_eq!(negated.history, expected_history);
        assert_eq!(negated.future.unwrap(), expected_future);
    }

    #[test]
    fn future_labels_do_not_change_fitted_statistics() {
        let history = [1.0, 2.0, 3.0];
        let past = labels(&["a", "b", "a"]);
        let short_future = labels(&["a", "unseen"]);
        let mut long_future = short_future.clone();
        long_future.extend([Some("future_only".into()), None, Some("b".into())]);

        let short = encode_categorical(&history, None, &past, Some(&short_future), 3).unwrap();
        let long = encode_categorical(&history, None, &past, Some(&long_future), 3).unwrap();

        assert_eq!(short.history, long.history);
        assert_eq!(short.future.unwrap(), long.future.unwrap()[..2]);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let past = labels(&["a"]);
        let cases = [
            (
                "empty target history",
                encode_categorical(&[], None, &[], None, 1),
                "nonempty target history",
            ),
            (
                "zero context length",
                encode_categorical(&[1.0], None, &past, None, 0),
                "max_context must be positive",
            ),
            (
                "misaligned categorical history",
                encode_categorical(&[1.0, 2.0], None, &past, None, 2),
                "Categorical history length",
            ),
            (
                "misaligned target mask",
                encode_categorical(&[1.0], Some(&[]), &past, None, 1),
                "Observation mask length",
            ),
        ];

        for (name, result, expected_error) in cases {
            let error = result.expect_err(name);

            assert!(
                error.to_string().contains(expected_error),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn validates_utf8_label_boundaries() {
        let at_limit = "é".repeat(128); // 256 bytes.
        let oversized = "é".repeat(129); // 258 bytes, despite only 129 characters.
        let cases = [
            ("historical label at limit", at_limit.as_str(), None, true),
            (
                "oversized historical label",
                oversized.as_str(),
                None,
                false,
            ),
            ("future label at limit", "a", Some(at_limit.as_str()), true),
            (
                "oversized future label",
                "a",
                Some(oversized.as_str()),
                false,
            ),
        ];

        for (name, historical_label, future_label, valid) in cases {
            let past = labels(&[historical_label]);
            let future = future_label.map(|label| labels(&[label]));

            let result = encode_categorical(&[1.0], None, &past, future.as_deref(), 1);

            if valid {
                let encoded = result.expect(name);
                assert_eq!(encoded.history, [1.0], "{name}");
                assert_eq!(encoded.future, future_label.map(|_| vec![1.0]), "{name}");
            } else {
                let error = result.expect_err(name);
                assert!(
                    error.to_string().contains("Category labels"),
                    "{name}: {error}"
                );
            }
        }
    }

    #[test]
    fn validates_distinct_category_boundaries() {
        let limit = MAX_CATEGORIES_PER_COVARIATE;
        let cases = [
            ("at historical limit", limit, vec![], true),
            ("too many historical categories", limit + 1, vec![], false),
            (
                "future label exceeds limit",
                limit,
                vec![Some("extra".into())],
                false,
            ),
            ("missing label does not count", limit, vec![None], true),
        ];

        for (name, historical_count, future, valid) in cases {
            let past: Vec<_> = (0..historical_count)
                .map(|index| Some(index.to_string()))
                .collect();
            let history = vec![1.0; historical_count];

            let result = encode_categorical(&history, None, &past, Some(&future), history.len());

            if valid {
                let encoded = result.expect(name);
                assert_eq!(encoded.history, history, "{name}");
                assert_eq!(encoded.future.unwrap(), vec![1.0; future.len()], "{name}");
            } else {
                let error = result.expect_err(name);
                assert!(
                    error.to_string().contains("Distinct category labels"),
                    "{name}: {error}"
                );
            }
        }
    }

    #[derive(Deserialize)]
    struct PythonFixtures {
        reference_commit: String,
        source_sha256: BTreeMap<String, String>,
        cases: Vec<EncodingCase>,
        reduction_pattern: Vec<Option<f32>>,
        mean_reductions: Vec<SumCase>,
    }

    #[derive(Deserialize)]
    struct SumCase {
        length: usize,
        expected_sum_bits: String,
    }

    #[derive(Deserialize)]
    struct EncodingCase {
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

    fn python_fixtures() -> PythonFixtures {
        let fixtures: PythonFixtures =
            serde_json::from_str(include_str!("../tests/fixtures/categorical-encoding.json"))
                .unwrap();
        assert_eq!(
            fixtures.reference_commit,
            "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
        );
        assert_eq!(fixtures.source_sha256.len(), 2);
        assert!(fixtures.source_sha256.values().all(|hash| hash.len() == 64));
        assert!(!fixtures.cases.is_empty(), "Missing Python encoding cases");
        assert!(
            !fixtures.reduction_pattern.is_empty(),
            "Missing reduction pattern"
        );
        assert!(
            !fixtures.mean_reductions.is_empty(),
            "Missing NumPy sum cases"
        );
        fixtures
    }

    fn assert_matches_fixture(case: &str, stage: &str, actual: &[f32], expected: &[Option<f32>]) {
        assert_eq!(actual.len(), expected.len(), "{case}/{stage}: length");
        for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            match expected {
                None => assert!(
                    actual.is_nan(),
                    "{case}/{stage}[{index}]: expected NaN, got {actual}"
                ),
                Some(expected) => {
                    // Preserve the reference fixture tolerance: 1e-6 in target units,
                    // scaled by magnitude above one. Missing masks must match exactly.
                    let tolerance = 1e-6 * expected.abs().max(1.0);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{case}/{stage}[{index}]: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn target_sums_match_numpy_fixtures() {
        let fixtures = python_fixtures();

        for case in fixtures.mean_reductions {
            let target: Vec<f32> = fixtures
                .reduction_pattern
                .iter()
                .cycle()
                .take(case.length)
                .map(|value| value.unwrap_or(f32::NAN))
                .collect();
            let expected_sum =
                f64::from_bits(u64::from_str_radix(&case.expected_sum_bits, 16).unwrap());

            let actual_sum = numpy_target_sum(&target);

            assert_eq!(
                actual_sum, expected_sum,
                "NumPy float64 sum for length {}",
                case.length
            );
        }
    }

    #[test]
    fn encodings_match_python_fixtures() {
        let fixtures = python_fixtures();

        for case in fixtures.cases {
            let history: Vec<f32> = case
                .history
                .iter()
                .map(|value| value.unwrap_or(f32::NAN))
                .collect();

            let encoded = encode_categorical(
                &history,
                case.observation_mask.as_deref(),
                &case.past,
                case.future.as_deref(),
                case.max_context,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));

            let fitting_target =
                encoding_target(&history, case.observation_mask.as_deref(), case.max_context);
            assert_matches_fixture(
                &case.name,
                "fitting_target",
                &fitting_target,
                &case.encoding_target,
            );
            assert_matches_fixture(&case.name, "history", &encoded.history, &case.encoded_past);
            match (&encoded.future, &case.encoded_future) {
                (Some(actual), Some(expected)) => {
                    assert_matches_fixture(&case.name, "future", actual, expected)
                }
                (None, None) => {}
                _ => panic!(
                    "{}: future output presence differs from reference",
                    case.name
                ),
            }
        }
    }
}
