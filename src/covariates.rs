//! Numeric and categorical covariate inputs, validation, and allocation bounds.
use crate::categorical::validate_categorical_labels;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub const DEFAULT_ATTENTION_ELEMENTS: usize = 8_388_608;
/// Initial validated CPU release; matches the existing VPS forecast limit.
pub const MAX_COVARIATE_HORIZON: usize = 256;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CovariateAttention {
    pub patch_size: usize,
    pub heads: usize,
    pub block_size: usize,
}

impl CovariateAttention {
    /// Conservative maximum of simultaneously materialized time + variable
    /// attention scores over the request, allowing context growth between blocks.
    pub fn elements(
        self,
        rows: usize,
        history: usize,
        horizon: usize,
        max_context: usize,
    ) -> Result<usize> {
        ensure!(
            rows > 0
                && history > 0
                && horizon > 0
                && max_context > 0
                && self.patch_size > 0
                && self.heads > 0
                && self.block_size > 0,
            "Invalid attention layout"
        );
        let context = history.min(max_context).div_ceil(self.patch_size);
        let future = horizon.div_ceil(self.patch_size);
        let block = future.min(self.block_size);
        let context = if future > block {
            context
                .checked_add(future - block)
                .context("Attention size overflow")?
                .min(max_context.div_ceil(self.patch_size))
        } else {
            context
        };
        let tokens = context
            .checked_add(block)
            .and_then(|n| n.checked_mul(2))
            .context("Attention size overflow")?;
        rows.checked_mul(tokens)
            .and_then(|n| n.checked_mul(tokens.checked_add(rows)?))
            .and_then(|n| n.checked_mul(self.heads))
            .context("Attention size overflow")
    }
}

/// Aligned numeric and categorical rows in stable name order. Numeric NaNs
/// retain explicit missing masks. Categorical encoding happens in forecast preparation.
#[derive(Clone, Debug, Default)]
pub struct Covariates {
    pub past: BTreeMap<String, Vec<f32>>,
    pub known: BTreeMap<String, KnownCovariate>,
    pub categorical_past: BTreeMap<String, Vec<Option<String>>>,
    pub categorical_known: BTreeMap<String, KnownCategoricalCovariate>,
}

#[derive(Clone, Debug)]
pub struct KnownCovariate {
    pub history: Vec<f32>,
    pub future: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct KnownCategoricalCovariate {
    pub history: Vec<Option<String>>,
    pub future: Vec<Option<String>>,
}

impl Covariates {
    pub fn past_count(&self) -> usize {
        self.past.len() + self.categorical_past.len()
    }

    pub fn known_count(&self) -> usize {
        self.known.len() + self.categorical_known.len()
    }

    pub fn is_empty(&self) -> bool {
        self.past_count() == 0 && self.known_count() == 0
    }
}

/// Check row/input-slot bounds before allocating or converting request values.
pub fn validate_covariate_limits(
    past_count: usize,
    known_count: usize,
    history_len: usize,
    horizon: usize,
    max_covariates: usize,
    max_input_values: usize,
) -> Result<()> {
    let count = past_count
        .checked_add(known_count)
        .context("Covariate count overflow")?;
    ensure!(
        count <= max_covariates,
        "Covariate count exceeds max_covariates"
    );
    let total = count
        .checked_add(1)
        .and_then(|n| n.checked_mul(history_len))
        .and_then(|n| {
            known_count
                .checked_mul(horizon)
                .and_then(|future| n.checked_add(future))
        })
        .context("Total input value count overflow")?;
    ensure!(
        total <= max_input_values,
        "Target and covariate values exceed max_input_values"
    );
    Ok(())
}

impl Covariates {
    /// Validate library inputs independently of any HTTP transport.
    pub fn validate(
        &self,
        history_len: usize,
        horizon: usize,
        max_covariates: usize,
        max_input_values: usize,
    ) -> Result<()> {
        validate_covariate_limits(
            self.past_count(),
            self.known_count(),
            history_len,
            horizon,
            max_covariates,
            max_input_values,
        )?;
        let mut names = BTreeSet::new();
        for name in self
            .past
            .keys()
            .chain(self.known.keys())
            .chain(self.categorical_past.keys())
            .chain(self.categorical_known.keys())
        {
            ensure!(
                names.insert(name),
                "Covariate names must be unique across all numeric and categorical inputs"
            );
            ensure!(!name.is_empty() && name.len() <= 64 && name.trim() == name
                && !name.chars().any(char::is_control),
                "Covariate names must be 1..=64 UTF-8 bytes, without control characters or surrounding whitespace");
        }
        for values in self.past.values() {
            ensure!(
                values.len() == history_len,
                "Past covariate length must match history length"
            );
            ensure!(
                values.iter().all(|v| !v.is_infinite()),
                "Covariate values must be finite or NaN for missing data"
            );
        }
        for values in self.known.values() {
            ensure!(
                values.history.len() == history_len,
                "Known covariate history length must match history length"
            );
            ensure!(
                values.future.len() == horizon,
                "Known covariate future length must equal horizon"
            );
            ensure!(
                values
                    .history
                    .iter()
                    .chain(&values.future)
                    .all(|v| !v.is_infinite()),
                "Covariate values must be finite or NaN for missing data"
            );
        }
        for values in self.categorical_past.values() {
            ensure!(
                values.len() == history_len,
                "Categorical past covariate length must match history length"
            );
            validate_categorical_labels(values, None)?;
        }
        for values in self.categorical_known.values() {
            ensure!(
                values.history.len() == history_len,
                "Categorical known covariate history length must match history length"
            );
            ensure!(
                values.future.len() == horizon,
                "Categorical known covariate future length must equal horizon"
            );
            validate_categorical_labels(&values.history, Some(&values.future))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_budget_bounds_every_block_and_detects_overflow() {
        let layout = CovariateAttention {
            patch_size: 32,
            heads: 6,
            block_size: 32,
        };
        for history in [1usize, 33, 257, 4096] {
            for horizon in [1usize, 33, 257, 1025, 2049] {
                let estimate = layout.elements(4, history, horizon, 4096).unwrap();
                let mut context = history.div_ceil(32);
                let mut remaining = horizon.div_ceil(32);
                while remaining > 0 {
                    let block = remaining.min(32);
                    let tokens = 2 * (context + block);
                    assert!(4 * 6 * tokens * (tokens + 4) <= estimate);
                    remaining -= block;
                    context = (context + block).min(128);
                }
            }
        }
        assert!(layout.elements(usize::MAX, 1, 1, 4096).is_err());
    }
}
