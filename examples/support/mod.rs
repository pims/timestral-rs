use anyhow::{ensure, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use timestral_rs::covariates::{Covariates, KnownCategoricalCovariate, KnownCovariate};

// Diagnostic JSON transport for constructing public library inputs.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovariatesInput {
    #[serde(default)]
    past: BTreeMap<String, Vec<Option<f32>>>,
    #[serde(default)]
    known: BTreeMap<String, KnownInput<f32>>,
    #[serde(default)]
    categorical_past: BTreeMap<String, Vec<Option<String>>>,
    #[serde(default)]
    categorical_known: BTreeMap<String, KnownInput<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KnownInput<T> {
    history: Vec<Option<T>>,
    future: Vec<Option<T>>,
}

pub fn numeric(values: Vec<Option<f32>>) -> Result<Vec<f32>> {
    values
        .into_iter()
        .map(|value| match value {
            None => Ok(f32::NAN),
            Some(value) => {
                ensure!(
                    value.is_finite(),
                    "Numeric values must fit finite float32; use null for missing data"
                );
                Ok(value)
            }
        })
        .collect()
}

impl CovariatesInput {
    pub fn into_covariates(self) -> Result<Covariates> {
        Ok(Covariates {
            past: self
                .past
                .into_iter()
                .map(|(name, values)| Ok((name, numeric(values)?)))
                .collect::<Result<_>>()?,
            known: self
                .known
                .into_iter()
                .map(|(name, row)| {
                    Ok((
                        name,
                        KnownCovariate {
                            history: numeric(row.history)?,
                            future: numeric(row.future)?,
                        },
                    ))
                })
                .collect::<Result<_>>()?,
            categorical_past: self.categorical_past,
            categorical_known: self
                .categorical_known
                .into_iter()
                .map(|(name, row)| {
                    (
                        name,
                        KnownCategoricalCovariate {
                            history: row.history,
                            future: row.future,
                        },
                    )
                })
                .collect(),
        })
    }
}
