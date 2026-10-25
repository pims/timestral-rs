//! Timestral-5M and Timestral-28M inference with Candle: original-unit univariate forecasting,
//! checkpoint loading, time/variable transformer layers and a monotone quantile head.

#[cfg(feature = "server")]
pub mod api;
pub mod config;
pub mod forecaster;
pub mod head;
pub mod layers;
pub mod loading;
pub mod masks;
pub mod model;
pub mod scaler;
pub mod transformer;
#[cfg(feature = "server")]
pub mod worker;
