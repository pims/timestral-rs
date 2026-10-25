//! One model-owning thread and a bounded inference queue. Candle values stay on
//! the worker thread; only ordinary Rust request/response data crosses channels.
use crate::{forecaster::TimestralForecaster, loading::load_model};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub model: String,
    pub server_version: String,
    pub device: String,
    pub dtype: String,
    pub parameter_count: usize,
    pub context_length: usize,
    pub quantile_levels: Vec<f64>,
    pub weights_sha256: String,
    pub config_sha256: String,
    pub fused_rmsnorm: bool,
    pub singleton_variate_attention: bool,
}

#[derive(Debug)]
pub struct ForecastInput {
    pub history: Vec<f32>,
    pub horizon: usize,
    pub quantiles: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct Timing {
    pub queue: f64,
    pub inference: f64,
}

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub request_id: String,
    pub model: String,
    pub horizon: usize,
    pub quantiles: BTreeMap<String, Vec<f32>>,
    pub timing_ms: Timing,
}

#[derive(Debug)]
pub enum WorkerFailure {
    Expired,
    Inference,
}
#[derive(Debug)]
pub enum SubmitError {
    Busy,
    Unavailable,
}

struct Job {
    input: ForecastInput,
    request_id: String,
    deadline: Instant,
    queued: Instant,
    reply: oneshot::Sender<std::result::Result<ForecastResponse, WorkerFailure>>,
}
enum Message {
    Forecast(Job),
    Stop,
}

#[derive(Clone)]
pub struct WorkerClient {
    tx: mpsc::Sender<Message>,
    ready: Arc<AtomicBool>,
    pub info: Arc<ModelInfo>,
    pub queue_capacity: usize,
}

impl WorkerClient {
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire) && !self.tx.is_closed()
    }
    pub fn begin_shutdown(&self) {
        self.ready.store(false, Ordering::Release);
    }
    pub fn queued_jobs(&self) -> usize {
        self.queue_capacity - self.tx.capacity()
    }

    pub fn submit(
        &self,
        input: ForecastInput,
        request_id: String,
        deadline: Instant,
    ) -> std::result::Result<
        oneshot::Receiver<std::result::Result<ForecastResponse, WorkerFailure>>,
        SubmitError,
    > {
        if !self.is_ready() {
            return Err(SubmitError::Unavailable);
        }
        let (reply, receiver) = oneshot::channel();
        let job = Job {
            input,
            request_id,
            deadline,
            queued: Instant::now(),
            reply,
        };
        self.tx
            .try_send(Message::Forecast(job))
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => SubmitError::Busy,
                mpsc::error::TrySendError::Closed(_) => SubmitError::Unavailable,
            })?;
        Ok(receiver)
    }
}

pub struct WorkerHandle {
    client: WorkerClient,
    join: thread::JoinHandle<()>,
}
impl WorkerHandle {
    /// Stop new admissions, finish queued jobs whose receivers/deadlines are still
    /// valid, and wait for the current forecast. Started inference is not aborted.
    pub async fn shutdown(self) -> Result<()> {
        self.client.begin_shutdown();
        let _ = self.client.tx.send(Message::Stop).await;
        tokio::task::spawn_blocking(move || self.join.join())
            .await?
            .map_err(|_| anyhow::anyhow!("Inference worker panicked"))?;
        Ok(())
    }
}

/// No Send requirement on the engine: it is created and used on its owning thread.
pub trait InferenceEngine {
    fn info(&self) -> ModelInfo;
    fn forecast(&mut self, history: &[f32], horizon: usize) -> Result<Vec<f32>>;
}

struct ReadyGuard(Arc<AtomicBool>);
impl Drop for ReadyGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Synchronous startup (model load + warmup). Call from outside Tokio, or from
/// spawn_blocking. A failed initialization never exposes a ready worker.
pub fn spawn_engine<F>(queue_capacity: usize, factory: F) -> Result<(WorkerClient, WorkerHandle)>
where
    F: FnOnce() -> Result<Box<dyn InferenceEngine>> + Send + 'static,
{
    ensure!(
        (1..=4096).contains(&queue_capacity),
        "queue_capacity must be in 1..=4096"
    );
    let (tx, mut rx) = mpsc::channel(queue_capacity);
    let ready = Arc::new(AtomicBool::new(false));
    let worker_ready = ready.clone();
    let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel(1);
    let join = thread::Builder::new().name("timestral-inference".into()).spawn(move || {
        let _guard = ReadyGuard(worker_ready.clone());
        let mut engine = match factory() {
            Ok(engine) => engine,
            Err(error) => { let _ = startup_tx.send(Err(format!("{error:#}"))); return; }
        };
        let info = engine.info();
        worker_ready.store(true, Ordering::Release);
        if startup_tx.send(Ok(info.clone())).is_err() { return; }
        while let Some(message) = rx.blocking_recv() {
            let Message::Forecast(job) = message else { break; };
            if job.reply.is_closed() { continue; }
            let started = Instant::now();
            if started >= job.deadline {
                let _ = job.reply.send(Err(WorkerFailure::Expired));
                continue;
            }
            let queue_ms = started.duration_since(job.queued).as_secs_f64() * 1000.0;
            let result = engine.forecast(&job.input.history, job.input.horizon).and_then(|values| {
                ensure!(job.input.horizon.checked_mul(info.quantile_levels.len()) == Some(values.len()), "Unexpected forecast shape");
                ensure!(values.iter().all(|v| v.is_finite()), "Non-finite forecast");
                let mut quantiles = BTreeMap::new();
                for level in &job.input.quantiles {
                    let index = info.quantile_levels.iter().position(|q| q == level).context("Unsupported quantile")?;
                    quantiles.insert(level.to_string(), values.chunks_exact(info.quantile_levels.len()).map(|step| step[index]).collect());
                }
                Ok(ForecastResponse { request_id: job.request_id.clone(), model: info.model.clone(),
                    horizon: job.input.horizon, quantiles,
                    timing_ms: Timing { queue: queue_ms, inference: started.elapsed().as_secs_f64() * 1000.0 } })
            });
            let result = result.map_err(|error| {
                eprintln!("{}", serde_json::json!({"event":"inference_error", "request_id":job.request_id, "error":format!("{error:#}")}));
                WorkerFailure::Inference
            });
            let _ = job.reply.send(result);
        }
    })?;
    let info = match startup_rx.recv() {
        Ok(Ok(info)) => info,
        Ok(Err(error)) => {
            let _ = join.join();
            anyhow::bail!("Worker initialization failed: {error}");
        }
        Err(_) => {
            let _ = join.join();
            anyhow::bail!("Worker exited during initialization (check device availability)");
        }
    };
    let client = WorkerClient {
        tx,
        ready,
        info: Arc::new(info),
        queue_capacity,
    };
    Ok((client.clone(), WorkerHandle { client, join }))
}

fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

struct CandleEngine {
    forecaster: TimestralForecaster,
    info: ModelInfo,
}
impl InferenceEngine for CandleEngine {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }
    fn forecast(&mut self, history: &[f32], horizon: usize) -> Result<Vec<f32>> {
        Ok(self
            .forecaster
            .forecast(history, horizon)?
            .quantiles
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }
}

pub fn spawn_model(
    checkpoint: PathBuf,
    device_name: String,
    queue_capacity: usize,
) -> Result<(WorkerClient, WorkerHandle)> {
    spawn_engine(queue_capacity, move || {
        let device = match device_name.as_str() {
            "cpu" => candle_core::Device::Cpu,
            "metal" => {
                #[cfg(feature = "metal")]
                {
                    candle_core::Device::new_metal(0)?
                }
                #[cfg(not(feature = "metal"))]
                {
                    anyhow::bail!("Build with --features server,metal to select Metal");
                }
            }
            _ => anyhow::bail!("Device must be cpu or metal"),
        };
        let load_started = Instant::now();
        let model = load_model(&checkpoint, &device)?;
        device.synchronize()?;
        eprintln!(
            "Loaded {} on {device_name}: {} parameters ({:.2} ms)",
            model.config.variant()?.name(),
            model.parameter_count,
            load_started.elapsed().as_secs_f64() * 1000.0
        );
        let info = ModelInfo {
            model: model.config.variant()?.repo_id().into(),
            server_version: env!("CARGO_PKG_VERSION").into(),
            device: device_name.clone(),
            dtype: "float32".into(),
            parameter_count: model.parameter_count,
            context_length: model.config.context_length,
            quantile_levels: model.config.quantile_knots.clone(),
            weights_sha256: file_sha256(&checkpoint.join("model.safetensors"))?,
            config_sha256: file_sha256(&checkpoint.join("config.json"))?,
            fused_rmsnorm: device_name == "metal" && !cfg!(feature = "unfused-rmsnorm"),
            singleton_variate_attention: !cfg!(feature = "full-variate-attention"),
        };
        let warmup_started = Instant::now();
        let forecaster = TimestralForecaster::new(model);
        let history: Vec<f32> = (0..129).map(|i| 10.0 + i as f32 * 0.01).collect();
        let forecast = forecaster.forecast(&history, 96)?;
        let _ = forecast
            .quantiles
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        eprintln!(
            "Warmup complete ({:.2} ms)",
            warmup_started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Box::new(CandleEngine { forecaster, info }))
    })
}
