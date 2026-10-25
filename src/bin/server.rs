//! Usage: server <checkpoint> [--bind 127.0.0.1:8080] [--device cpu|metal]
use anyhow::{ensure, Context, Result};
use std::{net::SocketAddr, path::PathBuf, time::Instant};
use timestral_rs::{
    api::{self, AppState, Limits},
    worker,
};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let checkpoint = PathBuf::from(args.next().context("Usage: server <checkpoint> [--bind 127.0.0.1:8080] [--device cpu|metal] [--queue-capacity 32] [--timeout-ms 30000] [--max-history 32768] [--max-horizon 1024] [--max-body-bytes 1048576]")?);
    let mut bind: SocketAddr = "127.0.0.1:8080".parse()?;
    let mut device = String::from("cpu");
    let mut capacity = 32usize;
    let mut limits = Limits::default();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--bind" => bind = value.parse()?,
            "--device" => device = value,
            "--queue-capacity" => capacity = value.parse()?,
            "--timeout-ms" => limits.request_timeout_ms = value.parse()?,
            "--max-history" => limits.max_history = value.parse()?,
            "--max-horizon" => limits.max_horizon = value.parse()?,
            "--max-body-bytes" => limits.max_body_bytes = value.parse()?,
            _ => anyhow::bail!("Unknown option: {flag}"),
        }
    }
    ensure!(
        matches!(device.as_str(), "cpu" | "metal"),
        "Device must be cpu or metal"
    );
    eprintln!(
        "Loading and warming {} on {device}...",
        checkpoint.display()
    );
    let startup_started = Instant::now();
    let (client, handle) =
        tokio::task::spawn_blocking(move || worker::spawn_model(checkpoint, device, capacity))
            .await??;
    let result = async {
        let state = AppState::new(client.clone(), limits)?;
        let listener = tokio::net::TcpListener::bind(bind).await?;
        eprintln!(
            "Timestral ready at http://{} (startup: {:.2} ms)",
            listener.local_addr()?,
            startup_started.elapsed().as_secs_f64() * 1000.0
        );
        let shutdown_client = client.clone();
        axum::serve(listener, api::router(state))
            .with_graceful_shutdown(async move {
                shutdown_signal().await;
                shutdown_client.begin_shutdown();
            })
            .await?;
        Ok::<_, anyhow::Error>(())
    }
    .await;
    let stopped = handle.shutdown().await;
    result?;
    stopped
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
