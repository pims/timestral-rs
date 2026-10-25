#![cfg(feature = "server")]
use anyhow::Result;
use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};
use timestral_rs::{
    api::{router, AppState, Limits},
    worker::{self, ForecastInput, InferenceEngine, ModelInfo, WorkerClient, WorkerHandle},
};
use tokio::sync::Notify;
use tower::ServiceExt;

#[derive(Default)]
struct Gate {
    started: Notify,
    released: Mutex<bool>,
    condition: Condvar,
}
impl Gate {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.condition.notify_all();
    }
}
struct FakeEngine {
    calls: Arc<AtomicUsize>,
    gate: Option<Arc<Gate>>,
    fail: bool,
}
impl InferenceEngine for FakeEngine {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            model: "test-model".into(),
            server_version: "test".into(),
            device: "cpu".into(),
            dtype: "float32".into(),
            parameter_count: 0,
            context_length: 32768,
            quantile_levels: vec![0.1, 0.5, 0.9],
            weights_sha256: "test".into(),
            config_sha256: "test".into(),
            fused_rmsnorm: false,
            singleton_variate_attention: true,
        }
    }
    fn forecast(&mut self, history: &[f32], horizon: usize) -> Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.started.notify_one();
            let mut released = gate.released.lock().unwrap();
            while !*released {
                released = gate.condition.wait(released).unwrap();
            }
        }
        if self.fail {
            anyhow::bail!("private internal details");
        }
        let last = *history.iter().rev().find(|v| v.is_finite()).unwrap();
        Ok((0..horizon)
            .flat_map(|step| (0..3).map(move |q| last + step as f32 + q as f32))
            .collect())
    }
}
fn setup(
    limits: Limits,
    capacity: usize,
    gate: Option<Arc<Gate>>,
    fail: bool,
) -> (Router, WorkerClient, WorkerHandle, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let fake_calls = calls.clone();
    let (client, handle) = worker::spawn_engine(capacity, move || {
        Ok(Box::new(FakeEngine {
            calls: fake_calls,
            gate,
            fail,
        }))
    })
    .unwrap();
    let app = router(AppState::new(client.clone(), limits).unwrap());
    (app, client, handle, calls)
}
fn request(path: &str, body: Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
async fn value(response: axum::response::Response) -> Value {
    serde_json::from_slice(
        &to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}
async fn wait_queued(client: &WorkerClient, count: usize) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while client.queued_jobs() != count {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
fn payload() -> Value {
    json!({"history":[1,2,null,4],"horizon":2})
}

#[tokio::test]
async fn routes_forecast_selection_and_metadata() {
    let (app, client, handle, calls) = setup(Limits::default(), 2, None, false);
    for path in ["/healthz", "/readyz"] {
        assert_eq!(
            app.clone()
                .oneshot(Request::get(path).body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
    let metadata = value(
        app.clone()
            .oneshot(Request::get("/v1/model").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(metadata["device"], "cpu");
    assert_eq!(metadata["queue_capacity"], 2);
    let response = app
        .clone()
        .oneshot(request("/v1/forecast", payload()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let data = value(response).await;
    assert_eq!(data["request_id"], id);
    assert_eq!(data["quantiles"]["0.5"], json!([5.0, 6.0]));
    assert!(data["timing_ms"]["queue"].as_f64().unwrap() >= 0.0);
    let selected = value(
        app.clone()
            .oneshot(request(
                "/v1/forecast",
                json!({"history":[4],"horizon":1,"quantiles":[0.9]}),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(selected["quantiles"].as_object().unwrap().len(), 1);
    assert_eq!(selected["quantiles"]["0.9"], json!([6.0]));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let missing = app
        .clone()
        .oneshot(Request::get("/missing").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    let method = app
        .clone()
        .oneshot(Request::get("/v1/forecast").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(method.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(value(method).await["error"]["code"], "method_not_allowed");
    handle.shutdown().await.unwrap();
    assert!(!client.is_ready());
    assert_eq!(
        app.oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn validation_rejects_bad_values_lengths_levels_and_json() {
    let limits = Limits {
        max_history: 4,
        max_horizon: 3,
        ..Limits::default()
    };
    let (app, _, handle, calls) = setup(limits, 2, None, false);
    for body in [
        json!({"history":[],"horizon":1}),
        json!({"history":[null],"horizon":1}),
        json!({"history":[1],"horizon":0}),
        json!({"history":[1],"horizon":4}),
        json!({"history":[1,2,3,4,5],"horizon":1}),
        json!({"history":[1e40],"horizon":1}),
        json!({"history":[1],"horizon":1,"quantiles":[]}),
        json!({"history":[1],"horizon":1,"quantiles":[0.2]}),
        json!({"history":[1],"horizon":1,"quantiles":[0.5,0.5]}),
        json!({"history":[1],"horizon":1,"extra":true}),
    ] {
        let response = app
            .clone()
            .oneshot(request("/v1/forecast", body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(value(response).await["error"]["request_id"].is_string());
    }
    let malformed = Request::post("/v1/forecast")
        .header("content-type", "application/json")
        .body(Body::from("{broken"))
        .unwrap();
    assert_eq!(
        app.clone().oneshot(malformed).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    let no_type = Request::post("/v1/forecast")
        .body(Body::from(payload().to_string()))
        .unwrap();
    assert_eq!(
        app.oneshot(no_type).await.unwrap().status(),
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn body_limit_is_enforced_before_inference() {
    let (app, _, handle, calls) = setup(
        Limits {
            max_body_bytes: 64,
            ..Limits::default()
        },
        1,
        None,
        false,
    );
    let response = app
        .oneshot(request(
            "/v1/forecast",
            json!({"history":vec![1;100],"horizon":1}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(value(response).await["error"]["request_id"].is_string());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn bounded_queue_rejects_overload_while_health_stays_responsive_and_shutdown_drains() {
    let gate = Arc::new(Gate::default());
    let (app, client, handle, calls) = setup(Limits::default(), 1, Some(gate.clone()), false);
    let first = tokio::spawn(app.clone().oneshot(request("/v1/forecast", payload())));
    tokio::time::timeout(Duration::from_secs(3), gate.started.notified())
        .await
        .unwrap();
    let second = tokio::spawn(app.clone().oneshot(request("/v1/forecast", payload())));
    wait_queued(&client, 1).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let overloaded = app
        .clone()
        .oneshot(request("/v1/forecast", payload()))
        .await
        .unwrap();
    assert_eq!(overloaded.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(overloaded.headers()["retry-after"], "1");
    assert_eq!(
        app.clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    client.begin_shutdown();
    assert_eq!(
        app.clone()
            .oneshot(request("/v1/forecast", payload()))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    let shutdown = tokio::spawn(handle.shutdown());
    gate.release();
    assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
    assert_eq!(second.await.unwrap().unwrap().status(), StatusCode::OK);
    shutdown.await.unwrap().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(!client.is_ready());
}

#[tokio::test]
async fn timeout_does_not_abort_running_work_and_canceled_queued_work_is_skipped() {
    let gate = Arc::new(Gate::default());
    let (app, client, handle, calls) = setup(
        Limits {
            request_timeout_ms: 50,
            ..Limits::default()
        },
        2,
        Some(gate.clone()),
        false,
    );
    let first = tokio::spawn(app.clone().oneshot(request("/v1/forecast", payload())));
    tokio::time::timeout(Duration::from_secs(3), gate.started.notified())
        .await
        .unwrap();
    assert_eq!(
        first.await.unwrap().unwrap().status(),
        StatusCode::GATEWAY_TIMEOUT
    );
    let second = app
        .clone()
        .oneshot(request("/v1/forecast", payload()))
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(client.queued_jobs(), 1);
    gate.release();
    wait_queued(&client, 0).await;
    assert_eq!(
        app.oneshot(request("/v1/forecast", payload()))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn expired_jobs_are_not_executed_and_worker_errors_are_sanitized() {
    let (app, client, handle, calls) = setup(Limits::default(), 2, None, true);
    let reply = client
        .submit(
            ForecastInput {
                history: vec![1.0],
                horizon: 1,
                quantiles: vec![0.5],
            },
            "expired".into(),
            Instant::now() - Duration::from_secs(1),
        )
        .unwrap();
    assert!(matches!(
        reply.await.unwrap(),
        Err(worker::WorkerFailure::Expired)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let response = app
        .oneshot(request("/v1/forecast", payload()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = value(response).await;
    assert_eq!(body["error"]["code"], "inference_failed");
    assert!(!body.to_string().contains("private internal details"));
    assert!(client.is_ready());
    handle.shutdown().await.unwrap();
}

#[test]
fn failed_startup_is_reported() {
    assert!(worker::spawn_engine(1, || anyhow::bail!("missing checkpoint")).is_err());
}

#[tokio::test]
#[ignore = "requires the downloaded local Timestral-5M checkpoint"]
async fn real_model_matches_direct_forecast() {
    check_real_model("Timestral-5M", 5_208_960).await;
}

#[tokio::test]
#[ignore = "requires the downloaded local Timestral-28M checkpoint"]
async fn real_28m_model_matches_direct_forecast() {
    check_real_model("Timestral-28M", 27_821_184).await;
}

async fn check_real_model(name: &str, parameter_count: usize) {
    use candle_core::Device;
    use timestral_rs::{
        forecaster::TimestralForecaster,
        loading::{load_5m, load_model},
    };
    let checkpoint = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name);
    if name == "Timestral-28M" {
        let error = load_5m(&checkpoint, &Device::Cpu)
            .err()
            .expect("5M-only loader must reject 28M");
        assert!(error.to_string().contains("Expected Timestral-5M"));
    }
    let history = [1f32, 2.0, f32::NAN, 3.0, 2.0, 1.0];
    let direct = TimestralForecaster::new(load_model(&checkpoint, &Device::Cpu).unwrap())
        .forecast(&history, 5)
        .unwrap();
    let direct = direct
        .quantiles
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (client, handle) =
        tokio::task::spawn_blocking(move || worker::spawn_model(checkpoint, "cpu".into(), 2))
            .await
            .unwrap()
            .unwrap();
    assert_eq!(client.info.weights_sha256.len(), 64);
    assert_eq!(client.info.model, format!("mistralai/{name}"));
    assert_eq!(client.info.parameter_count, parameter_count);
    let app = router(AppState::new(client, Limits::default()).unwrap());
    let response = app
        .oneshot(request(
            "/v1/forecast",
            json!({"history":[1,2,null,3,2,1],"horizon":5}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = value(response).await;
    assert_eq!(body["model"], format!("mistralai/{name}"));
    for (level, index) in [("0.1", 2), ("0.5", 10), ("0.9", 18)] {
        let values: Vec<f32> = body["quantiles"][level]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(
            values,
            (0..5)
                .map(|step| direct[step * 21 + index])
                .collect::<Vec<_>>()
        );
    }
    handle.shutdown().await.unwrap();
}
