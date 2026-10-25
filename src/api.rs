//! HTTP contract, bounded admissions, and request deadlines.
use crate::worker::{
    ForecastInput, ForecastResponse, ModelInfo, SubmitError, WorkerClient, WorkerFailure,
};
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Request, State},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Extension, Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

#[derive(Clone, Debug, Serialize)]
pub struct Limits {
    pub max_history: usize,
    pub max_horizon: usize,
    pub max_body_bytes: usize,
    pub request_timeout_ms: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_history: 32768,
            max_horizon: 1024,
            max_body_bytes: 1024 * 1024,
            request_timeout_ms: 30000,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub worker: WorkerClient,
    pub limits: Limits,
    admissions: Arc<Semaphore>,
    sequence: Arc<AtomicU64>,
}
impl AppState {
    pub fn new(worker: WorkerClient, limits: Limits) -> anyhow::Result<Self> {
        anyhow::ensure!(
            limits.max_history > 0 && limits.max_history <= worker.info.context_length,
            "max_history must be in 1..={}",
            worker.info.context_length
        );
        anyhow::ensure!(
            limits.max_horizon > 0 && limits.max_body_bytes > 0 && limits.request_timeout_ms > 0,
            "Horizon, body size, and timeout limits must be positive"
        );
        anyhow::ensure!(
            Instant::now()
                .checked_add(Duration::from_millis(limits.request_timeout_ms))
                .is_some(),
            "Request timeout is too large"
        );
        let admissions = Arc::new(Semaphore::new(worker.queue_capacity + 1));
        Ok(Self {
            worker,
            limits,
            admissions,
            sequence: Arc::new(AtomicU64::new(1)),
        })
    }
}

#[derive(Clone)]
struct RequestContext {
    id: String,
    deadline: Instant,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastRequest {
    history: Vec<Option<f64>>,
    horizon: usize,
    #[serde(default = "default_quantiles")]
    quantiles: Vec<f64>,
}
fn default_quantiles() -> Vec<f64> {
    vec![0.1, 0.5, 0.9]
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    request_id: Option<String>,
}
impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: &'static str) -> Self {
        Self {
            status,
            code,
            message,
            request_id: None,
        }
    }
    fn request(mut self, id: &str) -> Self {
        self.request_id = Some(id.into());
        self
    }
    fn busy() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "busy",
            "Inference capacity is full; retry later",
        )
    }
    fn unavailable() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "Inference worker is not ready",
        )
    }
    fn timeout() -> Self {
        Self::new(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            "Request deadline exceeded",
        )
    }
    fn validation(message: &'static str) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(serde_json::json!({"error": {
                "code": self.code, "message": self.message, "request_id": self.request_id
            }})),
        )
            .into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            response
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("1"));
        }
        response
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route(
            "/healthz",
            get(|| async { Json(serde_json::json!({"status":"ok"})) }),
        )
        .route("/readyz", get(readiness))
        .route("/v1/model", get(model_info))
        .route(
            "/v1/forecast",
            post(forecast)
                .fallback(|| async {
                    ApiError::new(
                        StatusCode::METHOD_NOT_ALLOWED,
                        "method_not_allowed",
                        "Use POST for forecasts",
                    )
                })
                .layer(middleware::from_fn_with_state(state.clone(), admission)),
        )
        .fallback(|| async {
            ApiError::new(StatusCode::NOT_FOUND, "not_found", "Unknown endpoint")
        })
        .layer(DefaultBodyLimit::max(state.limits.max_body_bytes))
        .with_state(state)
}

async fn readiness(State(state): State<AppState>) -> Response {
    if state.worker.is_ready() {
        Json(serde_json::json!({"status":"ready", "queued_jobs":state.worker.queued_jobs()}))
            .into_response()
    } else {
        ApiError::unavailable().into_response()
    }
}

#[derive(Serialize)]
struct ModelResponse {
    #[serde(flatten)]
    info: ModelInfo,
    limits: Limits,
    queue_capacity: usize,
}
async fn model_info(State(state): State<AppState>) -> Json<ModelResponse> {
    Json(ModelResponse {
        info: (*state.worker.info).clone(),
        limits: state.limits,
        queue_capacity: state.worker.queue_capacity,
    })
}

async fn admission(State(state): State<AppState>, mut request: Request, next: Next) -> Response {
    let id = format!("{:016x}", state.sequence.fetch_add(1, Ordering::Relaxed));
    let started = Instant::now();
    let duration = Duration::from_millis(state.limits.request_timeout_ms);
    let mut response = if !state.worker.is_ready() {
        ApiError::unavailable().request(&id).into_response()
    } else if let Ok(_permit) = state.admissions.clone().try_acquire_owned() {
        request.extensions_mut().insert(RequestContext {
            id: id.clone(),
            deadline: started + duration,
        });
        match tokio::time::timeout(duration, next.run(request)).await {
            Ok(response) => response,
            Err(_) => ApiError::timeout().request(&id).into_response(),
        }
    } else {
        ApiError::busy().request(&id).into_response()
    };
    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&id).expect("hex ID"));
    eprintln!(
        "{}",
        serde_json::json!({"event":"forecast_request", "request_id": id,
        "status": response.status().as_u16(), "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0})
    );
    response
}

async fn forecast(
    State(state): State<AppState>,
    Extension(context): Extension<RequestContext>,
    request: Result<Json<ForecastRequest>, JsonRejection>,
) -> Result<Json<ForecastResponse>, ApiError> {
    let Json(request) = request.map_err(|error| {
        let (code, message) = if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ("payload_too_large", "Request body exceeds max_body_bytes")
        } else {
            (
                "invalid_json",
                "Expected JSON history, horizon, and optional quantiles",
            )
        };
        ApiError::new(error.status(), code, message).request(&context.id)
    })?;
    let input = validate(request, &state).map_err(|error| error.request(&context.id))?;
    let reply = state
        .worker
        .submit(input, context.id.clone(), context.deadline)
        .map_err(|error| match error {
            SubmitError::Busy => ApiError::busy().request(&context.id),
            SubmitError::Unavailable => ApiError::unavailable().request(&context.id),
        })?;
    match reply.await {
        Ok(Ok(response)) => Ok(Json(response)),
        Ok(Err(WorkerFailure::Expired)) => Err(ApiError::timeout().request(&context.id)),
        Ok(Err(WorkerFailure::Inference)) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "Forecast could not be generated",
        )
        .request(&context.id)),
        Err(_) => Err(ApiError::unavailable().request(&context.id)),
    }
}

fn validate(request: ForecastRequest, state: &AppState) -> Result<ForecastInput, ApiError> {
    if request.history.is_empty() || request.history.len() > state.limits.max_history {
        return Err(ApiError::validation(
            "History must be nonempty and within max_history",
        ));
    }
    if request.horizon == 0 || request.horizon > state.limits.max_horizon {
        return Err(ApiError::validation(
            "Horizon must be positive and within max_horizon",
        ));
    }
    if request.quantiles.is_empty()
        || request.quantiles.len() > state.worker.info.quantile_levels.len()
    {
        return Err(ApiError::validation(
            "Select at least one supported quantile",
        ));
    }
    for (i, level) in request.quantiles.iter().enumerate() {
        if !state.worker.info.quantile_levels.contains(level)
            || request.quantiles[..i].contains(level)
        {
            return Err(ApiError::validation(
                "Quantile levels must be supported and unique",
            ));
        }
    }
    let mut observed = false;
    let mut history = Vec::with_capacity(request.history.len());
    for value in request.history {
        match value {
            Some(value) if value.is_finite() && (value as f32).is_finite() => {
                observed = true;
                history.push(value as f32);
            }
            None => history.push(f32::NAN),
            _ => {
                return Err(ApiError::validation(
                    "History values must fit finite float32; use null for missing data",
                ))
            }
        }
    }
    if !observed {
        return Err(ApiError::validation(
            "History must contain at least one observed value",
        ));
    }
    Ok(ForecastInput {
        history,
        horizon: request.horizon,
        quantiles: request.quantiles,
    })
}
