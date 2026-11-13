//! Dashboard HTTP server (axum) with the embedded single-page UI.
//!
//! Routes:
//! * `GET /`               — dashboard
//! * `GET /api/snapshot`   — full JSON snapshot
//! * `GET /api/predict`    — predictions for `?current=`
//! * `POST /api/observe`   — `{"path": "..."}` feed an open event
//! * `POST /api/prefetch`  — `{"paths": [...]}` force-warm
//! * `POST /api/budget`    — `{"bytes": N}` or `{"size": "2G"}`
//! * `GET /api/rag/type`   — `?prefix=...` streaming RAG hint
//! * `GET /api/rag/query`  — `?q=...` real RAG query
//! * `GET /api/health`     — liveness probe

use crate::config::{human_bytes, parse_size};
use crate::core::Core;
use crate::histogram::Snapshot as HistogramSnapshot;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;

const INDEX_HTML: &str = include_str!("../../dashboard/index.html");
const APP_JS: &str = include_str!("../../dashboard/app.js");
const STYLE_CSS: &str = include_str!("../../dashboard/style.css");

pub fn router(core: Arc<Core>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/style.css", get(style_css))
        .route("/favicon.svg", get(favicon))
        .route("/api/snapshot", get(snapshot))
        .route("/api/predict", get(predict))
        .route("/api/observe", post(observe))
        .route("/api/prefetch", post(prefetch))
        .route("/api/budget", post(budget))
        .route("/api/rag/type", get(rag_type))
        .route("/api/rag/query", get(rag_query))
        .route("/api/stream", get(sse))
        .route("/metrics", get(prometheus))
        .route("/api/health", get(health))
        .with_state(core)
}

pub async fn serve(core: Arc<Core>) -> anyhow::Result<()> {
    let port = core.config.port;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let app = router(core);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        INDEX_HTML,
    )
}

async fn app_js() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        APP_JS,
    )
}

async fn style_css() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLE_CSS,
    )
}

async fn favicon() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "image/svg+xml")],
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#0b0d12"/><path d="M7 22V10l5 7 4-7 4 7 5-7v12" fill="none" stroke="#6ee7b7" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>"##,
    )
}

async fn snapshot(State(core): State<Arc<Core>>) -> Json<Value> {
    let core2 = Arc::clone(&core);
    match tokio::task::spawn_blocking(move || core2.snapshot()).await {
        Ok(value) => Json(value),
        Err(e) => Json(json!({"error": format!("{e}")})),
    }
}

/// Server-sent events: one JSON snapshot every 900 ms, no client-side polling.
async fn sse(
    State(core): State<Arc<Core>>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
    tokio::spawn(async move {
        loop {
            let core = Arc::clone(&core);
            let snapshot = match tokio::task::spawn_blocking(move || core.snapshot()).await {
                Ok(value) => value,
                Err(_) => break,
            };
            let payload = serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".into());
            if tx.send(Ok(Event::default().data(payload))).await.is_err() {
                break; // client went away
            }
            tokio::time::sleep(Duration::from_millis(900)).await;
        }
    });
    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// Prometheus text exposition (scrape with `curl localhost:8765/metrics`).
async fn prometheus(State(core): State<Arc<Core>>) -> Response {
    let core = Arc::clone(&core);
    let (snapshot, warm, rag) = tokio::task::spawn_blocking(move || {
        (
            core.snapshot(),
            core.warm_latency_snapshot(),
            core.rag_latency_snapshot(),
        )
    })
    .await
    .unwrap_or_else(|_| {
        (
            Value::Null,
            HistogramSnapshot::default(),
            HistogramSnapshot::default(),
        )
    });
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        render_prometheus(&snapshot, &warm, &rag),
    )
        .into_response()
}

fn gauge(out: &mut String, name: &str, help: &str, value: f64) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push_str("\n# TYPE ");
    out.push_str(name);
    out.push_str(" gauge\n");
    out.push_str(name);
    out.push(' ');
    out.push_str(&format!("{value}"));
    out.push('\n');
}

pub fn render_prometheus(
    snapshot: &Value,
    warm: &HistogramSnapshot,
    rag: &HistogramSnapshot,
) -> String {
    let f = |path: &str| {
        snapshot
            .pointer(path)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    };
    let mut out = String::with_capacity(2048);

    gauge(
        &mut out,
        "mext_uptime_seconds",
        "Daemon uptime",
        f("/uptime_s"),
    );
    gauge(
        &mut out,
        "mext_memory_total_bytes",
        "Installed RAM",
        f("/memory/total"),
    );
    gauge(
        &mut out,
        "mext_memory_available_bytes",
        "RAM available to new allocations",
        f("/memory/available"),
    );
    gauge(
        &mut out,
        "mext_memory_used_ratio",
        "Share of RAM in use (0..1)",
        f("/memory/used_ratio"),
    );
    gauge(
        &mut out,
        "mext_swap_used_bytes",
        "Swap in use",
        f("/pressure/swap_used"),
    );

    gauge(
        &mut out,
        "mext_pool_budget_bytes",
        "Hard compressed-pool budget",
        f("/pool/budget_bytes"),
    );
    gauge(
        &mut out,
        "mext_pool_resident_bytes",
        "Real RSS of the pool (arena pages + mappings)",
        f("/pool/resident_bytes"),
    );
    gauge(
        &mut out,
        "mext_pool_logical_bytes",
        "Virtual volume served from the pool",
        f("/pool/logical_bytes"),
    );
    gauge(
        &mut out,
        "mext_pool_compressed_bytes",
        "LZ4 payload bytes",
        f("/pool/compressed_bytes"),
    );
    gauge(
        &mut out,
        "mext_pool_arena_pages",
        "Packed arena pages held",
        f("/pool/arena_pages"),
    );
    gauge(
        &mut out,
        "mext_pool_arena_frames",
        "Frames packed into shared pages",
        f("/pool/arena_frames"),
    );
    gauge(
        &mut out,
        "mext_pool_ratio",
        "Compression ratio (logical / compressed)",
        f("/pool/ratio"),
    );
    gauge(
        &mut out,
        "mext_pool_files",
        "Frames resident",
        f("/pool/files"),
    );
    gauge(
        &mut out,
        "mext_pool_evictions_total",
        "Frames evicted",
        f("/pool/evictions"),
    );
    gauge(
        &mut out,
        "mext_pool_released_bytes_total",
        "Bytes returned to macOS via madvise",
        f("/pool/released_bytes"),
    );
    gauge(
        &mut out,
        "mext_pool_ssd_read_bytes_total",
        "Bytes read from SSD into the pool",
        f("/pool/ssd_read_bytes"),
    );

    gauge(
        &mut out,
        "mext_predictor_hit_rate",
        "Online prediction accuracy (0..1)",
        f("/predictor/hit_rate"),
    );
    gauge(
        &mut out,
        "mext_predictor_attempts_total",
        "Scored predictions",
        f("/predictor/attempts"),
    );
    gauge(
        &mut out,
        "mext_predictor_hits_total",
        "Correct predictions",
        f("/predictor/hits"),
    );
    gauge(
        &mut out,
        "mext_predictor_edges",
        "Learned transitions",
        f("/predictor/edges"),
    );

    gauge(
        &mut out,
        "mext_activity_opens_total",
        "Observed file accesses",
        f("/activity/opens"),
    );
    gauge(
        &mut out,
        "mext_activity_warms_total",
        "Prefetch warms",
        f("/activity/warms"),
    );
    gauge(
        &mut out,
        "mext_activity_warm_avg_ms",
        "Average warm latency",
        f("/activity/warm_avg_ms"),
    );
    gauge(
        &mut out,
        "mext_rss_bytes",
        "Resident set size of the daemon",
        f("/rss_bytes"),
    );

    let neural_ok = snapshot
        .pointer("/neural/available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    gauge(
        &mut out,
        "mext_neural_available",
        "1 if the MLX layer loader is up",
        if neural_ok { 1.0 } else { 0.0 },
    );
    gauge(
        &mut out,
        "mext_neural_hot_bytes",
        "Weights pinned in unified memory",
        f("/neural/hot_bytes"),
    );
    gauge(
        &mut out,
        "mext_neural_cold_bytes",
        "Weights stored on SSD",
        f("/neural/cold_bytes"),
    );
    gauge(
        &mut out,
        "mext_neural_resident_bytes",
        "Cold bytes currently cached in RAM",
        f("/neural/resident_bytes"),
    );
    gauge(
        &mut out,
        "mext_neural_streamed_bytes_total",
        "Bytes streamed from SSD",
        f("/neural/streamed_bytes"),
    );
    gauge(
        &mut out,
        "mext_neural_prefetch_hit_rate",
        "Layer prefetch accuracy (0..1)",
        f("/neural/prefetch_hit_rate"),
    );
    gauge(
        &mut out,
        "mext_neural_tokens_per_second",
        "Generation throughput",
        f("/neural/tokens_per_s"),
    );
    gauge(
        &mut out,
        "mext_neural_experts_total",
        "MoE experts known",
        f("/neural/experts_total"),
    );
    gauge(
        &mut out,
        "mext_neural_experts_resident",
        "MoE experts cached in RAM",
        f("/neural/experts_resident"),
    );
    gauge(
        &mut out,
        "mext_neural_expert_hit_rate",
        "Router-guided expert prefetch accuracy",
        f("/neural/expert_hit_rate"),
    );

    let rag_ok = snapshot
        .pointer("/rag/available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    gauge(
        &mut out,
        "mext_rag_available",
        "1 if the RAG engine is ready",
        if rag_ok { 1.0 } else { 0.0 },
    );
    gauge(
        &mut out,
        "mext_rag_chunks",
        "Indexed chunks",
        f("/rag/chunks"),
    );
    gauge(
        &mut out,
        "mext_rag_clusters",
        "IVF clusters",
        f("/rag/clusters"),
    );
    gauge(
        &mut out,
        "mext_rag_prefetch_hit_rate",
        "Query prefetch accuracy (0..1)",
        f("/rag/prefetch_hit_rate"),
    );
    gauge(
        &mut out,
        "mext_rag_latency_ms",
        "Last query latency",
        f("/rag/latency_ms"),
    );

    out.push_str(&warm.render_prometheus(
        "mext_pool_warm_latency_ms",
        "Warm latency in milliseconds (SSD read + LZ4 + ARC insert)",
    ));
    out.push_str(&rag.render_prometheus(
        "mext_rag_query_latency_ms",
        "RAG query latency in milliseconds",
    ));

    let level = snapshot
        .pointer("/pressure/level")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    out.push_str("# HELP mext_pressure_info macOS memory pressure level\n");
    out.push_str("# TYPE mext_pressure_info gauge\n");
    out.push_str(&format!("mext_pressure_info{{level=\"{level}\"}} 1\n"));
    out
}

async fn health(State(core): State<Arc<Core>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": core.snapshot()["uptime_s"],
        "workspace": core.config.workspace.display().to_string(),
    }))
}

fn default_k() -> usize {
    4
}

async fn predict(
    State(core): State<Arc<Core>>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<Value> {
    let current = params.get("current").cloned().unwrap_or_default();
    let k = params
        .get("k")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(default_k);
    let path = PathBuf::from(&current);
    let path = if path.is_absolute() {
        path
    } else {
        core.config.workspace.join(path)
    };
    Json(core.predict(&path, k.min(16)))
}

async fn observe(State(core): State<Arc<Core>>, Json(body): Json<Value>) -> Json<Value> {
    let Some(path) = body.get("path").and_then(|v| v.as_str()) else {
        return Json(json!({"accepted": false, "error": "missing path"}));
    };
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        core.config.workspace.join(path)
    };
    let core2 = Arc::clone(&core);
    match tokio::task::spawn_blocking(move || core2.observe(&path, crate::monitor::Source::Cli))
        .await
    {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"accepted": false, "error": format!("{e}")})),
    }
}

async fn prefetch(State(core): State<Arc<Core>>, Json(body): Json<Value>) -> Json<Value> {
    let paths: Vec<PathBuf> = body
        .get("paths")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|p| {
                    let p = PathBuf::from(p);
                    if p.is_absolute() {
                        p
                    } else {
                        core.config.workspace.join(p)
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let core2 = Arc::clone(&core);
    match tokio::task::spawn_blocking(move || core2.warm_paths(&paths)).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": format!("{e}")})),
    }
}

async fn budget(State(core): State<Arc<Core>>, Json(body): Json<Value>) -> Response {
    let bytes = body.get("bytes").and_then(|v| v.as_u64()).or_else(|| {
        body.get("size")
            .and_then(|v| v.as_str())
            .and_then(|s| parse_size(s).ok())
    });
    match bytes {
        Some(bytes) => {
            core.set_budget(bytes);
            Json(json!({"budget": bytes, "human": human_bytes(bytes)})).into_response()
        }
        None => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "provide bytes or size"})),
        )
            .into_response(),
    }
}

async fn rag_type(
    State(core): State<Arc<Core>>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<Value> {
    let prefix = params.get("prefix").cloned().unwrap_or_default();
    let core2 = Arc::clone(&core);
    match tokio::task::spawn_blocking(move || {
        core2
            .ai
            .rag_type(&prefix)
            .unwrap_or_else(|| json!({"available": false}))
    })
    .await
    {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": format!("{e}")})),
    }
}

async fn rag_query(
    State(core): State<Arc<Core>>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<Value> {
    let query = params.get("q").cloned().unwrap_or_default();
    let k = params
        .get("k")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let core2 = Arc::clone(&core);
    let started = Instant::now();
    match tokio::task::spawn_blocking(move || {
        core2
            .ai
            .rag_commit(&query, k)
            .unwrap_or_else(|| json!({"available": false}))
    })
    .await
    {
        Ok(mut v) => {
            let http_ms = started.elapsed().as_secs_f64() * 1000.0;
            let engine_ms = v
                .get("latency_ms")
                .and_then(|value| value.as_f64())
                .filter(|value| *value > 0.0);
            core.observe_rag_latency(engine_ms.unwrap_or(http_ms));
            if let Some(obj) = v.as_object_mut() {
                obj.insert("http_ms".into(), json!(http_ms));
            }
            Json(v)
        }
        Err(e) => Json(json!({"error": format!("{e}")})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prometheus_text_has_core_metrics() {
        let snapshot = json!({
            "uptime_s": 12.5,
            "memory": {"total": 8589934592u64, "available": 2147483648u64, "used_ratio": 0.75},
            "pressure": {"level": "yellow", "swap_used": 1024},
            "pool": {
                "budget_bytes": 2147483648u64, "resident_bytes": 1048576,
                "logical_bytes": 4194304, "compressed_bytes": 900000,
                "arena_pages": 12, "ratio": 4.66, "files": 3, "evictions": 7,
                "released_bytes": 999, "ssd_read_bytes": 555, "arena_frames": 5
            },
            "predictor": {"hit_rate": 0.42, "attempts": 10, "hits": 4, "edges": 9},
            "activity": {"opens": 20, "warms": 5, "warm_avg_ms": 1.5},
            "neural": {"available": true, "hot_bytes": 10, "cold_bytes": 20, "resident_bytes": 5,
                       "streamed_bytes": 30, "prefetch_hit_rate": 0.9, "tokens_per_s": 12.0,
                       "experts_total": 8, "experts_resident": 2, "expert_hit_rate": 0.75},
            "rag": {"available": true, "chunks": 100, "clusters": 6, "prefetch_hit_rate": 0.5, "latency_ms": 3.2}
        });
        let text = render_prometheus(
            &snapshot,
            &HistogramSnapshot::default(),
            &HistogramSnapshot::default(),
        );
        assert!(text.contains("mext_uptime_seconds 12.5"));
        assert!(text.contains("mext_pool_resident_bytes 1048576"));
        assert!(text.contains("mext_pool_arena_pages 12"));
        assert!(text.contains("mext_predictor_hit_rate 0.42"));
        assert!(text.contains("mext_neural_expert_hit_rate 0.75"));
        assert!(text.contains("mext_pressure_info{level=\"yellow\"} 1"));
        assert!(text.contains("# TYPE mext_pool_budget_bytes gauge"));
        assert!(text.contains("# TYPE mext_pool_warm_latency_ms histogram"));
        assert!(text.contains("mext_pool_warm_latency_ms_bucket{le=\"+Inf\"} 0"));
        assert!(text.contains("mext_rag_query_latency_ms_count 0"));
        assert!(text.contains("mext_pool_warm_latency_ms_sum 0"));
        // Every HELP line is followed by a TYPE line.
        let helps = text.lines().filter(|l| l.starts_with("# HELP")).count();
        let types = text.lines().filter(|l| l.starts_with("# TYPE")).count();
        assert_eq!(helps, types);
    }
}

/// Small helper used by the daemon to probe the port before printing the URL.
pub async fn wait_ready(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
    false
}
