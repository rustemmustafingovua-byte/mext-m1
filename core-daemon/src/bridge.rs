//! Bridge to the Python AI engine (MLX neural loader + LanceDB RAG).
//!
//! The AI engine is optional: if it is not running every call degrades
//! gracefully to `{"available": false, ...}` so the Rust core and the dashboard
//! keep working.

use crate::ipc::RpcClient;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct AiBridge {
    client: RpcClient,
    cache: Mutex<Option<(Instant, Value)>>,
    timeout: Duration,
    cache_ttl: Duration,
}

impl AiBridge {
    pub fn new(socket: PathBuf) -> Self {
        Self {
            client: RpcClient::with_timeout(socket, Duration::from_millis(400)),
            cache: Mutex::new(None),
            timeout: Duration::from_millis(600),
            cache_ttl: Duration::from_millis(900),
        }
    }

    pub fn socket(&self) -> PathBuf {
        self.client.socket().to_path_buf()
    }

    pub fn reachable(&self) -> bool {
        self.client.reachable()
    }

    /// Aggregated AI metrics; cached for ~1 s so the dashboard can poll fast.
    pub fn metrics(&self) -> Value {
        if let Ok(guard) = self.cache.lock() {
            if let Some((at, value)) = guard.as_ref() {
                if at.elapsed() < self.cache_ttl {
                    return value.clone();
                }
            }
        }
        let value = match self.client.call("metrics", json!({})) {
            Ok(v) => v,
            Err(e) => json!({
                "available": false,
                "error": format!("{e}"),
                "socket": self.socket().display().to_string(),
            }),
        };
        if let Ok(mut guard) = self.cache.lock() {
            *guard = Some((Instant::now(), value.clone()));
        }
        value
    }

    /// Tell the neural loader which model layers are likely next.
    pub fn prefetch_layers(&self, layers: &[u32]) -> bool {
        if layers.is_empty() {
            return false;
        }
        self.client
            .call("neural.prefetch", json!({"layers": layers}))
            .is_ok()
    }

    /// Warm RAG clusters for an in-progress query prefix.
    pub fn rag_type(&self, prefix: &str) -> Option<Value> {
        self.client.call("rag.type", json!({"prefix": prefix})).ok()
    }

    /// Execute the real query (the user pressed Enter).
    pub fn rag_commit(&self, query: &str, k: usize) -> Option<Value> {
        self.client
            .call("rag.commit", json!({"query": query, "k": k}))
            .ok()
    }

    /// Ask the AI side to warm files it knows about (LanceDB metadata cache).
    pub fn prefetch(&self, paths: &[String]) -> bool {
        if paths.is_empty() {
            return false;
        }
        self.client
            .call("prefetch", json!({"paths": paths}))
            .is_ok()
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn health(&self) -> Value {
        self.client
            .call("health", json!({}))
            .unwrap_or_else(|e| json!({"available": false, "error": format!("{e}")}))
    }
}
