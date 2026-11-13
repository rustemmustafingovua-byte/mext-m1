//! The core actor: wires monitor events → predictor → compressed pool → AI.
//!
//! Everything is synchronous behind mutexes; the dashboard and the RPC server
//! only read snapshots, and heavy work (compression, AI prefetch) happens on
//! dedicated threads so no request ever waits on a 40 MiB LZ4 pass.

use crate::bridge::AiBridge;
use crate::config::{human_bytes, Config};
use crate::metrics::{unix_now, EventLog, PlatformSampler, Rate};
use crate::monitor::{is_watched, Source};
use crate::pool::CompressedMemoryPool;
use crate::predictor::{hour_of_day, Predictor};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// How many candidates we prefetch per observed file.
pub const TOP_K: usize = 4;
/// Shrinking below this is pointless and hurts hit rate.
const MIN_BUDGET: u64 = 32 * 1024 * 1024;
/// Under red pressure we release everything when the headroom is below this.
const PANIC_HEADROOM: u64 = 192 * 1024 * 1024;

#[derive(Default)]
struct RuntimeStats {
    opens: u64,
    warms: u64,
    warm_ms_total: f64,
    warm_failures: u64,
    pool_hits: u64,
}

impl RuntimeStats {
    fn warm_avg_ms(&self) -> f64 {
        if self.warms == 0 {
            0.0
        } else {
            self.warm_ms_total / self.warms as f64
        }
    }
}

pub struct Core {
    pub config: Config,
    pool: Mutex<CompressedMemoryPool>,
    predictor: Mutex<Predictor>,
    pub ai: AiBridge,
    events: Mutex<EventLog>,
    sampler: Mutex<PlatformSampler>,
    ssd_rate: Mutex<Rate>,
    stats: Mutex<RuntimeStats>,
    started: Instant,
    inflight: AtomicUsize,
    effective_budget: AtomicI64,
    /// Budget chosen explicitly through RPC/HTTP (`0` = follow `--ram-limit`).
    manual_budget: AtomicI64,
    shutdown: AtomicBool,
    last_persist: Mutex<Instant>,
}

impl Core {
    pub fn new(config: Config) -> Result<Self> {
        let pool =
            CompressedMemoryPool::new(config.ram_budget, config.chunk_size, config.max_file_bytes);
        let predictor = Predictor::load_or_new(&config.predictor_state());
        let ai = AiBridge::new(config.ai_socket.clone());
        let mut log = EventLog::new(400);
        log.push(
            "start",
            &config.workspace.display().to_string(),
            format!(
                "budget {} · chunk {} · model {}",
                human_bytes(config.ram_budget),
                human_bytes(config.chunk_size as u64),
                config.model
            ),
        );
        Ok(Self {
            pool: Mutex::new(pool),
            predictor: Mutex::new(predictor),
            ai,
            events: Mutex::new(log),
            sampler: Mutex::new(PlatformSampler::new(std::time::Duration::from_millis(900))),
            ssd_rate: Mutex::new(Rate::new()),
            stats: Mutex::new(RuntimeStats::default()),
            started: Instant::now(),
            inflight: AtomicUsize::new(0),
            effective_budget: AtomicI64::new(config.ram_budget as i64),
            manual_budget: AtomicI64::new(0),
            shutdown: AtomicBool::new(false),
            last_persist: Mutex::new(Instant::now()),
            config,
        })
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    pub fn workspace(&self) -> &Path {
        &self.config.workspace
    }

    /// Workspace-relative, forward-slash key used by the predictor.
    pub fn rel(&self, path: &Path) -> String {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        canonical
            .strip_prefix(&self.config.workspace)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn abs(&self, key: &str) -> PathBuf {
        let p = PathBuf::from(key);
        if p.is_absolute() {
            p
        } else {
            self.config.workspace.join(p)
        }
    }

    // ------------------------------------------------------------------
    // The main pipeline
    // ------------------------------------------------------------------

    /// Handle a filesystem access: learn, predict, warm, notify AI.
    pub fn observe(self: &Arc<Self>, path: &Path, source: Source) -> Value {
        // FSEvents and editors may hand us `/var/...` while the workspace was
        // canonicalised to `/private/var/...`; normalise once here.
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let path = canonical.as_path();
        if !is_watched(path, &self.config.workspace) {
            return json!({"accepted": false, "reason": "not watched"});
        }
        let key = self.rel(path);
        let abs = self.abs(&key);
        let hour = hour_of_day();

        let report = {
            let mut p = self.predictor.lock().expect("predictor poisoned");
            let report = p.observe(&key, hour);
            let predictions = p.predict(&key, hour, TOP_K);
            let pending: Vec<String> = predictions.iter().map(|i| i.path.clone()).collect();
            p.set_pending(pending);
            (report, predictions)
        };
        let (report, predictions) = report;

        let pool_hit = {
            let mut pool = self.pool.lock().expect("pool poisoned");
            pool.touch(&abs)
        };

        {
            let mut stats = self.stats.lock().expect("stats poisoned");
            stats.opens += 1;
            if pool_hit {
                stats.pool_hits += 1;
            }
        }

        let mut detail = String::new();
        if report.transition {
            detail.push_str("transition · ");
        }
        detail.push_str(match report.hit {
            Some(true) => "prediction HIT",
            Some(false) => "prediction MISS",
            None => "cold",
        });
        detail.push_str(if pool_hit {
            " · pool hit"
        } else {
            " · pool miss"
        });
        self.events
            .lock()
            .expect("events poisoned")
            .push(source.as_str(), &key, detail);

        // Fire-and-forget prefetch of the predicted next files.
        let candidates: Vec<PathBuf> = predictions
            .iter()
            .map(|p| self.abs(&p.path))
            .filter(|p| *p != abs)
            .collect();
        let predicted_keys: Vec<String> = predictions.iter().map(|p| p.path.clone()).collect();
        self.schedule_warm(candidates, &key);
        self.notify_ai(predicted_keys, source);

        json!({
            "accepted": true,
            "key": key,
            "source": source.as_str(),
            "transition": report.transition,
            "prediction_hit": report.hit,
            "pool_hit": pool_hit,
            "predictions": predictions,
        })
    }

    /// Warm the predicted paths on background threads, bounded by
    /// `warm_concurrency` so we do not thrash the NVMe controller.
    fn schedule_warm(self: &Arc<Self>, paths: Vec<PathBuf>, origin: &str) {
        let limit = self.config.warm_concurrency.max(1);
        for path in paths {
            if self.inflight.load(Ordering::SeqCst) >= limit {
                break;
            }
            if self.pool.lock().map(|p| p.contains(&path)).unwrap_or(false) {
                continue;
            }
            self.inflight.fetch_add(1, Ordering::SeqCst);
            let this = Arc::clone(self);
            let origin = origin.to_string();
            let spawned = std::thread::Builder::new()
                .name("mext-warm".into())
                .spawn(move || {
                    let started = Instant::now();
                    let outcome = this.pool.lock().expect("pool poisoned").warm(&path);
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    let key = this.rel(&path);
                    match outcome {
                        Ok(res) => {
                            if res.outcome == crate::pool::WarmOutcome::Warmed {
                                let mut stats = this.stats.lock().expect("stats poisoned");
                                stats.warms += 1;
                                stats.warm_ms_total += elapsed;
                                drop(stats);
                                this.events.lock().expect("events poisoned").push(
                                    "warm",
                                    &key,
                                    format!(
                                        "{} → {} in {:.1} ms ({}×) for {}",
                                        human_bytes(res.logical_bytes),
                                        human_bytes(res.compressed_bytes),
                                        elapsed,
                                        if res.compressed_bytes == 0 {
                                            1.0
                                        } else {
                                            res.logical_bytes as f64 / res.compressed_bytes as f64
                                        },
                                        origin
                                    ),
                                );
                            }
                        }
                        Err(e) => {
                            let mut stats = this.stats.lock().expect("stats poisoned");
                            stats.warm_failures += 1;
                            drop(stats);
                            this.events.lock().expect("events poisoned").push(
                                "error",
                                &key,
                                format!("warm failed: {e}"),
                            );
                        }
                    }
                    this.inflight.fetch_sub(1, Ordering::SeqCst);
                });
            if spawned.is_err() {
                self.inflight.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }

    /// Ask the AI engine to pre-stage layers / LanceDB clusters for these files.
    fn notify_ai(self: &Arc<Self>, predicted: Vec<String>, source: Source) {
        if !self.config.enable_neural && !self.config.enable_rag {
            return;
        }
        if !self.ai.reachable() {
            return;
        }
        let this = Arc::clone(self);
        std::thread::Builder::new()
            .name("mext-ai-notify".into())
            .spawn(move || {
                this.ai.prefetch(&predicted);
                // Prefetch model layers speculatively: a file switch usually
                // precedes a new completion request, so warm the first layers.
                let lookahead = this.config.lookahead_layers as u32;
                let layers: Vec<u32> = (0..lookahead).collect();
                this.ai.prefetch_layers(&layers);
                if this.ai.reachable() {
                    this.events.lock().expect("events poisoned").push(
                        "ai",
                        "-",
                        format!("prefetch {predicted:?} ← {}", source.as_str()),
                    );
                }
            })
            .ok();
    }

    // ------------------------------------------------------------------
    // Budget control (pressure-aware)
    // ------------------------------------------------------------------

    /// Recompute the effective pool budget from macOS memory pressure.
    /// The user limit stays a hard ceiling; we only shrink below it.
    pub fn rebalance(&self) -> Value {
        let (info, pressure, swap) = self.sampler.lock().expect("sampler poisoned").sample();
        // A manual budget stays the ceiling: pressure may shrink below it, but
        // going back to green must not undo the user's `POST /api/budget`.
        let manual = self.manual_budget.load(Ordering::SeqCst);
        let configured = if manual > 0 {
            manual as u64
        } else {
            self.config.ram_budget
        };
        let factor = pressure.shrink_factor();
        let mut target = ((configured as f64) * factor) as u64;
        target = target.max(MIN_BUDGET.min(configured));
        target = target.min(configured);

        let previous = self.effective_budget.swap(target as i64, Ordering::SeqCst) as u64;
        let mut released = 0u64;
        if target.abs_diff(previous) > configured / 40 {
            let mut pool = self.pool.lock().expect("pool poisoned");
            pool.set_budget(target);
            if pressure == crate::platform::Pressure::Red && info.available() < PANIC_HEADROOM {
                released = pool.release_all();
            }
            drop(pool);
            self.events.lock().expect("events poisoned").push(
                "budget",
                "-",
                format!(
                    "{} → {} ({}, free {}){}",
                    human_bytes(previous),
                    human_bytes(target),
                    pressure.as_str(),
                    human_bytes(info.available()),
                    if released > 0 {
                        format!(" · released {}", human_bytes(released))
                    } else {
                        String::new()
                    }
                ),
            );
        }
        json!({
            "budget": target,
            "pressure": pressure.as_str(),
            "free": info.available(),
            "swap_used": swap.used,
        })
    }

    pub fn set_budget(&self, bytes: u64) {
        self.manual_budget.store(bytes as i64, Ordering::SeqCst);
        self.pool.lock().expect("pool poisoned").set_budget(bytes);
        self.effective_budget.store(bytes as i64, Ordering::SeqCst);
        self.events.lock().expect("events poisoned").push(
            "budget",
            "-",
            format!("manual limit → {}", human_bytes(bytes)),
        );
    }

    pub fn warm_paths(&self, paths: &[PathBuf]) -> Value {
        let mut warmed = 0;
        let mut hit = 0;
        let mut bytes = 0u64;
        for p in paths {
            if let Ok(res) = self.pool.lock().expect("pool poisoned").warm(p) {
                match res.outcome {
                    crate::pool::WarmOutcome::Hit => hit += 1,
                    crate::pool::WarmOutcome::Warmed => {
                        warmed += 1;
                        bytes += res.logical_bytes;
                    }
                    crate::pool::WarmOutcome::Skipped => {}
                }
            }
        }
        json!({"warmed": warmed, "hits": hit, "bytes": bytes})
    }

    pub fn predict(&self, current: &Path, k: usize) -> Value {
        let key = self.rel(current);
        let hour = hour_of_day();
        let p = self.predictor.lock().expect("predictor poisoned");
        let items = p.predict(&key, hour, k);
        json!({
            "current": key,
            "hour": hour,
            "hit_rate": p.hit_rate(),
            "predictions": items,
        })
    }

    /// Periodic save, throttled to once per 20 s.
    pub fn persist(&self) {
        {
            let mut at = self.last_persist.lock().expect("persist poisoned");
            if at.elapsed() < std::time::Duration::from_secs(20) {
                return;
            }
            *at = Instant::now();
        }
        self.persist_now();
    }

    /// Unconditional save — used on shutdown and by `mext stop`.
    pub fn persist_now(&self) {
        let predictor = self.predictor.lock().expect("predictor poisoned");
        if let Err(e) = predictor.save(&self.config.predictor_state()) {
            eprintln!("[core] cannot persist predictor: {e}");
        }
    }

    pub fn release_all(&self) {
        let released = self.pool.lock().expect("pool poisoned").release_all();
        self.events.lock().expect("events poisoned").push(
            "release",
            "-",
            format!(
                "madvise(MADV_DONTNEED) → {} returned to macOS",
                human_bytes(released)
            ),
        );
    }

    // ------------------------------------------------------------------
    // Snapshot for dashboard / CLI
    // ------------------------------------------------------------------

    pub fn snapshot(&self) -> Value {
        let (info, pressure, swap) = self.sampler.lock().expect("sampler poisoned").sample();

        let (pool_stats, entries) = {
            let pool = self.pool.lock().expect("pool poisoned");
            (pool.stats(), pool.entries(12))
        };
        let ssd_mbps = {
            let mut rate = self.ssd_rate.lock().expect("rate poisoned");
            rate.update(pool_stats.ssd_read_bytes) / 1_000_000.0
        };
        let (pred_stats, pending, hot, observations) = {
            let p = self.predictor.lock().expect("predictor poisoned");
            (
                p.stats(),
                p.pending().to_vec(),
                p.hot_paths(8),
                p.observations(),
            )
        };
        let (stats, events) = {
            let s = self.stats.lock().expect("stats poisoned");
            let events = self.events.lock().expect("events poisoned").recent(30);
            (
                (
                    s.opens,
                    s.warms,
                    s.warm_avg_ms(),
                    s.warm_failures,
                    s.pool_hits,
                ),
                events,
            )
        };
        let ai = if self.config.enable_neural || self.config.enable_rag {
            self.ai.metrics()
        } else {
            json!({"available": false, "disabled": true})
        };
        let neural = ai
            .get("neural")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or_else(|| json!({"available": false}));
        let rag = ai
            .get("rag")
            .cloned()
            .filter(|v| !v.is_null())
            .unwrap_or_else(|| json!({"available": false}));
        let effective_budget = self.effective_budget.load(Ordering::SeqCst).max(0) as u64;
        let inflight = self.inflight.load(Ordering::SeqCst);

        json!({
            "ts": unix_now(),
            "uptime_s": self.started.elapsed().as_secs_f64(),
            "version": env!("CARGO_PKG_VERSION"),
            "workspace": self.config.workspace.display().to_string(),
            "model": self.config.model,
            "port": self.config.port,
            "profile": self.config.profile_name(),
            "trace_open": self.config.trace_open.as_str(),
            "demo": self.config.demo,
            "ram_limit": self.config.ram_budget,
            "effective_budget": effective_budget,
            "memory": {
                "total": info.total,
                "free": info.free,
                "cached": info.cached,
                "app": info.app,
                "wired": info.wired,
                "compressor_occupied": info.compressor_occupied,
                "compressor_data": info.compressor_data,
                "available": info.available(),
                "used_ratio": info.used_ratio(),
            },
            "pressure": {
                "level": pressure.as_str(),
                "shrink_factor": pressure.shrink_factor(),
                "swap_total": swap.total,
                "swap_used": swap.used,
            },
            "pool": {
                "budget_bytes": pool_stats.budget_bytes,
                "resident_bytes": pool_stats.resident_bytes,
                "logical_bytes": pool_stats.logical_bytes,
                "compressed_bytes": pool_stats.compressed_bytes,
                "overhead_bytes": pool_stats.overhead_bytes,
                "arena_bytes": pool_stats.arena_bytes,
                "arena_pages": pool_stats.arena_pages,
                "arena_frames": pool_stats.arena_frames,
                "dedicated_bytes": pool_stats.dedicated_bytes,
                "dedicated_frames": pool_stats.dedicated_frames,
                "virtual_bytes": pool_stats.logical_bytes,
                "saved_bytes": pool_stats.logical_bytes.saturating_sub(pool_stats.resident_bytes),
                "files": pool_stats.files,
                "hits": pool_stats.hits,
                "misses": pool_stats.misses,
                "evictions": pool_stats.evictions,
                "released_bytes": pool_stats.released_bytes,
                "ssd_read_bytes": pool_stats.ssd_read_bytes,
                "decompressions": pool_stats.decompressions,
                "warmed": pool_stats.warmed,
                "warm_ms_last": pool_stats.warm_ms_last,
                "ratio": pool_stats.ratio,
                "arc_p_ratio": pool_stats.arc_p_ratio,
                "ssd_mbps": ssd_mbps,
                "entries": entries,
            },
            "predictor": {
                "transitions": pred_stats.transitions,
                "nodes": pred_stats.nodes,
                "edges": pred_stats.edges,
                "hits": pred_stats.hits,
                "attempts": pred_stats.attempts,
                "hit_rate": pred_stats.hit_rate,
                "last": pred_stats.last,
                "pending": pending,
                "observations": observations,
                "hot": hot.into_iter().map(|(p, f)| json!({"path": p, "weight": f})).collect::<Vec<_>>(),
            },
            "activity": {
                "opens": stats.0,
                "warms": stats.1,
                "warm_avg_ms": stats.2,
                "warm_failures": stats.3,
                "pool_hits": stats.4,
                "inflight": inflight,
            },
            "neural": neural,
            "rag": rag,
            "ai": {
                "reachable": !ai.get("available").and_then(|v| v.as_bool()).is_some_and(|a| !a),
                "python": ai.get("python").cloned().unwrap_or(Value::Null),
                "error": ai.get("error").cloned().unwrap_or(Value::Null),
                "socket": self.config.ai_socket.display().to_string(),
            },
            "pids": {"mext": std::process::id()},
            "rss_bytes": crate::metrics::process_rss(),
            "events": events,
        })
    }

    // ------------------------------------------------------------------
    // RPC dispatch
    // ------------------------------------------------------------------

    pub fn handle_rpc(self: &Arc<Self>, method: &str, params: &Value) -> Result<Value, String> {
        let as_str = |key: &str| -> Option<String> {
            params
                .get(key)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        };
        match method {
            "ping" => Ok(json!({"pong": true, "version": env!("CARGO_PKG_VERSION")})),
            "status" | "snapshot" => Ok(self.snapshot()),
            "observe" => {
                let Some(path) = as_str("path") else {
                    return Err("missing `path`".into());
                };
                let source = match as_str("source").as_deref() {
                    Some("cli") | None => Source::Cli,
                    Some("demo") => Source::Demo,
                    Some("ai") => Source::Ai,
                    Some("fsevents") => Source::FsEvents,
                    Some(_) => Source::Cli,
                };
                Ok(self.observe(&self.abs(&path), source))
            }
            "predict" => {
                let path = as_str("current").unwrap_or_default();
                let k = params.get("k").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
                Ok(self.predict(&self.abs(&path), k))
            }
            "pool.warm" => {
                let paths: Vec<PathBuf> = params
                    .get("paths")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| self.abs(s))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(self.warm_paths(&paths))
            }
            "pool.read" => {
                let Some(path) = as_str("path") else {
                    return Err("missing `path`".into());
                };
                let offset = params.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
                let len = params.get("len").and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
                let data = self
                    .pool
                    .lock()
                    .map_err(|_| "pool poisoned".to_string())?
                    .read(&self.abs(&path), offset, len)
                    .map_err(|e| format!("{e}"))?;
                Ok(json!({
                    "bytes": data.len(),
                    "preview": String::from_utf8_lossy(&data[..data.len().min(400)]),
                }))
            }
            "budget" => {
                let bytes = params
                    .get("bytes")
                    .and_then(|v| v.as_u64())
                    .or_else(|| {
                        params
                            .get("size")
                            .and_then(|v| v.as_str())
                            .and_then(|s| crate::config::parse_size(s).ok())
                    })
                    .ok_or_else(|| "missing `bytes`/`size`".to_string())?;
                self.set_budget(bytes);
                Ok(json!({"budget": bytes}))
            }
            "rag.type" => {
                let prefix = as_str("prefix").unwrap_or_default();
                Ok(self
                    .ai
                    .rag_type(&prefix)
                    .unwrap_or_else(|| json!({"available": false})))
            }
            "rag.commit" => {
                let query = as_str("query").unwrap_or_default();
                let k = params.get("k").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
                Ok(self
                    .ai
                    .rag_commit(&query, k)
                    .unwrap_or_else(|| json!({"available": false})))
            }
            "ai.health" => Ok(self.ai.health()),
            "shutdown" => {
                self.request_shutdown();
                Ok(json!({"stopping": true}))
            }
            other => Err(format!("unknown method `{other}`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn workspace(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mext-core-{tag}-{}-{}",
            std::process::id(),
            unix_now() as u64
        ));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        for i in 0..6 {
            let mut f = std::fs::File::create(dir.join(format!("src/f{i}.rs"))).unwrap();
            writeln!(f, "fn f{i}() {{}}").unwrap();
        }
        dir
    }

    fn core_for(dir: &Path, budget: u64) -> Arc<Core> {
        std::env::set_var("MEXT_HOME", dir.join(".mext"));
        let config = Config::new(dir, budget).unwrap();
        Arc::new(Core::new(config).unwrap())
    }

    #[test]
    fn observe_learns_and_predicts() {
        let dir = workspace("learn");
        let core = core_for(&dir, 32 << 20);
        for _ in 0..12 {
            for i in 0..4 {
                core.observe(&dir.join(format!("src/f{i}.rs")), Source::Cli);
            }
        }
        let out = core.predict(&dir.join("src/f2.rs"), 3);
        let preds = out["predictions"].as_array().unwrap();
        assert!(!preds.is_empty());
        assert_eq!(preds[0]["path"], "src/f3.rs");
        assert!(out["hit_rate"].as_f64().unwrap() > 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn snapshot_has_all_sections() {
        let dir = workspace("snap");
        let core = core_for(&dir, 8 << 20);
        core.observe(&dir.join("src/f0.rs"), Source::Cli);
        std::thread::sleep(std::time::Duration::from_millis(150));
        let snap = core.snapshot();
        for key in [
            "memory",
            "pressure",
            "pool",
            "predictor",
            "activity",
            "events",
        ] {
            assert!(snap.get(key).is_some(), "missing {key}");
        }
        assert!(snap["pool"]["budget_bytes"].as_u64().unwrap() == 8 << 20);
        assert!(snap["memory"]["total"].as_u64().unwrap() > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn budget_rpc_and_warm_rpc() {
        let dir = workspace("rpc");
        let core = core_for(&dir, 8 << 20);
        let resp = core
            .handle_rpc("pool.warm", &json!({"paths": ["src/f0.rs"]}))
            .unwrap();
        assert_eq!(resp["warmed"], 1);
        let stats = core.handle_rpc("snapshot", &json!({})).unwrap();
        assert!(stats["pool"]["resident_bytes"].as_u64().unwrap() > 0);
        core.handle_rpc("budget", &json!({"size": "4M"})).unwrap();
        assert_eq!(core.snapshot()["pool"]["budget_bytes"], 4 << 20);
        assert!(core.handle_rpc("nope", &json!({})).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
