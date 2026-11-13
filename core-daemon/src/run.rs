//! Daemon orchestration: `mext start` / `mextd`.
//!
//! Boot order matters: dashboard first (so the browser has something to load),
//! then the IPC socket, then the FSEvents monitor, then the Python AI engine.
//! Every component degrades independently.

use crate::config::{human_bytes, parse_size, Config};
use crate::core::Core;
use crate::dashboard;
use crate::ipc::{Request, RpcClient};
use crate::monitor::{self, TraceBackend};
use anyhow::{bail, Context, Result};
use clap::Args;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Args, Debug, Clone)]
pub struct StartArgs {
    /// Directory to watch, index and prefetch (`--workspace ./my_project`).
    #[arg(long = "workspace", short = 'w', value_name = "DIR")]
    pub workspace: Option<PathBuf>,

    /// Same as `--workspace`, accepted positionally (`mext start ./my_project`).
    #[arg(value_name = "DIR", hide = true)]
    pub workspace_pos: Option<PathBuf>,

    /// Hard ceiling for the compressed RAM pool (2G, 512M, 4GB, ...).
    #[arg(long = "ram-limit", alias = "ram-budget", default_value = "2G")]
    pub ram_limit: String,

    /// Model id or local path for the MLX neural loader (informational + AI engine flag).
    #[arg(long, default_value = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")]
    pub model: String,

    /// Dashboard port (http://localhost:<port>).
    #[arg(long, default_value_t = 8765)]
    pub port: u16,

    /// Simulate file activity (useful for demos and screenshots).
    #[arg(long)]
    pub demo: bool,

    /// Disable the MLX neural layer loader.
    #[arg(long = "no-neural")]
    pub no_neural: bool,

    /// Disable the LanceDB RAG engine.
    #[arg(long = "no-rag")]
    pub no_rag: bool,

    /// Do not spawn the Python AI engine at all (Rust core only).
    #[arg(long = "no-ai")]
    pub no_ai: bool,

    /// Trace real `open(2)` calls: auto | eslogger | fs_usage | off (needs sudo).
    #[arg(long = "trace-open", value_name = "BACKEND", num_args = 0..=1, default_missing_value = "auto")]
    pub trace_open: Option<String>,

    /// Python interpreter override.
    #[arg(long)]
    pub python: Option<PathBuf>,

    /// Parallel prefetch workers (NVMe queue depth).
    #[arg(long = "warm-concurrency", default_value_t = 2)]
    pub warm_concurrency: usize,

    /// Model layers to pre-stage ahead of the current one.
    #[arg(long, default_value_t = 2)]
    pub lookahead: usize,

    /// Demo the router-guided MoE expert pager on the synthetic model.
    #[arg(long)]
    pub moe: bool,

    /// Maximum file size to warm, in MiB.
    #[arg(long = "max-file-mb", default_value_t = 512)]
    pub max_file_mb: u64,
}

impl StartArgs {
    pub fn workspace(&self) -> PathBuf {
        self.workspace
            .clone()
            .or_else(|| self.workspace_pos.clone())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    pub fn to_config(&self) -> Result<Config> {
        let budget = parse_size(&self.ram_limit)
            .with_context(|| format!("invalid --ram-limit `{}`", self.ram_limit))?;
        let mut config = Config::new(self.workspace(), budget)?;
        config.model = self.model.clone();
        config.port = self.port;
        config.demo = self.demo;
        config.enable_neural = !self.no_neural && !self.no_ai;
        config.enable_rag = !self.no_rag && !self.no_ai;
        config.trace_open = self
            .trace_open
            .as_deref()
            .map(TraceBackend::parse)
            .unwrap_or(TraceBackend::Off);
        config.python = self.python.clone();
        config.warm_concurrency = self.warm_concurrency.max(1);
        config.lookahead_layers = self.lookahead.max(1);
        config.synthetic_moe = if self.moe { 2 } else { 0 };
        config.max_file_bytes = self.max_file_mb.max(1) * 1024 * 1024;
        Ok(config)
    }
}

/// Locate the `ai-engine` directory that contains the `mext_ai` package.
pub fn ai_engine_dir(config: &Config) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    roots.push(config.workspace.clone());
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..4 {
            let Some(d) = dir else { break };
            roots.push(d.clone());
            dir = d.parent().map(|p| p.to_path_buf());
        }
    }
    for root in roots {
        let mut cur = Some(root.as_path());
        while let Some(dir) = cur {
            for cand in [dir.join("ai-engine"), dir.to_path_buf()] {
                if cand.join("mext_ai/__init__.py").is_file() {
                    return Some(cand);
                }
            }
            cur = dir.parent();
        }
    }
    None
}

/// Build a `python -m mext_ai ...` command with the right `PYTHONPATH`.
pub fn python_command(config: &Config, args: &[&str]) -> Result<Command> {
    let python = config
        .resolve_python()
        .context("no Python interpreter found (create .venv with setup.sh or pass --python)")?;
    let engine = ai_engine_dir(config)
        .context("cannot find ai-engine/mext_ai next to the binary or workspace")?;
    let mut cmd = Command::new(python);
    cmd.arg("-m").arg("mext_ai");
    for a in args {
        cmd.arg(a);
    }
    let mut path = engine.display().to_string();
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        path.push(':');
        path.push_str(&existing.to_string_lossy());
    }
    cmd.env("PYTHONPATH", path);
    cmd.env("PYTHONUNBUFFERED", "1");
    Ok(cmd)
}

fn spawn_ai(config: &Config) -> Option<Child> {
    if !(config.enable_neural || config.enable_rag) {
        return None;
    }
    let mut cmd = match python_command(
        config,
        &[
            "serve",
            "--workspace",
            &config.workspace.display().to_string(),
            "--socket",
            &config.ai_socket.display().to_string(),
            "--state-dir",
            &config.ai_state().display().to_string(),
            "--ram-budget",
            &config.ram_budget.to_string(),
            "--model",
            &config.model,
            "--lookahead",
            &config.lookahead_layers.to_string(),
        ],
    ) {
        Ok(cmd) => cmd,
        Err(e) => {
            eprintln!("[mext] AI engine disabled: {e}");
            return None;
        }
    };
    if config.demo {
        cmd.arg("--demo");
    }
    if config.synthetic_moe > 0 {
        cmd.arg("--synthetic-moe")
            .arg(config.synthetic_moe.to_string());
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.log_file())
        .ok();
    match (log.as_ref().and_then(|f| f.try_clone().ok()), log) {
        (Some(stderr), Some(stdout)) => {
            cmd.stdout(Stdio::from(stdout));
            cmd.stderr(Stdio::from(stderr));
        }
        _ => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    cmd.stdin(Stdio::null());
    match cmd.spawn() {
        Ok(child) => {
            eprintln!(
                "[mext] AI engine started (pid {}) · log {}",
                child.id(),
                config.log_file().display()
            );
            Some(child)
        }
        Err(e) => {
            eprintln!("[mext] cannot start AI engine: {e} (Rust core keeps running)");
            None
        }
    }
}

/// Entry point shared by `mext start` and `mextd`.
pub async fn daemon(config: Config) -> Result<()> {
    // Refuse to start twice on the same socket.
    if let Ok(ping) =
        RpcClient::with_timeout(config.core_socket.clone(), Duration::from_millis(250))
            .call("ping", serde_json::json!({}))
    {
        bail!(
            "another mext daemon is already running on {} ({}). Use `mext stop` first.",
            config.core_socket.display(),
            ping
        );
    }

    let core = Arc::new(Core::new(config.clone())?);
    // Establish the pressure-aware budget before anyone reads it.
    core.rebalance();

    // ---- IPC -----------------------------------------------------------
    let (tx, mut rx) = mpsc::unbounded_channel::<Request>();
    let socket = config.core_socket.clone();
    tokio::spawn(async move {
        if let Err(e) = crate::ipc::serve(socket, tx).await {
            eprintln!("[mext] ipc server stopped: {e}");
        }
    });
    let rpc_core = Arc::clone(&core);
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            let core = Arc::clone(&rpc_core);
            let method = req.method.clone();
            let params = req.params.clone();
            let result = tokio::task::spawn_blocking(move || core.handle_rpc(&method, &params))
                .await
                .unwrap_or_else(|e| Err(format!("task join error: {e}")));
            let _ = req.reply.send(result);
        }
    });

    // ---- Dashboard -----------------------------------------------------
    let dash_core = Arc::clone(&core);
    tokio::spawn(async move {
        if let Err(e) = dashboard::serve(dash_core).await {
            eprintln!("[mext] dashboard stopped: {e}");
        }
    });

    // ---- Filesystem monitor -------------------------------------------
    let mon = monitor::spawn(&config.workspace, config.trace_open)?;
    let tracer = mon.tracer();
    {
        let core = Arc::clone(&core);
        std::thread::Builder::new()
            .name("mext-monitor".into())
            .spawn(move || {
                while let Ok(event) = mon.rx().recv() {
                    if core.shutdown_requested() {
                        break;
                    }
                    core.observe(&event.path, event.source);
                }
            })
            .context("spawning monitor thread")?;
    }

    // ---- Python AI engine ---------------------------------------------
    let mut ai_child = spawn_ai(&config);

    // ---- Demo generator ------------------------------------------------
    if config.demo {
        let files = crate::bench::collect_files(&config.workspace, 400, 8 * 1024 * 1024);
        crate::demo::spawn(Arc::clone(&core), files, 0x9E37_79B9_7F4A_7C15);
    }

    print_banner(&config, ai_child.as_ref().map(|c| c.id()), tracer);

    // ---- Supervision loop ----------------------------------------------
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                println!("\n[mext] shutting down…");
                break;
            }
            _ = tick.tick() => {
                core.rebalance();
                core.persist();
                if core.shutdown_requested() {
                    println!("[mext] shutdown requested via RPC");
                    break;
                }
                if let Some(child) = ai_child.as_mut() {
                    if let Ok(Some(status)) = child.try_wait() {
                        eprintln!("[mext] AI engine exited ({status}); restarting in 2s");
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        ai_child = spawn_ai(&config);
                    }
                }
            }
        }
    }

    core.persist_now();
    core.release_all();
    if let Some(mut child) = ai_child {
        let _ = child.kill();
        let _ = child.wait();
    }
    let _ = std::fs::remove_file(&config.core_socket);
    println!(
        "[mext] stopped · predictor saved to {}",
        config.predictor_state().display()
    );
    Ok(())
}

fn print_banner(config: &Config, ai_pid: Option<u32>, tracer: TraceBackend) {
    let port = config.port;
    println!();
    println!(
        "  MEXT-M1 v{} · predictive memory tiering",
        env!("CARGO_PKG_VERSION")
    );
    println!("  {}", "─".repeat(66));
    println!("  workspace     {}", config.workspace.display());
    println!(
        "  RAM pool      {} (hard cap, LZ4 + ARC)",
        human_bytes(config.ram_budget)
    );
    println!("  pressure cap  adaptive: shrinks under yellow/red macOS pressure");
    println!("  dashboard     http://localhost:{port}");
    println!("  RPC socket    {}", config.core_socket.display());
    println!(
        "  AI engine     {}",
        if config.enable_neural || config.enable_rag {
            match ai_pid {
                Some(pid) => format!(
                    "python pid {pid} · {} · {}",
                    config.ai_socket.display(),
                    if config.enable_neural && config.enable_rag {
                        "MLX layers + LanceDB RAG"
                    } else if config.enable_neural {
                        "MLX layers only"
                    } else {
                        "LanceDB RAG only"
                    }
                ),
                None => "unavailable (Rust core only)".to_string(),
            }
        } else {
            "disabled (--no-ai)".to_string()
        }
    );
    println!("  model         {}", config.model);
    println!(
        "  open tracing  {}",
        match tracer {
            TraceBackend::Off => "off (FSEvents + `mext open` hooks only)".to_string(),
            other => other.as_str().to_string(),
        }
    );
    if config.demo {
        println!("  demo mode     synthetic activity generator ON");
    }
    if config.synthetic_moe > 0 {
        println!(
            "  MoE demo      {} router-guided expert layers (top-k=2, 8 experts)",
            config.synthetic_moe
        );
    }
    println!("  {}", "─".repeat(66));
    println!("  hints: mext status · mext open <file> · mext predict <file> · curl -s localhost:{port}/api/snapshot | jq .pool");
    println!();
}

/// Resolve the core socket for CLI commands (honours `MEXT_HOME`).
pub fn default_core_socket() -> PathBuf {
    match std::env::var_os("MEXT_HOME") {
        Some(dir) => PathBuf::from(dir).join("core.sock"),
        None => {
            Path::new(&std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".mext/core.sock")
        }
    }
}

/// Local helper used by `mext doctor`.
pub fn python_module_check(config: &Config, module: &str) -> Result<String> {
    let python = config.resolve_python().context("no python")?;
    let out = Command::new(python)
        .args([
            "-c",
            &format!("import {module} as m; print(getattr(m,'__version__','?'))"),
        ])
        .output()?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(anyhow::anyhow!(
            "{}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .next()
                .unwrap_or("import failed")
        ))
    }
}
