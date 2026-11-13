//! `mext` — the single CLI entry point for MEXT-M1.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use mext_core::bench::{self, BenchOptions};
use mext_core::config::{human_bytes, normalize_profile_name, parse_size, Config};
use mext_core::ipc::RpcClient;
use mext_core::platform::{memory_info, memory_pressure, swap_usage};
use mext_core::predictor::Predictor;
use mext_core::run::{self, StartArgs};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "mext",
    version,
    about = "MEXT-M1 — predictive memory tiering for Apple Silicon",
    long_about = "Userspace analogue of AMD MEXT for macOS/M1: a predictive LZ4 compressed \
                  RAM pool, an MLX layer streamer and a zero-RAM LanceDB RAG, all behind one \
                  CLI and a live dashboard."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the daemon (Rust core + dashboard + Python AI engine).
    Start(StartArgs),
    /// Print a live snapshot of the running daemon.
    Status {
        #[arg(long)]
        socket: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Tell the predictor that a file was opened (editor / wrapper hook).
    Open {
        path: PathBuf,
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Predict the next files after <path>.
    Predict {
        path: PathBuf,
        #[arg(short, long, default_value_t = 4)]
        k: usize,
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Warm a directory (or a single file) into the compressed pool.
    Warm {
        path: PathBuf,
        #[arg(long, default_value_t = 400)]
        limit: usize,
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Offline benchmark: OS page cache baseline vs MEXT compressed pool.
    Bench {
        #[arg(default_value = ".")]
        workspace: PathBuf,
        #[arg(long = "ram-limit", alias = "ram-budget", default_value = "2G")]
        ram_limit: String,
        #[arg(long, default_value_t = 256)]
        files: usize,
        #[arg(long, default_value_t = 3)]
        passes: usize,
        #[arg(long)]
        json: bool,
    },
    /// Build or refresh the LanceDB index (delegates to the Python engine).
    Index {
        #[arg(default_value = ".")]
        workspace: PathBuf,
        #[arg(long)]
        python: Option<PathBuf>,
    },
    /// Ask the RAG engine a question from the terminal.
    Query {
        query: String,
        #[arg(short, long, default_value_t = 8)]
        k: usize,
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Environment diagnostics (Rust, Python, MLX, LanceDB, pressure).
    Doctor {
        #[arg(long)]
        json: bool,
    },
    /// Stop the running daemon.
    Stop {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Inspect and merge per-workspace predictor profiles (no daemon needed).
    Profiles {
        #[command(subcommand)]
        action: Option<ProfilesCmd>,
    },
}

#[derive(Subcommand)]
enum ProfilesCmd {
    /// List learned profiles, newest first (default when no action is given).
    List {
        #[arg(long)]
        json: bool,
    },
    /// Show the statistics of one profile.
    Show {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Merge every other profile into <into>, scaling their evidence by weight.
    Merge {
        #[arg(long)]
        into: String,
        #[arg(long, default_value_t = 0.35)]
        weight: f32,
        /// Keep the source profiles on disk.
        #[arg(long)]
        keep: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Start(args) => {
            let config = args.to_config()?;
            run::daemon(config).await
        }
        Cmd::Status { socket, json } => {
            let client = client(socket);
            let snap = client
                .call("status", json!({}))
                .context("daemon is not running (start it with `mext start`)")?;
            if json {
                println!("{}", serde_json::to_string_pretty(&snap)?);
            } else {
                print_status(&snap);
            }
            Ok(())
        }
        Cmd::Open { path, socket } => {
            let abs = absolutize(&path)?;
            let resp = client(socket).call("observe", json!({"path": abs, "source": "cli"}))?;
            if resp.get("accepted").and_then(|v| v.as_bool()) == Some(false) {
                bail!(
                    "file is not watched ({}). It must live under the workspace and be a known type.",
                    resp.get("reason").cloned().unwrap_or(json!("unknown"))
                );
            }
            println!("opened {}", resp["key"]);
            if let Some(preds) = resp["predictions"].as_array() {
                println!("next:");
                for p in preds {
                    println!(
                        "  {:>5.1}%  {}  ({})",
                        p["score"].as_f64().unwrap_or(0.0) * 100.0,
                        p["path"].as_str().unwrap_or("?"),
                        p["reason"].as_str().unwrap_or("")
                    );
                }
            }
            Ok(())
        }
        Cmd::Predict { path, k, socket } => {
            let abs = absolutize(&path)?;
            let resp = client(socket).call("predict", json!({"current": abs, "k": k}))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&resp["predictions"]).unwrap_or_default()
            );
            println!(
                "predictor hit rate: {:.1}%",
                resp["hit_rate"].as_f64().unwrap_or(0.0) * 100.0
            );
            Ok(())
        }
        Cmd::Warm {
            path,
            limit,
            socket,
        } => {
            let abs = absolutize(&path)?;
            let files = if abs.is_file() {
                vec![abs.clone()]
            } else {
                mext_core::bench::collect_files(&abs, limit.max(1), 512 * 1024 * 1024)
            };
            if files.is_empty() {
                bail!("no warmable files under {}", abs.display());
            }
            let client = RpcClient::with_timeout(
                socket.unwrap_or_else(run::default_core_socket),
                Duration::from_secs(300),
            );
            let mut warmed = 0u64;
            let mut hits = 0u64;
            let mut bytes = 0u64;
            for chunk in files.chunks(64) {
                let paths: Vec<String> = chunk.iter().map(|p| p.display().to_string()).collect();
                let resp = client.call("pool.warm", json!({"paths": paths}))?;
                warmed += resp["warmed"].as_u64().unwrap_or(0);
                hits += resp["hits"].as_u64().unwrap_or(0);
                bytes += resp["bytes"].as_u64().unwrap_or(0);
            }
            println!(
                "warmed {} files ({} new, {} already resident) · {} of data",
                files.len(),
                warmed,
                hits,
                human_bytes(bytes)
            );
            let snap = client.call("status", json!({}))?;
            println!(
                "pool now: {} resident · {} virtual · {:.2}× · {} files · {} evictions · {} released",
                human_bytes(snap["pool"]["resident_bytes"].as_u64().unwrap_or(0)),
                human_bytes(snap["pool"]["logical_bytes"].as_u64().unwrap_or(0)),
                snap["pool"]["ratio"].as_f64().unwrap_or(0.0),
                snap["pool"]["files"].as_u64().unwrap_or(0),
                snap["pool"]["evictions"].as_u64().unwrap_or(0),
                human_bytes(snap["pool"]["released_bytes"].as_u64().unwrap_or(0))
            );
            Ok(())
        }
        Cmd::Bench {
            workspace,
            ram_limit,
            files,
            passes,
            json,
        } => {
            let budget = parse_size(&ram_limit)?;
            let mut opts = BenchOptions::new(&workspace, budget);
            opts.max_files = files.max(1);
            opts.passes = passes.max(1);
            let report = bench::run(&opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            Ok(())
        }
        Cmd::Index { workspace, python } => {
            let mut config = Config::new(&workspace, 2 << 30)?;
            if let Some(p) = python {
                config.python = Some(p);
            }
            let engine = run::ai_engine_dir(&config)
                .context("ai-engine/mext_ai not found near the binary or workspace")?;
            eprintln!(
                "[mext] indexing {} with {}",
                config.workspace.display(),
                engine.display()
            );
            let status = run::python_command(
                &config,
                &[
                    "index",
                    "--workspace",
                    &config.workspace.display().to_string(),
                ],
            )?
            .status()?;
            if !status.success() {
                bail!("index command failed with {status}");
            }
            Ok(())
        }
        Cmd::Query { query, k, socket } => {
            let resp = client(socket).call("rag.commit", json!({"query": query, "k": k}))?;
            if resp.get("available").and_then(|v| v.as_bool()) == Some(false) {
                bail!(
                    "RAG engine unavailable: {}",
                    resp.get("error").cloned().unwrap_or(json!("unknown"))
                );
            }
            if let Some(results) = resp["results"].as_array() {
                for (i, r) in results.iter().enumerate() {
                    println!(
                        "{:>2}. {:.3}  {}  #{}",
                        i + 1,
                        r["score"].as_f64().unwrap_or(0.0),
                        r["path"].as_str().unwrap_or("?"),
                        r["chunk"].as_u64().unwrap_or(0)
                    );
                    let text = r["text"].as_str().unwrap_or("");
                    let preview: String = text.chars().take(180).collect();
                    println!("    {}", preview.replace('\n', " "));
                }
            }
            println!(
                "latency {:.1} ms · prefetch hit {}",
                resp["latency_ms"].as_f64().unwrap_or(0.0),
                resp["prefetch_hit"].as_bool().unwrap_or(false)
            );
            Ok(())
        }
        Cmd::Doctor { json } => doctor(json),
        Cmd::Profiles { action } => profiles_command(action),
        Cmd::Stop { socket } => {
            let resp = client(socket).call("shutdown", json!({}))?;
            println!("{}", serde_json::to_string(&resp)?);
            Ok(())
        }
    }
}

fn profiles_config() -> Result<Config> {
    Config::new(".", 2 << 30)
}

fn profiles_command(action: Option<ProfilesCmd>) -> Result<()> {
    let config = profiles_config()?;
    match action.unwrap_or(ProfilesCmd::List { json: false }) {
        ProfilesCmd::List { json } => profiles_list(&config, json),
        ProfilesCmd::Show { name, json } => profiles_show(&config, &name, json),
        ProfilesCmd::Merge { into, weight, keep } => profiles_merge(&config, &into, weight, keep),
    }
}

fn profiles_list(config: &Config, json: bool) -> Result<()> {
    let profiles = config.list_profiles();
    if json {
        println!("{}", serde_json::to_string_pretty(&profiles)?);
        return Ok(());
    }
    if profiles.is_empty() {
        println!(
            "no profiles yet · they appear under {} after `mext start`",
            config.profiles_dir().display()
        );
        return Ok(());
    }
    println!(
        "  {:<28} {:>7} {:>7} {:>8} {:>9}  modified",
        "profile", "obs", "edges", "hit %", "size"
    );
    for profile in profiles {
        let modified = if profile.modified_ms == 0 {
            "-".to_string()
        } else {
            format!(
                "{}s ago",
                (mext_core::metrics::unix_now() * 1000.0 - profile.modified_ms as f64).max(0.0)
                    as u64
                    / 1000
            )
        };
        println!(
            "  {:<28} {:>7} {:>7} {:>7.1}% {:>9}  {}",
            profile.name,
            profile.observations,
            profile.edges,
            profile.hit_rate * 100.0,
            human_bytes(profile.bytes),
            modified
        );
    }
    Ok(())
}

fn profiles_show(config: &Config, name: &str, json: bool) -> Result<()> {
    let predictor = config
        .load_profile(name)
        .with_context(|| format!("profile `{name}` not found (try `mext profiles list`)"))?;
    let stats = predictor.stats();
    if json {
        let value = json!({
            "name": normalize_profile_name(name),
            "stats": stats,
            "edges": predictor.edge_count(),
            "nodes": predictor.node_count(),
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    println!("profile {}", normalize_profile_name(name));
    println!("  observations  {}", predictor.observations());
    println!("  transitions   {}", predictor.transitions);
    println!(
        "  predictions   {} hits / {} scored ({:.1}%)",
        predictor.hits,
        predictor.attempts,
        predictor.hit_rate() * 100.0
    );
    println!(
        "  graph         {} edges · {} nodes",
        predictor.edge_count(),
        predictor.node_count()
    );
    Ok(())
}

fn profiles_merge(config: &Config, into: &str, weight: f32, keep: bool) -> Result<()> {
    if !weight.is_finite() || !(0.0..=10.0).contains(&weight) {
        bail!("--weight must be a finite number between 0 and 10 (got {weight})");
    }
    let target_name = normalize_profile_name(into);
    if target_name.is_empty() {
        bail!("invalid profile name: {into:?}");
    }

    let mut target = config
        .load_profile(&target_name)
        .unwrap_or_else(|_| Predictor::default());
    let before = (
        target.transitions,
        target.observations(),
        target.edge_count(),
    );
    let mut merged = 0usize;
    let mut removed = 0usize;

    for profile in config.list_profiles() {
        if profile.name == target_name {
            continue;
        }
        let other = match config.load_profile(&profile.name) {
            Ok(value) => value,
            Err(error) => {
                eprintln!(
                    "[mext] skipping unreadable profile {}: {error}",
                    profile.name
                );
                continue;
            }
        };
        target.merge_from(&other, weight);
        merged += 1;
        if !keep && std::fs::remove_dir_all(config.profiles_dir().join(&profile.name)).is_ok() {
            removed += 1;
        }
    }

    config.save_profile(&target_name, &target)?;
    println!("merged {merged} profile(s) into `{target_name}` (source weight {weight:.2})");
    println!("  transitions   {} → {}", before.0, target.transitions);
    println!("  observations  {} → {}", before.1, target.observations());
    println!("  edges         {} → {}", before.2, target.edge_count());
    if removed > 0 {
        println!("  removed {removed} source profile(s); pass --keep to retain them");
    }
    if merged == 0 {
        println!("  (no other profiles found — target saved as-is)");
    }
    Ok(())
}

fn client(socket: Option<PathBuf>) -> RpcClient {
    let path = socket.unwrap_or_else(run::default_core_socket);
    RpcClient::with_timeout(path, Duration::from_secs(10))
}

fn absolutize(path: &PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.clone())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn print_status(s: &Value) {
    let n = |p: &str| s.pointer(p).and_then(|v| v.as_u64()).unwrap_or(0);
    let f = |p: &str| s.pointer(p).and_then(|v| v.as_f64()).unwrap_or(0.0);
    let txt = |p: &str| {
        s.pointer(p)
            .and_then(|v| v.as_str())
            .unwrap_or("-")
            .to_string()
    };
    let uptime = n("/uptime_s");
    let mem_used = f("/memory/used_ratio") * 100.0;
    let pressure = txt("/pressure/level");
    println!(
        "MEXT-M1 v{} · pid {} · up {}m {:02}s",
        txt("/version"),
        n("/pids/mext"),
        uptime / 60,
        uptime % 60
    );
    println!("  workspace  {}", txt("/workspace"));
    println!(
        "  memory     {:.0}% used · free {} · swap {} · pressure {}",
        mem_used,
        human_bytes(n("/memory/available")),
        human_bytes(n("/pressure/swap_used")),
        pressure.to_uppercase()
    );
    println!(
        "  pool       cap {} · resident {} · virtual {} · {:.2}× · {} files · {} hits / {} misses",
        human_bytes(n("/pool/budget_bytes")),
        human_bytes(n("/pool/resident_bytes")),
        human_bytes(n("/pool/logical_bytes")),
        f("/pool/ratio"),
        n("/pool/files"),
        n("/pool/hits"),
        n("/pool/misses")
    );
    println!(
        "  predictor  {:.1}% hit rate ({} / {}) · {} edges · {} nodes · last {}",
        f("/predictor/hit_rate") * 100.0,
        n("/predictor/hits"),
        n("/predictor/attempts"),
        n("/predictor/edges"),
        n("/predictor/nodes"),
        txt("/predictor/last")
    );
    println!(
        "  activity   {} opens · {} warms · avg {:.1} ms · inflight {} · SSD {:.1} MB/s · released {}",
        n("/activity/opens"),
        n("/activity/warms"),
        f("/activity/warm_avg_ms"),
        n("/activity/inflight"),
        f("/pool/ssd_mbps"),
        human_bytes(n("/pool/released_bytes"))
    );
    let neural_ok = s
        .pointer("/neural/available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if neural_ok {
        println!(
            "  neural     hot {} · cold {} · resident {} · {:.1} tok/s · lookahead {}",
            human_bytes(
                s.pointer("/neural/hot_bytes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
            ),
            human_bytes(
                s.pointer("/neural/cold_bytes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
            ),
            human_bytes(
                s.pointer("/neural/resident_bytes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
            ),
            s.pointer("/neural/tokens_per_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            n("/neural/lookahead")
        );
    } else {
        println!("  neural     unavailable (see `mext doctor`)");
    }
    let rag_ok = s
        .pointer("/rag/available")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if rag_ok {
        println!(
            "  rag        {} chunks · {} clusters · prefetch {:.0}% · {:.1} ms",
            n("/rag/chunks"),
            n("/rag/clusters"),
            s.pointer("/rag/prefetch_hit_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                * 100.0,
            s.pointer("/rag/latency_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        );
    } else {
        println!("  rag        unavailable (see `mext doctor`)");
    }
    if let Some(events) = s["events"].as_array() {
        println!("  events");
        for e in events.iter().rev().take(5) {
            println!(
                "    {:>9}  {:>10}  {}  {}",
                e["kind"].as_str().unwrap_or(""),
                e["ts"]
                    .as_f64()
                    .map(|t| {
                        let secs = (mext_core::metrics::unix_now() - t).max(0.0);
                        format!("{secs:.0}s ago")
                    })
                    .unwrap_or_default(),
                e["path"].as_str().unwrap_or(""),
                e["detail"].as_str().unwrap_or("")
            );
        }
    }
    let port = n("/port");
    println!(
        "  dashboard  http://localhost:{}",
        if port == 0 { 8765 } else { port }
    );
}

fn doctor(json_out: bool) -> Result<()> {
    let mut checks: Vec<Value> = Vec::new();
    let mut add = |name: &str, ok: bool, detail: String| {
        checks.push(json!({"check": name, "ok": ok, "detail": detail}));
    };

    let arch = std::env::consts::ARCH;
    add(
        "architecture",
        arch == "aarch64",
        format!("{arch} · {}", std::env::consts::OS),
    );

    let info = memory_info()?;
    add(
        "physical RAM",
        info.total >= 4 * 1024 * 1024 * 1024,
        human_bytes(info.total),
    );
    let pressure = memory_pressure(Some(&info));
    let swap = swap_usage();
    add(
        "memory pressure",
        pressure.as_str() != "red",
        format!(
            "{} · free {} · swap {} / {}",
            pressure.as_str(),
            human_bytes(info.available()),
            human_bytes(swap.used),
            human_bytes(swap.total)
        ),
    );

    let config = Config::new(".", 2 << 30)?;
    let python = config.resolve_python();
    add(
        "python",
        python.is_some(),
        python
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found — run ./setup.sh".into()),
    );
    for module in ["mlx.core", "lancedb", "psutil", "numpy"] {
        let (ok, detail) = match run::python_module_check(&config, module) {
            Ok(v) => (true, v),
            Err(e) => (false, e.to_string()),
        };
        add(&format!("python module {module}"), ok, detail);
    }
    match run::ai_engine_dir(&config) {
        Some(dir) => add("ai-engine", true, dir.display().to_string()),
        None => add(
            "ai-engine",
            false,
            "ai-engine/mext_ai not found (run from the repo root or rebuild)".into(),
        ),
    }

    let socket = run::default_core_socket();
    let reachable = RpcClient::with_timeout(socket.clone(), Duration::from_millis(250)).reachable();
    add(
        "core socket",
        true,
        format!(
            "{} ({})",
            socket.display(),
            if reachable {
                "daemon running"
            } else {
                "daemon stopped"
            }
        ),
    );

    if json_out {
        println!("{}", serde_json::to_string_pretty(&checks)?);
    } else {
        println!("\n  MEXT-M1 doctor\n  {}", "─".repeat(64));
        for c in &checks {
            let ok = c["ok"].as_bool().unwrap_or(false);
            println!(
                "  {} {:<32} {}",
                if ok { "✓" } else { "✗" },
                c["check"].as_str().unwrap_or("?"),
                c["detail"].as_str().unwrap_or("")
            );
        }
        println!("  {}", "─".repeat(64));
        let failed = checks
            .iter()
            .filter(|c| c["ok"].as_bool() != Some(true))
            .count();
        if failed == 0 {
            println!("  everything is in place — `mext start --demo`\n");
        } else {
            println!("  {failed} check(s) failed. Run ./setup.sh to fix the Python side.\n");
        }
    }
    Ok(())
}
