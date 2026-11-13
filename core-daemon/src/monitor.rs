//! Filesystem activity monitor.
//!
//! Three event sources, merged into one channel:
//!
//! 1. **FSEvents** via `notify` — always on. Reports create/modify/remove/rename.
//! 2. **`open(2)` tracing** — optional, needs root:
//!    * `eslogger open` (EndpointSecurity, macOS 12+): JSON, narrow, cheap;
//!      still requires root/TCC, and a real ES client would need an Apple
//!      entitlement, which a CLI cannot obtain — so we shell out.
//!    * `fs_usage -w -f filesys`: classic kdebug tracer with the same sudo
//!      requirement.
//! 3. **`mext open <path>` / HTTP `/api/observe`** — explicit hints from
//!    editors, wrappers or the AI engine.

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

/// Extensions we are willing to predict and warm.
const WATCHED_EXT: &[&str] = &[
    "rs",
    "py",
    "ts",
    "tsx",
    "js",
    "jsx",
    "mjs",
    "cjs",
    "go",
    "c",
    "cc",
    "cpp",
    "h",
    "hpp",
    "swift",
    "m",
    "mm",
    "java",
    "kt",
    "rb",
    "php",
    "cs",
    "scala",
    "ex",
    "exs",
    "erl",
    "hs",
    "lua",
    "zig",
    "nim",
    "jl",
    "r",
    "sql",
    "sh",
    "bash",
    "zsh",
    "toml",
    "yaml",
    "yml",
    "json",
    "jsonc",
    "ini",
    "cfg",
    "conf",
    "md",
    "mdx",
    "rst",
    "txt",
    "tex",
    "pdf",
    "csv",
    "tsv",
    "parquet",
    "arrow",
    "ipynb",
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "less",
    "svg",
    "xml",
    "proto",
    "graphql",
    "gql",
    "dockerfile",
    "makefile",
    "cmake",
    "lock",
    "env",
    "safetensors",
    "gguf",
    "npy",
    "npz",
    "bin",
    "onnx",
    "wasm",
    "log",
];

/// Directories that only create noise.
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "DerivedData",
    ".mext",
    "Library",
    ".Trash",
    "site-packages",
    ".idea",
    ".vscode",
    "Pods",
    ".terraform",
    ".gradle",
    ".dart_tool",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    FsEvents,
    TraceOpen,
    Cli,
    Demo,
    Ai,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::FsEvents => "fsevents",
            Source::TraceOpen => "trace-open",
            Source::Cli => "cli",
            Source::Demo => "demo",
            Source::Ai => "ai",
        }
    }
}

/// Which `open(2)` tracer to use (`--trace-open [backend]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceBackend {
    Off,
    Auto,
    EsLogger,
    FsUsage,
}

impl TraceBackend {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "eslogger" | "es" | "endpointsecurity" => Self::EsLogger,
            "fs_usage" | "fsusage" | "kdebug" | "fs" => Self::FsUsage,
            "off" | "none" | "no" => Self::Off,
            other => {
                eprintln!(
                    "[monitor] unknown --trace-open backend `{other}` \
                     (auto|eslogger|fs_usage|off); using auto"
                );
                Self::Auto
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::EsLogger => "eslogger",
            Self::FsUsage => "fs_usage",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FsEvent {
    pub path: PathBuf,
    pub kind: &'static str,
    pub source: Source,
}

pub struct Monitor {
    rx: Receiver<FsEvent>,
    _watcher: notify::RecommendedWatcher,
    root: PathBuf,
    tracer: TraceBackend,
}

impl Monitor {
    pub fn rx(&self) -> &Receiver<FsEvent> {
        &self.rx
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Backend that actually ended up running (for the dashboard/logs).
    pub fn tracer(&self) -> TraceBackend {
        self.tracer
    }
}

/// Start watching `workspace`. The returned struct must be kept alive.
pub fn spawn(workspace: &Path, trace: TraceBackend) -> Result<Monitor> {
    let (tx, rx) = channel::<FsEvent>();
    // FSEvents reports real paths (`/private/var/...`); canonicalise so that
    // `starts_with` and the dashboard keys match.
    let root = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());

    let mut debounce: HashMap<PathBuf, Instant> = HashMap::new();
    let root_for_cb = root.clone();
    let watcher_tx = tx.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        let Ok(event) = res else { return };
        let kind = match event.kind {
            notify::EventKind::Create(_) => "create",
            notify::EventKind::Modify(_) => "modify",
            notify::EventKind::Remove(_) => "remove",
            notify::EventKind::Access(_) => "access",
            _ => return,
        };
        for path in event.paths {
            if !is_watched(&path, &root_for_cb) {
                continue;
            }
            let now = Instant::now();
            if let Some(prev) = debounce.get(&path) {
                if now.duration_since(*prev) < Duration::from_millis(120) {
                    continue;
                }
            }
            if debounce.len() > 4096 {
                debounce.clear();
            }
            debounce.insert(path.clone(), now);
            let _ = watcher_tx.send(FsEvent {
                path,
                kind,
                source: Source::FsEvents,
            });
        }
    })
    .context("creating FSEvents watcher")?;

    watcher
        .watch(&root, RecursiveMode::Recursive)
        .with_context(|| format!("watching {}", root.display()))?;

    let tracer = match trace {
        TraceBackend::Off => TraceBackend::Off,
        TraceBackend::Auto => {
            if !sudo_available() {
                eprintln!("[monitor] open tracing off: {SUDO_HINT}");
                TraceBackend::Off
            } else if which_exists("/usr/bin/eslogger") {
                match spawn_eslogger(&root, tx.clone()) {
                    Ok(()) => TraceBackend::EsLogger,
                    Err(e) => {
                        eprintln!("[monitor] eslogger unavailable ({e}); trying fs_usage");
                        match spawn_fs_usage(&root, tx.clone()) {
                            Ok(()) => TraceBackend::FsUsage,
                            Err(e) => {
                                eprintln!("[monitor] open tracing disabled: {e}");
                                TraceBackend::Off
                            }
                        }
                    }
                }
            } else if let Err(e) = spawn_fs_usage(&root, tx.clone()) {
                eprintln!("[monitor] open tracing disabled: {e}");
                TraceBackend::Off
            } else {
                TraceBackend::FsUsage
            }
        }
        TraceBackend::EsLogger => match spawn_eslogger(&root, tx.clone()) {
            Ok(()) => TraceBackend::EsLogger,
            Err(e) => {
                eprintln!("[monitor] eslogger unavailable: {e}");
                TraceBackend::Off
            }
        },
        TraceBackend::FsUsage => match spawn_fs_usage(&root, tx.clone()) {
            Ok(()) => TraceBackend::FsUsage,
            Err(e) => {
                eprintln!("[monitor] fs_usage unavailable: {e}");
                TraceBackend::Off
            }
        },
    };

    Ok(Monitor {
        rx,
        _watcher: watcher,
        root,
        tracer,
    })
}

fn which_exists(path: &str) -> bool {
    Path::new(path).is_file()
}

/// `sudo -n true`: can we run tracers without a password prompt?
fn sudo_available() -> bool {
    Command::new("sudo")
        .args(["-n", "true"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Should this path take part in prediction/warming?
pub fn is_watched(path: &Path, root: &Path) -> bool {
    if !path.starts_with(root) {
        return false;
    }
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };
    if name.starts_with('.') && name != ".env" {
        return false;
    }
    for component in path.components() {
        if let std::path::Component::Normal(part) = component {
            if let Some(part) = part.to_str() {
                if IGNORED_DIRS.iter().any(|d| d.eq_ignore_ascii_case(part)) {
                    return false;
                }
            }
        }
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext {
        Some(ext) => WATCHED_EXT.contains(&ext.as_str()),
        None => matches!(
            name.to_ascii_lowercase().as_str(),
            "dockerfile" | "makefile" | "cmakelists.txt" | "justfile" | "brewfile"
        ),
    }
}

// ---------------------------------------------------------------------------
// open(2) tracers (both need passwordless sudo)
// ---------------------------------------------------------------------------

const SUDO_HINT: &str = "passwordless sudo is required, e.g. add to sudoers:\n  \
     admin ALL=(root) NOPASSWD: /usr/bin/eslogger open, /usr/bin/fs_usage";

/// Read the first stderr line to detect a sudo prompt. Returns `true` when the
/// tracer cannot run, so the caller can back off instead of parsing garbage.
fn privileged_stderr(stderr: Option<ChildStderr>, label: &str) -> bool {
    let Some(stderr) = stderr else { return false };
    let mut line = String::new();
    if BufReader::new(stderr).read_line(&mut line).is_err() {
        return false;
    }
    let lower = line.to_ascii_lowercase();
    if lower.contains("password")
        || lower.contains("not allowed")
        || lower.contains("a terminal is required")
    {
        eprintln!("[monitor] {label} disabled: {SUDO_HINT}");
        return true;
    }
    if !line.trim().is_empty() {
        eprintln!("[monitor] {label}: {}", line.trim());
    }
    false
}

fn spawn_eslogger(root: &Path, tx: Sender<FsEvent>) -> Result<()> {
    if !sudo_available() {
        anyhow::bail!(SUDO_HINT);
    }
    let mut child: Child = Command::new("sudo")
        .args(["-n", "/usr/bin/eslogger", "open"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning eslogger (needs passwordless sudo)")?;
    let stderr = child.stderr.take();
    let root = root.to_path_buf();
    std::thread::Builder::new()
        .name("mext-eslogger".into())
        .spawn(move || {
            if privileged_stderr(stderr, "eslogger") {
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
            let Some(stdout) = child.stdout.take() else {
                return;
            };
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Some(path) = parse_eslogger_open(&line, &root) {
                    if tx
                        .send(FsEvent {
                            path,
                            kind: "open",
                            source: Source::TraceOpen,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        })
        .context("spawning eslogger reader thread")?;
    Ok(())
}

/// Parse one JSON line from `eslogger open`.
///
/// Shape (abridged):
/// `{"event":{"open":{"path":"/Users/me/p/x.rs","flags":0}},"process":{...}}`
pub fn parse_eslogger_open(line: &str, root: &Path) -> Option<PathBuf> {
    let value: Value = serde_json::from_str(line).ok()?;
    let candidate = value
        .pointer("/event/open/path")
        .and_then(|v| v.as_str())
        .or_else(|| value.pointer("/event/path").and_then(|v| v.as_str()))
        .or_else(|| find_path_field(&value))?;
    let path = PathBuf::from(candidate);
    if is_watched(&path, root) {
        Some(path)
    } else {
        None
    }
}

fn find_path_field(value: &Value) -> Option<&str> {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(path)) = map.get("path") {
                return Some(path);
            }
            map.values().find_map(find_path_field)
        }
        Value::Array(items) => items.iter().find_map(find_path_field),
        _ => None,
    }
}

fn spawn_fs_usage(root: &Path, tx: Sender<FsEvent>) -> Result<()> {
    if !sudo_available() {
        anyhow::bail!(SUDO_HINT);
    }
    let mut child = Command::new("sudo")
        .args(["-n", "/usr/bin/fs_usage", "-w", "-f", "filesys"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning fs_usage (needs passwordless sudo)")?;
    let stderr = child.stderr.take();
    let root_str = root.display().to_string();
    let root = root.to_path_buf();
    std::thread::Builder::new()
        .name("mext-fs_usage".into())
        .spawn(move || {
            if privileged_stderr(stderr, "fs_usage") {
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
            let Some(stdout) = child.stdout.take() else {
                return;
            };
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if !line.contains("open") {
                    continue;
                }
                let Some(pos) = line.find(&root_str) else {
                    continue;
                };
                let raw = line[pos..].trim_end();
                let path = PathBuf::from(raw);
                if is_watched(&path, &root)
                    && tx
                        .send(FsEvent {
                            path,
                            kind: "open",
                            source: Source::TraceOpen,
                        })
                        .is_err()
                {
                    break;
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        })
        .context("spawning fs_usage reader thread")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_paths() {
        let root = Path::new("/tmp/ws");
        assert!(is_watched(Path::new("/tmp/ws/src/main.rs"), root));
        assert!(is_watched(Path::new("/tmp/ws/README.md"), root));
        assert!(is_watched(Path::new("/tmp/ws/run.log"), root));
        assert!(!is_watched(Path::new("/tmp/ws/target/debug/x.rs"), root));
        assert!(!is_watched(Path::new("/tmp/ws/node_modules/a/b.js"), root));
        assert!(!is_watched(Path::new("/tmp/ws/.git/config"), root));
        assert!(!is_watched(Path::new("/other/main.rs"), root));
        assert!(is_watched(Path::new("/tmp/ws/Dockerfile"), root));
    }

    #[test]
    fn parses_eslogger_lines() {
        let root = Path::new("/tmp/ws");
        let inside = r#"{"event":{"open":{"path":"/tmp/ws/src/lib.rs","flags":0}},"process":{"executable":{"path":"/usr/bin/ed"}},"seq_num":7}"#;
        assert_eq!(
            parse_eslogger_open(inside, root),
            Some(PathBuf::from("/tmp/ws/src/lib.rs"))
        );

        let outside = r#"{"event":{"open":{"path":"/etc/hosts"}}}"#;
        assert_eq!(parse_eslogger_open(outside, root), None);

        let nested = r#"{"event":{"close":{"target":{"path":"/tmp/ws/docs/a.md"}}}}"#;
        assert_eq!(
            parse_eslogger_open(nested, root),
            Some(PathBuf::from("/tmp/ws/docs/a.md"))
        );

        let ignored = r#"{"event":{"open":{"path":"/tmp/ws/target/x.rs"}}}"#;
        assert_eq!(parse_eslogger_open(ignored, root), None);
        assert_eq!(parse_eslogger_open("not json", root), None);
    }

    #[test]
    fn parses_backend_names() {
        assert_eq!(TraceBackend::parse(""), TraceBackend::Auto);
        assert_eq!(TraceBackend::parse("auto"), TraceBackend::Auto);
        assert_eq!(TraceBackend::parse("ESLogger"), TraceBackend::EsLogger);
        assert_eq!(TraceBackend::parse("fs_usage"), TraceBackend::FsUsage);
        assert_eq!(TraceBackend::parse("off"), TraceBackend::Off);
        assert_eq!(TraceBackend::parse("bogus"), TraceBackend::Auto);
        assert_eq!(TraceBackend::EsLogger.as_str(), "eslogger");
    }

    #[test]
    fn watcher_sees_new_file() {
        let dir = std::env::temp_dir().join(format!("mext-mon-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mon = spawn(&dir, TraceBackend::Off).unwrap();
        assert_eq!(mon.tracer(), TraceBackend::Off);
        std::thread::sleep(Duration::from_millis(350));
        std::fs::write(dir.join("hello.rs"), "fn main() {}").unwrap();

        let mut saw = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(ev) = mon.rx().recv_timeout(Duration::from_millis(300)) {
                if ev.path.file_name().and_then(|n| n.to_str()) == Some("hello.rs") {
                    saw = true;
                    break;
                }
            }
        }
        assert!(saw, "FSEvents did not report hello.rs");
        std::fs::remove_dir_all(&dir).ok();
    }
}
