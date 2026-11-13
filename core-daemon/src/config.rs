//! Runtime configuration and human-friendly size parsing.

use crate::monitor::TraceBackend;
use crate::predictor::Predictor;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Default dashboard port.
pub const DEFAULT_PORT: u16 = 8765;
/// Default LZ4 pool chunk size: 1 MiB. Small enough for random reads,
/// large enough to keep the LZ4 ratio decent.
pub const DEFAULT_CHUNK: usize = 1024 * 1024;
/// Files larger than this are not warmed (they would evict the whole pool).
pub const DEFAULT_MAX_FILE: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct Config {
    pub workspace: PathBuf,
    pub ram_budget: u64,
    pub model: String,
    pub port: u16,
    pub state_dir: PathBuf,
    pub chunk_size: usize,
    pub max_file_bytes: u64,
    pub demo: bool,
    pub enable_neural: bool,
    pub enable_rag: bool,
    pub python: Option<PathBuf>,
    pub ai_socket: PathBuf,
    pub core_socket: PathBuf,
    pub trace_open: TraceBackend,
    pub warm_concurrency: usize,
    pub lookahead_layers: usize,
    /// Turn the last N synthetic layers into router-guided MoE blocks.
    pub synthetic_moe: usize,
}

impl Config {
    /// Build a config, resolving paths (`~`, `$MEXT_HOME`, `./.venv`) and
    /// creating the state directory.
    pub fn new(workspace: impl AsRef<Path>, ram_budget: u64) -> Result<Self> {
        let workspace = workspace.as_ref();
        let workspace = if workspace.exists() {
            workspace
                .canonicalize()
                .with_context(|| format!("cannot resolve workspace {}", workspace.display()))?
        } else {
            bail!("workspace does not exist: {}", workspace.display());
        };

        let state_dir = match std::env::var_os("MEXT_HOME") {
            Some(v) => expand_path(&v.to_string_lossy()),
            None => home_dir()?.join(".mext"),
        };
        std::fs::create_dir_all(&state_dir)
            .with_context(|| format!("cannot create {}", state_dir.display()))?;

        Ok(Self {
            workspace,
            ram_budget,
            model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".to_string(),
            port: DEFAULT_PORT,
            core_socket: state_dir.join("core.sock"),
            ai_socket: state_dir.join("ai.sock"),
            state_dir,
            chunk_size: DEFAULT_CHUNK,
            max_file_bytes: DEFAULT_MAX_FILE,
            demo: false,
            enable_neural: true,
            enable_rag: true,
            python: None,
            trace_open: TraceBackend::Off,
            warm_concurrency: 2,
            lookahead_layers: 2,
            synthetic_moe: 0,
        })
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_demo(mut self, demo: bool) -> Self {
        self.demo = demo;
        self
    }

    /// Stable, human-readable profile id for this workspace.
    ///
    /// Prediction graphs are per project: `~/code/api` and `~/data/corpus`
    /// share neither file names nor access order. `MEXT_PROFILE` overrides.
    pub fn profile_name(&self) -> String {
        if let Ok(raw) = std::env::var("MEXT_PROFILE") {
            let slug = sanitize_slug(&raw);
            if !slug.is_empty() {
                return slug;
            }
        }
        let base = self
            .workspace
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("workspace");
        let hash = fnv1a(self.workspace.to_string_lossy().as_bytes()) & 0xffff;
        format!("{}-{hash:04x}", sanitize_slug(base))
    }

    pub fn profiles_dir(&self) -> PathBuf {
        self.state_dir.join("profiles")
    }

    pub fn predictor_state(&self) -> PathBuf {
        self.profiles_dir()
            .join(self.profile_name())
            .join("predictor.json")
    }

    /// All learned profiles, newest first.
    pub fn list_profiles(&self) -> Vec<ProfileInfo> {
        list_profiles_in(&self.profiles_dir())
    }

    pub fn load_profile(&self, name: &str) -> Result<Predictor> {
        Predictor::load(&profile_path(&self.profiles_dir(), name)?)
    }

    pub fn save_profile(&self, name: &str, predictor: &Predictor) -> Result<()> {
        predictor.save(&profile_path(&self.profiles_dir(), name)?)
    }

    pub fn ai_state(&self) -> PathBuf {
        self.state_dir.join("ai")
    }

    pub fn log_file(&self) -> PathBuf {
        self.state_dir.join("mext.log")
    }

    /// Locate a Python interpreter that has the `mext_ai` package installed.
    ///
    /// Order: explicit `--python`, `$MEXT_PYTHON`, repo-local `.venv`,
    /// `python3.12`/`python3.11`, then `python3`.
    pub fn resolve_python(&self) -> Option<PathBuf> {
        if let Some(p) = &self.python {
            return Some(p.clone());
        }
        if let Some(p) = std::env::var_os("MEXT_PYTHON") {
            return Some(PathBuf::from(p));
        }
        // Walk up from the workspace and from the binary looking for .venv.
        let mut candidates: Vec<PathBuf> = Vec::new();
        let exe = std::env::current_exe().ok();
        for base in [
            Some(self.workspace.as_path()),
            exe.as_deref().and_then(Path::parent),
        ] {
            let mut cur = base;
            while let Some(dir) = cur {
                candidates.push(dir.join(".venv/bin/python"));
                candidates.push(dir.join("ai-engine/.venv/bin/python"));
                cur = dir.parent();
            }
        }
        for c in &candidates {
            if c.is_file() {
                return Some(c.clone());
            }
        }
        for name in ["python3.12", "python3.11", "python3"] {
            if let Some(p) = which(name) {
                return Some(p);
            }
        }
        None
    }
}

/// One learned predictor profile on disk.
#[derive(Debug, Clone, Serialize)]
pub struct ProfileInfo {
    pub name: String,
    pub path: String,
    pub observations: u64,
    pub transitions: u64,
    pub hit_rate: f64,
    pub edges: usize,
    pub nodes: usize,
    pub bytes: u64,
    pub modified_ms: u64,
}

/// Scan `profiles/*/predictor.json`, skipping unreadable or corrupt files.
pub fn list_profiles_in(dir: &Path) -> Vec<ProfileInfo> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        let path = entry.path().join("predictor.json");
        let Ok(file_meta) = std::fs::metadata(&path) else {
            continue;
        };
        let Ok(predictor) = Predictor::load(&path) else {
            continue;
        };
        let modified_ms = file_meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        out.push(ProfileInfo {
            name: entry.file_name().to_string_lossy().to_string(),
            path: path.display().to_string(),
            observations: predictor.observations(),
            transitions: predictor.transitions,
            hit_rate: predictor.hit_rate(),
            edges: predictor.edge_count(),
            nodes: predictor.node_count(),
            bytes: file_meta.len(),
            modified_ms,
        });
    }
    out.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms));
    out
}

/// Public helper so the CLI can compare user input with stored directory names.
pub fn normalize_profile_name(name: &str) -> String {
    sanitize_slug(name)
}

fn profile_path(dir: &Path, name: &str) -> Result<PathBuf> {
    let slug = sanitize_slug(name);
    if slug.is_empty() {
        bail!("profile name is empty after sanitizing: {name:?}");
    }
    Ok(dir.join(slug).join("predictor.json"))
}

fn home_dir() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")
}

fn sanitize_slug(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
        if out.len() >= 32 {
            break;
        }
    }
    out.trim_matches(|c| c == '-' || c == '_').to_string()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

pub fn expand_path(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Ok(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(raw)
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// Parse `2GB`, `4G`, `512M`, `1.5G`, `1048576` (bytes).
pub fn parse_size(input: &str) -> Result<u64> {
    let s = input.trim();
    if s.is_empty() {
        bail!("empty size");
    }
    let mut num = String::new();
    let mut unit = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() || ch == '.' || ch == '_' {
            if !unit.is_empty() {
                bail!("invalid size `{input}`: number after unit");
            }
            if ch != '_' {
                num.push(ch);
            }
        } else {
            unit.push(ch);
        }
    }
    let value: f64 = num
        .parse()
        .with_context(|| format!("invalid number in size `{input}`"))?;
    let mult: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" | "ki" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => bail!("unknown size unit `{other}` in `{input}`"),
    };
    let bytes = value * mult;
    if bytes <= 0.0 {
        bail!("size must be positive");
    }
    Ok(bytes as u64)
}

/// `1.4 GiB` / `812 MiB` / `12 KiB` — used by the CLI and the dashboard.
pub fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= KIB * KIB * KIB {
        format!("{:.2} GiB", b / (KIB * KIB * KIB))
    } else if b >= KIB * KIB {
        format!("{:.1} MiB", b / (KIB * KIB))
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

pub fn parse_bool(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "y"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_size("2GB").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("4G").unwrap(), 4 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("512M").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_size("1.5g").unwrap(), 3 * 512 * 1024 * 1024);
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("1 Ki").unwrap(), 1024);
        assert!(parse_size("abc").is_err());
        assert!(parse_size("12XB").is_err());
    }

    #[test]
    fn profile_is_per_workspace_and_safe() {
        std::env::remove_var("MEXT_PROFILE");
        let a = Config::new("/tmp", 1 << 20).unwrap();
        let slug = a.profile_name();
        assert!(slug.starts_with("tmp-"), "{slug}");
        assert!(slug
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert!(a.predictor_state().ends_with("predictor.json"));
        assert!(a.predictor_state().to_string_lossy().contains("/profiles/"));

        assert_eq!(sanitize_slug("my project/one"), "my-project-one");
        assert_eq!(sanitize_slug("__weird//name__"), "weird-name");
        assert!(sanitize_slug(&"x".repeat(80)).len() <= 32);
    }

    #[test]
    fn lists_profiles_and_skips_corrupt() {
        let dir = std::env::temp_dir().join(format!("mext-profiles-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("alpha")).unwrap();
        std::fs::create_dir_all(dir.join("broken")).unwrap();
        let mut p = Predictor::new();
        p.observe("a", 9);
        p.observe("b", 9);
        p.save(&dir.join("alpha/predictor.json")).unwrap();
        std::fs::write(dir.join("broken/predictor.json"), b"{not json").unwrap();

        let profiles = list_profiles_in(&dir);
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].name, "alpha");
        assert_eq!(profiles[0].transitions, 1);
        assert!(profiles[0].bytes > 0);
        assert_eq!(normalize_profile_name("My Project/One"), "my-project-one");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn humanizes() {
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1024 * 1024), "1.0 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024 / 2), "1.50 GiB");
    }
}
