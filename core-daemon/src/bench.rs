//! Benchmark: cold FS access vs MEXT compressed pool.
//!
//! The question we answer is *not* "is LZ4 faster than reading from page cache"
//! (it usually is not, ~0.3 ms/MiB vs ~0.05 ms/MiB of RAM copy). It is:
//!
//! > how much physical RAM do we need to keep a working set resident, and what
//! > does that cost in latency and CPU?
//!
//! Baseline = the OS page cache approach: every file you want hot must live in
//! RAM at full size. MEXT = the same files live in RAM compressed, so the RAM
//! bill drops by the LZ4 ratio.

use crate::config::{human_bytes, parse_size};
use crate::metrics::process_rss;
use crate::monitor::is_watched;
use crate::pool::CompressedMemoryPool;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub workspace: PathBuf,
    pub budget: u64,
    pub max_files: usize,
    pub passes: usize,
    pub chunk_size: usize,
    pub max_file_bytes: u64,
}

impl BenchOptions {
    pub fn new(workspace: impl AsRef<Path>, budget: u64) -> Self {
        Self {
            workspace: workspace.as_ref().to_path_buf(),
            budget,
            max_files: 256,
            passes: 3,
            chunk_size: crate::config::DEFAULT_CHUNK,
            max_file_bytes: 64 * 1024 * 1024,
        }
    }
}

pub fn collect_files(root: &Path, max_files: usize, max_bytes: u64) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file()
                && meta.len() > 0
                && meta.len() <= max_bytes
                && is_watched(&path, root)
            {
                out.push(path);
                if out.len() >= max_files {
                    out.sort();
                    return out;
                }
            }
        }
    }
    out.sort();
    out
}

fn checksum(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut i = 0;
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
        i += 1024;
    }
    hash
}

/// Run the benchmark, print a human table and return machine-readable JSON.
pub fn run(opts: &BenchOptions) -> Result<Value> {
    let files = collect_files(&opts.workspace, opts.max_files, opts.max_file_bytes);
    anyhow::ensure!(
        !files.is_empty(),
        "no files to benchmark in {}",
        opts.workspace.display()
    );
    let total_bytes: u64 = files
        .iter()
        .map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0))
        .sum();

    // ---- baseline: OS page cache (files at full size) -------------------
    let rss_before = process_rss();
    let started = Instant::now();
    let mut baseline_checksum = 0u64;
    for _ in 0..opts.passes {
        for f in &files {
            let data = std::fs::read(f).with_context(|| format!("read {}", f.display()))?;
            baseline_checksum ^= checksum(&data);
        }
    }
    let baseline = started.elapsed();
    let baseline_ms = baseline.as_secs_f64() * 1000.0;

    // ---- MEXT: compressed pool -----------------------------------------
    let mut pool = CompressedMemoryPool::new(opts.budget, opts.chunk_size, opts.max_file_bytes);
    let warm_started = Instant::now();
    let mut skipped = 0usize;
    for f in &files {
        let res = pool.warm(f)?;
        if res.outcome == crate::pool::WarmOutcome::Skipped {
            skipped += 1;
        }
    }
    let warm_ms = warm_started.elapsed().as_secs_f64() * 1000.0;
    let after_warm = pool.stats();

    let started = Instant::now();
    let mut mext_checksum = 0u64;
    let mut reads = 0u64;
    for _ in 0..opts.passes {
        for f in &files {
            let data = pool.read_all(f)?;
            mext_checksum ^= checksum(&data);
            reads += 1;
        }
    }
    let mext = started.elapsed();
    let mext_ms = mext.as_secs_f64() * 1000.0;
    let stats = pool.stats();
    let rss_after = process_rss();

    let baseline_mbps = (total_bytes * opts.passes as u64) as f64 / 1e6 / baseline.as_secs_f64();
    let mext_mbps = (total_bytes * opts.passes as u64) as f64 / 1e6 / mext.as_secs_f64();
    let speed_ratio = baseline_ms / mext_ms.max(0.001);
    let ram_saved = after_warm
        .logical_bytes
        .saturating_sub(after_warm.resident_bytes);
    let ram_saved_pct = if after_warm.logical_bytes == 0 {
        0.0
    } else {
        ram_saved as f64 / after_warm.logical_bytes as f64 * 100.0
    };

    let report = json!({
        "files": files.len(),
        "skipped": skipped,
        "working_set_bytes": total_bytes,
        "budget_bytes": opts.budget,
        "passes": opts.passes,
        "baseline": {
            "ms": baseline_ms,
            "mbps": baseline_mbps,
            "ram_bytes": total_bytes,
            "checksum": baseline_checksum,
        },
        "mext": {
            "ms": mext_ms,
            "mbps": mext_mbps,
            "warm_ms": warm_ms,
            "resident_bytes": after_warm.resident_bytes,
            "logical_bytes": after_warm.logical_bytes,
            "compression_ratio": after_warm.ratio,
            "evictions": after_warm.evictions,
            "hits": stats.hits,
            "misses": stats.misses,
            "decompressions": stats.decompressions,
            "reads": reads,
            "released_bytes": after_warm.released_bytes,
            "checksum": mext_checksum,
        },
        "speed_ratio": speed_ratio,
        "ram_saved_bytes": ram_saved,
        "ram_saved_pct": ram_saved_pct,
        "rss_bytes": rss_after.saturating_sub(rss_before),
        "rss_total_bytes": rss_after,
        "workspace": opts.workspace.display().to_string(),
    });

    print_table(&report);
    Ok(report)
}

fn print_table(report: &Value) {
    let n = |v: &Value, p: &str| v.pointer(p).and_then(|x| x.as_u64()).unwrap_or(0);
    let f = |v: &Value, p: &str| v.pointer(p).and_then(|x| x.as_f64()).unwrap_or(0.0);

    let files = n(report, "/files");
    let ws = n(report, "/working_set_bytes");
    let base_ms = f(report, "/baseline/ms");
    let base_mbps = f(report, "/baseline/mbps");
    let mext_ms = f(report, "/mext/ms");
    let mext_mbps = f(report, "/mext/mbps");
    let warm_ms = f(report, "/mext/warm_ms");
    let resident = n(report, "/mext/resident_bytes");
    let ratio = f(report, "/mext/compression_ratio");
    let evictions = n(report, "/mext/evictions");
    let saved = n(report, "/ram_saved_bytes");
    let saved_pct = f(report, "/ram_saved_pct");
    let speed = f(report, "/speed_ratio");

    println!();
    println!(
        "  MEXT-M1 benchmark · {} files · working set {}",
        files,
        human_bytes(ws)
    );
    println!("  {}", "─".repeat(74));
    println!(
        "  {:34} {:>16} {:>18}",
        "", "baseline (page cache)", "mext (LZ4 pool)"
    );
    println!(
        "  {:34} {:>16} {:>18}",
        "wall time",
        format!("{base_ms:.1} ms"),
        format!("{mext_ms:.1} ms")
    );
    println!(
        "  {:34} {:>16} {:>18}",
        "throughput",
        format!("{base_mbps:.0} MB/s"),
        format!("{mext_mbps:.0} MB/s")
    );
    println!(
        "  {:34} {:>16} {:>18}",
        "RAM to keep set resident",
        human_bytes(ws),
        human_bytes(resident)
    );
    println!("  {}", "─".repeat(74));
    println!(
        "  compression ratio {ratio:.2}× · RAM saved {} ({saved_pct:.0}%) · evictions {evictions}",
        human_bytes(saved)
    );
    println!("  warm-up pass: {warm_ms:.1} ms · latency ratio {speed:.2}× (baseline/mext)");
    println!(
        "  note: the baseline number is the OS page cache, i.e. the best case for \
         uncompressed RAM."
    );
    println!(
        "        MEXT trades ~{:.1}× read latency for {:.0}% less resident RAM.",
        speed.max(1.0),
        saved_pct
    );
    println!();
}

/// Parse a budget string with a fallback, used by the CLI.
pub fn budget_or_default(raw: &str, fallback: u64) -> u64 {
    parse_size(raw).unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn collects_only_watched_files() {
        let dir = std::env::temp_dir().join(format!("mext-bench-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("target")).unwrap();
        let mut f = std::fs::File::create(dir.join("src/a.rs")).unwrap();
        writeln!(f, "fn a() {{}}").unwrap();
        std::fs::write(dir.join("target/junk.rs"), "x".repeat(1000)).unwrap();
        std::fs::write(dir.join("notes.xyz"), vec![7u8; 100]).unwrap();
        let files = collect_files(&dir, 10, 1 << 20);
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("src/a.rs"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bench_smoke() {
        let dir = std::env::temp_dir().join(format!("mext-benchrun-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        for i in 0..8 {
            std::fs::write(
                dir.join(format!("src/m{i}.md")),
                format!("# doc {i}\n{}", "lorem ipsum ".repeat(4000)),
            )
            .unwrap();
        }
        let mut opts = BenchOptions::new(&dir, 2 << 20);
        opts.passes = 1;
        let report = run(&opts).unwrap();
        assert_eq!(report["files"], 8);
        assert!(report["baseline"]["ms"].as_f64().unwrap() > 0.0);
        assert!(report["mext"]["resident_bytes"].as_u64().unwrap() > 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
