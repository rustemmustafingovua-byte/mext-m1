//! Platform sampling helpers shared by the dashboard and the CLI.

use crate::platform::{memory_info, memory_pressure, swap_usage, MemoryInfo, Pressure, SwapInfo};
use serde::Serialize;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Caches `vm_stat`/`sysctl` results for a short while: shelling out at 1 Hz is
/// cheap, but the dashboard also serves HTTP burst requests.
pub struct PlatformSampler {
    last_at: Option<Instant>,
    ttl: Duration,
    info: MemoryInfo,
    pressure: Pressure,
    swap: SwapInfo,
}

impl PlatformSampler {
    pub fn new(ttl: Duration) -> Self {
        Self {
            last_at: None,
            ttl,
            info: MemoryInfo::default(),
            pressure: Pressure::Unknown,
            swap: SwapInfo::default(),
        }
    }

    pub fn sample(&mut self) -> (MemoryInfo, Pressure, SwapInfo) {
        let fresh = self
            .last_at
            .map(|at| at.elapsed() >= self.ttl)
            .unwrap_or(true);
        if fresh {
            if let Ok(info) = memory_info() {
                self.info = info;
            }
            self.pressure = memory_pressure(Some(&self.info));
            self.swap = swap_usage();
            self.last_at = Some(Instant::now());
        }
        (self.info, self.pressure, self.swap)
    }

    pub fn pressure(&mut self) -> Pressure {
        self.sample().1
    }
}

/// Simple derivative tracker for throughput counters (bytes/s).
pub struct Rate {
    last_total: u64,
    last_at: Instant,
    value: f64,
}

impl Rate {
    pub fn new() -> Self {
        Self {
            last_total: 0,
            last_at: Instant::now(),
            value: 0.0,
        }
    }

    pub fn update(&mut self, total: u64) -> f64 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_at).as_secs_f64();
        if dt >= 0.05 {
            let delta = total.saturating_sub(self.last_total);
            let instantaneous = delta as f64 / dt;
            // Light smoothing keeps the sparkline readable.
            self.value = 0.65 * instantaneous + 0.35 * self.value;
            self.last_total = total;
            self.last_at = now;
        }
        self.value
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Default for Rate {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EventEntry {
    /// Unix seconds (float) — convenient for JS.
    pub ts: f64,
    pub kind: String,
    pub path: String,
    pub detail: String,
}

/// Ring buffer of recent activity for the dashboard event feed.
pub struct EventLog {
    entries: VecDeque<EventEntry>,
    capacity: usize,
}

impl EventLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, kind: &str, path: &str, detail: impl Into<String>) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(EventEntry {
            ts: unix_now(),
            kind: kind.to_string(),
            path: path.to_string(),
            detail: detail.into(),
        });
    }

    pub fn recent(&self, limit: usize) -> Vec<EventEntry> {
        let skip = self.entries.len().saturating_sub(limit);
        self.entries.iter().skip(skip).cloned().collect()
    }
}

pub fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Resident set size of this process in bytes (via `ps`, ~2 ms).
pub fn process_rss() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();
    match out {
        Ok(out) => String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .map(|kb| kb * 1024)
            .unwrap_or(0),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_reports_ram() {
        let mut s = PlatformSampler::new(Duration::from_millis(10));
        let (info, pressure, swap) = s.sample();
        assert!(info.total > 0);
        assert!(matches!(
            pressure,
            Pressure::Green | Pressure::Yellow | Pressure::Red | Pressure::Unknown
        ));
        assert!(swap.total >= swap.used);
    }

    #[test]
    fn rate_computes_delta() {
        let mut r = Rate::new();
        r.update(0);
        std::thread::sleep(Duration::from_millis(80));
        let v = r.update(10 * 1024 * 1024);
        assert!(v > 0.0);
    }

    #[test]
    fn event_log_caps() {
        let mut log = EventLog::new(3);
        for i in 0..5 {
            log.push("open", &format!("f{i}"), "");
        }
        let recent = log.recent(10);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent.last().unwrap().path, "f4");
    }

    #[test]
    fn rss_is_plausible() {
        let rss = process_rss();
        assert!(rss > 1024 * 1024, "rss={rss}");
    }
}
