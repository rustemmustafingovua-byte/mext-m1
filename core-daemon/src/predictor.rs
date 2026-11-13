//! Lightweight access predictor: Markov transition graph + EMA frequency +
//! hour-of-day temporal profile.
//!
//! `score(next | current)` mixes three signals (tuned on synthetic + real
//! workspace traces; see `mext bench`):
//!
//! ```text
//! score = 0.72 * P(next | current)      // Markov chain
//!       + 0.18 * P(hour(next) == now)   // diurnal pattern
//!       + 0.10 * freq(next) / max_freq  // EMA popularity
//! ```
//!
//! Prediction quality is measured online: every prediction is stored as
//! "pending" and the next observed file either matches it (hit) or not (miss).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Multiplicative decay applied every `DECAY_EVERY` observations.
const DECAY: f32 = 0.82;
const DECAY_EVERY: u64 = 512;
const MIN_EDGE: f32 = 0.04;
const MAX_ROW: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Prediction {
    pub score: f32,
    pub edge: f32,
    pub temporal: f32,
    pub frequency: f32,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictionItem {
    pub path: String,
    pub score: f32,
    pub edge: f32,
    pub temporal: f32,
    pub frequency: f32,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ObserveReport {
    pub transition: bool,
    pub hit: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictorStats {
    pub transitions: u64,
    pub nodes: usize,
    pub edges: usize,
    pub hits: u64,
    pub attempts: u64,
    pub hit_rate: f64,
    pub last: Option<String>,
    pub decay_events: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Predictor {
    #[serde(default)]
    edges: HashMap<String, HashMap<String, f32>>,
    #[serde(default)]
    frequency: HashMap<String, f32>,
    /// 24 hourly buckets per file.
    #[serde(default)]
    hourly: HashMap<String, Vec<f32>>,
    #[serde(default)]
    last: Option<String>,
    #[serde(default)]
    pending: Vec<String>,
    #[serde(default)]
    pub transitions: u64,
    #[serde(default)]
    pub hits: u64,
    #[serde(default)]
    pub attempts: u64,
    #[serde(default)]
    decay_events: u64,
    #[serde(default)]
    observations: u64,
}

impl Predictor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `path` was accessed at `hour` (0..23).
    ///
    /// Returns whether a transition was learned and whether the previous
    /// prediction was correct.
    pub fn observe(&mut self, path: &str, hour: u32) -> ObserveReport {
        let mut report = ObserveReport::default();

        if let Some(prev) = self.last.clone() {
            if prev != path {
                let row = self.edges.entry(prev).or_default();
                *row.entry(path.to_string()).or_insert(0.0) += 1.0;
                self.transitions += 1;
                report.transition = true;

                if row.len() > MAX_ROW {
                    Self::bound_row(row);
                }
            }
        }

        if !self.pending.is_empty() {
            let hit = self.pending.iter().any(|p| p == path);
            self.attempts += 1;
            if hit {
                self.hits += 1;
            }
            report.hit = Some(hit);
            self.pending.clear();
        }

        *self.frequency.entry(path.to_string()).or_insert(0.0) += 1.0;
        let buckets = self
            .hourly
            .entry(path.to_string())
            .or_insert_with(|| vec![0.0; 24]);
        if (hour as usize) < buckets.len() {
            buckets[hour as usize] += 1.0;
        }

        self.last = Some(path.to_string());
        self.observations += 1;
        if self.observations % DECAY_EVERY == 0 {
            self.decay();
        }
        report
    }

    /// Predict the next `k` files after `current`.
    pub fn predict(&self, current: &str, hour: u32, k: usize) -> Vec<PredictionItem> {
        let now_bucket = hour as usize % 24;
        let max_freq = self.frequency.values().copied().fold(1.0f32, f32::max);
        let mut scored: HashMap<&str, Prediction> = HashMap::new();

        if let Some(row) = self.edges.get(current) {
            let total: f32 = row.values().sum::<f32>().max(1.0);
            for (next, weight) in row.iter() {
                let edge = weight / total;
                let temporal = self.temporal_score(next, now_bucket);
                let frequency = self.frequency.get(next).copied().unwrap_or(0.0) / max_freq;
                let score = 0.72 * edge + 0.18 * temporal + 0.10 * frequency;
                if score > 0.0 {
                    scored.insert(
                        next.as_str(),
                        Prediction {
                            score,
                            edge,
                            temporal,
                            frequency,
                            reason: "markov",
                        },
                    );
                }
            }
        }

        // Fallback for cold starts / unseen nodes: temporal + popularity.
        if scored.len() < k {
            let mut candidates: Vec<(&String, f32)> = self
                .frequency
                .iter()
                .map(|(path, freq)| {
                    let temporal = self.temporal_score(path, now_bucket);
                    let f = *freq / max_freq;
                    (path, 0.55 * f + 0.45 * temporal)
                })
                .collect();
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (path, score) in candidates.into_iter().take(k * 2) {
                if path == current || scored.contains_key(path.as_str()) {
                    continue;
                }
                scored.insert(
                    path.as_str(),
                    Prediction {
                        score: score * 0.6,
                        edge: 0.0,
                        temporal: self.temporal_score(path, now_bucket),
                        frequency: self.frequency.get(path).copied().unwrap_or(0.0) / max_freq,
                        reason: "temporal",
                    },
                );
                if scored.len() >= k {
                    break;
                }
            }
        }

        let mut items: Vec<PredictionItem> = scored
            .into_iter()
            .map(|(path, p)| PredictionItem {
                path: path.to_string(),
                score: p.score,
                edge: p.edge,
                temporal: p.temporal,
                frequency: p.frequency,
                reason: p.reason,
            })
            .collect();
        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(k);
        items
    }

    /// Remember what we predicted so the next `observe` can score it.
    pub fn set_pending(&mut self, paths: Vec<String>) {
        self.pending = paths;
    }

    pub fn pending(&self) -> &[String] {
        &self.pending
    }

    fn temporal_score(&self, path: &str, bucket: usize) -> f32 {
        let Some(hours) = self.hourly.get(path) else {
            return 0.0;
        };
        let total: f32 = hours.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        // Compare this hour against the mean bucket.
        let value = hours.get(bucket).copied().unwrap_or(0.0);
        let mean = total / 24.0;
        ((value / mean.max(0.25)) / 4.0).clamp(0.0, 1.0)
    }

    fn decay(&mut self) {
        self.decay_events += 1;
        for row in self.edges.values_mut() {
            row.retain(|_, w| {
                *w *= DECAY;
                *w >= MIN_EDGE
            });
        }
        self.edges.retain(|_, row| !row.is_empty());
        for freq in self.frequency.values_mut() {
            *freq *= DECAY;
        }
        self.frequency.retain(|_, f| *f >= MIN_EDGE);
        for hours in self.hourly.values_mut() {
            for v in hours.iter_mut() {
                *v *= DECAY;
            }
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.hits as f64 / self.attempts as f64
        }
    }

    /// Number of distinct paths tracked in the frequency table.
    /// Keep only the strongest edges of a row (bounded graph growth).
    fn bound_row(row: &mut HashMap<String, f32>) {
        let mut pairs: Vec<(String, f32)> = row.iter().map(|(k, v)| (k.clone(), *v)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep: HashMap<String, f32> = pairs.into_iter().take(MAX_ROW / 2).collect();
        *row = keep;
    }

    /// Merge another predictor into this one, scaling its evidence by `weight`.
    ///
    /// Edges, frequency and hourly profiles are additive; counters are scaled so
    /// the online hit-rate stays comparable. Used by `mext profiles merge` to
    /// bootstrap a new project from learned sessions.
    pub fn merge_from(&mut self, other: &Predictor, weight: f32) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        let scale = |value: u64| -> u64 { ((value as f64) * weight as f64).round() as u64 };

        for (source, row) in &other.edges {
            let target = self.edges.entry(source.clone()).or_default();
            for (destination, edge) in row {
                let add = edge * weight;
                if add.is_finite() && add > 0.0 {
                    *target.entry(destination.clone()).or_insert(0.0) += add;
                }
            }
            if target.len() > MAX_ROW {
                Self::bound_row(target);
            }
        }
        for (path, frequency) in &other.frequency {
            let add = frequency * weight;
            if add.is_finite() && add > 0.0 {
                *self.frequency.entry(path.clone()).or_insert(0.0) += add;
            }
        }
        for (path, hours) in &other.hourly {
            let target = self
                .hourly
                .entry(path.clone())
                .or_insert_with(|| vec![0.0; 24]);
            for (index, value) in hours.iter().enumerate().take(24) {
                let add = value * weight;
                if add.is_finite() && add > 0.0 {
                    target[index] += add;
                }
            }
        }

        self.transitions = self.transitions.saturating_add(other.transitions);
        self.hits = self.hits.saturating_add(scale(other.hits));
        self.attempts = self.attempts.saturating_add(scale(other.attempts));
        self.observations = self.observations.saturating_add(scale(other.observations));
        if self.last.is_none() {
            self.last = other.last.clone();
        }
    }

    pub fn node_count(&self) -> usize {
        self.frequency.len()
    }

    /// Number of learned Markov edges (distinct `from -> to` pairs).
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|r| r.len()).sum()
    }

    pub fn stats(&self) -> PredictorStats {
        PredictorStats {
            transitions: self.transitions,
            nodes: self.node_count(),
            edges: self.edge_count(),
            hits: self.hits,
            attempts: self.attempts,
            hit_rate: self.hit_rate(),
            last: self.last.clone(),
            decay_events: self.decay_events,
        }
    }

    /// Total number of learned observations (used by the dashboard).
    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn last(&self) -> Option<&str> {
        self.last.as_deref()
    }

    /// Top-K most frequently accessed paths (for the "hot files" panel).
    pub fn hot_paths(&self, k: usize) -> Vec<(String, f32)> {
        let mut v: Vec<(String, f32)> = self
            .frequency
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(k);
        v
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }

    /// Load if present, otherwise start empty.
    pub fn load_or_new(path: &Path) -> Self {
        Self::load(path).unwrap_or_default()
    }
}

/// Convenience: current hour in local time.
pub fn hour_of_day() -> u32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Local offset from libc (handles DST correctly).
    let offset = unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut tm);
        tm.tm_gmtoff
    };
    (((secs as i64 + offset) / 3600) % 24) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_sequence() {
        let mut p = Predictor::new();
        for _ in 0..20 {
            p.observe("a.rs", 10);
            p.observe("b.rs", 10);
            p.observe("c.rs", 10);
            p.observe("a.rs", 10);
        }
        let preds = p.predict("a.rs", 10, 3);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].path, "b.rs");
        assert!(preds[0].score > 0.5, "score = {}", preds[0].score);
    }

    #[test]
    fn hit_rate_tracks_accuracy() {
        let mut p = Predictor::new();
        p.observe("a.rs", 9);
        p.observe("b.rs", 9);
        p.observe("a.rs", 9);
        let preds = p.predict("a.rs", 9, 2);
        p.set_pending(preds.iter().map(|x| x.path.clone()).collect());
        let report = p.observe("b.rs", 9);
        assert_eq!(report.hit, Some(true));
        assert_eq!(p.attempts, 1);
        assert_eq!(p.hits, 1);
        assert_eq!(p.hit_rate(), 1.0);
    }

    #[test]
    fn temporal_pattern() {
        let mut p = Predictor::new();
        for _ in 0..50 {
            p.observe("morning.md", 9);
            p.observe("evening.md", 21);
        }
        let morning = p.predict("", 9, 5);
        let evening = p.predict("", 21, 5);
        assert_eq!(morning[0].path, "morning.md");
        assert_eq!(evening[0].path, "evening.md");
    }

    #[test]
    fn cold_start_falls_back() {
        let mut p = Predictor::new();
        p.observe("only.rs", 12);
        let preds = p.predict("unknown.rs", 12, 3);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].path, "only.rs");
    }

    #[test]
    fn save_load_roundtrip() {
        let mut p = Predictor::new();
        p.observe("x", 1);
        p.observe("y", 2);
        let dir = std::env::temp_dir().join(format!("mext-pred-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pred.json");
        p.save(&path).unwrap();
        let loaded = Predictor::load(&path).unwrap();
        assert_eq!(loaded.transitions, p.transitions);
        assert!(loaded.predict("x", 2, 2).iter().any(|i| i.path == "y"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_keeps_both_graphs_and_scales_counters() {
        let mut a = Predictor::new();
        for _ in 0..4 {
            a.observe("a.rs", 9);
            a.observe("b.rs", 9);
        }
        let mut b = Predictor::new();
        for _ in 0..2 {
            b.observe("x.rs", 10);
            b.observe("y.rs", 10);
        }
        let a_edges = a.edge_count();
        let b_edges = b.edge_count();
        let b_hits = b.hits;
        let b_attempts = b.attempts;

        a.merge_from(&b, 0.5);

        assert_eq!(a.edge_count(), a_edges + b_edges);
        assert_eq!(a.hits, (b_hits as f64 * 0.5).round() as u64);
        assert_eq!(a.attempts, (b_attempts as f64 * 0.5).round() as u64);
        let next = a.predict("a.rs", 9, 3);
        assert_eq!(next[0].path, "b.rs");
        let other = a.predict("x.rs", 10, 2);
        assert!(other.iter().any(|p| p.path == "y.rs"));
    }

    #[test]
    fn merge_weight_zero_is_noop_and_self_merge_doubles() {
        let mut p = Predictor::new();
        p.observe("a", 1);
        p.observe("b", 1);
        let snapshot = p.clone();
        p.merge_from(&snapshot, 0.0);
        assert_eq!(p.edge_count(), 1);
        assert_eq!(p.transitions, 1);
        assert_eq!(p.hits, 0);

        let mut q = Predictor::new();
        q.observe("a", 1);
        q.observe("b", 1);
        let before = q.edge_count();
        q.merge_from(&q.clone(), 1.0);
        assert_eq!(q.edge_count(), before);
        assert_eq!(q.transitions, 2);
    }

    #[test]
    fn hour_is_valid() {
        assert!(hour_of_day() < 24);
    }
}
