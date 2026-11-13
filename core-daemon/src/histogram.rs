//! Thread-safe fixed-bucket histogram for Prometheus exposition.

use std::sync::Mutex;

/// Default latency buckets in milliseconds.
const DEFAULT_BUCKETS_MS: &[f64] = &[
    0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];

/// A cumulative histogram protected by a mutex so it can be observed from many
/// threads and rendered from an HTTP handler.
pub struct Histogram {
    buckets: Vec<f64>,
    counts: Mutex<Vec<u64>>,
    sum: Mutex<f64>,
    count: Mutex<u64>,
}

impl Histogram {
    /// Create a histogram with the default millisecond buckets plus a `+Inf`
    /// catch-all bucket appended at snapshot time.
    pub fn new() -> Self {
        Self::with_buckets(DEFAULT_BUCKETS_MS)
    }

    /// Create a histogram with custom bucket upper bounds (exclusive).
    pub fn with_buckets(buckets: &[f64]) -> Self {
        let buckets = buckets.to_vec();
        Self {
            counts: Mutex::new(vec![0u64; buckets.len()]),
            sum: Mutex::new(0.0),
            count: Mutex::new(0),
            buckets,
        }
    }

    /// Record a single observation. Negative values are clamped to zero; NaN is
    /// ignored.
    pub fn observe(&self, value: f64) {
        if value.is_nan() {
            return;
        }
        let value = value.max(0.0);
        let idx = self
            .buckets
            .iter()
            .position(|&upper| value <= upper)
            .unwrap_or(self.buckets.len());
        if let Ok(mut counts) = self.counts.lock() {
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }
        if let Ok(mut sum) = self.sum.lock() {
            *sum += value;
        }
        if let Ok(mut count) = self.count.lock() {
            *count += 1;
        }
    }

    /// Return an immutable snapshot of the histogram state.
    pub fn snapshot(&self) -> Snapshot {
        let counts = self
            .counts
            .lock()
            .map(|c| c.clone())
            .unwrap_or_else(|_| vec![]);
        Snapshot {
            buckets: self.buckets.clone(),
            counts,
            sum: self.sum.lock().map(|value| *value).unwrap_or(0.0),
            count: self.count.lock().map(|value| *value).unwrap_or(0),
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable copy of a [`Histogram`] suitable for rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Default for Snapshot {
    fn default() -> Self {
        Histogram::new().snapshot()
    }
}

impl Snapshot {
    /// Render the snapshot in Prometheus text exposition format.
    ///
    /// Output includes cumulative `_bucket{le="..."}` lines for each bucket,
    /// including `+Inf`, followed by `_sum` and `_count`.
    pub fn render_prometheus(&self, name: &str, help: &str) -> String {
        let mut out = String::new();
        out.push_str("# HELP ");
        out.push_str(name);
        out.push(' ');
        out.push_str(help);
        out.push('\n');
        out.push_str("# TYPE ");
        out.push_str(name);
        out.push_str(" histogram\n");

        let mut cumulative = 0u64;
        for (upper, count) in self.buckets.iter().zip(self.counts.iter()) {
            cumulative += *count;
            out.push_str(name);
            out.push_str("_bucket{le=\"");
            // Format without trailing zeros for whole-number boundaries.
            if upper.fract() == 0.0 {
                out.push_str(&format!("{:.0}", upper));
            } else {
                out.push_str(&format!("{}", upper));
            }
            out.push_str("\"} ");
            out.push_str(&cumulative.to_string());
            out.push('\n');
        }
        out.push_str(name);
        out.push_str("_bucket{le=\"+Inf\"} ");
        out.push_str(&self.count.to_string());
        out.push('\n');

        out.push_str(name);
        out.push_str("_sum ");
        out.push_str(&format!("{}", self.sum));
        out.push('\n');

        out.push_str(name);
        out.push_str("_count ");
        out.push_str(&self.count.to_string());
        out.push('\n');

        out
    }

    /// Approximate quantile in milliseconds (upper bound of the bucket that
    /// contains the observation). Returns 0 for an empty histogram.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let q = q.clamp(0.0, 1.0);
        let target = ((q * self.count as f64).ceil() as u64).max(1);
        let mut cumulative = 0u64;
        for (upper, count) in self.buckets.iter().zip(self.counts.iter()) {
            cumulative += *count;
            if cumulative >= target {
                return *upper;
            }
        }
        *self.buckets.last().unwrap_or(&0.0)
    }

    /// Total number of observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sum of all observations.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Bucket upper bounds and per-bucket (non-cumulative) counts.
    pub fn buckets(&self) -> Vec<(f64, u64)> {
        self.buckets
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram_renders_zero() {
        let h = Histogram::new();
        let text = h.snapshot().render_prometheus("mext_test", "test help");
        assert!(text.contains("# HELP mext_test test help"));
        assert!(text.contains("# TYPE mext_test histogram"));
        assert!(text.contains("mext_test_bucket{le=\"+Inf\"} 0"));
        assert!(text.contains("mext_test_sum 0"));
        assert!(text.contains("mext_test_count 0"));
    }

    #[test]
    fn bucket_placement() {
        let h = Histogram::new();
        // Default buckets: 0.5, 1, 2, 5, 10, ...
        h.observe(0.3); // <= 0.5
        h.observe(0.7); // <= 1
        h.observe(1.5); // <= 2
        h.observe(3.0); // <= 5
        h.observe(7.0); // <= 10
        let snap = h.snapshot();
        let pairs = snap.buckets();
        assert_eq!(pairs[0], (0.5, 1));
        assert_eq!(pairs[1], (1.0, 1));
        assert_eq!(pairs[2], (2.0, 1));
        assert_eq!(pairs[3], (5.0, 1));
        assert_eq!(pairs[4], (10.0, 1));
    }

    #[test]
    fn cumulative_and_monotonic() {
        let h = Histogram::new();
        for v in [0.5, 1.5, 3.0, 12.0] {
            h.observe(v);
        }
        let text = h.snapshot().render_prometheus("mext_test", "test help");
        let mut last: u64 = 0;
        for line in text.lines() {
            if let Some(start) = line.find("_bucket{le=") {
                let num = line[start..]
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                assert!(num >= last, "cumulative buckets must not decrease");
                last = num;
            }
        }
        assert_eq!(last, 4);
        assert!(text.contains("mext_test_count 4"));
        assert!(text.contains("mext_test_sum 17"));
    }

    #[test]
    fn sum_and_count_match() {
        let h = Histogram::new();
        let values = [1.2, 2.5, 5.0, 100.0];
        let expected_sum: f64 = values.iter().sum();
        for v in values {
            h.observe(v);
        }
        let snap = h.snapshot();
        assert_eq!(snap.count(), 4);
        assert!((snap.sum() - expected_sum).abs() < 1e-9);
    }

    #[test]
    fn quantiles_use_cumulative_buckets() {
        let h = Histogram::new();
        for value in [0.4, 1.0, 2.5, 100.0] {
            h.observe(value);
        }
        let snap = h.snapshot();
        assert_eq!(snap.quantile(0.5), 1.0);
        assert_eq!(snap.quantile(0.99), 100.0);
        assert_eq!(snap.quantile(0.0), 0.5);
        assert_eq!(Histogram::new().snapshot().quantile(0.95), 0.0);
    }

    #[test]
    fn nan_ignored_and_negatives_clamped() {
        let h = Histogram::new();
        h.observe(f64::NAN);
        h.observe(-5.0);
        h.observe(0.0);
        h.observe(0.4);
        let snap = h.snapshot();
        assert_eq!(snap.count(), 3);
        // -5.0 clamped to 0, 0.0 and 0.4 all fall into the first bucket.
        assert_eq!(snap.buckets()[0].1, 3);
        assert_eq!(snap.sum(), 0.4);
    }
}
