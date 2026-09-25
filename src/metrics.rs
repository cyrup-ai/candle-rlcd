//! Prometheus metrics for `candle-rlcd serve` (`GET /metrics`, text exposition format 0.0.4).
//!
//! Lock-free counters and fixed-bucket histograms, cheap enough to update on every request.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Latency buckets in seconds, from a fast single question to a long multi-question request.
pub const LATENCY_BUCKETS: [f64; 14] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

/// A cumulative histogram over [`LATENCY_BUCKETS`].
#[derive(Default)]
pub struct Histogram {
    /// Per-bucket (non-cumulative) counts; the last slot is `+Inf`.
    counts: [AtomicU64; LATENCY_BUCKETS.len() + 1],
    sum_us: AtomicU64,
}

impl Histogram {
    pub fn observe(&self, d: Duration) {
        let s = d.as_secs_f64();
        let i = LATENCY_BUCKETS
            .iter()
            .position(|b| s <= *b)
            .unwrap_or(LATENCY_BUCKETS.len());
        self.counts[i].fetch_add(1, Ordering::Relaxed);
        self.sum_us
            .fetch_add(d.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Appends `name_bucket`, `name_sum` and `name_count` lines with the given labels
    /// (`labels` is empty or `key="value",...` without braces).
    pub fn render(&self, out: &mut String, name: &str, labels: &str) {
        let sep = if labels.is_empty() { "" } else { "," };
        let mut cum = 0;
        for (i, c) in self.counts.iter().enumerate() {
            cum += c.load(Ordering::Relaxed);
            let le = LATENCY_BUCKETS
                .get(i)
                .map_or("+Inf".to_string(), |b| b.to_string());
            let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"{le}\"}} {cum}");
        }
        let braces = if labels.is_empty() {
            String::new()
        } else {
            format!("{{{labels}}}")
        };
        let sum = self.sum_us.load(Ordering::Relaxed) as f64 / 1e6;
        let _ = writeln!(out, "{name}_sum{braces} {sum}");
        let _ = writeln!(out, "{name}_count{braces} {cum}");
    }
}

/// HTTP-level metrics, keyed by route and status.
#[derive(Default)]
pub struct HttpMetrics {
    /// `(route, status)` → count; few distinct keys, so a small locked vector is enough.
    responses: Mutex<Vec<(String, u16, u64)>>,
    /// Request latency per route, in the order routes were first seen.
    latency: Mutex<Vec<(String, std::sync::Arc<Histogram>)>>,
}

impl HttpMetrics {
    pub fn record(&self, route: &str, status: u16, took: Duration) {
        {
            let mut r = self.responses.lock().expect("metrics lock");
            match r.iter_mut().find(|(p, s, _)| p == route && *s == status) {
                Some(e) => e.2 += 1,
                None => r.push((route.to_string(), status, 1)),
            }
        }
        let h = {
            let mut l = self.latency.lock().expect("metrics lock");
            match l.iter().find(|(p, _)| p == route) {
                Some((_, h)) => h.clone(),
                None => {
                    let h = std::sync::Arc::new(Histogram::default());
                    l.push((route.to_string(), h.clone()));
                    h
                }
            }
        };
        h.observe(took);
    }

    pub fn render(&self, out: &mut String) {
        out.push_str(
            "# HELP candle_rlcd_http_requests_total HTTP responses by route and status.\n",
        );
        out.push_str("# TYPE candle_rlcd_http_requests_total counter\n");
        for (route, status, n) in self.responses.lock().expect("metrics lock").iter() {
            let _ = writeln!(
                out,
                "candle_rlcd_http_requests_total{{route=\"{route}\",status=\"{status}\"}} {n}"
            );
        }
        out.push_str(
            "# HELP candle_rlcd_http_request_duration_seconds Time from request to response.\n",
        );
        out.push_str("# TYPE candle_rlcd_http_request_duration_seconds histogram\n");
        for (route, h) in self.latency.lock().expect("metrics lock").iter() {
            h.render(
                out,
                "candle_rlcd_http_request_duration_seconds",
                &format!("route=\"{route}\""),
            );
        }
    }
}

/// Escapes a Prometheus label value.
pub fn label(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_is_cumulative() {
        let h = Histogram::default();
        h.observe(Duration::from_millis(3));
        h.observe(Duration::from_millis(300));
        h.observe(Duration::from_secs(1000));
        let mut s = String::new();
        h.render(&mut s, "x", "m=\"a\"");
        assert!(s.contains("x_bucket{m=\"a\",le=\"0.005\"} 1\n"), "{s}");
        assert!(s.contains("x_bucket{m=\"a\",le=\"0.5\"} 2\n"), "{s}");
        assert!(s.contains("x_bucket{m=\"a\",le=\"+Inf\"} 3\n"), "{s}");
        assert!(s.contains("x_count{m=\"a\"} 3\n"), "{s}");
        assert_eq!(h.count(), 3);
    }
}
