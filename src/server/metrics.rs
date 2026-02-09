//! Prometheus-compatible metrics for the transcoding server
//!
//! This module provides metrics collection and export for monitoring
//! the coordinator and worker performance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Prometheus-compatible metrics for the transcoding server
#[derive(Debug)]
pub struct Metrics {
    /// Total number of jobs submitted
    pub jobs_total: AtomicU64,

    /// Number of currently running jobs
    pub jobs_running: AtomicU64,

    /// Number of completed jobs
    pub jobs_completed: AtomicU64,

    /// Number of failed jobs
    pub jobs_failed: AtomicU64,

    /// Number of cancelled jobs
    pub jobs_cancelled: AtomicU64,

    /// Number of registered workers
    pub worker_count: AtomicU64,

    /// Current transcoding FPS (aggregate)
    pub transcode_fps: AtomicU64,

    /// Total bytes processed
    pub bytes_processed: AtomicU64,

    /// Total encoding time in milliseconds
    pub encoding_time_ms: AtomicU64,

    /// Queue depth (pending jobs)
    pub queue_depth: AtomicU64,
}

impl Metrics {
    /// Create a new metrics instance
    pub fn new() -> Self {
        Self {
            jobs_total: AtomicU64::new(0),
            jobs_running: AtomicU64::new(0),
            jobs_completed: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            jobs_cancelled: AtomicU64::new(0),
            worker_count: AtomicU64::new(0),
            transcode_fps: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            encoding_time_ms: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
        }
    }

    /// Get the total number of jobs submitted
    pub fn total_jobs(&self) -> u64 {
        self.jobs_total.load(Ordering::Relaxed)
    }

    /// Get the number of running jobs
    pub fn running_jobs(&self) -> u64 {
        self.jobs_running.load(Ordering::Relaxed)
    }

    /// Get the number of completed jobs
    pub fn completed_jobs(&self) -> u64 {
        self.jobs_completed.load(Ordering::Relaxed)
    }

    /// Get the number of failed jobs
    pub fn failed_jobs(&self) -> u64 {
        self.jobs_failed.load(Ordering::Relaxed)
    }

    /// Get the number of cancelled jobs
    pub fn cancelled_jobs(&self) -> u64 {
        self.jobs_cancelled.load(Ordering::Relaxed)
    }

    /// Get the number of workers
    pub fn workers(&self) -> u64 {
        self.worker_count.load(Ordering::Relaxed)
    }

    /// Get the current FPS
    pub fn fps(&self) -> u64 {
        self.transcode_fps.load(Ordering::Relaxed)
    }

    /// Get total bytes processed
    pub fn total_bytes(&self) -> u64 {
        self.bytes_processed.load(Ordering::Relaxed)
    }

    /// Get total encoding time
    pub fn total_encoding_time_ms(&self) -> u64 {
        self.encoding_time_ms.load(Ordering::Relaxed)
    }

    /// Get queue depth
    pub fn pending_jobs(&self) -> u64 {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Increment job count
    pub fn inc_jobs_total(&self) {
        self.jobs_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment running jobs
    pub fn inc_jobs_running(&self) {
        self.jobs_running.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement running jobs
    pub fn dec_jobs_running(&self) {
        self.jobs_running.fetch_sub(1, Ordering::Relaxed);
    }

    /// Increment completed jobs
    pub fn inc_jobs_completed(&self) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment failed jobs
    pub fn inc_jobs_failed(&self) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment cancelled jobs
    pub fn inc_jobs_cancelled(&self) {
        self.jobs_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment worker count
    pub fn inc_workers(&self) {
        self.worker_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement worker count
    pub fn dec_workers(&self) {
        self.worker_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Set current FPS
    pub fn set_fps(&self, fps: u64) {
        self.transcode_fps.store(fps, Ordering::Relaxed);
    }

    /// Add bytes processed
    pub fn add_bytes(&self, bytes: u64) {
        self.bytes_processed.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Add encoding time
    pub fn add_encoding_time(&self, ms: u64) {
        self.encoding_time_ms.fetch_add(ms, Ordering::Relaxed);
    }

    /// Set queue depth
    pub fn set_queue_depth(&self, depth: u64) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Record job completion with stats
    pub fn record_job_completion(&self, encoding_time_ms: u64, output_bytes: u64) {
        self.inc_jobs_completed();
        self.dec_jobs_running();
        self.add_encoding_time(encoding_time_ms);
        self.add_bytes(output_bytes);
    }

    /// Record job failure
    pub fn record_job_failure(&self) {
        self.inc_jobs_failed();
        self.dec_jobs_running();
    }

    /// Record job cancellation
    pub fn record_job_cancellation(&self) {
        self.inc_jobs_cancelled();
        // May or may not have been running
    }

    /// Export metrics in Prometheus text format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::with_capacity(2048);

        // Jobs total
        output.push_str("# HELP zvd_jobs_total Total number of jobs submitted\n");
        output.push_str("# TYPE zvd_jobs_total counter\n");
        output.push_str(&format!("zvd_jobs_total {}\n\n", self.total_jobs()));

        // Jobs running
        output.push_str("# HELP zvd_jobs_running Number of currently running jobs\n");
        output.push_str("# TYPE zvd_jobs_running gauge\n");
        output.push_str(&format!("zvd_jobs_running {}\n\n", self.running_jobs()));

        // Jobs completed
        output.push_str("# HELP zvd_jobs_completed Total number of completed jobs\n");
        output.push_str("# TYPE zvd_jobs_completed counter\n");
        output.push_str(&format!("zvd_jobs_completed {}\n\n", self.completed_jobs()));

        // Jobs failed
        output.push_str("# HELP zvd_jobs_failed Total number of failed jobs\n");
        output.push_str("# TYPE zvd_jobs_failed counter\n");
        output.push_str(&format!("zvd_jobs_failed {}\n\n", self.failed_jobs()));

        // Jobs cancelled
        output.push_str("# HELP zvd_jobs_cancelled Total number of cancelled jobs\n");
        output.push_str("# TYPE zvd_jobs_cancelled counter\n");
        output.push_str(&format!("zvd_jobs_cancelled {}\n\n", self.cancelled_jobs()));

        // Worker count
        output.push_str("# HELP zvd_worker_count Number of registered workers\n");
        output.push_str("# TYPE zvd_worker_count gauge\n");
        output.push_str(&format!("zvd_worker_count {}\n\n", self.workers()));

        // Transcode FPS
        output.push_str("# HELP zvd_transcode_fps Current transcoding FPS (aggregate)\n");
        output.push_str("# TYPE zvd_transcode_fps gauge\n");
        output.push_str(&format!("zvd_transcode_fps {}\n\n", self.fps()));

        // Bytes processed
        output.push_str("# HELP zvd_bytes_processed_total Total bytes processed\n");
        output.push_str("# TYPE zvd_bytes_processed_total counter\n");
        output.push_str(&format!("zvd_bytes_processed_total {}\n\n", self.total_bytes()));

        // Encoding time
        output.push_str("# HELP zvd_encoding_time_seconds_total Total encoding time in seconds\n");
        output.push_str("# TYPE zvd_encoding_time_seconds_total counter\n");
        output.push_str(&format!(
            "zvd_encoding_time_seconds_total {:.3}\n\n",
            self.total_encoding_time_ms() as f64 / 1000.0
        ));

        // Queue depth
        output.push_str("# HELP zvd_queue_depth Number of pending jobs in queue\n");
        output.push_str("# TYPE zvd_queue_depth gauge\n");
        output.push_str(&format!("zvd_queue_depth {}\n", self.pending_jobs()));

        output
    }

    /// Export metrics as JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "jobs_total": self.total_jobs(),
            "jobs_running": self.running_jobs(),
            "jobs_completed": self.completed_jobs(),
            "jobs_failed": self.failed_jobs(),
            "jobs_cancelled": self.cancelled_jobs(),
            "worker_count": self.workers(),
            "transcode_fps": self.fps(),
            "bytes_processed": self.total_bytes(),
            "encoding_time_seconds": self.total_encoding_time_ms() as f64 / 1000.0,
            "queue_depth": self.pending_jobs(),
        })
    }

    /// Reset all metrics to zero
    pub fn reset(&self) {
        self.jobs_total.store(0, Ordering::Relaxed);
        self.jobs_running.store(0, Ordering::Relaxed);
        self.jobs_completed.store(0, Ordering::Relaxed);
        self.jobs_failed.store(0, Ordering::Relaxed);
        self.jobs_cancelled.store(0, Ordering::Relaxed);
        self.worker_count.store(0, Ordering::Relaxed);
        self.transcode_fps.store(0, Ordering::Relaxed);
        self.bytes_processed.store(0, Ordering::Relaxed);
        self.encoding_time_ms.store(0, Ordering::Relaxed);
        self.queue_depth.store(0, Ordering::Relaxed);
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Metrics {
    fn clone(&self) -> Self {
        Self {
            jobs_total: AtomicU64::new(self.jobs_total.load(Ordering::Relaxed)),
            jobs_running: AtomicU64::new(self.jobs_running.load(Ordering::Relaxed)),
            jobs_completed: AtomicU64::new(self.jobs_completed.load(Ordering::Relaxed)),
            jobs_failed: AtomicU64::new(self.jobs_failed.load(Ordering::Relaxed)),
            jobs_cancelled: AtomicU64::new(self.jobs_cancelled.load(Ordering::Relaxed)),
            worker_count: AtomicU64::new(self.worker_count.load(Ordering::Relaxed)),
            transcode_fps: AtomicU64::new(self.transcode_fps.load(Ordering::Relaxed)),
            bytes_processed: AtomicU64::new(self.bytes_processed.load(Ordering::Relaxed)),
            encoding_time_ms: AtomicU64::new(self.encoding_time_ms.load(Ordering::Relaxed)),
            queue_depth: AtomicU64::new(self.queue_depth.load(Ordering::Relaxed)),
        }
    }
}

/// Worker-specific metrics
#[derive(Debug)]
pub struct WorkerMetrics {
    /// Worker ID
    pub worker_id: String,

    /// Jobs processed by this worker
    pub jobs_processed: AtomicU64,

    /// Current FPS
    pub current_fps: AtomicU64,

    /// Total encoding time
    pub encoding_time_ms: AtomicU64,

    /// Total bytes output
    pub bytes_output: AtomicU64,

    /// Number of failures
    pub failures: AtomicU64,
}

impl WorkerMetrics {
    /// Create new worker metrics
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            jobs_processed: AtomicU64::new(0),
            current_fps: AtomicU64::new(0),
            encoding_time_ms: AtomicU64::new(0),
            bytes_output: AtomicU64::new(0),
            failures: AtomicU64::new(0),
        }
    }

    /// Record job completion
    pub fn record_completion(&self, encoding_time_ms: u64, output_bytes: u64) {
        self.jobs_processed.fetch_add(1, Ordering::Relaxed);
        self.encoding_time_ms.fetch_add(encoding_time_ms, Ordering::Relaxed);
        self.bytes_output.fetch_add(output_bytes, Ordering::Relaxed);
    }

    /// Record job failure
    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Set current FPS
    pub fn set_fps(&self, fps: u64) {
        self.current_fps.store(fps, Ordering::Relaxed);
    }

    /// Get average encoding speed (bytes per second)
    pub fn avg_encoding_speed(&self) -> f64 {
        let bytes = self.bytes_output.load(Ordering::Relaxed) as f64;
        let time_s = self.encoding_time_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        if time_s > 0.0 {
            bytes / time_s
        } else {
            0.0
        }
    }

    /// Export as JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "worker_id": self.worker_id,
            "jobs_processed": self.jobs_processed.load(Ordering::Relaxed),
            "current_fps": self.current_fps.load(Ordering::Relaxed),
            "encoding_time_seconds": self.encoding_time_ms.load(Ordering::Relaxed) as f64 / 1000.0,
            "bytes_output": self.bytes_output.load(Ordering::Relaxed),
            "failures": self.failures.load(Ordering::Relaxed),
            "avg_encoding_speed_bps": self.avg_encoding_speed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_basic() {
        let metrics = Metrics::new();

        metrics.inc_jobs_total();
        metrics.inc_jobs_total();
        assert_eq!(metrics.total_jobs(), 2);

        metrics.inc_jobs_running();
        assert_eq!(metrics.running_jobs(), 1);

        metrics.record_job_completion(1000, 1024);
        assert_eq!(metrics.completed_jobs(), 1);
        assert_eq!(metrics.running_jobs(), 0);
        assert_eq!(metrics.total_encoding_time_ms(), 1000);
        assert_eq!(metrics.total_bytes(), 1024);
    }

    #[test]
    fn test_metrics_prometheus_format() {
        let metrics = Metrics::new();
        metrics.inc_jobs_total();
        metrics.inc_workers();

        let output = metrics.to_prometheus();
        assert!(output.contains("zvd_jobs_total 1"));
        assert!(output.contains("zvd_worker_count 1"));
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
    }

    #[test]
    fn test_metrics_json() {
        let metrics = Metrics::new();
        metrics.inc_jobs_total();
        metrics.inc_jobs_completed();

        let json = metrics.to_json();
        assert_eq!(json["jobs_total"], 1);
        assert_eq!(json["jobs_completed"], 1);
    }

    #[test]
    fn test_worker_metrics() {
        let worker_metrics = WorkerMetrics::new("worker-1");

        worker_metrics.record_completion(2000, 2048);
        assert_eq!(worker_metrics.jobs_processed.load(Ordering::Relaxed), 1);

        let speed = worker_metrics.avg_encoding_speed();
        assert!((speed - 1024.0).abs() < 0.1);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new();
        metrics.inc_jobs_total();
        metrics.inc_workers();

        metrics.reset();

        assert_eq!(metrics.total_jobs(), 0);
        assert_eq!(metrics.workers(), 0);
    }
}
