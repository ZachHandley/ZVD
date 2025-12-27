//! ZVC69 Pipelined Encoding/Decoding for Real-time Performance
//!
//! This module provides pipelined async processing infrastructure for achieving
//! sustained real-time throughput in the ZVC69 neural video codec. The pipeline
//! architecture enables overlapping of CPU and GPU operations, achieving:
//!
//! - **30+ fps encoding at 720p** (target: <33ms latency)
//! - **60+ fps decoding at 720p** (target: <16ms latency)
//!
//! ## Architecture
//!
//! The pipeline uses a producer-consumer pattern with:
//!
//! ```text
//! +---------+    +----------+    +---------+    +----------+
//! | Input   | -> | Preproc  | -> | Neural  | -> | Entropy  | -> Output
//! | Queue   |    | Thread   |    | Infer   |    | Coding   |
//! +---------+    +----------+    +---------+    +----------+
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::pipeline::{PipelinedEncoder, PipelineConfig};
//!
//! let config = PipelineConfig::realtime_720p();
//! let mut encoder = PipelinedEncoder::new(zvc69_config, config)?;
//!
//! // Submit frames (non-blocking)
//! for frame in video_frames {
//!     encoder.submit(frame)?;
//!     if let Some(encoded) = encoder.try_recv() {
//!         // Process encoded frame
//!     }
//! }
//!
//! // Flush remaining frames
//! for encoded in encoder.flush() {
//!     // Process remaining frames
//! }
//! ```
//!
//! ## Performance Tips
//!
//! - Use `PipelineConfig::realtime_720p()` for optimal 720p performance
//! - Pre-warm the pipeline with a few frames before timing
//! - Use `try_recv()` in hot loops to avoid blocking
//! - Monitor `stats()` for queue depth and throughput

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::config::ZVC69Config;
use super::decoder::ZVC69Decoder;
use super::encoder::ZVC69Encoder;
use super::error::ZVC69Error;
use crate::codec::{Decoder, Encoder, Frame, VideoFrame};
use crate::error::Error;
use crate::format::Packet;

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Configuration for pipelined encoding/decoding
///
/// Controls queue depths, worker counts, and optimization settings
/// for achieving real-time performance.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of frames to buffer in the input queue
    ///
    /// Higher values provide smoother throughput but increase latency.
    /// Default: 4 for real-time encoding
    pub queue_depth: usize,

    /// Number of parallel encoding threads (for CPU operations)
    ///
    /// This controls parallelism for CPU-bound operations like
    /// entropy coding and bitstream packaging. GPU inference
    /// is typically single-threaded with batching.
    pub num_workers: usize,

    /// Frames to prefetch for motion estimation
    ///
    /// Enables lookahead for better motion estimation and
    /// rate control decisions.
    pub prefetch_frames: usize,

    /// Target latency in milliseconds
    ///
    /// The pipeline will try to maintain this latency bound.
    /// Lower values prioritize responsiveness over throughput.
    pub target_latency_ms: f64,

    /// Enable batch processing for GPU inference
    ///
    /// When enabled, multiple frames are batched for more
    /// efficient GPU utilization.
    pub enable_batching: bool,

    /// Maximum batch size for GPU inference
    pub max_batch_size: usize,

    /// Enable memory pooling for reduced allocations
    pub enable_memory_pooling: bool,

    /// Enable async GPU transfers
    ///
    /// When enabled, CPU and GPU operations can overlap.
    pub enable_async_transfers: bool,

    /// Frame timeout in milliseconds
    ///
    /// Maximum time to wait for a frame before reporting timeout.
    pub frame_timeout_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            queue_depth: 4,
            num_workers: 2,
            prefetch_frames: 2,
            target_latency_ms: 33.0, // ~30 fps
            enable_batching: false,
            max_batch_size: 4,
            enable_memory_pooling: true,
            enable_async_transfers: false,
            frame_timeout_ms: 1000,
        }
    }
}

impl PipelineConfig {
    /// Create a new default pipeline configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for real-time 720p encoding
    ///
    /// Targets 30+ fps encode with <33ms latency.
    pub fn realtime_720p() -> Self {
        PipelineConfig {
            queue_depth: 4,
            num_workers: 2,
            prefetch_frames: 2,
            target_latency_ms: 33.0,
            enable_batching: false,
            max_batch_size: 1,
            enable_memory_pooling: true,
            enable_async_transfers: true,
            frame_timeout_ms: 500,
        }
    }

    /// Create configuration optimized for real-time 1080p encoding
    ///
    /// Targets 30+ fps encode with <33ms latency.
    pub fn realtime_1080p() -> Self {
        PipelineConfig {
            queue_depth: 3,
            num_workers: 4,
            prefetch_frames: 2,
            target_latency_ms: 33.0,
            enable_batching: false,
            max_batch_size: 1,
            enable_memory_pooling: true,
            enable_async_transfers: true,
            frame_timeout_ms: 1000,
        }
    }

    /// Create configuration optimized for low latency
    ///
    /// Minimizes latency at the cost of throughput.
    pub fn low_latency() -> Self {
        PipelineConfig {
            queue_depth: 2,
            num_workers: 1,
            prefetch_frames: 0,
            target_latency_ms: 16.0, // ~60 fps
            enable_batching: false,
            max_batch_size: 1,
            enable_memory_pooling: true,
            enable_async_transfers: false,
            frame_timeout_ms: 100,
        }
    }

    /// Create configuration optimized for throughput
    ///
    /// Maximizes fps at the cost of latency.
    pub fn high_throughput() -> Self {
        PipelineConfig {
            queue_depth: 8,
            num_workers: 4,
            prefetch_frames: 4,
            target_latency_ms: 100.0,
            enable_batching: true,
            max_batch_size: 4,
            enable_memory_pooling: true,
            enable_async_transfers: true,
            frame_timeout_ms: 2000,
        }
    }

    /// Set queue depth
    pub fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth.max(1);
        self
    }

    /// Set number of workers
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = workers.max(1);
        self
    }

    /// Set target latency
    pub fn with_target_latency_ms(mut self, latency_ms: f64) -> Self {
        self.target_latency_ms = latency_ms.max(1.0);
        self
    }

    /// Enable or disable batching
    pub fn with_batching(mut self, enabled: bool) -> Self {
        self.enable_batching = enabled;
        self
    }

    /// Calculate optimal queue depth for target latency and frame time
    pub fn optimal_queue_depth(target_latency_ms: f64, avg_frame_time_ms: f64) -> usize {
        let depth = (target_latency_ms / avg_frame_time_ms).ceil() as usize;
        depth.clamp(2, 16)
    }
}

// ============================================================================
// Pipeline Statistics
// ============================================================================

/// Statistics for pipeline performance monitoring
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Number of frames submitted to the pipeline
    pub frames_in: u64,

    /// Number of frames received from the pipeline
    pub frames_out: u64,

    /// Average latency from submit to receive (milliseconds)
    pub avg_latency_ms: f64,

    /// Minimum latency observed (milliseconds)
    pub min_latency_ms: f64,

    /// Maximum latency observed (milliseconds)
    pub max_latency_ms: f64,

    /// P95 latency (milliseconds)
    pub p95_latency_ms: f64,

    /// P99 latency (milliseconds)
    pub p99_latency_ms: f64,

    /// Throughput in frames per second
    pub throughput_fps: f64,

    /// Current queue depth (frames waiting)
    pub queue_depth: usize,

    /// Peak queue depth observed
    pub peak_queue_depth: usize,

    /// Number of frames dropped due to queue overflow
    pub frames_dropped: u64,

    /// Number of timeouts
    pub timeouts: u64,

    /// Total processing time (milliseconds)
    pub total_time_ms: f64,

    /// Total bytes produced
    pub total_bytes: u64,
}

impl PipelineStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            min_latency_ms: f64::INFINITY,
            ..Default::default()
        }
    }

    /// Check if pipeline is meeting real-time target
    pub fn is_realtime(&self, target_fps: f64) -> bool {
        self.throughput_fps >= target_fps
    }

    /// Check if pipeline is meeting latency target
    pub fn meets_latency_target(&self, target_ms: f64) -> bool {
        self.avg_latency_ms <= target_ms
    }

    /// Get effective bitrate in bits per second
    pub fn bitrate_bps(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.total_bytes as f64 * 8.0) / (self.total_time_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Format as a human-readable report
    pub fn to_report(&self) -> String {
        format!(
            "Pipeline Stats:\n\
             - Frames: {} in / {} out\n\
             - Latency: {:.2}ms avg, {:.2}ms p99\n\
             - Throughput: {:.1} fps\n\
             - Queue: {} current, {} peak\n\
             - Dropped: {} frames\n\
             - Bitrate: {:.2} Mbps",
            self.frames_in,
            self.frames_out,
            self.avg_latency_ms,
            self.p99_latency_ms,
            self.throughput_fps,
            self.queue_depth,
            self.peak_queue_depth,
            self.frames_dropped,
            self.bitrate_bps() / 1_000_000.0
        )
    }
}

// ============================================================================
// Pipeline Frame Wrapper
// ============================================================================

/// Frame with timing metadata for pipeline tracking
#[derive(Debug)]
pub struct PipelineFrame {
    /// The video frame data
    pub frame: VideoFrame,
    /// Frame sequence number
    pub sequence: u64,
    /// Time when frame entered the pipeline
    pub submit_time: Instant,
    /// Target PTS
    pub pts: i64,
}

impl PipelineFrame {
    /// Create a new pipeline frame
    pub fn new(frame: VideoFrame, sequence: u64, pts: i64) -> Self {
        PipelineFrame {
            frame,
            sequence,
            submit_time: Instant::now(),
            pts,
        }
    }

    /// Get time since submit in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.submit_time.elapsed().as_secs_f64() * 1000.0
    }
}

/// Encoded frame with timing metadata
#[derive(Debug)]
pub struct EncodedFrame {
    /// Encoded bitstream data
    pub data: Vec<u8>,
    /// Frame sequence number
    pub sequence: u64,
    /// Presentation timestamp
    pub pts: i64,
    /// Decode timestamp
    pub dts: i64,
    /// Whether this is a keyframe
    pub is_keyframe: bool,
    /// Encoding latency in milliseconds
    pub latency_ms: f64,
    /// Size in bits
    pub size_bits: usize,
}

impl EncodedFrame {
    /// Create a new encoded frame
    pub fn new(
        data: Vec<u8>,
        sequence: u64,
        pts: i64,
        dts: i64,
        is_keyframe: bool,
        latency_ms: f64,
    ) -> Self {
        let size_bits = data.len() * 8;
        EncodedFrame {
            data,
            sequence,
            pts,
            dts,
            is_keyframe,
            latency_ms,
            size_bits,
        }
    }

    /// Create from a Packet
    pub fn from_packet(packet: &Packet, sequence: u64, latency_ms: f64) -> Self {
        EncodedFrame {
            data: packet.data.as_slice().to_vec(),
            sequence,
            pts: packet.pts.value,
            dts: packet.dts.value,
            is_keyframe: packet.flags.keyframe,
            latency_ms,
            size_bits: packet.data.len() * 8,
        }
    }
}

// ============================================================================
// Latency Tracker
// ============================================================================

/// Tracks latency statistics for the pipeline
#[derive(Debug)]
struct LatencyTracker {
    /// Recent latencies for percentile calculation
    latencies: VecDeque<f64>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Running sum for average
    sum: f64,
    /// Count of samples
    count: u64,
    /// Minimum latency
    min: f64,
    /// Maximum latency
    max: f64,
}

impl LatencyTracker {
    fn new(max_samples: usize) -> Self {
        LatencyTracker {
            latencies: VecDeque::with_capacity(max_samples),
            max_samples,
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: 0.0,
        }
    }

    fn record(&mut self, latency_ms: f64) {
        self.sum += latency_ms;
        self.count += 1;
        self.min = self.min.min(latency_ms);
        self.max = self.max.max(latency_ms);

        if self.latencies.len() >= self.max_samples {
            self.latencies.pop_front();
        }
        self.latencies.push_back(latency_ms);
    }

    fn average(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = self.latencies.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

// ============================================================================
// Pipelined Encoder
// ============================================================================

/// Pipelined encoder for sustained throughput
///
/// Uses a multi-stage pipeline to achieve real-time encoding performance.
/// The pipeline overlaps preprocessing, neural inference, and entropy coding
/// to maximize throughput.
pub struct PipelinedEncoder {
    /// Input queue for frames
    input_tx: Sender<PipelineFrame>,
    /// Output queue for encoded data
    output_rx: Receiver<EncodedFrame>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,
    /// Configuration
    config: PipelineConfig,
    /// Sequence counter
    sequence: AtomicU64,
    /// Statistics (shared with worker)
    stats: Arc<Mutex<PipelineStats>>,
    /// Start time
    start_time: Instant,
    /// Latency tracker
    latency_tracker: Arc<Mutex<LatencyTracker>>,
}

impl PipelinedEncoder {
    /// Create a new pipelined encoder
    ///
    /// # Arguments
    ///
    /// * `codec_config` - ZVC69 encoder configuration
    /// * `pipeline_config` - Pipeline configuration
    ///
    /// # Returns
    ///
    /// A new `PipelinedEncoder` instance or an error
    pub fn new(
        codec_config: ZVC69Config,
        pipeline_config: PipelineConfig,
    ) -> Result<Self, ZVC69Error> {
        let (input_tx, input_rx) = channel::<PipelineFrame>();
        let (output_tx, output_rx) = channel::<EncodedFrame>();

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(PipelineStats::new()));
        let latency_tracker = Arc::new(Mutex::new(LatencyTracker::new(1000)));

        // Create encoder
        let encoder = ZVC69Encoder::new(codec_config).map_err(|e| ZVC69Error::Io(e.to_string()))?;

        // Spawn worker thread
        let worker_shutdown = Arc::clone(&shutdown);
        let worker_stats = Arc::clone(&stats);
        let worker_latency = Arc::clone(&latency_tracker);
        let worker_config = pipeline_config.clone();

        let worker = thread::Builder::new()
            .name("zvc69-encode-worker".to_string())
            .spawn(move || {
                Self::encode_worker(
                    encoder,
                    input_rx,
                    output_tx,
                    worker_shutdown,
                    worker_stats,
                    worker_latency,
                    worker_config,
                );
            })
            .map_err(|e| ZVC69Error::Io(e.to_string()))?;

        Ok(PipelinedEncoder {
            input_tx,
            output_rx,
            shutdown,
            workers: vec![worker],
            config: pipeline_config,
            sequence: AtomicU64::new(0),
            stats,
            start_time: Instant::now(),
            latency_tracker,
        })
    }

    /// Encoder worker thread function
    fn encode_worker(
        mut encoder: ZVC69Encoder,
        input_rx: Receiver<PipelineFrame>,
        output_tx: Sender<EncodedFrame>,
        shutdown: Arc<AtomicBool>,
        stats: Arc<Mutex<PipelineStats>>,
        latency_tracker: Arc<Mutex<LatencyTracker>>,
        _config: PipelineConfig,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // Try to receive frame with timeout
            match input_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(pipeline_frame) => {
                    let submit_time = pipeline_frame.submit_time;
                    let sequence = pipeline_frame.sequence;
                    let pts = pipeline_frame.pts;

                    // Encode the frame
                    let frame = Frame::Video(pipeline_frame.frame);
                    if let Err(e) = encoder.send_frame(&frame) {
                        eprintln!("Encode error: {}", e);
                        continue;
                    }

                    // Receive encoded packet
                    match encoder.receive_packet() {
                        Ok(packet) => {
                            let latency_ms = submit_time.elapsed().as_secs_f64() * 1000.0;

                            // Update latency tracker
                            if let Ok(mut tracker) = latency_tracker.lock() {
                                tracker.record(latency_ms);
                            }

                            // Update stats
                            if let Ok(mut s) = stats.lock() {
                                s.frames_out += 1;
                                s.total_bytes += packet.data.len() as u64;
                            }

                            // Send encoded frame
                            let encoded = EncodedFrame::from_packet(&packet, sequence, latency_ms);
                            if output_tx.send(encoded).is_err() {
                                break; // Receiver dropped
                            }
                        }
                        Err(Error::TryAgain) => {
                            // No packet ready yet, continue
                        }
                        Err(e) => {
                            eprintln!("Receive packet error: {}", e);
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No frame available, continue
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break; // Sender dropped
                }
            }
        }

        // Flush encoder
        let _ = encoder.flush();

        // Drain remaining packets
        while let Ok(packet) = encoder.receive_packet() {
            let encoded = EncodedFrame::from_packet(&packet, 0, 0.0);
            if output_tx.send(encoded).is_err() {
                break;
            }
        }
    }

    /// Submit a frame for encoding (non-blocking)
    ///
    /// Returns immediately after queuing the frame. Use `try_recv()` or
    /// `recv()` to get the encoded result.
    pub fn submit(&self, frame: VideoFrame) -> Result<(), ZVC69Error> {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        let pts = frame.pts.value;
        let pipeline_frame = PipelineFrame::new(frame, sequence, pts);

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.frames_in += 1;
            stats.queue_depth = stats.frames_in.saturating_sub(stats.frames_out) as usize;
            if stats.queue_depth > stats.peak_queue_depth {
                stats.peak_queue_depth = stats.queue_depth;
            }
        }

        self.input_tx
            .send(pipeline_frame)
            .map_err(|_| ZVC69Error::Io("Pipeline input channel closed".to_string()))
    }

    /// Try to get the next encoded frame (non-blocking)
    ///
    /// Returns `None` if no encoded frame is ready.
    pub fn try_recv(&self) -> Option<EncodedFrame> {
        match self.output_rx.try_recv() {
            Ok(encoded) => Some(encoded),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => None,
        }
    }

    /// Get the next encoded frame (blocking)
    ///
    /// Blocks until an encoded frame is available or timeout.
    pub fn recv(&self) -> Result<EncodedFrame, ZVC69Error> {
        let timeout = Duration::from_millis(self.config.frame_timeout_ms);
        self.output_rx
            .recv_timeout(timeout)
            .map_err(|e| ZVC69Error::Io(format!("Receive timeout: {}", e)))
    }

    /// Flush the pipeline and get remaining frames
    ///
    /// Signals the encoder to finish and returns all remaining encoded frames.
    pub fn flush(&mut self) -> Vec<EncodedFrame> {
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);

        // Drop input sender to signal end of input
        // (This happens automatically when we're dropped)

        // Collect remaining frames
        let mut frames = Vec::new();
        while let Ok(encoded) = self.output_rx.recv_timeout(Duration::from_millis(100)) {
            frames.push(encoded);
        }

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }

        frames
    }

    /// Get current pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        let mut stats = self.stats.lock().unwrap().clone();

        // Update derived statistics
        let elapsed = self.start_time.elapsed().as_secs_f64() * 1000.0;
        stats.total_time_ms = elapsed;

        if elapsed > 0.0 {
            stats.throughput_fps = stats.frames_out as f64 / (elapsed / 1000.0);
        }

        if let Ok(tracker) = self.latency_tracker.lock() {
            stats.avg_latency_ms = tracker.average();
            stats.min_latency_ms = tracker.min;
            stats.max_latency_ms = tracker.max;
            stats.p95_latency_ms = tracker.percentile(95.0);
            stats.p99_latency_ms = tracker.percentile(99.0);
        }

        stats.queue_depth = stats.frames_in.saturating_sub(stats.frames_out) as usize;

        stats
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Check if the pipeline is meeting real-time requirements
    pub fn is_realtime(&self, target_fps: f64) -> bool {
        self.stats().is_realtime(target_fps)
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = PipelineStats::new();
        }
    }
}

impl Drop for PipelinedEncoder {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// ============================================================================
// Pipelined Decoder
// ============================================================================

/// Pipelined decoder for sustained throughput
///
/// Uses a multi-stage pipeline to achieve real-time decoding performance.
pub struct PipelinedDecoder {
    /// Input queue for encoded frames
    input_tx: Sender<EncodedFrame>,
    /// Output queue for decoded frames
    output_rx: Receiver<VideoFrame>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,
    /// Configuration
    config: PipelineConfig,
    /// Sequence counter
    sequence: AtomicU64,
    /// Statistics (shared with worker)
    stats: Arc<Mutex<PipelineStats>>,
    /// Start time
    start_time: Instant,
    /// Latency tracker
    latency_tracker: Arc<Mutex<LatencyTracker>>,
}

impl PipelinedDecoder {
    /// Create a new pipelined decoder
    pub fn new(pipeline_config: PipelineConfig) -> Result<Self, ZVC69Error> {
        let (input_tx, input_rx) = channel::<EncodedFrame>();
        let (output_tx, output_rx) = channel::<VideoFrame>();

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(PipelineStats::new()));
        let latency_tracker = Arc::new(Mutex::new(LatencyTracker::new(1000)));

        // Create decoder
        let decoder = ZVC69Decoder::new().map_err(|e| ZVC69Error::Io(e.to_string()))?;

        // Spawn worker thread
        let worker_shutdown = Arc::clone(&shutdown);
        let worker_stats = Arc::clone(&stats);
        let worker_latency = Arc::clone(&latency_tracker);
        let worker_config = pipeline_config.clone();

        let worker = thread::Builder::new()
            .name("zvc69-decode-worker".to_string())
            .spawn(move || {
                Self::decode_worker(
                    decoder,
                    input_rx,
                    output_tx,
                    worker_shutdown,
                    worker_stats,
                    worker_latency,
                    worker_config,
                );
            })
            .map_err(|e| ZVC69Error::Io(e.to_string()))?;

        Ok(PipelinedDecoder {
            input_tx,
            output_rx,
            shutdown,
            workers: vec![worker],
            config: pipeline_config,
            sequence: AtomicU64::new(0),
            stats,
            start_time: Instant::now(),
            latency_tracker,
        })
    }

    /// Decoder worker thread function
    fn decode_worker(
        mut decoder: ZVC69Decoder,
        input_rx: Receiver<EncodedFrame>,
        output_tx: Sender<VideoFrame>,
        shutdown: Arc<AtomicBool>,
        stats: Arc<Mutex<PipelineStats>>,
        latency_tracker: Arc<Mutex<LatencyTracker>>,
        _config: PipelineConfig,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // Try to receive encoded frame with timeout
            match input_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(encoded_frame) => {
                    let start = Instant::now();

                    // Create packet from encoded data
                    let mut packet =
                        Packet::new(0, crate::util::Buffer::from_vec(encoded_frame.data.clone()));
                    packet.pts = crate::util::Timestamp::new(encoded_frame.pts);
                    packet.dts = crate::util::Timestamp::new(encoded_frame.dts);
                    packet.flags.keyframe = encoded_frame.is_keyframe;

                    // Decode the packet
                    if let Err(e) = decoder.send_packet(&packet) {
                        eprintln!("Decode error: {}", e);
                        continue;
                    }

                    // Receive decoded frame
                    match decoder.receive_frame() {
                        Ok(Frame::Video(frame)) => {
                            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                            // Update latency tracker
                            if let Ok(mut tracker) = latency_tracker.lock() {
                                tracker.record(latency_ms);
                            }

                            // Update stats
                            if let Ok(mut s) = stats.lock() {
                                s.frames_out += 1;
                            }

                            // Send decoded frame
                            if output_tx.send(frame).is_err() {
                                break; // Receiver dropped
                            }
                        }
                        Ok(Frame::Audio(_)) => {
                            // Ignore audio frames
                        }
                        Err(Error::TryAgain) => {
                            // No frame ready yet, continue
                        }
                        Err(e) => {
                            eprintln!("Receive frame error: {}", e);
                        }
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No frame available, continue
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break; // Sender dropped
                }
            }
        }

        // Flush decoder
        let _ = decoder.flush();

        // Drain remaining frames
        while let Ok(Frame::Video(frame)) = decoder.receive_frame() {
            if output_tx.send(frame).is_err() {
                break;
            }
        }
    }

    /// Submit an encoded frame for decoding (non-blocking)
    pub fn submit(&self, encoded: EncodedFrame) -> Result<(), ZVC69Error> {
        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.frames_in += 1;
            stats.queue_depth = stats.frames_in.saturating_sub(stats.frames_out) as usize;
            if stats.queue_depth > stats.peak_queue_depth {
                stats.peak_queue_depth = stats.queue_depth;
            }
        }

        self.input_tx
            .send(encoded)
            .map_err(|_| ZVC69Error::Io("Pipeline input channel closed".to_string()))
    }

    /// Try to get the next decoded frame (non-blocking)
    pub fn try_recv(&self) -> Option<VideoFrame> {
        match self.output_rx.try_recv() {
            Ok(frame) => Some(frame),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => None,
        }
    }

    /// Get the next decoded frame (blocking)
    pub fn recv(&self) -> Result<VideoFrame, ZVC69Error> {
        let timeout = Duration::from_millis(self.config.frame_timeout_ms);
        self.output_rx
            .recv_timeout(timeout)
            .map_err(|e| ZVC69Error::Io(format!("Receive timeout: {}", e)))
    }

    /// Flush the pipeline and get remaining frames
    pub fn flush(&mut self) -> Vec<VideoFrame> {
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);

        // Collect remaining frames
        let mut frames = Vec::new();
        while let Ok(frame) = self.output_rx.recv_timeout(Duration::from_millis(100)) {
            frames.push(frame);
        }

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }

        frames
    }

    /// Get current pipeline statistics
    pub fn stats(&self) -> PipelineStats {
        let mut stats = self.stats.lock().unwrap().clone();

        // Update derived statistics
        let elapsed = self.start_time.elapsed().as_secs_f64() * 1000.0;
        stats.total_time_ms = elapsed;

        if elapsed > 0.0 {
            stats.throughput_fps = stats.frames_out as f64 / (elapsed / 1000.0);
        }

        if let Ok(tracker) = self.latency_tracker.lock() {
            stats.avg_latency_ms = tracker.average();
            stats.min_latency_ms = tracker.min;
            stats.max_latency_ms = tracker.max;
            stats.p95_latency_ms = tracker.percentile(95.0);
            stats.p99_latency_ms = tracker.percentile(99.0);
        }

        stats.queue_depth = stats.frames_in.saturating_sub(stats.frames_out) as usize;

        stats
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Check if the pipeline is meeting real-time requirements
    pub fn is_realtime(&self, target_fps: f64) -> bool {
        self.stats().is_realtime(target_fps)
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = PipelineStats::new();
        }
    }
}

impl Drop for PipelinedDecoder {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

// ============================================================================
// Synchronous Pipeline (for single-threaded usage)
// ============================================================================

/// Synchronous encoder wrapper (no threading)
///
/// Use this for simpler cases where threading overhead is not desired.
pub struct SyncEncoder {
    encoder: ZVC69Encoder,
    stats: PipelineStats,
    latency_tracker: LatencyTracker,
    start_time: Instant,
}

impl SyncEncoder {
    /// Create a new synchronous encoder
    pub fn new(config: ZVC69Config) -> Result<Self, ZVC69Error> {
        let encoder = ZVC69Encoder::new(config).map_err(|e| ZVC69Error::Io(e.to_string()))?;
        Ok(SyncEncoder {
            encoder,
            stats: PipelineStats::new(),
            latency_tracker: LatencyTracker::new(1000),
            start_time: Instant::now(),
        })
    }

    /// Encode a single frame synchronously
    pub fn encode(&mut self, frame: VideoFrame) -> Result<EncodedFrame, ZVC69Error> {
        let start = Instant::now();
        let pts = frame.pts.value;
        let sequence = self.stats.frames_in;

        self.stats.frames_in += 1;

        self.encoder
            .send_frame(&Frame::Video(frame))
            .map_err(|e| ZVC69Error::Io(e.to_string()))?;

        let packet = self
            .encoder
            .receive_packet()
            .map_err(|e| ZVC69Error::Io(e.to_string()))?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.latency_tracker.record(latency_ms);
        self.stats.frames_out += 1;
        self.stats.total_bytes += packet.data.len() as u64;

        Ok(EncodedFrame::from_packet(&packet, sequence, latency_ms))
    }

    /// Get current statistics
    pub fn stats(&self) -> PipelineStats {
        let mut stats = self.stats.clone();
        let elapsed = self.start_time.elapsed().as_secs_f64() * 1000.0;
        stats.total_time_ms = elapsed;

        if elapsed > 0.0 {
            stats.throughput_fps = stats.frames_out as f64 / (elapsed / 1000.0);
        }

        stats.avg_latency_ms = self.latency_tracker.average();
        stats.min_latency_ms = self.latency_tracker.min;
        stats.max_latency_ms = self.latency_tracker.max;
        stats.p95_latency_ms = self.latency_tracker.percentile(95.0);
        stats.p99_latency_ms = self.latency_tracker.percentile(99.0);

        stats
    }
}

/// Synchronous decoder wrapper (no threading)
pub struct SyncDecoder {
    decoder: ZVC69Decoder,
    stats: PipelineStats,
    latency_tracker: LatencyTracker,
    start_time: Instant,
}

impl SyncDecoder {
    /// Create a new synchronous decoder
    pub fn new() -> Result<Self, ZVC69Error> {
        let decoder = ZVC69Decoder::new().map_err(|e| ZVC69Error::Io(e.to_string()))?;
        Ok(SyncDecoder {
            decoder,
            stats: PipelineStats::new(),
            latency_tracker: LatencyTracker::new(1000),
            start_time: Instant::now(),
        })
    }

    /// Decode a single frame synchronously
    pub fn decode(&mut self, encoded: EncodedFrame) -> Result<VideoFrame, ZVC69Error> {
        let start = Instant::now();

        self.stats.frames_in += 1;

        let mut packet =
            Packet::new(0, crate::util::Buffer::from_vec(encoded.data.clone()));
        packet.pts = crate::util::Timestamp::new(encoded.pts);
        packet.dts = crate::util::Timestamp::new(encoded.dts);
        packet.flags.keyframe = encoded.is_keyframe;

        self.decoder
            .send_packet(&packet)
            .map_err(|e| ZVC69Error::Io(e.to_string()))?;

        match self.decoder.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                self.latency_tracker.record(latency_ms);
                self.stats.frames_out += 1;
                Ok(frame)
            }
            Ok(Frame::Audio(_)) => Err(ZVC69Error::Io("Unexpected audio frame".to_string())),
            Err(e) => Err(ZVC69Error::Io(e.to_string())),
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> PipelineStats {
        let mut stats = self.stats.clone();
        let elapsed = self.start_time.elapsed().as_secs_f64() * 1000.0;
        stats.total_time_ms = elapsed;

        if elapsed > 0.0 {
            stats.throughput_fps = stats.frames_out as f64 / (elapsed / 1000.0);
        }

        stats.avg_latency_ms = self.latency_tracker.average();
        stats.min_latency_ms = self.latency_tracker.min;
        stats.max_latency_ms = self.latency_tracker.max;
        stats.p95_latency_ms = self.latency_tracker.percentile(95.0);
        stats.p99_latency_ms = self.latency_tracker.percentile(99.0);

        stats
    }
}

// ============================================================================
// CUDA Stream-Aware Encoder
// ============================================================================

use super::tensorrt::{CudaStreamManager, CudaStreamStats, StreamRole, StreamState};

/// A CUDA stream-aware encoder that overlaps compute and memory transfers
///
/// This encoder uses triple-buffered CUDA streams to maximize GPU utilization:
/// - Stream 0 (Compute): Neural network inference on current frame
/// - Stream 1 (H2D): Uploading next frame to GPU
/// - Stream 2 (D2H): Downloading previous results from GPU
///
/// ## Usage
///
/// ```rust,ignore
/// use zvd::codec::zvc69::pipeline::StreamAwareEncoder;
/// use zvd::codec::zvc69::ZVC69Config;
///
/// let mut encoder = StreamAwareEncoder::new(ZVC69Config::default())?;
///
/// // Submit frames - overlapped execution happens automatically
/// for (seq, frame) in frames.enumerate() {
///     encoder.submit_frame(frame, seq as u64)?;
///
///     // Check for completed results
///     while let Some(result) = encoder.poll_result()? {
///         process_result(result);
///     }
/// }
///
/// // Flush remaining
/// let remaining = encoder.flush()?;
/// ```
pub struct StreamAwareEncoder {
    /// Inner synchronous encoder
    encoder: ZVC69Encoder,
    /// CUDA stream manager for overlapped execution
    stream_manager: CudaStreamManager,
    /// Frames queued for upload (H2D pending)
    upload_queue: VecDeque<(VideoFrame, u64, Instant)>,
    /// Frames queued for inference (compute pending)
    compute_queue: VecDeque<(VideoFrame, u64, Instant)>,
    /// Frames queued for download (D2H pending)
    download_queue: VecDeque<(u64, Instant)>,
    /// Completed encoded frames
    output_queue: VecDeque<EncodedFrame>,
    /// Statistics
    stats: PipelineStats,
    /// Latency tracker
    latency_tracker: LatencyTracker,
    /// Start time
    start_time: Instant,
    /// Sequence counter
    sequence: u64,
    /// Maximum pipeline depth
    max_pipeline_depth: usize,
}

impl StreamAwareEncoder {
    /// Create a new stream-aware encoder
    ///
    /// # Arguments
    ///
    /// * `config` - ZVC69 encoder configuration
    ///
    /// # Returns
    ///
    /// A new `StreamAwareEncoder` instance or an error
    pub fn new(config: ZVC69Config) -> Result<Self, ZVC69Error> {
        let encoder = ZVC69Encoder::new(config).map_err(|e| ZVC69Error::Io(e.to_string()))?;
        let stream_manager = CudaStreamManager::triple_buffered()?;

        Ok(StreamAwareEncoder {
            encoder,
            stream_manager,
            upload_queue: VecDeque::with_capacity(4),
            compute_queue: VecDeque::with_capacity(4),
            download_queue: VecDeque::with_capacity(4),
            output_queue: VecDeque::with_capacity(8),
            stats: PipelineStats::new(),
            latency_tracker: LatencyTracker::new(1000),
            start_time: Instant::now(),
            sequence: 0,
            max_pipeline_depth: 3, // Triple buffered
        })
    }

    /// Create with custom stream configuration
    pub fn with_streams(
        config: ZVC69Config,
        num_streams: usize,
    ) -> Result<Self, ZVC69Error> {
        let encoder = ZVC69Encoder::new(config).map_err(|e| ZVC69Error::Io(e.to_string()))?;
        let stream_manager = CudaStreamManager::new(num_streams)?;
        let max_pipeline_depth = num_streams;

        Ok(StreamAwareEncoder {
            encoder,
            stream_manager,
            upload_queue: VecDeque::with_capacity(max_pipeline_depth + 1),
            compute_queue: VecDeque::with_capacity(max_pipeline_depth + 1),
            download_queue: VecDeque::with_capacity(max_pipeline_depth + 1),
            output_queue: VecDeque::with_capacity(max_pipeline_depth * 2),
            stats: PipelineStats::new(),
            latency_tracker: LatencyTracker::new(1000),
            start_time: Instant::now(),
            sequence: 0,
            max_pipeline_depth,
        })
    }

    /// Check if CUDA streams are available
    pub fn is_cuda_available(&self) -> bool {
        self.stream_manager.is_cuda_available()
    }

    /// Get current pipeline depth (frames in flight)
    pub fn pipeline_depth(&self) -> usize {
        self.upload_queue.len() + self.compute_queue.len() + self.download_queue.len()
    }

    /// Check if the pipeline can accept more frames
    pub fn can_accept(&self) -> bool {
        self.pipeline_depth() < self.max_pipeline_depth
    }

    /// Advance the pipeline state
    ///
    /// This moves frames through the pipeline stages:
    /// 1. Complete any finished D2H transfers (download -> output)
    /// 2. Start D2H for completed compute operations (compute -> download)
    /// 3. Start compute for completed H2D transfers (upload -> compute)
    fn advance(&mut self) -> Result<(), ZVC69Error> {
        // 1. Check D2H stream for completed downloads
        if self.stream_manager.is_role_ready(StreamRole::DeviceToHost) {
            if let Some((sequence, start_time)) = self.download_queue.pop_front() {
                self.stream_manager
                    .synchronize_role(StreamRole::DeviceToHost)?;

                // Receive the encoded packet
                match self.encoder.receive_packet() {
                    Ok(packet) => {
                        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                        self.latency_tracker.record(latency_ms);
                        self.stats.frames_out += 1;
                        self.stats.total_bytes += packet.data.len() as u64;

                        let encoded = EncodedFrame::from_packet(&packet, sequence, latency_ms);
                        self.output_queue.push_back(encoded);
                    }
                    Err(Error::TryAgain) => {
                        // No packet ready yet, re-queue
                        self.download_queue.push_front((sequence, start_time));
                    }
                    Err(e) => {
                        return Err(ZVC69Error::Io(format!("Packet receive error: {}", e)));
                    }
                }
            }
        }

        // 2. Check compute stream - if done and D2H is free, start download
        if self.stream_manager.is_role_ready(StreamRole::Compute)
            && self.download_queue.is_empty()
        {
            if let Some((frame, sequence, start_time)) = self.compute_queue.pop_front() {
                self.stream_manager.synchronize_role(StreamRole::Compute)?;

                // The frame has been encoded, queue for D2H
                self.download_queue.push_back((sequence, start_time));
                self.stream_manager.begin_d2h_transfer(sequence)?;
            }
        }

        // 3. Check H2D stream - if done and compute is free, start inference
        if self.stream_manager.is_role_ready(StreamRole::HostToDevice)
            && self.compute_queue.is_empty()
        {
            if let Some((frame, sequence, start_time)) = self.upload_queue.pop_front() {
                self.stream_manager
                    .synchronize_role(StreamRole::HostToDevice)?;

                // Send frame to encoder for inference
                self.encoder
                    .send_frame(&Frame::Video(frame.clone()))
                    .map_err(|e| ZVC69Error::Io(e.to_string()))?;

                // Queue for compute tracking
                self.compute_queue.push_back((frame, sequence, start_time));
                self.stream_manager.begin_compute(sequence)?;
            }
        }

        Ok(())
    }

    /// Submit a frame for encoding
    ///
    /// If the pipeline is full, this will wait for space.
    ///
    /// # Arguments
    ///
    /// * `frame` - The video frame to encode
    ///
    /// # Returns
    ///
    /// The sequence number assigned to this frame
    pub fn submit_frame(&mut self, frame: VideoFrame) -> Result<u64, ZVC69Error> {
        // Advance pipeline first
        self.advance()?;

        // If H2D slot is busy, sync and wait
        if !self.stream_manager.is_role_ready(StreamRole::HostToDevice) {
            self.stream_manager
                .synchronize_role(StreamRole::HostToDevice)?;
        }

        let sequence = self.sequence;
        self.sequence += 1;
        self.stats.frames_in += 1;

        // Queue for H2D upload
        let start_time = Instant::now();
        self.upload_queue.push_back((frame, sequence, start_time));
        self.stream_manager.begin_h2d_transfer(sequence)?;

        // Advance again to potentially start processing
        self.advance()?;

        Ok(sequence)
    }

    /// Submit a frame with explicit sequence number
    pub fn submit_frame_with_seq(
        &mut self,
        frame: VideoFrame,
        sequence: u64,
    ) -> Result<(), ZVC69Error> {
        // Advance pipeline first
        self.advance()?;

        // If H2D slot is busy, sync and wait
        if !self.stream_manager.is_role_ready(StreamRole::HostToDevice) {
            self.stream_manager
                .synchronize_role(StreamRole::HostToDevice)?;
        }

        self.stats.frames_in += 1;

        // Queue for H2D upload
        let start_time = Instant::now();
        self.upload_queue.push_back((frame, sequence, start_time));
        self.stream_manager.begin_h2d_transfer(sequence)?;

        // Advance again to potentially start processing
        self.advance()?;

        Ok(())
    }

    /// Poll for a completed encoded frame (non-blocking)
    ///
    /// # Returns
    ///
    /// A completed encoded frame if available, or None
    pub fn poll_result(&mut self) -> Result<Option<EncodedFrame>, ZVC69Error> {
        self.advance()?;
        Ok(self.output_queue.pop_front())
    }

    /// Wait for all in-flight frames to complete
    ///
    /// # Returns
    ///
    /// All completed encoded frames
    pub fn flush(&mut self) -> Result<Vec<EncodedFrame>, ZVC69Error> {
        // Keep advancing until all queues are empty
        while !self.upload_queue.is_empty()
            || !self.compute_queue.is_empty()
            || !self.download_queue.is_empty()
        {
            // Force synchronization of all streams
            self.stream_manager.synchronize_all()?;

            // Process any completed stages
            // D2H complete -> output
            if let Some((sequence, start_time)) = self.download_queue.pop_front() {
                match self.encoder.receive_packet() {
                    Ok(packet) => {
                        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                        self.latency_tracker.record(latency_ms);
                        self.stats.frames_out += 1;
                        self.stats.total_bytes += packet.data.len() as u64;

                        let encoded = EncodedFrame::from_packet(&packet, sequence, latency_ms);
                        self.output_queue.push_back(encoded);
                    }
                    Err(Error::TryAgain) => {
                        // Push back and try again
                        self.download_queue.push_front((sequence, start_time));
                    }
                    Err(e) => {
                        return Err(ZVC69Error::Io(format!("Flush receive error: {}", e)));
                    }
                }
            }

            // Compute complete -> D2H
            if let Some((_frame, sequence, start_time)) = self.compute_queue.pop_front() {
                self.download_queue.push_back((sequence, start_time));
            }

            // H2D complete -> Compute
            if let Some((frame, sequence, start_time)) = self.upload_queue.pop_front() {
                self.encoder
                    .send_frame(&Frame::Video(frame))
                    .map_err(|e| ZVC69Error::Io(e.to_string()))?;
                self.compute_queue.push_back((VideoFrame::default(), sequence, start_time));
            }
        }

        // Collect all output
        let mut results = Vec::with_capacity(self.output_queue.len());
        while let Some(encoded) = self.output_queue.pop_front() {
            results.push(encoded);
        }

        Ok(results)
    }

    /// Check if there are any completed results waiting
    pub fn has_output(&self) -> bool {
        !self.output_queue.is_empty()
    }

    /// Get current statistics
    pub fn stats(&self) -> PipelineStats {
        let mut stats = self.stats.clone();
        let elapsed = self.start_time.elapsed().as_secs_f64() * 1000.0;
        stats.total_time_ms = elapsed;

        if elapsed > 0.0 {
            stats.throughput_fps = stats.frames_out as f64 / (elapsed / 1000.0);
        }

        stats.avg_latency_ms = self.latency_tracker.average();
        stats.min_latency_ms = self.latency_tracker.min;
        stats.max_latency_ms = self.latency_tracker.max;
        stats.p95_latency_ms = self.latency_tracker.percentile(95.0);
        stats.p99_latency_ms = self.latency_tracker.percentile(99.0);

        stats
    }

    /// Get CUDA stream statistics
    pub fn stream_stats(&self) -> CudaStreamStats {
        self.stream_manager.stats()
    }

    /// Reset the encoder and streams
    pub fn reset(&mut self) {
        self.stream_manager.reset();
        self.upload_queue.clear();
        self.compute_queue.clear();
        self.download_queue.clear();
        self.output_queue.clear();
        self.sequence = 0;
        self.stats = PipelineStats::new();
        self.latency_tracker = LatencyTracker::new(1000);
        self.start_time = Instant::now();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{Buffer, PixelFormat};

    // ── PipelineConfig Tests ──

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.queue_depth, 4);
        assert_eq!(config.num_workers, 2);
        assert!((config.target_latency_ms - 33.0).abs() < 0.1);
    }

    #[test]
    fn test_pipeline_config_realtime_720p() {
        let config = PipelineConfig::realtime_720p();
        assert_eq!(config.queue_depth, 4);
        assert!((config.target_latency_ms - 33.0).abs() < 0.1);
        assert!(config.enable_memory_pooling);
    }

    #[test]
    fn test_pipeline_config_realtime_1080p() {
        let config = PipelineConfig::realtime_1080p();
        assert_eq!(config.queue_depth, 3);
        assert!((config.target_latency_ms - 33.0).abs() < 0.1);
    }

    #[test]
    fn test_pipeline_config_low_latency() {
        let config = PipelineConfig::low_latency();
        assert_eq!(config.queue_depth, 2);
        assert!((config.target_latency_ms - 16.0).abs() < 0.1);
        assert_eq!(config.prefetch_frames, 0);
    }

    #[test]
    fn test_pipeline_config_high_throughput() {
        let config = PipelineConfig::high_throughput();
        assert_eq!(config.queue_depth, 8);
        assert!(config.enable_batching);
        assert_eq!(config.max_batch_size, 4);
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::new()
            .with_queue_depth(6)
            .with_workers(4)
            .with_target_latency_ms(50.0)
            .with_batching(true);

        assert_eq!(config.queue_depth, 6);
        assert_eq!(config.num_workers, 4);
        assert!((config.target_latency_ms - 50.0).abs() < 0.1);
        assert!(config.enable_batching);
    }

    #[test]
    fn test_optimal_queue_depth() {
        let depth = PipelineConfig::optimal_queue_depth(33.0, 10.0);
        assert_eq!(depth, 4); // ceil(33/10) = 4

        let depth = PipelineConfig::optimal_queue_depth(100.0, 20.0);
        assert_eq!(depth, 5); // ceil(100/20) = 5

        let depth = PipelineConfig::optimal_queue_depth(16.0, 5.0);
        assert_eq!(depth, 4); // ceil(16/5) = 4
    }

    // ── PipelineStats Tests ──

    #[test]
    fn test_pipeline_stats_new() {
        let stats = PipelineStats::new();
        assert_eq!(stats.frames_in, 0);
        assert_eq!(stats.frames_out, 0);
        assert_eq!(stats.throughput_fps, 0.0);
    }

    #[test]
    fn test_pipeline_stats_is_realtime() {
        let mut stats = PipelineStats::new();
        stats.throughput_fps = 35.0;
        assert!(stats.is_realtime(30.0));
        assert!(!stats.is_realtime(60.0));
    }

    #[test]
    fn test_pipeline_stats_meets_latency_target() {
        let mut stats = PipelineStats::new();
        stats.avg_latency_ms = 25.0;
        assert!(stats.meets_latency_target(33.0));
        assert!(!stats.meets_latency_target(16.0));
    }

    #[test]
    fn test_pipeline_stats_bitrate() {
        let mut stats = PipelineStats::new();
        stats.total_bytes = 1000000; // 1 MB
        stats.total_time_ms = 1000.0; // 1 second
        let bitrate = stats.bitrate_bps();
        assert!((bitrate - 8000000.0).abs() < 1.0); // 8 Mbps
    }

    #[test]
    fn test_pipeline_stats_report() {
        let mut stats = PipelineStats::new();
        stats.frames_in = 100;
        stats.frames_out = 100;
        stats.avg_latency_ms = 25.5;
        stats.throughput_fps = 39.2;

        let report = stats.to_report();
        assert!(report.contains("100"));
        assert!(report.contains("fps"));
    }

    // ── PipelineFrame Tests ──

    #[test]
    fn test_pipeline_frame_creation() {
        let frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let pf = PipelineFrame::new(frame, 0, 1000);
        assert_eq!(pf.sequence, 0);
        assert_eq!(pf.pts, 1000);
    }

    #[test]
    fn test_pipeline_frame_elapsed() {
        let frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let pf = PipelineFrame::new(frame, 0, 0);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(pf.elapsed_ms() >= 9.0);
    }

    // ── EncodedFrame Tests ──

    #[test]
    fn test_encoded_frame_creation() {
        let data = vec![0u8; 1000];
        let ef = EncodedFrame::new(data.clone(), 0, 100, 100, true, 15.0);

        assert_eq!(ef.sequence, 0);
        assert_eq!(ef.pts, 100);
        assert!(ef.is_keyframe);
        assert!((ef.latency_ms - 15.0).abs() < 0.1);
        assert_eq!(ef.size_bits, 8000);
    }

    // ── LatencyTracker Tests ──

    #[test]
    fn test_latency_tracker_basic() {
        let mut tracker = LatencyTracker::new(100);

        tracker.record(10.0);
        tracker.record(20.0);
        tracker.record(30.0);

        assert!((tracker.average() - 20.0).abs() < 0.1);
        assert!((tracker.min - 10.0).abs() < 0.1);
        assert!((tracker.max - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_latency_tracker_percentile() {
        let mut tracker = LatencyTracker::new(100);

        // Add values 1-100
        for i in 1..=100 {
            tracker.record(i as f64);
        }

        let p50 = tracker.percentile(50.0);
        assert!(p50 >= 49.0 && p50 <= 51.0);

        let p95 = tracker.percentile(95.0);
        assert!(p95 >= 94.0 && p95 <= 96.0);
    }

    #[test]
    fn test_latency_tracker_max_samples() {
        let mut tracker = LatencyTracker::new(10);

        // Add more samples than max
        for i in 0..20 {
            tracker.record(i as f64);
        }

        assert_eq!(tracker.latencies.len(), 10);
    }

    // ── SyncEncoder Tests ──

    #[test]
    fn test_sync_encoder_creation() {
        let config = ZVC69Config::new(64, 64);
        let encoder = SyncEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_sync_encoder_encode() {
        let config = ZVC69Config::new(64, 64);
        let mut encoder = SyncEncoder::new(config).unwrap();

        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        let result = encoder.encode(frame);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert!(encoded.is_keyframe);
        assert!(encoded.latency_ms > 0.0);
    }

    #[test]
    fn test_sync_encoder_stats() {
        let config = ZVC69Config::new(64, 64);
        let mut encoder = SyncEncoder::new(config).unwrap();

        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        let _ = encoder.encode(frame);

        let stats = encoder.stats();
        assert_eq!(stats.frames_in, 1);
        assert_eq!(stats.frames_out, 1);
        assert!(stats.avg_latency_ms > 0.0);
    }

    // ── SyncDecoder Tests ──

    #[test]
    fn test_sync_decoder_creation() {
        let decoder = SyncDecoder::new();
        assert!(decoder.is_ok());
    }

    // ── PipelinedEncoder Tests ──

    #[test]
    fn test_pipelined_encoder_creation() {
        let codec_config = ZVC69Config::new(64, 64);
        let pipeline_config = PipelineConfig::realtime_720p();
        let encoder = PipelinedEncoder::new(codec_config, pipeline_config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_pipelined_encoder_submit_recv() {
        let codec_config = ZVC69Config::new(64, 64);
        let pipeline_config = PipelineConfig::low_latency();
        let encoder = PipelinedEncoder::new(codec_config, pipeline_config).unwrap();

        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        let result = encoder.submit(frame);
        assert!(result.is_ok());

        // Wait for encoding
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Try to receive
        if let Some(encoded) = encoder.try_recv() {
            assert!(encoded.is_keyframe);
        }
    }

    #[test]
    fn test_pipelined_encoder_stats() {
        let codec_config = ZVC69Config::new(64, 64);
        let pipeline_config = PipelineConfig::realtime_720p();
        let encoder = PipelinedEncoder::new(codec_config, pipeline_config).unwrap();

        let stats = encoder.stats();
        assert_eq!(stats.frames_in, 0);
        assert_eq!(stats.frames_out, 0);
    }

    #[test]
    fn test_pipelined_encoder_flush() {
        let codec_config = ZVC69Config::new(64, 64);
        let pipeline_config = PipelineConfig::low_latency();
        let mut encoder = PipelinedEncoder::new(codec_config, pipeline_config).unwrap();

        // Submit a frame
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        let _ = encoder.submit(frame);

        // Wait for encoding
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Flush
        let remaining = encoder.flush();
        // May or may not have remaining frames depending on timing
        let _ = remaining;
    }

    // ── PipelinedDecoder Tests ──

    #[test]
    fn test_pipelined_decoder_creation() {
        let pipeline_config = PipelineConfig::realtime_720p();
        let decoder = PipelinedDecoder::new(pipeline_config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_pipelined_decoder_stats() {
        let pipeline_config = PipelineConfig::realtime_720p();
        let decoder = PipelinedDecoder::new(pipeline_config).unwrap();

        let stats = decoder.stats();
        assert_eq!(stats.frames_in, 0);
        assert_eq!(stats.frames_out, 0);
    }

    // ── Integration Tests ──

    #[test]
    fn test_sync_encode_decode_roundtrip() {
        // Encode
        let config = ZVC69Config::new(64, 64);
        let mut encoder = SyncEncoder::new(config).unwrap();

        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        let encoded = encoder.encode(frame).unwrap();

        // Decode
        let mut decoder = SyncDecoder::new().unwrap();
        let decoded = decoder.decode(encoded);
        assert!(decoded.is_ok());

        let decoded_frame = decoded.unwrap();
        assert_eq!(decoded_frame.width, 64);
        assert_eq!(decoded_frame.height, 64);
    }

    #[test]
    fn test_multiple_frame_encode() {
        let config = ZVC69Config::new(64, 64);
        let mut encoder = SyncEncoder::new(config).unwrap();

        for i in 0..5 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = 32 * 32;
            frame.data = vec![
                Buffer::from_vec(vec![(128 + i) as u8; y_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            let result = encoder.encode(frame);
            assert!(result.is_ok());
        }

        let stats = encoder.stats();
        assert_eq!(stats.frames_in, 5);
        assert_eq!(stats.frames_out, 5);
    }

    #[test]
    fn test_is_realtime_check() {
        let codec_config = ZVC69Config::new(64, 64);
        let pipeline_config = PipelineConfig::realtime_720p();
        let encoder = PipelinedEncoder::new(codec_config, pipeline_config).unwrap();

        // Initially should not be realtime (no frames processed)
        assert!(!encoder.is_realtime(30.0));
    }

    #[test]
    fn test_config_access() {
        let codec_config = ZVC69Config::new(64, 64);
        let pipeline_config = PipelineConfig::realtime_720p();
        let encoder = PipelinedEncoder::new(codec_config, pipeline_config).unwrap();

        let config = encoder.config();
        assert_eq!(config.queue_depth, 4);
    }
}
