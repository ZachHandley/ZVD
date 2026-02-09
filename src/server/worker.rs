//! Worker Daemon Implementation
//!
//! The worker daemon connects to a coordinator, receives transcode jobs,
//! executes them using the ZVD codec infrastructure, and reports progress.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use super::proto::{
    worker_service_client::WorkerServiceClient, GetJobRequest, HeartbeatRequest,
    JobCompletionRequest, ProgressReport, RegisterRequest, UnregisterRequest,
    WorkerCapabilities as ProtoCapabilities, HwAccelType as ProtoHwAccel,
    JobStatus as ProtoJobStatus, TranscodeJob as ProtoJob,
};
use super::protocol::{TranscodeJob, TranscodeParams, WorkerCapabilities};
use crate::hwaccel::{detect_hw_devices, HwAccelType};

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Coordinator gRPC address (e.g., "http://localhost:50051")
    pub coordinator_addr: String,

    /// Maximum concurrent jobs
    pub max_concurrent_jobs: u32,

    /// Heartbeat interval in seconds
    pub heartbeat_interval: u64,

    /// Job poll interval in seconds
    pub poll_interval: u64,

    /// Worker hostname (auto-detected if not provided)
    pub hostname: Option<String>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            coordinator_addr: "http://localhost:50051".to_string(),
            max_concurrent_jobs: 1,
            heartbeat_interval: 30,
            poll_interval: 5,
            hostname: None,
        }
    }
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerRunState {
    Starting,
    Running,
    Draining,
    Stopped,
}

/// Active job being processed
struct ActiveJob {
    job: ProtoJob,
    started_at: Instant,
    progress: f32,
}

/// The Worker daemon
pub struct Worker {
    config: WorkerConfig,
    id: Uuid,
    capabilities: WorkerCapabilities,
    session_token: Arc<RwLock<Option<String>>>,
    state: Arc<RwLock<WorkerRunState>>,
    active_jobs: Arc<RwLock<Vec<ActiveJob>>>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: mpsc::Receiver<()>,
}

impl Worker {
    /// Create a new worker with the given configuration
    pub fn new(config: WorkerConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        // Auto-detect capabilities
        let capabilities = detect_capabilities(config.max_concurrent_jobs);

        Self {
            config,
            id: Uuid::new_v4(),
            capabilities,
            session_token: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(WorkerRunState::Starting)),
            active_jobs: Arc::new(RwLock::new(Vec::new())),
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Get the worker ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Run the worker daemon
    pub async fn run(mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Starting worker {} connecting to {}",
            self.id, self.config.coordinator_addr
        );

        // Connect to coordinator
        let channel = Channel::from_shared(self.config.coordinator_addr.clone())?
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .connect()
            .await?;

        let mut client = WorkerServiceClient::new(channel.clone());

        // Register with coordinator
        let hostname = self.config.hostname.clone().unwrap_or_else(|| {
            hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        });

        let proto_caps = capabilities_to_proto(&self.capabilities, self.id, &hostname);
        let register_req = RegisterRequest {
            capabilities: Some(proto_caps),
        };

        let response = client.register(register_req).await?;
        let reg_response = response.into_inner();

        if !reg_response.success {
            return Err(format!("Registration failed: {}", reg_response.error).into());
        }

        *self.session_token.write() = Some(reg_response.session_token.clone());
        *self.state.write() = WorkerRunState::Running;

        info!("Worker registered successfully with coordinator");

        // Start heartbeat task
        let heartbeat_client = WorkerServiceClient::new(channel.clone());
        let heartbeat_id = self.id;
        let heartbeat_token = self.session_token.clone();
        let heartbeat_caps = self.capabilities.clone();
        let heartbeat_hostname = hostname.clone();
        let heartbeat_interval = Duration::from_secs(self.config.heartbeat_interval);
        let heartbeat_state = self.state.clone();

        let heartbeat_handle = tokio::spawn(async move {
            run_heartbeat_loop(
                heartbeat_client,
                heartbeat_id,
                heartbeat_token,
                heartbeat_caps,
                heartbeat_hostname,
                heartbeat_interval,
                heartbeat_state,
            )
            .await
        });

        // Start job polling task
        let poll_client = WorkerServiceClient::new(channel.clone());
        let poll_id = self.id;
        let poll_token = self.session_token.clone();
        let poll_interval = Duration::from_secs(self.config.poll_interval);
        let poll_state = self.state.clone();
        let poll_active_jobs = self.active_jobs.clone();
        let poll_max_jobs = self.config.max_concurrent_jobs;

        let poll_handle = tokio::spawn(async move {
            run_job_poll_loop(
                poll_client,
                poll_id,
                poll_token,
                poll_interval,
                poll_state,
                poll_active_jobs,
                poll_max_jobs,
            )
            .await
        });

        // Wait for shutdown signal
        tokio::select! {
            _ = self.shutdown_rx.recv() => {
                info!("Received shutdown signal");
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down");
            }
        }

        // Graceful shutdown
        *self.state.write() = WorkerRunState::Draining;
        info!("Draining active jobs...");

        // Wait for active jobs to complete (with timeout)
        let drain_deadline = Instant::now() + Duration::from_secs(60);
        while !self.active_jobs.read().is_empty() {
            if Instant::now() > drain_deadline {
                warn!("Drain timeout reached, forcing shutdown");
                break;
            }
            time::sleep(Duration::from_millis(500)).await;
        }

        // Unregister from coordinator
        let mut unregister_client = WorkerServiceClient::new(channel);
        let token = self.session_token.read().clone().unwrap_or_default();
        let _ = unregister_client
            .unregister(UnregisterRequest {
                worker_id: self.id.to_string(),
                session_token: token,
                reason: "Graceful shutdown".to_string(),
            })
            .await;

        // Stop background tasks
        heartbeat_handle.abort();
        poll_handle.abort();

        *self.state.write() = WorkerRunState::Stopped;
        info!("Worker stopped");

        Ok(())
    }

    /// Signal the worker to shut down
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.try_send(());
    }
}

/// Detect worker capabilities
fn detect_capabilities(max_concurrent_jobs: u32) -> WorkerCapabilities {
    let hw_devices = detect_hw_devices();

    // Get system info
    let cpu_cores = num_cpus();
    let available_memory = get_available_memory();
    let gpu_memory = get_gpu_memory(&hw_devices);

    // Determine supported codecs based on available features
    let mut video_codecs = vec![
        "copy".to_string(), // Stream copy is always available
    ];

    // Always available (pure Rust implementations)
    video_codecs.push("av1".to_string());

    #[cfg(feature = "h264")]
    video_codecs.push("h264".to_string());

    #[cfg(feature = "h265")]
    video_codecs.push("h265".to_string());

    #[cfg(feature = "vp8-codec")]
    video_codecs.push("vp8".to_string());

    #[cfg(feature = "vp9-codec")]
    video_codecs.push("vp9".to_string());

    // Professional codecs
    video_codecs.push("prores".to_string());
    video_codecs.push("dnxhd".to_string());

    let mut audio_codecs = vec![
        "copy".to_string(),
        "pcm".to_string(),
        "flac".to_string(),
        "mp3".to_string(),
    ];

    #[cfg(feature = "aac")]
    audio_codecs.push("aac".to_string());

    #[cfg(feature = "opus-codec")]
    audio_codecs.push("opus".to_string());

    WorkerCapabilities {
        hw_accel: hw_devices,
        max_concurrent_jobs,
        supported_video_codecs: video_codecs,
        supported_audio_codecs: audio_codecs,
        available_memory,
        cpu_cores,
        gpu_memory,
    }
}

/// Get number of CPU cores
fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1)
}

/// Get available system memory
fn get_available_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        if let Ok(val) = kb.parse::<u64>() {
                            return val * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Use sysctl on macOS
        // For simplicity, return a reasonable default
    }

    // Default: 4GB
    4 * 1024 * 1024 * 1024
}

/// Get GPU memory if available
fn get_gpu_memory(hw_devices: &[HwAccelType]) -> Option<u64> {
    if hw_devices.contains(&HwAccelType::NVENC) || hw_devices.contains(&HwAccelType::NVDEC) {
        // Try to detect NVIDIA GPU memory
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    if let Ok(mem_str) = String::from_utf8(output.stdout) {
                        if let Ok(mem_mb) = mem_str.trim().parse::<u64>() {
                            return Some(mem_mb * 1024 * 1024);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Convert capabilities to proto format
fn capabilities_to_proto(
    caps: &WorkerCapabilities,
    worker_id: Uuid,
    hostname: &str,
) -> ProtoCapabilities {
    let hw_accel: Vec<i32> = caps
        .hw_accel
        .iter()
        .map(|h| match h {
            HwAccelType::None => ProtoHwAccel::HwAccelNone as i32,
            HwAccelType::VAAPI => ProtoHwAccel::HwAccelVaapi as i32,
            HwAccelType::NVENC => ProtoHwAccel::HwAccelNvenc as i32,
            HwAccelType::NVDEC => ProtoHwAccel::HwAccelNvdec as i32,
            HwAccelType::QSV => ProtoHwAccel::HwAccelQsv as i32,
            HwAccelType::VideoToolbox => ProtoHwAccel::HwAccelVideotoolbox as i32,
            HwAccelType::AMF => ProtoHwAccel::HwAccelAmf as i32,
            HwAccelType::DXVA2 => ProtoHwAccel::HwAccelDxva2 as i32,
            HwAccelType::D3D11VA => ProtoHwAccel::HwAccelD3d11va as i32,
            HwAccelType::Vulkan => ProtoHwAccel::HwAccelVulkan as i32,
        })
        .collect();

    ProtoCapabilities {
        worker_id: worker_id.to_string(),
        hostname: hostname.to_string(),
        hw_accel,
        max_concurrent_jobs: caps.max_concurrent_jobs,
        current_jobs: 0,
        supported_video_codecs: caps.supported_video_codecs.clone(),
        supported_audio_codecs: caps.supported_audio_codecs.clone(),
        available_memory: caps.available_memory,
        cpu_cores: caps.cpu_cores,
        gpu_memory: caps.gpu_memory.unwrap_or(0),
        version: super::SERVER_VERSION.to_string(),
    }
}

/// Run heartbeat loop
async fn run_heartbeat_loop(
    mut client: WorkerServiceClient<Channel>,
    worker_id: Uuid,
    session_token: Arc<RwLock<Option<String>>>,
    capabilities: WorkerCapabilities,
    hostname: String,
    interval: Duration,
    state: Arc<RwLock<WorkerRunState>>,
) {
    let mut ticker = time::interval(interval);

    loop {
        ticker.tick().await;

        if *state.read() == WorkerRunState::Stopped {
            break;
        }

        let token = match session_token.read().clone() {
            Some(t) => t,
            None => continue,
        };

        let caps = capabilities_to_proto(&capabilities, worker_id, &hostname);
        let req = HeartbeatRequest {
            worker_id: worker_id.to_string(),
            session_token: token,
            capabilities: Some(caps),
        };

        match client.heartbeat(req).await {
            Ok(response) => {
                let resp = response.into_inner();
                if !resp.acknowledged {
                    warn!("Heartbeat not acknowledged");
                }
                // Handle commands from coordinator
                for cmd in resp.commands {
                    match cmd.as_str() {
                        "shutdown" => {
                            info!("Received shutdown command from coordinator");
                            *state.write() = WorkerRunState::Draining;
                        }
                        "pause" => {
                            info!("Received pause command from coordinator");
                            *state.write() = WorkerRunState::Draining;
                        }
                        _ => {
                            debug!("Unknown command from coordinator: {}", cmd);
                        }
                    }
                }
            }
            Err(e) => {
                error!("Heartbeat failed: {}", e);
            }
        }
    }
}

/// Run job polling loop
async fn run_job_poll_loop(
    mut client: WorkerServiceClient<Channel>,
    worker_id: Uuid,
    session_token: Arc<RwLock<Option<String>>>,
    interval: Duration,
    state: Arc<RwLock<WorkerRunState>>,
    active_jobs: Arc<RwLock<Vec<ActiveJob>>>,
    max_jobs: u32,
) {
    let mut ticker = time::interval(interval);

    loop {
        ticker.tick().await;

        let current_state = *state.read();
        if current_state != WorkerRunState::Running {
            if current_state == WorkerRunState::Stopped {
                break;
            }
            continue;
        }

        // Check if we have capacity
        let current_job_count = active_jobs.read().len() as u32;
        if current_job_count >= max_jobs {
            continue;
        }

        let token = match session_token.read().clone() {
            Some(t) => t,
            None => continue,
        };

        let capacity = max_jobs - current_job_count;
        let req = GetJobRequest {
            worker_id: worker_id.to_string(),
            session_token: token.clone(),
            capacity,
        };

        match client.get_job(req).await {
            Ok(response) => {
                let resp = response.into_inner();
                if resp.has_job {
                    if let Some(job) = resp.job {
                        info!("Received job: {}", job.job_id);

                        // Add to active jobs
                        active_jobs.write().push(ActiveJob {
                            job: job.clone(),
                            started_at: Instant::now(),
                            progress: 0.0,
                        });

                        // Spawn job execution task
                        let job_client = client.clone();
                        let job_worker_id = worker_id;
                        let job_token = token.clone();
                        let job_active_jobs = active_jobs.clone();

                        tokio::spawn(async move {
                            execute_job(job_client, job_worker_id, job_token, job, job_active_jobs).await;
                        });
                    }
                }
            }
            Err(e) => {
                error!("Job poll failed: {}", e);
            }
        }
    }
}

/// Execute a transcode job
async fn execute_job(
    mut client: WorkerServiceClient<Channel>,
    worker_id: Uuid,
    session_token: String,
    job: ProtoJob,
    active_jobs: Arc<RwLock<Vec<ActiveJob>>>,
) {
    let job_id = job.job_id.clone();
    let start_time = Instant::now();

    info!("Starting job {}: {} -> {}", job_id, job.input_path, job.output_path);

    // Convert proto job to internal format
    let transcode_result = run_transcode(&job, worker_id, &mut client).await;

    let elapsed_ms = start_time.elapsed().as_millis() as u64;

    // Remove from active jobs
    active_jobs.write().retain(|j| j.job.job_id != job_id);

    // Report completion
    let completion_req = match transcode_result {
        Ok((output_size, output_path)) => {
            info!("Job {} completed in {}ms", job_id, elapsed_ms);
            JobCompletionRequest {
                job_id: job_id.clone(),
                worker_id: worker_id.to_string(),
                status: ProtoJobStatus::Completed as i32,
                error: String::new(),
                output_path,
                output_size,
                encoding_duration_ms: elapsed_ms,
                output_video_bitrate: 0,
                output_audio_bitrate: 0,
                output_resolution: String::new(),
                output_duration: 0.0,
            }
        }
        Err(e) => {
            error!("Job {} failed: {}", job_id, e);
            JobCompletionRequest {
                job_id: job_id.clone(),
                worker_id: worker_id.to_string(),
                status: ProtoJobStatus::Failed as i32,
                error: e.to_string(),
                output_path: String::new(),
                output_size: 0,
                encoding_duration_ms: elapsed_ms,
                output_video_bitrate: 0,
                output_audio_bitrate: 0,
                output_resolution: String::new(),
                output_duration: 0.0,
            }
        }
    };

    if let Err(e) = client.complete_job(completion_req).await {
        error!("Failed to report job completion: {}", e);
    }
}

/// Run the actual transcode operation
async fn run_transcode(
    job: &ProtoJob,
    worker_id: Uuid,
    client: &mut WorkerServiceClient<Channel>,
) -> Result<(u64, String), Box<dyn std::error::Error + Send + Sync>> {
    let job_id = job.job_id.clone();
    let worker_id_str = worker_id.to_string();
    let input_path = job.input_path.clone();
    let output_path = job.output_path.clone();

    // Set up progress reporting channel
    let (progress_tx, mut progress_rx) = mpsc::channel::<ProgressReport>(100);

    // Spawn progress reporting task
    let mut progress_client = client.clone();
    let progress_handle = tokio::spawn(async move {
        let mut reports: Vec<ProgressReport> = vec![];
        while let Some(report) = progress_rx.recv().await {
            reports.push(report);
            // Batch send every 10 reports or when buffer is full
            if reports.len() >= 10 {
                let batch: Vec<ProgressReport> = reports.drain(..).collect();
                let stream = tokio_stream::iter(batch);
                let _ = progress_client.report_progress(stream).await;
            }
        }
        // Send remaining reports
        if !reports.is_empty() {
            let batch: Vec<ProgressReport> = reports.drain(..).collect();
            let stream = tokio_stream::iter(batch);
            let _ = progress_client.report_progress(stream).await;
        }
    });

    // Run the actual transcoding in a blocking thread since demuxer/muxer aren't Send
    let transcode_result = tokio::task::spawn_blocking(move || {
        run_transcode_sync(
            &input_path,
            &output_path,
            &job_id,
            &worker_id_str,
            progress_tx,
        )
    })
    .await
    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    })??;

    // Wait for progress reporting to finish
    let _ = progress_handle.await;

    Ok(transcode_result)
}

/// Synchronous transcode implementation (runs in blocking thread)
fn run_transcode_sync(
    input_path: &str,
    output_path: &str,
    job_id: &str,
    worker_id: &str,
    progress_tx: mpsc::Sender<ProgressReport>,
) -> Result<(u64, String), Box<dyn std::error::Error + Send + Sync>> {
    use crate::format::demuxer::create_demuxer;
    use crate::format::muxer::create_muxer;
    use crate::format::detect_format_from_extension;

    let input = std::path::Path::new(input_path);
    let output = std::path::Path::new(output_path);

    // Validate input exists
    if !input.exists() {
        return Err(format!("Input file not found: {}", input_path).into());
    }

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Detect output format
    let output_format = detect_format_from_extension(output_path)
        .ok_or_else(|| format!("Unknown output format for: {}", output_path))?;

    // Open input
    let mut demuxer = create_demuxer(&input.to_path_buf())?;
    let streams = demuxer.streams();

    if streams.is_empty() {
        return Err("No streams in input file".into());
    }

    // Create output muxer
    let mut muxer = create_muxer(output_format)?;
    muxer.create(&output.to_path_buf())?;

    // Add streams to output
    for stream in streams {
        muxer.add_stream(stream.clone())?;
    }

    muxer.write_header()?;

    // Get total duration for progress calculation
    let total_duration = demuxer.streams()
        .iter()
        .filter_map(|s| {
            let dur = s.info.duration_seconds();
            if dur > 0.0 { Some(dur) } else { None }
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    // Process packets
    let mut packets_processed = 0u64;
    let mut bytes_processed = 0u64;
    let start_time = Instant::now();
    let mut last_progress_time = Instant::now();

    loop {
        let packet = match demuxer.read_packet() {
            Ok(p) => p,
            Err(crate::Error::EndOfStream) => break,
            Err(e) => return Err(e.into()),
        };

        bytes_processed += packet.data.len() as u64;
        packets_processed += 1;

        // For now, we just copy packets (actual transcoding would decode/encode here)
        muxer.write_packet(&packet)?;

        // Report progress periodically
        if last_progress_time.elapsed() > Duration::from_millis(500) {
            let elapsed = start_time.elapsed().as_secs_f64();
            let progress = if total_duration > 0.0 {
                // Estimate progress based on bytes processed
                (bytes_processed as f64 / (bytes_processed as f64 * 1.1)).min(99.0) as f32
            } else {
                (packets_processed as f32 / 100.0).min(99.0)
            };

            let fps = packets_processed as f32 / elapsed as f32;
            let eta = if progress > 0.0 {
                Some((100.0 - progress as f64) / (progress as f64 / elapsed))
            } else {
                None
            };

            let report = ProgressReport {
                job_id: job_id.to_string(),
                worker_id: worker_id.to_string(),
                progress,
                current_frame: packets_processed,
                total_frames: 0,
                fps,
                bitrate: (bytes_processed * 8) / (elapsed.max(1.0) as u64),
                time_position: elapsed,
                total_duration,
                eta: eta.unwrap_or(0.0),
                current_pass: 1,
                status_message: format!("Processing packet {}", packets_processed),
            };

            // Use blocking send since we're in a blocking context
            let _ = progress_tx.blocking_send(report);
            last_progress_time = Instant::now();
        }
    }

    // Finalize output
    muxer.write_trailer()?;

    // Get output file size
    let output_size = std::fs::metadata(output)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok((output_size, output_path.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.max_concurrent_jobs, 1);
        assert_eq!(config.heartbeat_interval, 30);
        assert_eq!(config.poll_interval, 5);
    }

    #[test]
    fn test_detect_capabilities() {
        let caps = detect_capabilities(2);
        assert_eq!(caps.max_concurrent_jobs, 2);
        assert!(caps.cpu_cores >= 1);
        assert!(caps.available_memory > 0);
    }
}
