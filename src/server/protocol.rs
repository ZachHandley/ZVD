//! Protocol types for the distributed transcoding system
//!
//! This module defines the core data structures used for communication
//! between the coordinator, workers, and dispatch clients.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::hwaccel::HwAccelType;

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobPriority {
    /// Lowest priority - background tasks
    Low = 0,
    /// Normal priority - default
    Normal = 1,
    /// High priority - urgent tasks
    High = 2,
    /// Critical priority - immediate processing
    Critical = 3,
}

impl Default for JobPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl From<u32> for JobPriority {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Critical,
        }
    }
}

impl From<JobPriority> for u32 {
    fn from(value: JobPriority) -> Self {
        value as u32
    }
}

/// State of a transcode job
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobState {
    /// Job is waiting in the queue
    Pending,
    /// Job is currently being processed
    Running {
        /// Current progress (0.0 - 100.0)
        progress: f32,
        /// Worker handling this job
        worker_id: Uuid,
        /// When processing started
        started_at: DateTime<Utc>,
        /// Current encoding FPS
        fps: f32,
        /// Estimated time remaining in seconds
        eta: Option<f64>,
    },
    /// Job completed successfully
    Complete {
        /// Path to the output file
        output_path: String,
        /// Size of the output file in bytes
        output_size: u64,
        /// When the job completed
        completed_at: DateTime<Utc>,
        /// Encoding duration in milliseconds
        encoding_duration_ms: u64,
    },
    /// Job failed
    Failed {
        /// Error message
        error: String,
        /// When the job failed
        failed_at: DateTime<Utc>,
        /// Worker that was processing (if any)
        worker_id: Option<Uuid>,
    },
    /// Job was cancelled
    Cancelled {
        /// Reason for cancellation
        reason: String,
        /// When the job was cancelled
        cancelled_at: DateTime<Utc>,
    },
}

impl JobState {
    /// Check if the job is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobState::Complete { .. } | JobState::Failed { .. } | JobState::Cancelled { .. }
        )
    }

    /// Check if the job is running
    pub fn is_running(&self) -> bool {
        matches!(self, JobState::Running { .. })
    }

    /// Check if the job is pending
    pub fn is_pending(&self) -> bool {
        matches!(self, JobState::Pending)
    }

    /// Get progress if job is running
    pub fn progress(&self) -> Option<f32> {
        match self {
            JobState::Running { progress, .. } => Some(*progress),
            JobState::Complete { .. } => Some(100.0),
            _ => None,
        }
    }
}

/// Job status summary for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    /// Job ID
    pub id: Uuid,
    /// Current state
    pub state: JobState,
    /// Job priority
    pub priority: JobPriority,
    /// Input file path
    pub input: String,
    /// Output file path
    pub output: String,
    /// When the job was created
    pub created_at: DateTime<Utc>,
    /// When the job was last updated
    pub updated_at: DateTime<Utc>,
}

/// Transcoding parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeParams {
    /// Video codec (e.g., "h264", "h265", "vp9", "av1")
    /// None means copy video stream
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_codec: Option<String>,

    /// Audio codec (e.g., "aac", "opus", "mp3")
    /// None means copy audio stream
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_codec: Option<String>,

    /// Video bitrate in bits per second
    /// None means use codec default or CRF mode
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_bitrate: Option<u64>,

    /// Audio bitrate in bits per second
    /// None means use codec default
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_bitrate: Option<u64>,

    /// Output width in pixels
    /// None means keep original width
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,

    /// Output height in pixels
    /// None means keep original height
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,

    /// Output frame rate
    /// None means keep original frame rate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_rate: Option<f64>,

    /// Pixel format (e.g., "yuv420p", "yuv444p")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_format: Option<String>,

    /// Encoder preset (e.g., "ultrafast", "fast", "medium", "slow")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,

    /// Quality value (CRF for x264/x265, CQ for NVENC)
    /// Lower values = higher quality
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<u32>,

    /// Enable two-pass encoding
    #[serde(default)]
    pub two_pass: bool,

    /// Preferred hardware acceleration type
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_accel: Option<HwAccelType>,

    /// Start time in seconds (for seeking)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_time: Option<f64>,

    /// Duration in seconds (for trimming)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,

    /// Number of audio channels
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_channels: Option<u32>,

    /// Audio sample rate in Hz
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_rate: Option<u32>,

    /// Additional codec-specific parameters
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra_params: HashMap<String, String>,
}

impl Default for TranscodeParams {
    fn default() -> Self {
        Self {
            video_codec: None,
            audio_codec: None,
            video_bitrate: None,
            audio_bitrate: None,
            width: None,
            height: None,
            frame_rate: None,
            pixel_format: None,
            preset: None,
            quality: None,
            two_pass: false,
            hw_accel: None,
            start_time: None,
            duration: None,
            audio_channels: None,
            audio_sample_rate: None,
            extra_params: HashMap::new(),
        }
    }
}

impl TranscodeParams {
    /// Create new transcode params with video codec
    pub fn with_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = Some(codec.into());
        self
    }

    /// Create new transcode params with audio codec
    pub fn with_audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.audio_codec = Some(codec.into());
        self
    }

    /// Set video bitrate
    pub fn with_video_bitrate(mut self, bitrate: u64) -> Self {
        self.video_bitrate = Some(bitrate);
        self
    }

    /// Set audio bitrate
    pub fn with_audio_bitrate(mut self, bitrate: u64) -> Self {
        self.audio_bitrate = Some(bitrate);
        self
    }

    /// Set output resolution
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set quality (CRF value)
    pub fn with_quality(mut self, quality: u32) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Set encoder preset
    pub fn with_preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Enable two-pass encoding
    pub fn with_two_pass(mut self) -> Self {
        self.two_pass = true;
        self
    }

    /// Set hardware acceleration preference
    pub fn with_hw_accel(mut self, hw_accel: HwAccelType) -> Self {
        self.hw_accel = Some(hw_accel);
        self
    }

    /// Check if this is a copy operation (no transcoding)
    pub fn is_copy(&self) -> bool {
        self.video_codec.is_none() && self.audio_codec.is_none()
    }

    /// Get list of required video codecs for matching workers
    pub fn required_video_codecs(&self) -> Vec<&str> {
        self.video_codec.as_deref().into_iter().collect()
    }

    /// Get list of required audio codecs for matching workers
    pub fn required_audio_codecs(&self) -> Vec<&str> {
        self.audio_codec.as_deref().into_iter().collect()
    }
}

/// A transcoding job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJob {
    /// Unique job identifier
    pub id: Uuid,

    /// Input file path (must be accessible to workers)
    pub input: String,

    /// Output file path
    pub output: String,

    /// Transcoding parameters
    pub params: TranscodeParams,

    /// Job priority
    #[serde(default)]
    pub priority: JobPriority,

    /// Current job state
    pub state: JobState,

    /// When the job was created
    pub created_at: DateTime<Utc>,

    /// When the job was last updated
    pub updated_at: DateTime<Utc>,

    /// Job timeout in seconds (0 = no timeout)
    #[serde(default)]
    pub timeout: u32,

    /// Callback URL for completion notification
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,

    /// Metadata for tracking purposes
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl TranscodeJob {
    /// Create a new transcode job
    pub fn new(input: impl Into<String>, output: impl Into<String>, params: TranscodeParams) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            input: input.into(),
            output: output.into(),
            params,
            priority: JobPriority::Normal,
            state: JobState::Pending,
            created_at: now,
            updated_at: now,
            timeout: super::DEFAULT_JOB_TIMEOUT,
            callback_url: None,
            metadata: HashMap::new(),
        }
    }

    /// Set job priority
    pub fn with_priority(mut self, priority: JobPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set job timeout
    pub fn with_timeout(mut self, timeout: u32) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set callback URL
    pub fn with_callback(mut self, url: impl Into<String>) -> Self {
        self.callback_url = Some(url.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Mark job as running
    pub fn mark_running(&mut self, worker_id: Uuid) {
        self.state = JobState::Running {
            progress: 0.0,
            worker_id,
            started_at: Utc::now(),
            fps: 0.0,
            eta: None,
        };
        self.updated_at = Utc::now();
    }

    /// Update job progress
    pub fn update_progress(&mut self, progress: f32, fps: f32, eta: Option<f64>) {
        if let JobState::Running {
            progress: ref mut p,
            fps: ref mut f,
            eta: ref mut e,
            ..
        } = self.state
        {
            *p = progress.clamp(0.0, 100.0);
            *f = fps;
            *e = eta;
            self.updated_at = Utc::now();
        }
    }

    /// Mark job as complete
    pub fn mark_complete(&mut self, output_path: String, output_size: u64, encoding_duration_ms: u64) {
        self.state = JobState::Complete {
            output_path,
            output_size,
            completed_at: Utc::now(),
            encoding_duration_ms,
        };
        self.updated_at = Utc::now();
    }

    /// Mark job as failed
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        let worker_id = match &self.state {
            JobState::Running { worker_id, .. } => Some(*worker_id),
            _ => None,
        };
        self.state = JobState::Failed {
            error: error.into(),
            failed_at: Utc::now(),
            worker_id,
        };
        self.updated_at = Utc::now();
    }

    /// Mark job as cancelled
    pub fn mark_cancelled(&mut self, reason: impl Into<String>) {
        self.state = JobState::Cancelled {
            reason: reason.into(),
            cancelled_at: Utc::now(),
        };
        self.updated_at = Utc::now();
    }

    /// Get job status summary
    pub fn status(&self) -> JobStatus {
        JobStatus {
            id: self.id,
            state: self.state.clone(),
            priority: self.priority,
            input: self.input.clone(),
            output: self.output.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }
}

/// State of a worker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    /// Worker is online and ready
    Ready,
    /// Worker is processing jobs
    Busy,
    /// Worker is draining (not accepting new jobs)
    Draining,
    /// Worker is offline
    Offline,
}

impl Default for WorkerState {
    fn default() -> Self {
        Self::Ready
    }
}

/// Worker capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Hardware acceleration support
    pub hw_accel: Vec<HwAccelType>,

    /// Maximum concurrent jobs
    pub max_concurrent_jobs: u32,

    /// Supported video codecs
    pub supported_video_codecs: Vec<String>,

    /// Supported audio codecs
    pub supported_audio_codecs: Vec<String>,

    /// Available memory in bytes
    pub available_memory: u64,

    /// Number of CPU cores
    pub cpu_cores: u32,

    /// GPU memory in bytes (if applicable)
    pub gpu_memory: Option<u64>,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            hw_accel: Vec::new(),
            max_concurrent_jobs: 1,
            supported_video_codecs: vec![
                "h264".to_string(),
                "h265".to_string(),
                "vp8".to_string(),
                "vp9".to_string(),
                "av1".to_string(),
            ],
            supported_audio_codecs: vec![
                "aac".to_string(),
                "opus".to_string(),
                "mp3".to_string(),
                "flac".to_string(),
                "pcm".to_string(),
            ],
            available_memory: 0,
            cpu_cores: 1,
            gpu_memory: None,
        }
    }
}

impl WorkerCapabilities {
    /// Check if worker supports a video codec
    pub fn supports_video_codec(&self, codec: &str) -> bool {
        self.supported_video_codecs
            .iter()
            .any(|c| c.eq_ignore_ascii_case(codec))
    }

    /// Check if worker supports an audio codec
    pub fn supports_audio_codec(&self, codec: &str) -> bool {
        self.supported_audio_codecs
            .iter()
            .any(|c| c.eq_ignore_ascii_case(codec))
    }

    /// Check if worker supports required hardware acceleration
    pub fn supports_hw_accel(&self, hw_accel: HwAccelType) -> bool {
        hw_accel == HwAccelType::None || self.hw_accel.contains(&hw_accel)
    }

    /// Check if worker can handle a job
    pub fn can_handle_job(&self, job: &TranscodeJob) -> bool {
        // Check video codec requirement
        if let Some(ref codec) = job.params.video_codec {
            if !self.supports_video_codec(codec) {
                return false;
            }
        }

        // Check audio codec requirement
        if let Some(ref codec) = job.params.audio_codec {
            if !self.supports_audio_codec(codec) {
                return false;
            }
        }

        // Check hardware acceleration requirement
        if let Some(hw_accel) = job.params.hw_accel {
            if !self.supports_hw_accel(hw_accel) {
                return false;
            }
        }

        true
    }
}

/// Information about a registered worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Unique worker identifier
    pub id: Uuid,

    /// Worker hostname
    pub hostname: String,

    /// Worker capabilities
    pub capabilities: WorkerCapabilities,

    /// Current state
    pub state: WorkerState,

    /// Current number of running jobs
    pub current_jobs: u32,

    /// Session token for authentication
    #[serde(skip_serializing)]
    pub session_token: String,

    /// When the worker registered
    pub registered_at: DateTime<Utc>,

    /// When the last heartbeat was received
    pub last_heartbeat: DateTime<Utc>,

    /// Worker version string
    pub version: String,

    /// IDs of jobs currently being processed
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub active_jobs: Vec<Uuid>,
}

impl WorkerInfo {
    /// Create new worker info
    pub fn new(
        id: Uuid,
        hostname: impl Into<String>,
        capabilities: WorkerCapabilities,
        version: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            hostname: hostname.into(),
            capabilities,
            state: WorkerState::Ready,
            current_jobs: 0,
            session_token: Uuid::new_v4().to_string(),
            registered_at: now,
            last_heartbeat: now,
            version: version.into(),
            active_jobs: Vec::new(),
        }
    }

    /// Update last heartbeat time
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Utc::now();
    }

    /// Check if worker is healthy (received heartbeat recently)
    pub fn is_healthy(&self, timeout_seconds: i64) -> bool {
        let elapsed = Utc::now()
            .signed_duration_since(self.last_heartbeat)
            .num_seconds();
        elapsed < timeout_seconds
    }

    /// Check if worker can accept more jobs
    pub fn can_accept_job(&self) -> bool {
        self.state == WorkerState::Ready
            && self.current_jobs < self.capabilities.max_concurrent_jobs
    }

    /// Get available capacity
    pub fn available_capacity(&self) -> u32 {
        if self.state != WorkerState::Ready {
            return 0;
        }
        self.capabilities
            .max_concurrent_jobs
            .saturating_sub(self.current_jobs)
    }

    /// Start processing a job
    pub fn start_job(&mut self, job_id: Uuid) {
        self.current_jobs += 1;
        self.active_jobs.push(job_id);
        if self.current_jobs >= self.capabilities.max_concurrent_jobs {
            self.state = WorkerState::Busy;
        }
    }

    /// Finish processing a job
    pub fn finish_job(&mut self, job_id: Uuid) {
        self.current_jobs = self.current_jobs.saturating_sub(1);
        self.active_jobs.retain(|id| *id != job_id);
        if self.state == WorkerState::Busy && self.current_jobs < self.capabilities.max_concurrent_jobs {
            self.state = WorkerState::Ready;
        }
    }
}

/// Request to create a new job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateJobRequest {
    /// Input file path
    pub input: String,

    /// Output file path
    pub output: String,

    /// Transcoding parameters
    #[serde(default)]
    pub params: TranscodeParams,

    /// Job priority
    #[serde(default)]
    pub priority: JobPriority,

    /// Job timeout in seconds
    #[serde(default)]
    pub timeout: Option<u32>,

    /// Callback URL
    #[serde(default)]
    pub callback_url: Option<String>,

    /// Metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl CreateJobRequest {
    /// Create a new job request
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            params: TranscodeParams::default(),
            priority: JobPriority::Normal,
            timeout: None,
            callback_url: None,
            metadata: HashMap::new(),
        }
    }

    /// Convert to a TranscodeJob
    pub fn into_job(self) -> TranscodeJob {
        let mut job = TranscodeJob::new(self.input, self.output, self.params)
            .with_priority(self.priority);

        if let Some(timeout) = self.timeout {
            job = job.with_timeout(timeout);
        }

        if let Some(callback) = self.callback_url {
            job = job.with_callback(callback);
        }

        for (key, value) in self.metadata {
            job = job.with_metadata(key, value);
        }

        job
    }
}

/// Response for job creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateJobResponse {
    /// Job ID
    pub id: Uuid,

    /// Job status
    pub status: JobStatus,
}

/// Response for listing workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerListResponse {
    /// List of workers
    pub workers: Vec<WorkerSummary>,

    /// Total count
    pub total: usize,
}

/// Summary of worker info for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSummary {
    /// Worker ID
    pub id: Uuid,

    /// Hostname
    pub hostname: String,

    /// Current state
    pub state: WorkerState,

    /// Current jobs
    pub current_jobs: u32,

    /// Max concurrent jobs
    pub max_concurrent_jobs: u32,

    /// Hardware acceleration
    pub hw_accel: Vec<HwAccelType>,

    /// Last heartbeat
    pub last_heartbeat: DateTime<Utc>,

    /// Version
    pub version: String,
}

impl From<&WorkerInfo> for WorkerSummary {
    fn from(info: &WorkerInfo) -> Self {
        Self {
            id: info.id,
            hostname: info.hostname.clone(),
            state: info.state,
            current_jobs: info.current_jobs,
            max_concurrent_jobs: info.capabilities.max_concurrent_jobs,
            hw_accel: info.capabilities.hw_accel.clone(),
            last_heartbeat: info.last_heartbeat,
            version: info.version.clone(),
        }
    }
}

/// Progress update sent via WebSocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    /// Job ID
    pub job_id: Uuid,

    /// Progress percentage (0.0 - 100.0)
    pub progress: f32,

    /// Current encoding FPS
    pub fps: f32,

    /// Current bitrate being achieved
    pub bitrate: Option<u64>,

    /// Current time position in seconds
    pub time_position: f64,

    /// Total duration in seconds
    pub total_duration: f64,

    /// Estimated time remaining in seconds
    pub eta: Option<f64>,

    /// Current pass (for two-pass encoding)
    pub current_pass: Option<u32>,

    /// Status message
    pub status_message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Low < JobPriority::Normal);
        assert!(JobPriority::Normal < JobPriority::High);
        assert!(JobPriority::High < JobPriority::Critical);
    }

    #[test]
    fn test_job_state_transitions() {
        let mut job = TranscodeJob::new("input.mp4", "output.webm", TranscodeParams::default());

        assert!(job.state.is_pending());
        assert!(!job.state.is_running());
        assert!(!job.state.is_terminal());

        let worker_id = Uuid::new_v4();
        job.mark_running(worker_id);

        assert!(!job.state.is_pending());
        assert!(job.state.is_running());
        assert!(!job.state.is_terminal());

        job.mark_complete("output.webm".to_string(), 1024, 5000);

        assert!(!job.state.is_pending());
        assert!(!job.state.is_running());
        assert!(job.state.is_terminal());
    }

    #[test]
    fn test_worker_capabilities() {
        let mut caps = WorkerCapabilities::default();
        caps.hw_accel.push(HwAccelType::NVENC);

        assert!(caps.supports_video_codec("h264"));
        assert!(caps.supports_video_codec("H264")); // Case insensitive
        assert!(!caps.supports_video_codec("unknown"));

        assert!(caps.supports_hw_accel(HwAccelType::None));
        assert!(caps.supports_hw_accel(HwAccelType::NVENC));
        assert!(!caps.supports_hw_accel(HwAccelType::VAAPI));
    }

    #[test]
    fn test_worker_job_tracking() {
        let caps = WorkerCapabilities {
            max_concurrent_jobs: 2,
            ..Default::default()
        };
        let mut worker = WorkerInfo::new(Uuid::new_v4(), "test-host", caps, "0.1.0");

        assert!(worker.can_accept_job());
        assert_eq!(worker.available_capacity(), 2);

        let job1 = Uuid::new_v4();
        worker.start_job(job1);
        assert!(worker.can_accept_job());
        assert_eq!(worker.available_capacity(), 1);
        assert_eq!(worker.state, WorkerState::Ready);

        let job2 = Uuid::new_v4();
        worker.start_job(job2);
        assert!(!worker.can_accept_job());
        assert_eq!(worker.available_capacity(), 0);
        assert_eq!(worker.state, WorkerState::Busy);

        worker.finish_job(job1);
        assert!(worker.can_accept_job());
        assert_eq!(worker.available_capacity(), 1);
        assert_eq!(worker.state, WorkerState::Ready);
    }

    #[test]
    fn test_transcode_params_builder() {
        let params = TranscodeParams::default()
            .with_video_codec("h264")
            .with_audio_codec("aac")
            .with_video_bitrate(5_000_000)
            .with_resolution(1920, 1080)
            .with_quality(23)
            .with_preset("medium");

        assert_eq!(params.video_codec, Some("h264".to_string()));
        assert_eq!(params.audio_codec, Some("aac".to_string()));
        assert_eq!(params.video_bitrate, Some(5_000_000));
        assert_eq!(params.width, Some(1920));
        assert_eq!(params.height, Some(1080));
        assert_eq!(params.quality, Some(23));
        assert_eq!(params.preset, Some("medium".to_string()));
    }
}
