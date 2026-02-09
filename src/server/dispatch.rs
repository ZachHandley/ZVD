//! Dispatch Client Implementation
//!
//! The dispatch client provides both a library API and CLI interface
//! for submitting transcode jobs to the coordinator.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};
use uuid::Uuid;

use super::protocol::{
    CreateJobRequest, CreateJobResponse, JobPriority, JobStatus, ProgressUpdate,
    TranscodeParams, WorkerListResponse,
};

/// Dispatch client configuration
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Coordinator REST API URL (e.g., "http://localhost:8080")
    pub coordinator_url: String,

    /// Request timeout in seconds
    pub timeout: u64,

    /// Enable verbose output
    pub verbose: bool,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            coordinator_url: format!("http://localhost:{}", super::DEFAULT_REST_PORT),
            timeout: 30,
            verbose: false,
        }
    }
}

/// Error type for dispatch operations
#[derive(Debug)]
pub enum DispatchError {
    /// Network error
    Network(String),
    /// Server error
    Server { status: u16, message: String },
    /// Serialization error
    Serialization(String),
    /// Job not found
    NotFound,
    /// Timeout
    Timeout,
    /// Invalid configuration
    Config(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::Network(msg) => write!(f, "Network error: {}", msg),
            DispatchError::Server { status, message } => {
                write!(f, "Server error ({}): {}", status, message)
            }
            DispatchError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            DispatchError::NotFound => write!(f, "Job not found"),
            DispatchError::Timeout => write!(f, "Request timed out"),
            DispatchError::Config(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<reqwest::Error> for DispatchError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            DispatchError::Timeout
        } else if err.is_connect() {
            DispatchError::Network(format!("Connection failed: {}", err))
        } else {
            DispatchError::Network(err.to_string())
        }
    }
}

impl From<serde_json::Error> for DispatchError {
    fn from(err: serde_json::Error) -> Self {
        DispatchError::Serialization(err.to_string())
    }
}

/// Result type for dispatch operations
pub type Result<T> = std::result::Result<T, DispatchError>;

/// Dispatch client for submitting and managing transcode jobs
#[derive(Clone)]
pub struct DispatchClient {
    config: DispatchConfig,
    client: Client,
}

impl DispatchClient {
    /// Create a new dispatch client with the given configuration
    pub fn new(config: DispatchConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| DispatchError::Network(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { config, client })
    }

    /// Create a new dispatch client with default configuration
    pub fn with_url(url: impl Into<String>) -> Result<Self> {
        let config = DispatchConfig {
            coordinator_url: url.into(),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Get the base URL
    fn base_url(&self) -> &str {
        &self.config.coordinator_url
    }

    /// Check if the coordinator is reachable
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url());
        let response = self.client.get(&url).send().await?;
        Ok(response.status().is_success())
    }

    /// Submit a new transcode job
    pub async fn submit_job(&self, request: CreateJobRequest) -> Result<CreateJobResponse> {
        let url = format!("{}/jobs", self.base_url());

        debug!("Submitting job: {} -> {}", request.input, request.output);

        let response = self.client.post(&url).json(&request).send().await?;

        let status = response.status();
        if status.is_success() {
            let body = response.json::<CreateJobResponse>().await?;
            info!("Job submitted: {}", body.id);
            Ok(body)
        } else {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(DispatchError::Server {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Submit a simple transcode job
    pub async fn transcode(
        &self,
        input: impl Into<String>,
        output: impl Into<String>,
        params: TranscodeParams,
    ) -> Result<JobStatus> {
        let request = CreateJobRequest {
            input: input.into(),
            output: output.into(),
            params,
            priority: JobPriority::Normal,
            timeout: None,
            callback_url: None,
            metadata: std::collections::HashMap::new(),
        };

        let response = self.submit_job(request).await?;
        Ok(response.status)
    }

    /// Get the status of a job
    pub async fn get_job_status(&self, job_id: Uuid) -> Result<JobStatus> {
        let url = format!("{}/jobs/{}", self.base_url(), job_id);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if status.is_success() {
            let body = response.json::<JobStatus>().await?;
            Ok(body)
        } else if status.as_u16() == 404 {
            Err(DispatchError::NotFound)
        } else {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(DispatchError::Server {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: Uuid) -> Result<()> {
        let url = format!("{}/jobs/{}", self.base_url(), job_id);

        let response = self.client.delete(&url).send().await?;

        let status = response.status();
        if status.is_success() || status.as_u16() == 204 {
            info!("Job {} cancelled", job_id);
            Ok(())
        } else if status.as_u16() == 404 {
            Err(DispatchError::NotFound)
        } else {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(DispatchError::Server {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// List all jobs
    pub async fn list_jobs(&self, limit: Option<usize>, offset: Option<usize>) -> Result<Vec<JobStatus>> {
        let mut url = format!("{}/jobs", self.base_url());

        // Add query parameters
        let mut params = vec![];
        if let Some(l) = limit {
            params.push(format!("limit={}", l));
        }
        if let Some(o) = offset {
            params.push(format!("offset={}", o));
        }
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if status.is_success() {
            let body = response.json::<Vec<JobStatus>>().await?;
            Ok(body)
        } else {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(DispatchError::Server {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// List all workers
    pub async fn list_workers(&self) -> Result<WorkerListResponse> {
        let url = format!("{}/workers", self.base_url());

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if status.is_success() {
            let body = response.json::<WorkerListResponse>().await?;
            Ok(body)
        } else {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(DispatchError::Server {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Get metrics from the coordinator
    pub async fn get_metrics(&self) -> Result<String> {
        let url = format!("{}/metrics", self.base_url());

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if status.is_success() {
            let body = response.text().await?;
            Ok(body)
        } else {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            Err(DispatchError::Server {
                status: status.as_u16(),
                message,
            })
        }
    }

    /// Wait for a job to complete, polling periodically
    pub async fn wait_for_job(
        &self,
        job_id: Uuid,
        poll_interval: Duration,
        timeout: Option<Duration>,
    ) -> Result<JobStatus> {
        let start = std::time::Instant::now();

        loop {
            let status = self.get_job_status(job_id).await?;

            if status.state.is_terminal() {
                return Ok(status);
            }

            // Check timeout
            if let Some(t) = timeout {
                if start.elapsed() > t {
                    return Err(DispatchError::Timeout);
                }
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Submit a job and wait for completion
    pub async fn submit_and_wait(
        &self,
        request: CreateJobRequest,
        poll_interval: Duration,
        timeout: Option<Duration>,
    ) -> Result<JobStatus> {
        let response = self.submit_job(request).await?;
        self.wait_for_job(response.id, poll_interval, timeout).await
    }
}

/// Builder for creating transcode requests with a fluent API
pub struct TranscodeRequestBuilder {
    input: String,
    output: String,
    params: TranscodeParams,
    priority: JobPriority,
    timeout: Option<u32>,
    callback_url: Option<String>,
    metadata: std::collections::HashMap<String, String>,
}

impl TranscodeRequestBuilder {
    /// Create a new builder
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            params: TranscodeParams::default(),
            priority: JobPriority::Normal,
            timeout: None,
            callback_url: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set video codec
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.params.video_codec = Some(codec.into());
        self
    }

    /// Set audio codec
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.params.audio_codec = Some(codec.into());
        self
    }

    /// Set video bitrate in bits per second
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        self.params.video_bitrate = Some(bitrate);
        self
    }

    /// Set audio bitrate in bits per second
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        self.params.audio_bitrate = Some(bitrate);
        self
    }

    /// Set output resolution
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.params.width = Some(width);
        self.params.height = Some(height);
        self
    }

    /// Set frame rate
    pub fn frame_rate(mut self, fps: f64) -> Self {
        self.params.frame_rate = Some(fps);
        self
    }

    /// Set quality (CRF value)
    pub fn quality(mut self, crf: u32) -> Self {
        self.params.quality = Some(crf);
        self
    }

    /// Set encoder preset
    pub fn preset(mut self, preset: impl Into<String>) -> Self {
        self.params.preset = Some(preset.into());
        self
    }

    /// Enable two-pass encoding
    pub fn two_pass(mut self) -> Self {
        self.params.two_pass = true;
        self
    }

    /// Set job priority
    pub fn priority(mut self, priority: JobPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set job timeout in seconds
    pub fn timeout(mut self, timeout: u32) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set callback URL
    pub fn callback(mut self, url: impl Into<String>) -> Self {
        self.callback_url = Some(url.into());
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set start time for seeking
    pub fn start_time(mut self, seconds: f64) -> Self {
        self.params.start_time = Some(seconds);
        self
    }

    /// Set duration for trimming
    pub fn duration(mut self, seconds: f64) -> Self {
        self.params.duration = Some(seconds);
        self
    }

    /// Set extra encoder parameter
    pub fn extra_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.extra_params.insert(key.into(), value.into());
        self
    }

    /// Build the request
    pub fn build(self) -> CreateJobRequest {
        CreateJobRequest {
            input: self.input,
            output: self.output,
            params: self.params,
            priority: self.priority,
            timeout: self.timeout,
            callback_url: self.callback_url,
            metadata: self.metadata,
        }
    }

    /// Submit the job using the provided client
    pub async fn submit(self, client: &DispatchClient) -> Result<CreateJobResponse> {
        client.submit_job(self.build()).await
    }

    /// Submit the job and wait for completion
    pub async fn submit_and_wait(
        self,
        client: &DispatchClient,
        poll_interval: Duration,
        timeout: Option<Duration>,
    ) -> Result<JobStatus> {
        client
            .submit_and_wait(self.build(), poll_interval, timeout)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_config_default() {
        let config = DispatchConfig::default();
        assert!(config.coordinator_url.contains("8080"));
        assert_eq!(config.timeout, 30);
    }

    #[test]
    fn test_request_builder() {
        let request = TranscodeRequestBuilder::new("input.mp4", "output.webm")
            .video_codec("vp9")
            .audio_codec("opus")
            .video_bitrate(5_000_000)
            .audio_bitrate(128_000)
            .resolution(1920, 1080)
            .quality(28)
            .preset("medium")
            .priority(JobPriority::High)
            .metadata("source", "test")
            .build();

        assert_eq!(request.input, "input.mp4");
        assert_eq!(request.output, "output.webm");
        assert_eq!(request.params.video_codec, Some("vp9".to_string()));
        assert_eq!(request.params.audio_codec, Some("opus".to_string()));
        assert_eq!(request.params.video_bitrate, Some(5_000_000));
        assert_eq!(request.params.audio_bitrate, Some(128_000));
        assert_eq!(request.params.width, Some(1920));
        assert_eq!(request.params.height, Some(1080));
        assert_eq!(request.params.quality, Some(28));
        assert_eq!(request.priority, JobPriority::High);
        assert_eq!(request.metadata.get("source"), Some(&"test".to_string()));
    }
}
