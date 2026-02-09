//! Distributed Transcoding Server Mode
//!
//! This module implements a distributed transcoding system similar to rffmpeg,
//! allowing multiple worker nodes to process transcode jobs from a central coordinator.
//!
//! # Architecture
//!
//! The system consists of three main components:
//!
//! - **Coordinator**: Central server that manages jobs and workers
//!   - REST API for job submission and status queries
//!   - gRPC server for worker communication
//!   - Priority-based job queue
//!   - Worker health monitoring
//!
//! - **Worker**: Daemon that executes transcode jobs
//!   - Auto-detects hardware capabilities on startup
//!   - Registers with coordinator via gRPC
//!   - Pulls jobs from queue based on capabilities
//!   - Reports progress via streaming gRPC
//!
//! - **Dispatch Client**: Library and CLI for job submission
//!   - REST client for coordinator API
//!   - FFmpeg-compatible argument parsing
//!
//! # Example Usage
//!
//! Start coordinator:
//! ```bash
//! zvd coordinator --port 8080 --grpc-port 50051
//! ```
//!
//! Start worker:
//! ```bash
//! zvd worker --coordinator http://localhost:8080
//! ```
//!
//! Submit job:
//! ```bash
//! zvd dispatch -i input.mp4 -o output.webm --coordinator http://localhost:8080
//! ```
//!
//! # FFmpeg Compatibility
//!
//! The dispatch client supports common FFmpeg arguments for easy migration:
//! - `-i input.mp4` - Input file
//! - `-c:v libx264` - Video codec
//! - `-c:a aac` - Audio codec
//! - `-b:v 5M` - Video bitrate
//! - `-b:a 128k` - Audio bitrate
//! - `-s 1920x1080` - Resolution
//! - `-r 30` - Frame rate

pub mod coordinator;
pub mod dispatch;
pub mod ffcompat;
pub mod metrics;
pub mod protocol;
pub mod worker;

// Re-export main types
pub use coordinator::{Coordinator, CoordinatorConfig};
pub use dispatch::{DispatchClient, DispatchConfig, TranscodeRequestBuilder};
pub use metrics::Metrics;
pub use protocol::{
    JobPriority, JobState, JobStatus, TranscodeJob, TranscodeParams, WorkerCapabilities,
    WorkerInfo, WorkerState,
};
pub use worker::{Worker, WorkerConfig};

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("zvd.worker");
}

use crate::error::{Error, Result};

/// Default port for REST API
pub const DEFAULT_REST_PORT: u16 = 8080;

/// Default port for gRPC
pub const DEFAULT_GRPC_PORT: u16 = 50051;

/// Default heartbeat interval in seconds
pub const DEFAULT_HEARTBEAT_INTERVAL: u32 = 30;

/// Default job timeout in seconds (1 hour)
pub const DEFAULT_JOB_TIMEOUT: u32 = 3600;

/// Server version string
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
