//! Coordinator Server Implementation
//!
//! The coordinator is the central server that manages:
//! - Job queue with priority scheduling
//! - Worker registration and health monitoring
//! - REST API for job submission and status queries
//! - gRPC server for worker communication
//! - WebSocket connections for live progress updates

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path, State, WebSocketUpgrade},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, delete},
    Json, Router,
};
use axum::extract::ws::{Message, WebSocket};
use chrono::Utc;
use futures::stream::StreamExt;
use futures::SinkExt;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};
use tokio::time;
use tonic::{Request, Response, Status};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

use super::metrics::Metrics;
use super::protocol::{
    CreateJobRequest, CreateJobResponse, JobPriority, JobState, JobStatus,
    ProgressUpdate, TranscodeJob, WorkerInfo, WorkerListResponse, WorkerState,
    WorkerSummary,
};
use super::proto::{
    worker_service_server::{WorkerService, WorkerServiceServer},
    GetJobRequest, GetJobResponse, HeartbeatRequest, HeartbeatResponse,
    JobCompletionRequest, JobCompletionResponse, ProgressAck, ProgressReport,
    RegisterRequest, RegisterResponse, TranscodeJob as ProtoJob,
    TranscodeParams as ProtoParams, UnregisterRequest, UnregisterResponse,
    WorkerCapabilities as ProtoCapabilities, HwAccelType as ProtoHwAccel,
    JobStatus as ProtoJobStatus,
};
use crate::hwaccel::HwAccelType;

/// Coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// REST API port
    pub rest_port: u16,

    /// gRPC port for worker communication
    pub grpc_port: u16,

    /// Heartbeat timeout in seconds (workers are marked offline after this)
    pub heartbeat_timeout: u64,

    /// Job timeout in seconds (default for new jobs)
    pub default_job_timeout: u32,

    /// Maximum jobs in queue
    pub max_queue_size: usize,

    /// Enable CORS
    pub enable_cors: bool,

    /// Bind address
    pub bind_addr: String,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            rest_port: super::DEFAULT_REST_PORT,
            grpc_port: super::DEFAULT_GRPC_PORT,
            heartbeat_timeout: 60,
            default_job_timeout: super::DEFAULT_JOB_TIMEOUT,
            max_queue_size: 10000,
            enable_cors: true,
            bind_addr: "0.0.0.0".to_string(),
        }
    }
}

/// Job wrapper for priority queue ordering
#[derive(Clone)]
struct PriorityJob {
    job: TranscodeJob,
}

impl PartialEq for PriorityJob {
    fn eq(&self, other: &Self) -> bool {
        self.job.id == other.job.id
    }
}

impl Eq for PriorityJob {}

impl PartialOrd for PriorityJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then older jobs first
        match self.job.priority.cmp(&other.job.priority) {
            Ordering::Equal => other.job.created_at.cmp(&self.job.created_at),
            ord => ord,
        }
    }
}

/// Shared coordinator state
pub struct CoordinatorState {
    /// Configuration
    config: CoordinatorConfig,

    /// Job queue (pending jobs)
    job_queue: RwLock<BinaryHeap<PriorityJob>>,

    /// All jobs by ID (including completed/failed)
    jobs: RwLock<HashMap<Uuid, TranscodeJob>>,

    /// Registered workers
    workers: RwLock<HashMap<Uuid, WorkerInfo>>,

    /// Progress broadcast channel
    progress_tx: broadcast::Sender<ProgressUpdate>,

    /// Metrics
    metrics: Arc<Metrics>,

    /// Shutdown signal
    shutdown_tx: mpsc::Sender<()>,
}

impl CoordinatorState {
    /// Create new coordinator state
    fn new(config: CoordinatorConfig, shutdown_tx: mpsc::Sender<()>) -> Self {
        let (progress_tx, _) = broadcast::channel(1000);
        Self {
            config,
            job_queue: RwLock::new(BinaryHeap::new()),
            jobs: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            progress_tx,
            metrics: Arc::new(Metrics::new()),
            shutdown_tx,
        }
    }

    /// Add a job to the queue
    fn enqueue_job(&self, job: TranscodeJob) -> Result<JobStatus, String> {
        let queue_len = self.job_queue.read().len();
        if queue_len >= self.config.max_queue_size {
            return Err("Job queue is full".to_string());
        }

        let status = job.status();
        let id = job.id;

        self.jobs.write().insert(id, job.clone());
        self.job_queue.write().push(PriorityJob { job });
        self.metrics.jobs_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(status)
    }

    /// Get a job for a worker
    fn get_job_for_worker(&self, worker_id: Uuid) -> Option<TranscodeJob> {
        let workers = self.workers.read();
        let worker = workers.get(&worker_id)?;

        if !worker.can_accept_job() {
            return None;
        }
        drop(workers);

        let mut queue = self.job_queue.write();
        let mut workers = self.workers.write();

        // Find a job this worker can handle
        let mut temp_queue = Vec::new();
        let mut found_job = None;

        while let Some(pj) = queue.pop() {
            if found_job.is_none() {
                let worker = workers.get(&worker_id)?;
                if worker.capabilities.can_handle_job(&pj.job) {
                    found_job = Some(pj.job);
                    continue;
                }
            }
            temp_queue.push(pj);
        }

        // Restore jobs we didn't take
        for pj in temp_queue {
            queue.push(pj);
        }

        // Update job and worker state
        if let Some(ref mut job) = found_job {
            job.mark_running(worker_id);
            self.jobs.write().insert(job.id, job.clone());

            if let Some(w) = workers.get_mut(&worker_id) {
                w.start_job(job.id);
            }

            self.metrics.jobs_running.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        found_job
    }

    /// Update job progress
    fn update_job_progress(&self, job_id: Uuid, progress: f32, fps: f32, eta: Option<f64>) {
        if let Some(job) = self.jobs.write().get_mut(&job_id) {
            job.update_progress(progress, fps, eta);

            // Broadcast progress update
            let update = ProgressUpdate {
                job_id,
                progress,
                fps,
                bitrate: None,
                time_position: 0.0,
                total_duration: 0.0,
                eta,
                current_pass: None,
                status_message: None,
            };
            let _ = self.progress_tx.send(update);
        }

        self.metrics.transcode_fps.store(fps as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Complete a job
    fn complete_job(
        &self,
        job_id: Uuid,
        worker_id: Uuid,
        output_path: String,
        output_size: u64,
        encoding_duration_ms: u64,
    ) {
        // Update job state
        if let Some(job) = self.jobs.write().get_mut(&job_id) {
            job.mark_complete(output_path, output_size, encoding_duration_ms);
        }

        // Update worker state
        if let Some(worker) = self.workers.write().get_mut(&worker_id) {
            worker.finish_job(job_id);
        }

        self.metrics.jobs_running.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Fail a job
    fn fail_job(&self, job_id: Uuid, worker_id: Uuid, error: String) {
        // Update job state
        if let Some(job) = self.jobs.write().get_mut(&job_id) {
            job.mark_failed(error);
        }

        // Update worker state
        if let Some(worker) = self.workers.write().get_mut(&worker_id) {
            worker.finish_job(job_id);
        }

        self.metrics.jobs_running.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Register a worker
    fn register_worker(&self, info: WorkerInfo) -> String {
        let token = info.session_token.clone();
        self.workers.write().insert(info.id, info);
        self.metrics.worker_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        token
    }

    /// Unregister a worker
    fn unregister_worker(&self, worker_id: Uuid) {
        if self.workers.write().remove(&worker_id).is_some() {
            self.metrics.worker_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Update worker heartbeat
    fn worker_heartbeat(&self, worker_id: Uuid) -> bool {
        if let Some(worker) = self.workers.write().get_mut(&worker_id) {
            worker.heartbeat();
            true
        } else {
            false
        }
    }

    /// Get job status
    fn get_job_status(&self, job_id: Uuid) -> Option<JobStatus> {
        self.jobs.read().get(&job_id).map(|j| j.status())
    }

    /// List all jobs
    fn list_jobs(&self, limit: usize, offset: usize) -> Vec<JobStatus> {
        self.jobs
            .read()
            .values()
            .skip(offset)
            .take(limit)
            .map(|j| j.status())
            .collect()
    }

    /// List workers
    fn list_workers(&self) -> WorkerListResponse {
        let workers = self.workers.read();
        let summaries: Vec<WorkerSummary> = workers.values().map(WorkerSummary::from).collect();
        let total = summaries.len();
        WorkerListResponse {
            workers: summaries,
            total,
        }
    }

    /// Cancel a job
    fn cancel_job(&self, job_id: Uuid, reason: String) -> bool {
        // Remove from queue if pending
        {
            let mut queue = self.job_queue.write();
            let jobs: Vec<_> = queue.drain().collect();
            for pj in jobs {
                if pj.job.id != job_id {
                    queue.push(pj);
                }
            }
        }

        // Update job state
        if let Some(job) = self.jobs.write().get_mut(&job_id) {
            if !job.state.is_terminal() {
                job.mark_cancelled(reason);
                return true;
            }
        }
        false
    }

    /// Check worker health and mark unhealthy workers as offline
    fn check_worker_health(&self) {
        let timeout = self.config.heartbeat_timeout as i64;

        // First pass: identify unhealthy workers and collect jobs to requeue
        let jobs_to_requeue: Vec<uuid::Uuid>;
        {
            let mut workers = self.workers.write();
            let mut jobs = Vec::new();

            for worker in workers.values_mut() {
                if !worker.is_healthy(timeout) && worker.state != WorkerState::Offline {
                    warn!("Worker {} ({}) is unhealthy, marking offline", worker.id, worker.hostname);
                    worker.state = WorkerState::Offline;

                    // Collect jobs to requeue
                    jobs.extend(worker.active_jobs.drain(..));
                }
            }
            jobs_to_requeue = jobs;
        }

        // Second pass: requeue collected jobs (outside the workers lock)
        if !jobs_to_requeue.is_empty() {
            let mut job_map = self.jobs.write();
            let mut queue = self.job_queue.write();

            for job_id in jobs_to_requeue {
                if let Some(job) = job_map.get_mut(&job_id) {
                    job.state = JobState::Pending;
                    queue.push(PriorityJob { job: job.clone() });
                }
            }
        }
    }
}

/// The Coordinator server
pub struct Coordinator {
    state: Arc<CoordinatorState>,
    shutdown_rx: mpsc::Receiver<()>,
}

impl Coordinator {
    /// Create a new coordinator with the given configuration
    pub fn new(config: CoordinatorConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        let state = Arc::new(CoordinatorState::new(config, shutdown_tx));
        Self { state, shutdown_rx }
    }

    /// Run the coordinator server
    pub async fn run(mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = self.state.config.clone();

        info!(
            "Starting coordinator on {}:{} (REST) and {}:{} (gRPC)",
            config.bind_addr, config.rest_port, config.bind_addr, config.grpc_port
        );

        // Start health check task
        let health_state = self.state.clone();
        let health_handle = tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                health_state.check_worker_health();
            }
        });

        // Build REST API router
        let rest_state = self.state.clone();
        let app = build_rest_api(rest_state);

        // Start REST API server
        let rest_addr: SocketAddr = format!("{}:{}", config.bind_addr, config.rest_port).parse()?;
        let rest_handle = tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(rest_addr).await.unwrap();
            info!("REST API listening on {}", rest_addr);
            axum::serve(listener, app).await.unwrap();
        });

        // Build and start gRPC server
        let grpc_state = self.state.clone();
        let grpc_addr: SocketAddr = format!("{}:{}", config.bind_addr, config.grpc_port).parse()?;
        let grpc_handle = tokio::spawn(async move {
            let service = WorkerServiceImpl::new(grpc_state);
            info!("gRPC server listening on {}", grpc_addr);
            tonic::transport::Server::builder()
                .add_service(WorkerServiceServer::new(service))
                .serve(grpc_addr)
                .await
                .unwrap();
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

        // Cleanup
        health_handle.abort();
        rest_handle.abort();
        grpc_handle.abort();

        Ok(())
    }

    /// Get a handle to submit jobs programmatically
    pub fn handle(&self) -> CoordinatorHandle {
        CoordinatorHandle {
            state: self.state.clone(),
        }
    }
}

/// Handle for interacting with the coordinator programmatically
#[derive(Clone)]
pub struct CoordinatorHandle {
    state: Arc<CoordinatorState>,
}

impl CoordinatorHandle {
    /// Submit a job
    pub fn submit_job(&self, job: TranscodeJob) -> Result<JobStatus, String> {
        self.state.enqueue_job(job)
    }

    /// Get job status
    pub fn get_job_status(&self, job_id: Uuid) -> Option<JobStatus> {
        self.state.get_job_status(job_id)
    }

    /// Cancel a job
    pub fn cancel_job(&self, job_id: Uuid, reason: impl Into<String>) -> bool {
        self.state.cancel_job(job_id, reason.into())
    }

    /// Subscribe to progress updates
    pub fn subscribe_progress(&self) -> broadcast::Receiver<ProgressUpdate> {
        self.state.progress_tx.subscribe()
    }

    /// Get metrics
    pub fn metrics(&self) -> Arc<Metrics> {
        self.state.metrics.clone()
    }
}

// ============================================================================
// REST API Implementation
// ============================================================================

fn build_rest_api(state: Arc<CoordinatorState>) -> Router {
    let cors = CorsLayer::permissive();

    Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(get_metrics))
        .route("/jobs", post(create_job).get(list_jobs))
        .route("/jobs/:id", get(get_job).delete(cancel_job))
        .route("/jobs/:id/progress", get(job_progress_websocket))
        .route("/workers", get(list_workers))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

async fn health_check() -> &'static str {
    "OK"
}

async fn get_metrics(State(state): State<Arc<CoordinatorState>>) -> impl IntoResponse {
    let metrics = &state.metrics;
    let output = format!(
        "# HELP zvd_jobs_total Total number of jobs submitted\n\
         # TYPE zvd_jobs_total counter\n\
         zvd_jobs_total {}\n\n\
         # HELP zvd_jobs_running Number of currently running jobs\n\
         # TYPE zvd_jobs_running gauge\n\
         zvd_jobs_running {}\n\n\
         # HELP zvd_worker_count Number of registered workers\n\
         # TYPE zvd_worker_count gauge\n\
         zvd_worker_count {}\n\n\
         # HELP zvd_transcode_fps Current transcoding FPS\n\
         # TYPE zvd_transcode_fps gauge\n\
         zvd_transcode_fps {}\n",
        metrics.jobs_total.load(std::sync::atomic::Ordering::Relaxed),
        metrics.jobs_running.load(std::sync::atomic::Ordering::Relaxed),
        metrics.worker_count.load(std::sync::atomic::Ordering::Relaxed),
        metrics.transcode_fps.load(std::sync::atomic::Ordering::Relaxed),
    );
    (StatusCode::OK, [("content-type", "text/plain")], output)
}

#[derive(Deserialize)]
struct ListJobsQuery {
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn create_job(
    State(state): State<Arc<CoordinatorState>>,
    Json(req): Json<CreateJobRequest>,
) -> impl IntoResponse {
    let job = req.into_job();
    match state.enqueue_job(job) {
        Ok(status) => {
            let response = CreateJobResponse {
                id: status.id,
                status,
            };
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => {
            let error = serde_json::json!({ "error": e });
            (StatusCode::SERVICE_UNAVAILABLE, Json(error)).into_response()
        }
    }
}

async fn list_jobs(
    State(state): State<Arc<CoordinatorState>>,
    axum::extract::Query(query): axum::extract::Query<ListJobsQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(100);
    let offset = query.offset.unwrap_or(0);
    let jobs = state.list_jobs(limit, offset);
    Json(jobs)
}

async fn get_job(
    State(state): State<Arc<CoordinatorState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.get_job_status(id) {
        Some(status) => Json(status).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Job not found" }))).into_response(),
    }
}

async fn cancel_job(
    State(state): State<Arc<CoordinatorState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if state.cancel_job(id, "Cancelled by user".to_string()) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Job not found or already completed" }))).into_response()
    }
}

async fn job_progress_websocket(
    State(state): State<Arc<CoordinatorState>>,
    Path(job_id): Path<Uuid>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_progress_socket(socket, state, job_id))
}

async fn handle_progress_socket(socket: WebSocket, state: Arc<CoordinatorState>, job_id: Uuid) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to progress updates
    let mut progress_rx = state.progress_tx.subscribe();

    // Send initial status
    if let Some(status) = state.get_job_status(job_id) {
        if let Ok(msg) = serde_json::to_string(&status) {
            let _ = sender.send(Message::Text(msg.into())).await;
        }
    }

    loop {
        tokio::select! {
            // Forward progress updates
            Ok(update) = progress_rx.recv() => {
                if update.job_id == job_id {
                    if let Ok(msg) = serde_json::to_string(&update) {
                        if sender.send(Message::Text(msg.into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
            // Handle client messages (close, ping, etc.)
            Some(msg) = receiver.next() => {
                match msg {
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
        }

        // Check if job is terminal
        if let Some(status) = state.get_job_status(job_id) {
            if status.state.is_terminal() {
                // Send final status and close
                if let Ok(msg) = serde_json::to_string(&status) {
                    let _ = sender.send(Message::Text(msg.into())).await;
                }
                break;
            }
        }
    }
}

async fn list_workers(State(state): State<Arc<CoordinatorState>>) -> impl IntoResponse {
    Json(state.list_workers())
}

// ============================================================================
// gRPC Service Implementation
// ============================================================================

struct WorkerServiceImpl {
    state: Arc<CoordinatorState>,
}

impl WorkerServiceImpl {
    fn new(state: Arc<CoordinatorState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl WorkerService for WorkerServiceImpl {
    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let req = request.into_inner();
        let caps = req.capabilities.ok_or_else(|| Status::invalid_argument("Missing capabilities"))?;

        let worker_id = Uuid::parse_str(&caps.worker_id)
            .map_err(|_| Status::invalid_argument("Invalid worker ID"))?;

        // Convert proto capabilities to our type
        let hw_accel: Vec<HwAccelType> = caps.hw_accel.iter().filter_map(|h| {
            match ProtoHwAccel::try_from(*h).ok()? {
                ProtoHwAccel::HwAccelNone => Some(HwAccelType::None),
                ProtoHwAccel::HwAccelVaapi => Some(HwAccelType::VAAPI),
                ProtoHwAccel::HwAccelNvenc => Some(HwAccelType::NVENC),
                ProtoHwAccel::HwAccelNvdec => Some(HwAccelType::NVDEC),
                ProtoHwAccel::HwAccelQsv => Some(HwAccelType::QSV),
                ProtoHwAccel::HwAccelVideotoolbox => Some(HwAccelType::VideoToolbox),
                ProtoHwAccel::HwAccelAmf => Some(HwAccelType::AMF),
                ProtoHwAccel::HwAccelDxva2 => Some(HwAccelType::DXVA2),
                ProtoHwAccel::HwAccelD3d11va => Some(HwAccelType::D3D11VA),
                ProtoHwAccel::HwAccelVulkan => Some(HwAccelType::Vulkan),
            }
        }).collect();

        let capabilities = super::protocol::WorkerCapabilities {
            hw_accel,
            max_concurrent_jobs: caps.max_concurrent_jobs,
            supported_video_codecs: caps.supported_video_codecs,
            supported_audio_codecs: caps.supported_audio_codecs,
            available_memory: caps.available_memory,
            cpu_cores: caps.cpu_cores,
            gpu_memory: if caps.gpu_memory > 0 { Some(caps.gpu_memory) } else { None },
        };

        let worker_info = WorkerInfo::new(
            worker_id,
            caps.hostname,
            capabilities,
            caps.version,
        );

        let session_token = self.state.register_worker(worker_info);

        info!("Worker {} registered", worker_id);

        Ok(Response::new(RegisterResponse {
            success: true,
            error: String::new(),
            heartbeat_interval: super::DEFAULT_HEARTBEAT_INTERVAL,
            session_token,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let worker_id = Uuid::parse_str(&req.worker_id)
            .map_err(|_| Status::invalid_argument("Invalid worker ID"))?;

        if self.state.worker_heartbeat(worker_id) {
            Ok(Response::new(HeartbeatResponse {
                acknowledged: true,
                commands: vec![],
            }))
        } else {
            Err(Status::not_found("Worker not registered"))
        }
    }

    async fn get_job(
        &self,
        request: Request<GetJobRequest>,
    ) -> Result<Response<GetJobResponse>, Status> {
        let req = request.into_inner();
        let worker_id = Uuid::parse_str(&req.worker_id)
            .map_err(|_| Status::invalid_argument("Invalid worker ID"))?;

        match self.state.get_job_for_worker(worker_id) {
            Some(job) => {
                let proto_job = job_to_proto(&job);
                Ok(Response::new(GetJobResponse {
                    has_job: true,
                    job: Some(proto_job),
                }))
            }
            None => {
                Ok(Response::new(GetJobResponse {
                    has_job: false,
                    job: None,
                }))
            }
        }
    }

    async fn report_progress(
        &self,
        request: Request<tonic::Streaming<ProgressReport>>,
    ) -> Result<Response<ProgressAck>, Status> {
        use futures::StreamExt;

        let mut stream = request.into_inner();
        let state = self.state.clone();

        // Process all incoming progress reports
        while let Some(report_result) = stream.next().await {
            let report = report_result?;
            let job_id = Uuid::parse_str(&report.job_id)
                .map_err(|_| Status::invalid_argument("Invalid job ID"))?;

            state.update_job_progress(
                job_id,
                report.progress,
                report.fps,
                if report.eta > 0.0 { Some(report.eta) } else { None },
            );
        }

        // Return final acknowledgment
        Ok(Response::new(ProgressAck {
            received: true,
            command: String::new(),
        }))
    }

    async fn complete_job(
        &self,
        request: Request<JobCompletionRequest>,
    ) -> Result<Response<JobCompletionResponse>, Status> {
        let req = request.into_inner();
        let job_id = Uuid::parse_str(&req.job_id)
            .map_err(|_| Status::invalid_argument("Invalid job ID"))?;
        let worker_id = Uuid::parse_str(&req.worker_id)
            .map_err(|_| Status::invalid_argument("Invalid worker ID"))?;

        match ProtoJobStatus::try_from(req.status).unwrap_or(ProtoJobStatus::Unknown) {
            ProtoJobStatus::Completed => {
                self.state.complete_job(
                    job_id,
                    worker_id,
                    req.output_path,
                    req.output_size,
                    req.encoding_duration_ms,
                );
                info!("Job {} completed by worker {}", job_id, worker_id);
            }
            ProtoJobStatus::Failed => {
                self.state.fail_job(job_id, worker_id, req.error);
                warn!("Job {} failed on worker {}", job_id, worker_id);
            }
            ProtoJobStatus::Cancelled => {
                self.state.cancel_job(job_id, "Cancelled by worker".to_string());
                info!("Job {} cancelled by worker {}", job_id, worker_id);
            }
            _ => {
                return Err(Status::invalid_argument("Unknown job status"));
            }
        }

        Ok(Response::new(JobCompletionResponse {
            acknowledged: true,
            message: "Job completion recorded".to_string(),
        }))
    }

    async fn unregister(
        &self,
        request: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        let req = request.into_inner();
        let worker_id = Uuid::parse_str(&req.worker_id)
            .map_err(|_| Status::invalid_argument("Invalid worker ID"))?;

        self.state.unregister_worker(worker_id);
        info!("Worker {} unregistered: {}", worker_id, req.reason);

        Ok(Response::new(UnregisterResponse { success: true }))
    }
}

// Helper function to convert job to proto format
fn job_to_proto(job: &TranscodeJob) -> ProtoJob {
    let params = ProtoParams {
        video_codec: job.params.video_codec.clone().unwrap_or_default(),
        audio_codec: job.params.audio_codec.clone().unwrap_or_default(),
        video_bitrate: job.params.video_bitrate.unwrap_or(0),
        audio_bitrate: job.params.audio_bitrate.unwrap_or(0),
        resolution: match (job.params.width, job.params.height) {
            (Some(w), Some(h)) => format!("{}x{}", w, h),
            _ => String::new(),
        },
        frame_rate: job.params.frame_rate.map(|f| f.to_string()).unwrap_or_default(),
        pixel_format: job.params.pixel_format.clone().unwrap_or_default(),
        preset: job.params.preset.clone().unwrap_or_default(),
        quality: job.params.quality.unwrap_or(0),
        two_pass: job.params.two_pass,
        hw_accel: job.params.hw_accel.map(|h| match h {
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
        }).unwrap_or(0),
        extra_params: job.params.extra_params.clone(),
        start_time: job.params.start_time.unwrap_or(0.0),
        duration: job.params.duration.unwrap_or(0.0),
        audio_channels: job.params.audio_channels.unwrap_or(0),
        audio_sample_rate: job.params.audio_sample_rate.unwrap_or(0),
    };

    ProtoJob {
        job_id: job.id.to_string(),
        input_path: job.input.clone(),
        output_path: job.output.clone(),
        params: Some(params),
        priority: job.priority as u32,
        created_at: job.created_at.timestamp_millis(),
        timeout: job.timeout,
        callback_url: job.callback_url.clone().unwrap_or_default(),
        metadata: job.metadata.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let mut heap = BinaryHeap::new();

        // Add jobs with different priorities
        let low = TranscodeJob::new("a", "b", Default::default()).with_priority(JobPriority::Low);
        let normal = TranscodeJob::new("c", "d", Default::default()).with_priority(JobPriority::Normal);
        let high = TranscodeJob::new("e", "f", Default::default()).with_priority(JobPriority::High);

        heap.push(PriorityJob { job: low });
        heap.push(PriorityJob { job: normal });
        heap.push(PriorityJob { job: high });

        // Should come out in priority order
        assert_eq!(heap.pop().unwrap().job.priority, JobPriority::High);
        assert_eq!(heap.pop().unwrap().job.priority, JobPriority::Normal);
        assert_eq!(heap.pop().unwrap().job.priority, JobPriority::Low);
    }

    #[tokio::test]
    async fn test_coordinator_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.rest_port, super::super::DEFAULT_REST_PORT);
        assert_eq!(config.grpc_port, super::super::DEFAULT_GRPC_PORT);
    }
}
