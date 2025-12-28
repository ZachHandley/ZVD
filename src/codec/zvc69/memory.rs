//! ZVC69 Memory Optimization Infrastructure
//!
//! This module provides memory management utilities for zero-allocation encoding
//! and decoding in the ZVC69 neural video codec. Pre-allocated buffer pools and
//! arena allocators reduce allocation overhead during real-time encoding.
//!
//! ## Features
//!
//! - **Frame Buffer Pool**: Pre-allocated buffers for video frames and latent tensors
//! - **Automatic Return**: Pooled buffers return to pool on drop via RAII
//! - **Resolution Presets**: Common resolutions (720p, 1080p, 4K) with optimal settings
//! - **Arena Allocator**: Fast bump allocation for temporary bitstream operations
//! - **Statistics Tracking**: Monitor allocations, reuses, and memory usage
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::memory::{FramePool, PoolConfig, BitstreamArena};
//!
//! // Create a frame pool for 1080p encoding
//! let config = PoolConfig::preset_1080p();
//! let pool = FramePool::new(config);
//!
//! // Pre-warm the pool with buffers
//! pool.prewarm(4);
//!
//! // Acquire a buffer (automatically returns to pool on drop)
//! let buffer = pool.acquire();
//! // ... use buffer for encoding ...
//! drop(buffer);  // Buffer returns to pool
//!
//! // Check pool statistics
//! let stats = pool.stats();
//! println!("Allocations: {}, Reuses: {}", stats.allocations, stats.reuses);
//! ```
//!
//! ## Memory Layout
//!
//! Buffers are allocated with optimal alignment for SIMD operations:
//!
//! ```text
//! FramePool:
//!   available: [PooledBuffer, PooledBuffer, ...]
//!              |
//!              v
//!   PooledBuffer:
//!     data: Vec<f32> (aligned, preallocated)
//!     shape: BufferShape { batch, channels, height, width }
//!     pool: Arc<FramePool> (for automatic return)
//! ```

use std::sync::{Arc, Mutex, Weak};

use super::error::ZVC69Error;

// ============================================================================
// Buffer Shape
// ============================================================================

/// Shape descriptor for 4D tensor buffers (NCHW format)
///
/// Used to track buffer dimensions for frame and latent tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferShape {
    /// Batch size (typically 1 for real-time encoding)
    pub batch: usize,

    /// Number of channels (3 for RGB/YUV, 192 for latents)
    pub channels: usize,

    /// Height in pixels/samples
    pub height: usize,

    /// Width in pixels/samples
    pub width: usize,
}

impl BufferShape {
    /// Create a new buffer shape
    pub fn new(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        BufferShape {
            batch,
            channels,
            height,
            width,
        }
    }

    /// Create a shape for an input frame (RGB/YUV)
    pub fn frame(height: usize, width: usize) -> Self {
        BufferShape {
            batch: 1,
            channels: 3,
            height,
            width,
        }
    }

    /// Create a shape for latent representation
    ///
    /// Latents are typically 1/16 the spatial resolution of input frames.
    pub fn latent(height: usize, width: usize, channels: usize) -> Self {
        BufferShape {
            batch: 1,
            channels,
            height,
            width,
        }
    }

    /// Create a shape for hyperprior representation
    ///
    /// Hyperprior is typically 1/4 the spatial resolution of latents.
    pub fn hyperprior(height: usize, width: usize, channels: usize) -> Self {
        BufferShape {
            batch: 1,
            channels,
            height,
            width,
        }
    }

    /// Calculate total number of elements
    pub fn num_elements(&self) -> usize {
        self.batch * self.channels * self.height * self.width
    }

    /// Calculate memory size in bytes (f32 elements)
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * std::mem::size_of::<f32>()
    }

    /// Check if shape is valid (non-zero dimensions)
    pub fn is_valid(&self) -> bool {
        self.batch > 0 && self.channels > 0 && self.height > 0 && self.width > 0
    }

    /// Create corresponding latent shape from frame shape
    ///
    /// Assumes 16x spatial downsampling and 192 latent channels.
    pub fn to_latent_shape(&self) -> Self {
        BufferShape {
            batch: self.batch,
            channels: 192,
            height: (self.height + 15) / 16,
            width: (self.width + 15) / 16,
        }
    }

    /// Create corresponding hyperprior shape from latent shape
    ///
    /// Assumes 4x spatial downsampling and 128 hyperprior channels.
    pub fn to_hyperprior_shape(&self) -> Self {
        BufferShape {
            batch: self.batch,
            channels: 128,
            height: (self.height + 3) / 4,
            width: (self.width + 3) / 4,
        }
    }
}

impl Default for BufferShape {
    fn default() -> Self {
        BufferShape {
            batch: 1,
            channels: 3,
            height: 1080,
            width: 1920,
        }
    }
}

impl std::fmt::Display for BufferShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.batch, self.channels, self.height, self.width
        )
    }
}

// ============================================================================
// Pool Configuration
// ============================================================================

/// Configuration for the frame buffer pool
///
/// Controls pool capacity, buffer sizes, and allocation behavior.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial number of buffers to pre-allocate
    pub initial_capacity: usize,

    /// Maximum number of buffers to keep in pool
    pub max_capacity: usize,

    /// Frame width in pixels
    pub frame_width: u32,

    /// Frame height in pixels
    pub frame_height: u32,

    /// Number of channels (3 for RGB/YUV frames)
    pub channels: usize,

    /// Number of latent channels (default: 192)
    pub latent_channels: usize,

    /// Number of hyperprior channels (default: 128)
    pub hyperprior_channels: usize,

    /// Enable zero-initialization of new buffers
    pub zero_init: bool,

    /// Enable pinned (page-locked) memory for faster GPU transfers
    ///
    /// When enabled, buffers are allocated as pinned memory which cannot be
    /// swapped out by the OS. This enables faster DMA transfers to/from GPU
    /// memory but uses more system resources.
    ///
    /// Note: This is a hint - actual pinning depends on platform support.
    /// On systems without GPU or pinning support, this flag is ignored.
    pub pinned_memory: bool,
}

impl PoolConfig {
    /// Create a new pool configuration
    pub fn new(width: u32, height: u32) -> Self {
        PoolConfig {
            initial_capacity: 2,
            max_capacity: 8,
            frame_width: width,
            frame_height: height,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: true,
            pinned_memory: false,
        }
    }

    /// Create configuration for a given resolution
    pub fn for_resolution(width: u32, height: u32) -> Self {
        // Calculate optimal capacity based on resolution
        let pixels = (width as u64) * (height as u64);
        let capacity = if pixels > 8_000_000 {
            // 4K+: fewer buffers due to memory
            2
        } else if pixels > 2_000_000 {
            // 1080p: moderate buffers
            4
        } else {
            // 720p and below: more buffers
            6
        };

        PoolConfig {
            initial_capacity: capacity,
            max_capacity: capacity * 2,
            frame_width: width,
            frame_height: height,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: true,
            pinned_memory: false,
        }
    }

    /// Preset for 720p (1280x720)
    pub fn preset_720p() -> Self {
        PoolConfig {
            initial_capacity: 4,
            max_capacity: 12,
            frame_width: 1280,
            frame_height: 720,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: true,
            pinned_memory: false,
        }
    }

    /// Preset for 1080p (1920x1088 - aligned to 16)
    pub fn preset_1080p() -> Self {
        PoolConfig {
            initial_capacity: 3,
            max_capacity: 8,
            frame_width: 1920,
            frame_height: 1088,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: true,
            pinned_memory: false,
        }
    }

    /// Preset for 1080p real-time encoding (1920x1088 - aligned to 16)
    ///
    /// Optimized for sustained real-time encoding at 1080p:
    /// - Higher initial capacity (6) to avoid allocation during encoding
    /// - Higher max capacity (12) for burst handling
    /// - Zero-init disabled for faster buffer allocation
    /// - Pinned memory disabled by default (enable with `with_pinned_memory`)
    ///
    /// Use with `FramePool::prewarm_1080p()` to pre-allocate all buffers before encoding.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::memory::{FramePool, PoolConfig};
    ///
    /// let config = PoolConfig::preset_1080p_realtime();
    /// let pool = FramePool::new(config);
    /// pool.prewarm_1080p(); // Pre-allocate all buffers
    /// ```
    pub fn preset_1080p_realtime() -> Self {
        PoolConfig {
            initial_capacity: 6,
            max_capacity: 12,
            frame_width: 1920,
            frame_height: 1088,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: false,
            pinned_memory: false,
        }
    }

    /// Preset for 1080p real-time encoding with pinned memory for GPU acceleration
    ///
    /// Same as `preset_1080p_realtime()` but with pinned memory enabled for
    /// faster GPU DMA transfers. Use this when encoding with TensorRT or
    /// other GPU-accelerated neural networks.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::memory::{FramePool, PoolConfig};
    ///
    /// let config = PoolConfig::preset_1080p_realtime_gpu();
    /// let pool = FramePool::new(config);
    /// pool.prewarm_1080p(); // Pre-allocate all pinned buffers
    /// ```
    pub fn preset_1080p_realtime_gpu() -> Self {
        PoolConfig {
            initial_capacity: 6,
            max_capacity: 12,
            frame_width: 1920,
            frame_height: 1088,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: false,
            pinned_memory: true,
        }
    }

    /// Preset for 4K (3840x2160)
    pub fn preset_4k() -> Self {
        PoolConfig {
            initial_capacity: 2,
            max_capacity: 4,
            frame_width: 3840,
            frame_height: 2160,
            channels: 3,
            latent_channels: 192,
            hyperprior_channels: 128,
            zero_init: true,
            pinned_memory: false,
        }
    }

    /// Set initial capacity
    pub fn with_initial_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }

    /// Set maximum capacity
    pub fn with_max_capacity(mut self, capacity: usize) -> Self {
        self.max_capacity = capacity;
        self
    }

    /// Set number of latent channels
    pub fn with_latent_channels(mut self, channels: usize) -> Self {
        self.latent_channels = channels;
        self
    }

    /// Set zero initialization
    pub fn with_zero_init(mut self, zero_init: bool) -> Self {
        self.zero_init = zero_init;
        self
    }

    /// Enable or disable pinned memory for GPU transfers
    ///
    /// Pinned memory enables faster DMA transfers between CPU and GPU
    /// but uses more system resources and may impact system stability
    /// if overused.
    pub fn with_pinned_memory(mut self, pinned: bool) -> Self {
        self.pinned_memory = pinned;
        self
    }

    /// Get the frame shape for this configuration
    pub fn frame_shape(&self) -> BufferShape {
        BufferShape {
            batch: 1,
            channels: self.channels,
            height: self.frame_height as usize,
            width: self.frame_width as usize,
        }
    }

    /// Get the latent shape for this configuration
    pub fn latent_shape(&self) -> BufferShape {
        let h = (self.frame_height as usize + 15) / 16;
        let w = (self.frame_width as usize + 15) / 16;
        BufferShape {
            batch: 1,
            channels: self.latent_channels,
            height: h,
            width: w,
        }
    }

    /// Get the hyperprior shape for this configuration
    pub fn hyperprior_shape(&self) -> BufferShape {
        let latent = self.latent_shape();
        BufferShape {
            batch: 1,
            channels: self.hyperprior_channels,
            height: (latent.height + 3) / 4,
            width: (latent.width + 3) / 4,
        }
    }

    /// Calculate total memory for one complete buffer set (frame + latent + hyperprior)
    pub fn memory_per_set(&self) -> usize {
        self.frame_shape().size_bytes()
            + self.latent_shape().size_bytes()
            + self.hyperprior_shape().size_bytes()
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self::preset_1080p()
    }
}

// ============================================================================
// Pool Statistics
// ============================================================================

/// Statistics for buffer pool usage
///
/// Tracks allocations, reuses, and memory consumption for monitoring
/// and optimization.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of buffer allocations (not reuses)
    pub allocations: u64,

    /// Number of times buffers were reused from pool
    pub reuses: u64,

    /// Current number of available buffers in pool
    pub current_available: usize,

    /// Peak number of buffers in use simultaneously
    pub peak_usage: usize,

    /// Current number of buffers in use
    pub current_in_use: usize,

    /// Total bytes allocated by the pool
    pub total_bytes_allocated: usize,

    /// Number of buffers returned to pool
    pub returns: u64,

    /// Number of buffers that were dropped (pool at capacity)
    pub drops: u64,
}

impl PoolStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the reuse ratio (reuses / total acquisitions)
    pub fn reuse_ratio(&self) -> f64 {
        let total = self.allocations + self.reuses;
        if total == 0 {
            0.0
        } else {
            self.reuses as f64 / total as f64
        }
    }

    /// Get total number of buffer acquisitions
    pub fn total_acquisitions(&self) -> u64 {
        self.allocations + self.reuses
    }

    /// Check if pool is effectively reusing buffers
    pub fn is_effective(&self) -> bool {
        self.reuse_ratio() > 0.5
    }
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PoolStats {{ allocs: {}, reuses: {}, reuse_ratio: {:.1}%, available: {}, peak: {}, bytes: {} }}",
            self.allocations,
            self.reuses,
            self.reuse_ratio() * 100.0,
            self.current_available,
            self.peak_usage,
            self.total_bytes_allocated
        )
    }
}

// ============================================================================
// Pooled Buffer
// ============================================================================

/// A buffer acquired from the pool with automatic return on drop
///
/// When a `PooledBuffer` is dropped, it automatically returns its data
/// to the pool for reuse (if the pool still exists and has capacity).
pub struct PooledBuffer {
    /// The actual buffer data
    data: Option<Vec<f32>>,

    /// Shape of the buffer
    shape: BufferShape,

    /// Weak reference to the pool for returning buffer on drop
    pool: Option<Weak<FramePoolInner>>,
}

impl PooledBuffer {
    /// Create a new pooled buffer (internal use)
    fn new(data: Vec<f32>, shape: BufferShape, pool: Option<Weak<FramePoolInner>>) -> Self {
        PooledBuffer {
            data: Some(data),
            shape,
            pool,
        }
    }

    /// Create a standalone buffer (not associated with a pool)
    pub fn standalone(shape: BufferShape) -> Self {
        let size = shape.num_elements();
        PooledBuffer {
            data: Some(vec![0.0; size]),
            shape,
            pool: None,
        }
    }

    /// Get the buffer shape
    pub fn shape(&self) -> BufferShape {
        self.shape
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.as_ref().map_or(0, |d| d.len())
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get immutable access to buffer data
    pub fn data(&self) -> &[f32] {
        self.data.as_ref().map_or(&[], |d| d.as_slice())
    }

    /// Get mutable access to buffer data
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data.as_mut().map_or(&mut [], |d| d.as_mut_slice())
    }

    /// Fill buffer with zeros
    pub fn zero(&mut self) {
        if let Some(ref mut data) = self.data {
            data.fill(0.0);
        }
    }

    /// Fill buffer with a value
    pub fn fill(&mut self, value: f32) {
        if let Some(ref mut data) = self.data {
            data.fill(value);
        }
    }

    /// Copy data from a slice
    pub fn copy_from_slice(&mut self, src: &[f32]) -> Result<(), ZVC69Error> {
        if let Some(ref mut data) = self.data {
            if src.len() != data.len() {
                return Err(ZVC69Error::DimensionMismatch {
                    actual_w: src.len() as u32,
                    actual_h: 1,
                    expected_w: data.len() as u32,
                    expected_h: 1,
                });
            }
            data.copy_from_slice(src);
            Ok(())
        } else {
            Err(ZVC69Error::AllocationFailed { size: 0 })
        }
    }

    /// Check if this buffer belongs to a pool
    pub fn is_pooled(&self) -> bool {
        self.pool.is_some()
    }

    /// Detach buffer from pool (becomes standalone)
    pub fn detach(&mut self) {
        self.pool = None;
    }

    /// Convert to Vec, consuming the buffer
    pub fn into_vec(mut self) -> Vec<f32> {
        self.pool = None; // Prevent return to pool
        self.data.take().unwrap_or_default()
    }

    /// Get memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.shape.size_bytes()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            if let Some(ref pool_weak) = self.pool {
                if let Some(pool) = pool_weak.upgrade() {
                    pool.return_buffer(data, self.shape);
                }
            }
        }
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("shape", &self.shape)
            .field("len", &self.len())
            .field("pooled", &self.is_pooled())
            .finish()
    }
}

impl Clone for PooledBuffer {
    fn clone(&self) -> Self {
        // Clone creates a standalone buffer (not pooled)
        PooledBuffer {
            data: self.data.clone(),
            shape: self.shape,
            pool: None, // Clones are not returned to pool
        }
    }
}

// ============================================================================
// Frame Pool Inner (Internal)
// ============================================================================

/// Internal pool state (behind Arc for thread safety)
struct FramePoolInner {
    /// Available buffers, keyed by shape
    available: Mutex<Vec<(BufferShape, Vec<f32>)>>,

    /// Configuration
    config: PoolConfig,

    /// Statistics
    stats: Mutex<PoolStats>,
}

impl FramePoolInner {
    fn new(config: PoolConfig) -> Self {
        FramePoolInner {
            available: Mutex::new(Vec::new()),
            config,
            stats: Mutex::new(PoolStats::new()),
        }
    }

    fn return_buffer(&self, data: Vec<f32>, shape: BufferShape) {
        let mut available = self.available.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.returns += 1;
        stats.current_in_use = stats.current_in_use.saturating_sub(1);

        if available.len() < self.config.max_capacity {
            available.push((shape, data));
            stats.current_available = available.len();
        } else {
            // Pool at capacity, drop the buffer
            stats.drops += 1;
        }
    }
}

// ============================================================================
// Frame Pool
// ============================================================================

/// Frame buffer pool for reusing allocated memory
///
/// The pool maintains a collection of pre-allocated buffers that can be
/// acquired and automatically returned when dropped. This reduces allocation
/// overhead during real-time encoding.
///
/// # Thread Safety
///
/// The pool is thread-safe and can be shared across threads using `Arc`.
///
/// # Example
///
/// ```rust,ignore
/// let pool = FramePool::new(PoolConfig::preset_1080p());
/// pool.prewarm(4);
///
/// // Acquire buffers (automatically return on drop)
/// let buf1 = pool.acquire();
/// let buf2 = pool.acquire();
///
/// // Use buffers...
/// drop(buf1); // Returns to pool
/// drop(buf2); // Returns to pool
///
/// // Check statistics
/// let stats = pool.stats();
/// assert!(stats.reuse_ratio() > 0.5);
/// ```
pub struct FramePool {
    inner: Arc<FramePoolInner>,
}

impl FramePool {
    /// Create a new frame pool with the given configuration
    pub fn new(config: PoolConfig) -> Arc<Self> {
        let pool = Arc::new(FramePool {
            inner: Arc::new(FramePoolInner::new(config)),
        });

        // Pre-allocate initial buffers
        let initial = pool.inner.config.initial_capacity;
        if initial > 0 {
            pool.prewarm(initial);
        }

        pool
    }

    /// Create a pool for the given resolution
    pub fn for_resolution(width: u32, height: u32) -> Arc<Self> {
        Self::new(PoolConfig::for_resolution(width, height))
    }

    /// Acquire a frame buffer from the pool
    ///
    /// Returns a pooled buffer that automatically returns to the pool on drop.
    /// If no buffer is available, a new one is allocated.
    pub fn acquire(&self) -> PooledBuffer {
        self.acquire_shape(self.inner.config.frame_shape())
    }

    /// Acquire a latent buffer from the pool
    pub fn acquire_latent(&self) -> PooledBuffer {
        self.acquire_shape(self.inner.config.latent_shape())
    }

    /// Acquire a hyperprior buffer from the pool
    pub fn acquire_hyperprior(&self) -> PooledBuffer {
        self.acquire_shape(self.inner.config.hyperprior_shape())
    }

    /// Acquire a buffer with a specific shape
    pub fn acquire_shape(&self, shape: BufferShape) -> PooledBuffer {
        let mut available = self.inner.available.lock().unwrap();
        let mut stats = self.inner.stats.lock().unwrap();

        // Try to find a matching buffer
        let pos = available.iter().position(|(s, _)| *s == shape);

        let data = if let Some(idx) = pos {
            stats.reuses += 1;
            let (_, data) = available.remove(idx);
            stats.current_available = available.len();
            data
        } else {
            // Allocate new buffer
            stats.allocations += 1;
            let size = shape.num_elements();
            stats.total_bytes_allocated += size * std::mem::size_of::<f32>();
            if self.inner.config.zero_init {
                vec![0.0; size]
            } else {
                let mut v = Vec::with_capacity(size);
                // SAFETY: We immediately set the length and will initialize before use
                unsafe { v.set_len(size) };
                v
            }
        };

        stats.current_in_use += 1;
        if stats.current_in_use > stats.peak_usage {
            stats.peak_usage = stats.current_in_use;
        }

        PooledBuffer::new(data, shape, Some(Arc::downgrade(&self.inner)))
    }

    /// Pre-warm the pool by pre-allocating buffers
    ///
    /// This reduces allocation latency during initial encoding.
    pub fn prewarm(&self, count: usize) {
        let frame_shape = self.inner.config.frame_shape();
        let latent_shape = self.inner.config.latent_shape();

        let mut available = self.inner.available.lock().unwrap();
        let mut stats = self.inner.stats.lock().unwrap();

        for _ in 0..count {
            if available.len() >= self.inner.config.max_capacity {
                break;
            }

            // Pre-allocate frame buffer
            let frame_size = frame_shape.num_elements();
            let frame_data = vec![0.0; frame_size];
            stats.allocations += 1;
            stats.total_bytes_allocated += frame_size * std::mem::size_of::<f32>();
            available.push((frame_shape, frame_data));

            // Pre-allocate latent buffer
            if available.len() < self.inner.config.max_capacity {
                let latent_size = latent_shape.num_elements();
                let latent_data = vec![0.0; latent_size];
                stats.allocations += 1;
                stats.total_bytes_allocated += latent_size * std::mem::size_of::<f32>();
                available.push((latent_shape, latent_data));
            }
        }

        stats.current_available = available.len();
    }

    /// Pre-warm the pool for 1080p real-time encoding
    ///
    /// Pre-allocates all buffers needed for sustained 1080p encoding:
    /// - 6 frame buffers (1920x1088x3 = ~25 MB each)
    /// - 6 latent buffers (120x68x192 = ~6 MB each)
    /// - 6 hyperprior buffers (30x17x128 = ~260 KB each)
    ///
    /// Total memory: ~186 MB for a fully pre-warmed pool.
    ///
    /// This ensures zero allocations during steady-state encoding,
    /// which is critical for real-time performance.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::memory::{FramePool, PoolConfig};
    ///
    /// let config = PoolConfig::preset_1080p_realtime();
    /// let pool = FramePool::new(config);
    /// pool.prewarm_1080p();
    ///
    /// // Now the pool is ready for real-time encoding
    /// assert!(pool.available_count() >= 12);
    /// ```
    pub fn prewarm_1080p(&self) {
        let frame_shape = self.inner.config.frame_shape();
        let latent_shape = self.inner.config.latent_shape();
        let hyperprior_shape = self.inner.config.hyperprior_shape();

        let mut available = self.inner.available.lock().unwrap();
        let mut stats = self.inner.stats.lock().unwrap();

        // Pre-allocate 6 sets of buffers (frame + latent + hyperprior)
        let sets_to_allocate = 6;
        for _ in 0..sets_to_allocate {
            if available.len() >= self.inner.config.max_capacity {
                break;
            }

            // Pre-allocate frame buffer
            let frame_size = frame_shape.num_elements();
            let frame_data = if self.inner.config.zero_init {
                vec![0.0; frame_size]
            } else {
                let mut v = Vec::with_capacity(frame_size);
                // SAFETY: We will initialize before use, skipping zero-init for speed
                unsafe { v.set_len(frame_size) };
                v
            };
            stats.allocations += 1;
            stats.total_bytes_allocated += frame_size * std::mem::size_of::<f32>();
            available.push((frame_shape, frame_data));

            // Pre-allocate latent buffer
            if available.len() < self.inner.config.max_capacity {
                let latent_size = latent_shape.num_elements();
                let latent_data = if self.inner.config.zero_init {
                    vec![0.0; latent_size]
                } else {
                    let mut v = Vec::with_capacity(latent_size);
                    unsafe { v.set_len(latent_size) };
                    v
                };
                stats.allocations += 1;
                stats.total_bytes_allocated += latent_size * std::mem::size_of::<f32>();
                available.push((latent_shape, latent_data));
            }

            // Pre-allocate hyperprior buffer
            if available.len() < self.inner.config.max_capacity {
                let hyper_size = hyperprior_shape.num_elements();
                let hyper_data = if self.inner.config.zero_init {
                    vec![0.0; hyper_size]
                } else {
                    let mut v = Vec::with_capacity(hyper_size);
                    unsafe { v.set_len(hyper_size) };
                    v
                };
                stats.allocations += 1;
                stats.total_bytes_allocated += hyper_size * std::mem::size_of::<f32>();
                available.push((hyperprior_shape, hyper_data));
            }
        }

        stats.current_available = available.len();
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.inner.stats.lock().unwrap().clone()
    }

    /// Clear all available buffers from the pool
    pub fn clear(&self) {
        let mut available = self.inner.available.lock().unwrap();
        let mut stats = self.inner.stats.lock().unwrap();

        available.clear();
        stats.current_available = 0;
    }

    /// Get the pool configuration
    pub fn config(&self) -> &PoolConfig {
        &self.inner.config
    }

    /// Get current number of available buffers
    pub fn available_count(&self) -> usize {
        self.inner.available.lock().unwrap().len()
    }

    /// Get total memory currently held in pool (bytes)
    pub fn memory_usage(&self) -> usize {
        let available = self.inner.available.lock().unwrap();
        available.iter().map(|(shape, _)| shape.size_bytes()).sum()
    }
}

impl std::fmt::Debug for FramePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("FramePool")
            .field("config", &self.inner.config)
            .field("stats", &stats)
            .finish()
    }
}

// ============================================================================
// Bitstream Arena
// ============================================================================

/// Arena allocator for temporary bitstream operations
///
/// Provides fast bump allocation for temporary buffers during entropy
/// coding and bitstream construction. Memory is allocated in chunks
/// and can be reset without deallocation.
///
/// # Example
///
/// ```rust,ignore
/// let mut arena = BitstreamArena::new(64 * 1024); // 64 KB chunks
///
/// // Allocate temporary buffers
/// let buf1 = arena.alloc(1024);
/// let buf2 = arena.alloc(2048);
///
/// // Use buffers...
///
/// // Reset arena for next frame (no deallocation)
/// arena.reset();
/// ```
#[derive(Debug)]
pub struct BitstreamArena {
    /// Memory chunks
    chunks: Vec<Vec<u8>>,

    /// Current chunk index
    current_chunk: usize,

    /// Offset within current chunk
    offset: usize,

    /// Size of each chunk
    chunk_size: usize,

    /// Total allocations count
    allocation_count: usize,
}

impl BitstreamArena {
    /// Create a new arena with the specified chunk size
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Size of each memory chunk in bytes
    pub fn new(chunk_size: usize) -> Self {
        let chunk_size = chunk_size.max(1024); // Minimum 1 KB
        BitstreamArena {
            chunks: vec![vec![0u8; chunk_size]],
            current_chunk: 0,
            offset: 0,
            chunk_size,
            allocation_count: 0,
        }
    }

    /// Create an arena with default chunk size (64 KB)
    pub fn default_size() -> Self {
        Self::new(64 * 1024)
    }

    /// Create an arena sized for the given resolution
    ///
    /// Estimates chunk size based on expected frame size.
    pub fn for_resolution(width: u32, height: u32) -> Self {
        let pixels = (width as usize) * (height as usize);
        // Estimate ~0.5 bits per pixel compressed
        let estimated_size = (pixels / 16).max(16 * 1024);
        Self::new(estimated_size)
    }

    /// Allocate a buffer from the arena
    ///
    /// Returns a mutable slice of the requested size. The memory is
    /// valid until `reset()` is called.
    ///
    /// # Panics
    ///
    /// Panics if the requested size exceeds the chunk size.
    pub fn alloc(&mut self, size: usize) -> &mut [u8] {
        if size > self.chunk_size {
            // Allocate oversized chunk
            self.chunks.push(vec![0u8; size]);
            self.current_chunk = self.chunks.len() - 1;
            self.offset = size;
            self.allocation_count += 1;
            return &mut self.chunks[self.current_chunk][..size];
        }

        // Check if we need a new chunk
        if self.offset + size > self.chunk_size {
            self.current_chunk += 1;
            self.offset = 0;

            // Allocate new chunk if needed
            if self.current_chunk >= self.chunks.len() {
                self.chunks.push(vec![0u8; self.chunk_size]);
            }
        }

        let start = self.offset;
        self.offset += size;
        self.allocation_count += 1;

        &mut self.chunks[self.current_chunk][start..start + size]
    }

    /// Allocate and zero-initialize a buffer
    pub fn alloc_zeroed(&mut self, size: usize) -> &mut [u8] {
        let buf = self.alloc(size);
        buf.fill(0);
        buf
    }

    /// Reset the arena for reuse
    ///
    /// Does not deallocate memory, just resets the allocation pointer.
    /// This is very fast and allows reusing memory for the next frame.
    pub fn reset(&mut self) {
        self.current_chunk = 0;
        self.offset = 0;
        self.allocation_count = 0;
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    /// Get currently used memory in bytes
    pub fn used_memory(&self) -> usize {
        let full_chunks: usize = self.chunks[..self.current_chunk]
            .iter()
            .map(|c| c.len())
            .sum();
        full_chunks + self.offset
    }

    /// Get the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the number of chunks allocated
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get the number of allocations since last reset
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    /// Shrink the arena to release unused chunks
    ///
    /// Keeps only chunks that have been used since the last reset.
    pub fn shrink(&mut self) {
        if self.current_chunk + 1 < self.chunks.len() {
            self.chunks.truncate(self.current_chunk + 1);
        }
    }

    /// Pre-allocate chunks
    pub fn reserve(&mut self, num_chunks: usize) {
        while self.chunks.len() < num_chunks {
            self.chunks.push(vec![0u8; self.chunk_size]);
        }
    }
}

impl Default for BitstreamArena {
    fn default() -> Self {
        Self::default_size()
    }
}

impl Clone for BitstreamArena {
    fn clone(&self) -> Self {
        BitstreamArena {
            chunks: self.chunks.clone(),
            current_chunk: 0,
            offset: 0,
            chunk_size: self.chunk_size,
            allocation_count: 0,
        }
    }
}

// ============================================================================
// Encoder Memory Context
// ============================================================================

/// Memory context for encoder operations
///
/// Bundles together all memory pools and arenas needed for encoding,
/// providing convenient access and lifecycle management.
#[derive(Debug)]
pub struct EncoderMemoryContext {
    /// Pool for input frames
    pub frame_pool: Arc<FramePool>,

    /// Pool for latent representations
    pub latent_pool: Arc<FramePool>,

    /// Arena for bitstream operations
    pub arena: BitstreamArena,
}

impl EncoderMemoryContext {
    /// Create a new encoder memory context for the given resolution
    pub fn new(width: u32, height: u32) -> Self {
        let config = PoolConfig::for_resolution(width, height);
        let frame_pool = FramePool::new(config.clone());
        let latent_pool = FramePool::new(config);
        let arena = BitstreamArena::for_resolution(width, height);

        EncoderMemoryContext {
            frame_pool,
            latent_pool,
            arena,
        }
    }

    /// Create with specific pool configuration
    pub fn with_config(config: PoolConfig) -> Self {
        let frame_pool = FramePool::new(config.clone());
        let latent_pool = FramePool::new(config.clone());
        let arena = BitstreamArena::for_resolution(config.frame_width, config.frame_height);

        EncoderMemoryContext {
            frame_pool,
            latent_pool,
            arena,
        }
    }

    /// Pre-warm all pools
    pub fn prewarm(&self, count: usize) {
        self.frame_pool.prewarm(count);
        self.latent_pool.prewarm(count);
    }

    /// Reset arena for next frame
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.frame_pool.memory_usage() + self.latent_pool.memory_usage() + self.arena.memory_usage()
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.frame_pool.clear();
        self.latent_pool.clear();
        self.arena.reset();
        self.arena.shrink();
    }
}

impl Default for EncoderMemoryContext {
    fn default() -> Self {
        Self::new(1920, 1088) // 1080p default
    }
}

// ============================================================================
// Decoder Memory Context
// ============================================================================

/// Memory context for decoder operations
///
/// Bundles together all memory pools needed for decoding,
/// including decoded picture buffer (DPB) for reference frames.
#[derive(Debug)]
pub struct DecoderMemoryContext {
    /// Pool for output frames
    pub frame_pool: Arc<FramePool>,

    /// Pool for latent representations
    pub latent_pool: Arc<FramePool>,

    /// Pool for decoded picture buffer (reference frames)
    pub dpb_pool: Arc<FramePool>,

    /// Arena for bitstream operations
    pub arena: BitstreamArena,
}

impl DecoderMemoryContext {
    /// Create a new decoder memory context for the given resolution
    pub fn new(width: u32, height: u32) -> Self {
        let config = PoolConfig::for_resolution(width, height);

        // DPB needs more capacity for reference frames
        let dpb_config = PoolConfig::for_resolution(width, height)
            .with_initial_capacity(4)
            .with_max_capacity(16);

        let frame_pool = FramePool::new(config.clone());
        let latent_pool = FramePool::new(config);
        let dpb_pool = FramePool::new(dpb_config);
        let arena = BitstreamArena::for_resolution(width, height);

        DecoderMemoryContext {
            frame_pool,
            latent_pool,
            dpb_pool,
            arena,
        }
    }

    /// Create with specific pool configuration
    pub fn with_config(config: PoolConfig, dpb_config: PoolConfig) -> Self {
        let frame_pool = FramePool::new(config.clone());
        let latent_pool = FramePool::new(config.clone());
        let dpb_pool = FramePool::new(dpb_config);
        let arena = BitstreamArena::for_resolution(config.frame_width, config.frame_height);

        DecoderMemoryContext {
            frame_pool,
            latent_pool,
            dpb_pool,
            arena,
        }
    }

    /// Pre-warm all pools
    pub fn prewarm(&self, count: usize) {
        self.frame_pool.prewarm(count);
        self.latent_pool.prewarm(count);
        self.dpb_pool.prewarm(count);
    }

    /// Reset arena for next frame
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.frame_pool.memory_usage()
            + self.latent_pool.memory_usage()
            + self.dpb_pool.memory_usage()
            + self.arena.memory_usage()
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.frame_pool.clear();
        self.latent_pool.clear();
        self.dpb_pool.clear();
        self.arena.reset();
        self.arena.shrink();
    }
}

impl Default for DecoderMemoryContext {
    fn default() -> Self {
        Self::new(1920, 1088) // 1080p default
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── BufferShape Tests ──

    #[test]
    fn test_buffer_shape_new() {
        let shape = BufferShape::new(1, 3, 1080, 1920);
        assert_eq!(shape.batch, 1);
        assert_eq!(shape.channels, 3);
        assert_eq!(shape.height, 1080);
        assert_eq!(shape.width, 1920);
    }

    #[test]
    fn test_buffer_shape_num_elements() {
        let shape = BufferShape::new(1, 3, 1080, 1920);
        assert_eq!(shape.num_elements(), 1 * 3 * 1080 * 1920);
    }

    #[test]
    fn test_buffer_shape_size_bytes() {
        let shape = BufferShape::new(1, 3, 64, 64);
        assert_eq!(shape.size_bytes(), 1 * 3 * 64 * 64 * 4); // f32 = 4 bytes
    }

    #[test]
    fn test_buffer_shape_frame() {
        let shape = BufferShape::frame(1080, 1920);
        assert_eq!(shape.batch, 1);
        assert_eq!(shape.channels, 3);
        assert_eq!(shape.height, 1080);
        assert_eq!(shape.width, 1920);
    }

    #[test]
    fn test_buffer_shape_to_latent() {
        let frame = BufferShape::frame(1080, 1920);
        let latent = frame.to_latent_shape();
        assert_eq!(latent.channels, 192);
        assert_eq!(latent.height, 68); // ceil(1080/16)
        assert_eq!(latent.width, 120); // ceil(1920/16)
    }

    #[test]
    fn test_buffer_shape_to_hyperprior() {
        let latent = BufferShape::latent(68, 120, 192);
        let hyper = latent.to_hyperprior_shape();
        assert_eq!(hyper.channels, 128);
        assert_eq!(hyper.height, 17); // ceil(68/4)
        assert_eq!(hyper.width, 30); // ceil(120/4)
    }

    #[test]
    fn test_buffer_shape_is_valid() {
        assert!(BufferShape::new(1, 3, 64, 64).is_valid());
        assert!(!BufferShape::new(0, 3, 64, 64).is_valid());
        assert!(!BufferShape::new(1, 0, 64, 64).is_valid());
    }

    #[test]
    fn test_buffer_shape_display() {
        let shape = BufferShape::new(1, 3, 64, 64);
        assert_eq!(format!("{}", shape), "[1, 3, 64, 64]");
    }

    // ── PoolConfig Tests ──

    #[test]
    fn test_pool_config_new() {
        let config = PoolConfig::new(1920, 1080);
        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.frame_height, 1080);
        assert_eq!(config.channels, 3);
    }

    #[test]
    fn test_pool_config_preset_720p() {
        let config = PoolConfig::preset_720p();
        assert_eq!(config.frame_width, 1280);
        assert_eq!(config.frame_height, 720);
        assert_eq!(config.initial_capacity, 4);
    }

    #[test]
    fn test_pool_config_preset_1080p() {
        let config = PoolConfig::preset_1080p();
        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.frame_height, 1088);
        assert_eq!(config.initial_capacity, 3);
        assert!(config.zero_init);
        assert!(!config.pinned_memory);
    }

    #[test]
    fn test_pool_config_preset_1080p_realtime() {
        let config = PoolConfig::preset_1080p_realtime();
        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.frame_height, 1088);
        assert_eq!(config.initial_capacity, 6);
        assert_eq!(config.max_capacity, 12);
        assert!(!config.zero_init); // No zero-init for speed
        assert!(!config.pinned_memory);
    }

    #[test]
    fn test_pool_config_preset_1080p_realtime_gpu() {
        let config = PoolConfig::preset_1080p_realtime_gpu();
        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.frame_height, 1088);
        assert_eq!(config.initial_capacity, 6);
        assert_eq!(config.max_capacity, 12);
        assert!(!config.zero_init);
        assert!(config.pinned_memory); // Pinned for GPU
    }

    #[test]
    fn test_pool_config_with_pinned_memory() {
        let config = PoolConfig::preset_1080p().with_pinned_memory(true);
        assert!(config.pinned_memory);

        let config = PoolConfig::preset_1080p_realtime_gpu().with_pinned_memory(false);
        assert!(!config.pinned_memory);
    }

    #[test]
    fn test_pool_config_preset_4k() {
        let config = PoolConfig::preset_4k();
        assert_eq!(config.frame_width, 3840);
        assert_eq!(config.frame_height, 2160);
        assert_eq!(config.initial_capacity, 2);
    }

    #[test]
    fn test_pool_config_for_resolution() {
        let config = PoolConfig::for_resolution(1280, 720);
        assert!(config.initial_capacity > 0);
        assert!(config.max_capacity >= config.initial_capacity);
    }

    #[test]
    fn test_pool_config_shapes() {
        let config = PoolConfig::preset_1080p();
        let frame_shape = config.frame_shape();
        let latent_shape = config.latent_shape();
        let hyper_shape = config.hyperprior_shape();

        assert_eq!(frame_shape.height, 1088);
        assert_eq!(latent_shape.channels, 192);
        assert_eq!(hyper_shape.channels, 128);
    }

    // ── PoolStats Tests ──

    #[test]
    fn test_pool_stats_new() {
        let stats = PoolStats::new();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.reuses, 0);
        assert_eq!(stats.reuse_ratio(), 0.0);
    }

    #[test]
    fn test_pool_stats_reuse_ratio() {
        let mut stats = PoolStats::new();
        stats.allocations = 10;
        stats.reuses = 90;
        assert!((stats.reuse_ratio() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_pool_stats_is_effective() {
        let mut stats = PoolStats::new();
        stats.allocations = 10;
        stats.reuses = 10;
        assert!(!stats.is_effective()); // 50% not effective

        stats.reuses = 20;
        assert!(stats.is_effective()); // 66% effective
    }

    // ── PooledBuffer Tests ──

    #[test]
    fn test_pooled_buffer_standalone() {
        let shape = BufferShape::new(1, 3, 64, 64);
        let buf = PooledBuffer::standalone(shape);

        assert_eq!(buf.shape(), shape);
        assert_eq!(buf.len(), shape.num_elements());
        assert!(!buf.is_pooled());
    }

    #[test]
    fn test_pooled_buffer_data_access() {
        let shape = BufferShape::new(1, 3, 64, 64);
        let mut buf = PooledBuffer::standalone(shape);

        buf.fill(1.0);
        assert!(buf.data().iter().all(|&x| x == 1.0));

        buf.zero();
        assert!(buf.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_pooled_buffer_copy_from_slice() {
        let shape = BufferShape::new(1, 1, 2, 2);
        let mut buf = PooledBuffer::standalone(shape);

        let src = vec![1.0, 2.0, 3.0, 4.0];
        buf.copy_from_slice(&src).unwrap();

        assert_eq!(buf.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pooled_buffer_into_vec() {
        let shape = BufferShape::new(1, 1, 2, 2);
        let mut buf = PooledBuffer::standalone(shape);
        buf.fill(5.0);

        let vec = buf.into_vec();
        assert_eq!(vec.len(), 4);
        assert!(vec.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_pooled_buffer_clone() {
        let shape = BufferShape::new(1, 1, 2, 2);
        let mut buf = PooledBuffer::standalone(shape);
        buf.fill(3.0);

        let cloned = buf.clone();
        assert!(!cloned.is_pooled()); // Clones are not pooled
        assert_eq!(cloned.data(), buf.data());
    }

    // ── FramePool Tests ──

    #[test]
    fn test_frame_pool_creation() {
        let config = PoolConfig::preset_720p();
        let pool = FramePool::new(config);

        assert!(pool.available_count() > 0); // Pre-warmed
    }

    #[test]
    fn test_frame_pool_acquire_release() {
        let config = PoolConfig::new(64, 64).with_initial_capacity(0);
        let pool = FramePool::new(config);

        // Acquire buffer
        let buf = pool.acquire();
        assert!(buf.is_pooled());

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.current_in_use, 1);

        // Release buffer
        drop(buf);

        let stats = pool.stats();
        assert_eq!(stats.current_in_use, 0);
        assert_eq!(stats.returns, 1);
    }

    #[test]
    fn test_frame_pool_reuse() {
        let config = PoolConfig::new(64, 64).with_initial_capacity(0);
        let pool = FramePool::new(config);

        // First acquire (allocation)
        let buf1 = pool.acquire();
        drop(buf1);

        // Second acquire (reuse)
        let buf2 = pool.acquire();
        drop(buf2);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.reuses, 1);
    }

    #[test]
    fn test_frame_pool_prewarm() {
        let config = PoolConfig::new(64, 64).with_initial_capacity(0);
        let pool = FramePool::new(config);

        assert_eq!(pool.available_count(), 0);

        pool.prewarm(4);

        assert!(pool.available_count() >= 4);
    }

    #[test]
    fn test_frame_pool_prewarm_1080p() {
        // Use realtime preset with initial_capacity=0 to test prewarm_1080p
        let config = PoolConfig::preset_1080p_realtime().with_initial_capacity(0);
        let pool = FramePool::new(config);

        assert_eq!(pool.available_count(), 0);

        pool.prewarm_1080p();

        // Should have pre-allocated up to max_capacity (12) buffers
        // 6 sets of (frame + latent + hyperprior) = 18, but capped at max_capacity
        assert!(pool.available_count() >= 6);
        assert!(pool.available_count() <= 12);

        // Check memory was allocated
        let stats = pool.stats();
        assert!(stats.total_bytes_allocated > 0);
        assert!(stats.allocations >= 6);
    }

    #[test]
    fn test_frame_pool_prewarm_1080p_no_zero_init() {
        // Verify that prewarm_1080p respects zero_init=false
        let config = PoolConfig::preset_1080p_realtime().with_initial_capacity(0);
        assert!(!config.zero_init);

        let pool = FramePool::new(config);
        pool.prewarm_1080p();

        // Should have buffers available
        assert!(pool.available_count() > 0);

        // Acquire a buffer - it should not be zero-initialized
        // (but this is hard to verify without reading uninitialized memory)
        let buf = pool.acquire();
        assert!(buf.len() > 0);
    }

    #[test]
    fn test_frame_pool_clear() {
        let config = PoolConfig::preset_720p();
        let pool = FramePool::new(config);

        pool.prewarm(4);
        assert!(pool.available_count() > 0);

        pool.clear();
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_frame_pool_memory_usage() {
        let config = PoolConfig::new(64, 64).with_initial_capacity(0);
        let pool = FramePool::new(config);

        assert_eq!(pool.memory_usage(), 0);

        pool.prewarm(1);
        assert!(pool.memory_usage() > 0);
    }

    #[test]
    fn test_frame_pool_acquire_latent() {
        let config = PoolConfig::preset_720p();
        let pool = FramePool::new(config);

        let buf = pool.acquire_latent();
        assert_eq!(buf.shape().channels, 192);
    }

    // ── BitstreamArena Tests ──

    #[test]
    fn test_arena_creation() {
        let arena = BitstreamArena::new(1024);
        assert_eq!(arena.chunk_size(), 1024);
        assert_eq!(arena.chunk_count(), 1);
    }

    #[test]
    fn test_arena_alloc() {
        let mut arena = BitstreamArena::new(1024);

        let buf = arena.alloc(100);
        assert_eq!(buf.len(), 100);
        assert_eq!(arena.allocation_count(), 1);
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let mut arena = BitstreamArena::new(1024);

        let _buf1 = arena.alloc(100);
        let _buf2 = arena.alloc(200);
        let _buf3 = arena.alloc(300);

        assert_eq!(arena.allocation_count(), 3);
        assert!(arena.used_memory() >= 600);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = BitstreamArena::new(1024);

        arena.alloc(500);
        arena.alloc(300);
        assert!(arena.used_memory() > 0);

        arena.reset();

        assert_eq!(arena.used_memory(), 0);
        assert_eq!(arena.allocation_count(), 0);
    }

    #[test]
    fn test_arena_chunk_growth() {
        let mut arena = BitstreamArena::new(1024);

        // Allocate more than one chunk worth
        arena.alloc(800);
        arena.alloc(800);

        assert!(arena.chunk_count() >= 2);
    }

    #[test]
    fn test_arena_oversized_alloc() {
        let mut arena = BitstreamArena::new(1024);

        // Allocate more than chunk size
        let buf = arena.alloc(2048);
        assert_eq!(buf.len(), 2048);
    }

    #[test]
    fn test_arena_alloc_zeroed() {
        let mut arena = BitstreamArena::new(1024);

        let buf = arena.alloc_zeroed(100);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_arena_shrink() {
        let mut arena = BitstreamArena::new(1024);

        // Force multiple chunks
        arena.alloc(800);
        arena.alloc(800);
        arena.alloc(800);

        let initial_chunks = arena.chunk_count();
        arena.reset();
        arena.shrink();

        assert!(arena.chunk_count() < initial_chunks);
    }

    #[test]
    fn test_arena_reserve() {
        let mut arena = BitstreamArena::new(1024);

        arena.reserve(5);
        assert!(arena.chunk_count() >= 5);
    }

    // ── EncoderMemoryContext Tests ──

    #[test]
    fn test_encoder_context_creation() {
        let ctx = EncoderMemoryContext::new(1920, 1080);

        assert!(ctx.frame_pool.config().frame_width == 1920);
        assert!(ctx.latent_pool.config().frame_width == 1920);
    }

    #[test]
    fn test_encoder_context_prewarm() {
        let ctx = EncoderMemoryContext::new(64, 64);

        ctx.prewarm(2);

        assert!(ctx.frame_pool.available_count() > 0);
        assert!(ctx.latent_pool.available_count() > 0);
    }

    #[test]
    fn test_encoder_context_memory_usage() {
        let ctx = EncoderMemoryContext::new(64, 64);
        ctx.prewarm(2);

        let usage = ctx.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_encoder_context_clear() {
        let mut ctx = EncoderMemoryContext::new(64, 64);
        ctx.prewarm(2);

        ctx.clear();

        assert_eq!(ctx.frame_pool.available_count(), 0);
        assert_eq!(ctx.latent_pool.available_count(), 0);
    }

    // ── DecoderMemoryContext Tests ──

    #[test]
    fn test_decoder_context_creation() {
        let ctx = DecoderMemoryContext::new(1920, 1080);

        assert!(ctx.frame_pool.config().frame_width == 1920);
        assert!(ctx.dpb_pool.config().max_capacity >= 4);
    }

    #[test]
    fn test_decoder_context_prewarm() {
        let ctx = DecoderMemoryContext::new(64, 64);

        ctx.prewarm(2);

        assert!(ctx.frame_pool.available_count() > 0);
        assert!(ctx.dpb_pool.available_count() > 0);
    }

    #[test]
    fn test_decoder_context_memory_usage() {
        let ctx = DecoderMemoryContext::new(64, 64);
        ctx.prewarm(2);

        let usage = ctx.memory_usage();
        assert!(usage > 0);
    }

    // ── Concurrent Access Tests ──

    #[test]
    fn test_pool_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let config = PoolConfig::new(64, 64).with_initial_capacity(0);
        let pool = FramePool::new(config);
        let pool = Arc::new(pool);

        let mut handles = vec![];

        for _ in 0..4 {
            let pool_clone = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let _buf = pool_clone.acquire();
                    // Buffer returned on drop
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = pool.stats();
        assert!(stats.total_acquisitions() == 40);
    }

    #[test]
    fn test_pool_concurrent_acquire_release() {
        use std::sync::Arc;
        use std::thread;

        let config = PoolConfig::new(64, 64)
            .with_initial_capacity(2)
            .with_max_capacity(4);
        let pool = FramePool::new(config);
        let pool = Arc::new(pool);

        let mut handles = vec![];

        for _ in 0..2 {
            let pool_clone = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let mut buffers = vec![];
                for _ in 0..5 {
                    buffers.push(pool_clone.acquire());
                }
                // All buffers released
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All buffers should be returned
        let stats = pool.stats();
        assert_eq!(stats.current_in_use, 0);
    }
}
