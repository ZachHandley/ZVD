//! ZVD - A multimedia processing library written in Rust
//!
//! ZVD is a reimplementation of FFMPEG's core functionality in pure Rust,
//! providing efficient and safe multimedia processing capabilities.
//!
//! # Architecture
//!
//! ZVD is organized into several key modules:
//!
//! - `format`: Container format handling (demuxing/muxing)
//! - `codec`: Audio and video codec implementations
//! - `filter`: Audio and video filtering and processing
//! - `util`: Common utilities and data structures
//! - `swscale`: Video scaling and color space conversion
//! - `swresample`: Audio resampling and format conversion
//! - `hwaccel`: Hardware acceleration support
//! - `streaming`: Streaming protocol support (RTMP, HLS, DASH, etc.)
//! - `subtitle`: Subtitle format support (SRT, WebVTT, ASS/SSA)

pub mod codec;
pub mod error;
pub mod filter;
pub mod format;
pub mod hwaccel;
pub mod streaming;
pub mod subtitle;
pub mod swresample;
pub mod swscale;
pub mod util;

/// Distributed transcoding server mode
///
/// This module provides a coordinator/worker architecture for distributed
/// transcoding, similar to rffmpeg. Enable with the "server" feature.
#[cfg(feature = "server")]
pub mod server;

pub use error::{Error, Result};

/// ZVD version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Configuration for the ZVD library
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum number of threads to use for parallel processing
    pub max_threads: Option<usize>,
    /// Enable verbose logging
    pub verbose: bool,
    /// Enable debug output
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_threads: None,
            verbose: false,
            debug: false,
        }
    }
}

/// Initialize the ZVD library with the given configuration
pub fn init(config: Config) -> Result<()> {
    // Initialize thread pool if max_threads is specified
    if let Some(threads) = config.max_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| Error::Init(format!("Failed to initialize thread pool: {}", e)))?;
    }

    // Initialize logging
    if config.verbose || config.debug {
        let level = if config.debug { "debug" } else { "info" };
        tracing_subscriber::fmt().with_env_filter(level).init();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION_MAJOR, 0);
        assert_eq!(VERSION_MINOR, 1);
        assert_eq!(VERSION_PATCH, 0);
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.max_threads, None);
        assert_eq!(config.verbose, false);
        assert_eq!(config.debug, false);
    }

    #[test]
    fn test_init() {
        let config = Config::default();
        assert!(init(config).is_ok());
    }
}
