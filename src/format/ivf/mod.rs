//! IVF container muxing (simple, video-only)
//!
//! IVF is a minimal container commonly used for AV1/VPx elementary streams.
//! We implement just enough to mux AV1 video with monotonically increasing
//! timestamps. Audio is not supported in IVF.

mod muxer;

pub use muxer::IvfMuxer;
