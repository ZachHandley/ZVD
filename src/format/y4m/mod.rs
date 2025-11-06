//! Y4M (YUV4MPEG2) format support
//!
//! This module provides support for the Y4M raw video format,
//! which is commonly used for uncompressed YUV video testing
//! and as input/output for video encoders.

pub mod demuxer;
pub mod muxer;

pub use demuxer::Y4mDemuxer;
pub use muxer::Y4mMuxer;
