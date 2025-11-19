//! WebM/Matroska format support
//!
//! This module provides support for WebM and Matroska container formats,
//! which can contain VP8, VP9, AV1 video and Vorbis, Opus audio.

pub mod demuxer;
pub mod muxer;

pub use demuxer::WebmDemuxer;
pub use muxer::WebmMuxer;
