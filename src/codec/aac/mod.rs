//! AAC (Advanced Audio Coding) codec implementation
//!
//! AAC is a lossy audio codec designed as the successor to MP3.
//! This module provides AAC decoding using Symphonia's AAC codec support.
//!
//! ## Features
//! - **Sample rates**: 8 to 96 kHz
//! - **Channels**: Up to 48 channels (typically 1-7.1)
//! - **Profiles**: LC-AAC (Low Complexity) only
//! - **Containers**: M4A, MP4, ADTS
//!
//! ## Limitations
//! - **LC-AAC only**: HE-AAC and HE-AACv2 not supported
//! - Symphonia's AAC decoder supports only the LC (Low Complexity) profile
//!
//! ## Usage
//!
//! AAC decoding in ZVD is done through the `SymphoniaAdapter`:
//!
//! ```no_run
//! use zvd_lib::format::{Demuxer, symphonia_adapter::SymphoniaDemuxer};
//! use std::path::Path;
//!
//! let mut demuxer = SymphoniaDemuxer::new("m4a");
//! demuxer.open(Path::new("audio.m4a"))?;
//!
//! while let Ok(packet) = demuxer.read_packet() {
//!     // Process decoded PCM audio (LC-AAC only)
//! }
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## Patent Notice
//! **AAC is patent-encumbered.** Commercial use may require patent licensing from:
//! - Via Licensing Corporation
//! - Other AAC patent holders
//!
//! See `CODEC_LICENSES.md` for detailed patent and licensing information.
//!
//! Enable AAC support with the `aac` feature flag.
//!
//! ## License
//! - Symphonia AAC decoder: MPL-2.0
//! - **Patent licensing may be required** for commercial use

pub mod decoder;

#[cfg(feature = "aac")]
pub use decoder::{AacDecoder, AacDecoderConfig};
