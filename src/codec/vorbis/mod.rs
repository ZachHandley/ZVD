//! Vorbis audio codec implementation using Symphonia
//!
//! Vorbis is a free, open-source lossy audio codec managed by the Xiph.Org Foundation.
//! It's patent-free and royalty-free with no licensing restrictions.
//!
//! ## Features
//! - **Sample rates**: 8 kHz to 192 kHz
//! - **Channels**: Up to 255 channels
//! - **Quality-based encoding**: -1 (low) to 10 (high)
//! - **Variable bitrate**: Adaptive quality
//! - **Container**: Typically used in Ogg container
//!
//! ## Usage
//!
//! Vorbis decoding in ZVD is done through the `SymphoniaAdapter` which handles
//! the Ogg container format and Vorbis decoding together:
//!
//! ```no_run
//! use zvd_lib::format::{Demuxer, symphonia_adapter::SymphoniaDemuxer};
//! use std::path::Path;
//!
//! let mut demuxer = SymphoniaDemuxer::new("ogg");
//! demuxer.open(Path::new("audio.ogg"))?;
//!
//! while let Ok(packet) = demuxer.read_packet() {
//!     // Process decoded PCM audio
//! }
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## License
//! - Vorbis specification: Public domain
//! - Reference implementation (libvorbis): BSD-3-Clause
//! - Symphonia Vorbis decoder: MPL-2.0
//!
//! All are royalty-free with no patent restrictions.

pub mod decoder;

pub use decoder::{VorbisDecoder, VorbisDecoderConfig};
