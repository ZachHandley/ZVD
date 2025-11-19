//! Vorbis audio codec implementation
//!
//! Vorbis is a free, open-source lossy audio codec managed by the Xiph.Org Foundation.
//! It's patent-free and royalty-free with no licensing restrictions.
//!
//! ## Important: Use Opus for New Projects
//!
//! **For new projects, use Opus instead of Vorbis.** Opus is superior in every way:
//! - Better quality at all bitrates
//! - Lower latency (5-60ms vs 50-200ms)
//! - More flexible frame sizes
//! - Better packet loss resilience
//! - Wider industry adoption
//!
//! Vorbis encoding is provided for:
//! - Compatibility with existing Ogg Vorbis workflows
//! - Legacy system support
//! - Projects requiring Ogg container format
//!
//! ## Features
//! - **Sample rates**: 8 kHz to 192 kHz
//! - **Channels**: Up to 255 channels
//! - **Quality-based encoding**: -1 (low) to 10 (high)
//! - **Variable bitrate**: Adaptive quality
//! - **Container**: Typically used in Ogg container
//!
//! ## Encoding Example
//!
//! ```no_run
//! use zvd_lib::codec::vorbis::VorbisEncoder;
//! use zvd_lib::codec::{Encoder, AudioFrame, Frame};
//! use zvd_lib::util::SampleFormat;
//!
//! // Note: Consider using Opus for better quality!
//! let mut encoder = VorbisEncoder::new(48000, 2)?;
//! encoder.set_quality(6.0)?; // Higher quality (~192 kbps)
//!
//! let mut frame = AudioFrame::new(1024, 2, SampleFormat::F32);
//! // ... fill with PCM data ...
//!
//! encoder.send_frame(&Frame::Audio(frame))?;
//! encoder.flush()?;
//!
//! while let Ok(packet) = encoder.receive_packet() {
//!     // packet.data contains Vorbis-encoded audio
//! }
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## Decoding Example
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
pub mod encoder;

pub use decoder::{VorbisDecoder, VorbisDecoderConfig};
pub use encoder::{VorbisEncoder, VorbisEncoderConfig};
