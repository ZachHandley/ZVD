//! FLAC audio codec implementation using Symphonia
//!
//! FLAC (Free Lossless Audio Codec) is an open-source lossless audio codec.
//! It's patent-free and royalty-free with no licensing restrictions.
//!
//! ## Features
//!
//! - **Lossless Compression**: Typically 40-60% compression for music
//! - **Sample Rates**: Up to 655,350 Hz
//! - **Channels**: Up to 8 channels
//! - **Bit Depths**: 4 to 32 bits per sample
//! - **Fast Decoding**: Efficient decompression algorithm
//! - **Metadata**: Supports extensive tagging via Vorbis comments
//!
//! ## Encoding Audio to FLAC
//!
//! FLAC encoding is available through `FlacEncoder`:
//!
//! ```no_run
//! use zvd_lib::codec::flac::FlacEncoder;
//! use zvd_lib::codec::{Encoder, AudioFrame, Frame};
//! use zvd_lib::util::SampleFormat;
//!
//! // Create encoder for 44.1kHz stereo
//! let mut encoder = FlacEncoder::new(44100, 2)?;
//!
//! // Set compression level (0-8, default 5)
//! encoder.set_compression_level(8)?; // Maximum compression
//!
//! // Encode audio frames
//! let mut frame = AudioFrame::new(4096, 2, SampleFormat::I16);
//! // ... fill with PCM data ...
//! encoder.send_frame(&Frame::Audio(frame))?;
//! encoder.flush()?;
//!
//! // Get encoded packets
//! while let Ok(packet) = encoder.receive_packet() {
//!     // packet.data contains FLAC-compressed audio
//! }
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## Decoding FLAC Files
//!
//! **FLAC decoding in ZVD is best done through the `SymphoniaAdapter`** at the
//! container/format level, not through packet-level codec decoding. This is because:
//!
//! 1. Symphonia's architecture tightly couples format reading and codec decoding
//! 2. FLAC files contain headers and metadata that must be parsed at container level
//! 3. The SymphoniaAdapter provides complete FLAC decoding with metadata support
//!
//! The `FlacDecoder` struct provides the interface for consistency, but will
//! return unsupported errors for packet-level decoding.
//!
//! ```no_run
//! use zvd_lib::format::{Demuxer, symphonia_adapter::SymphoniaDemuxer};
//! use std::path::Path;
//!
//! // Recommended: Use SymphoniaAdapter for FLAC files
//! let mut demuxer = SymphoniaDemuxer::new("flac");
//! demuxer.open(Path::new("audio.flac"))?;
//!
//! // Read decoded PCM data directly
//! while let Ok(packet) = demuxer.read_packet() {
//!     // packet.data contains decoded PCM samples
//! }
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## License
//!
//! FLAC reference implementation: BSD-3-Clause
//! Symphonia FLAC decoder: MPL-2.0
//!
//! Both are royalty-free with no patent restrictions.

pub mod decoder;
pub mod encoder;

pub use decoder::{FlacDecoder, FlacDecoderConfig};
pub use encoder::{FlacEncoder, FlacEncoderConfig};
