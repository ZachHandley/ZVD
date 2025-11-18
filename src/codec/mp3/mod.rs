//! MP3 audio codec implementation using Symphonia
//!
//! MP3 (MPEG-1 Audio Layer III) is a widely-used lossy audio codec.
//! All MP3 patents have expired worldwide as of 2017.
//!
//! ## Features
//! - **Sample rates**: 8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48 kHz
//! - **Bitrates**: 8 to 320 kbps
//! - **Modes**: Stereo, Joint Stereo, Dual Channel, Mono
//! - **CBR/VBR**: Constant and variable bitrate support
//! - **ID3 Tags**: Metadata support (v1, v2)
//!
//! ## Usage
//!
//! MP3 decoding in ZVD is done through the `SymphoniaAdapter`:
//!
//! ```no_run
//! use zvd_lib::format::{Demuxer, symphonia_adapter::SymphoniaDemuxer};
//! use std::path::Path;
//!
//! let mut demuxer = SymphoniaDemuxer::new("mp3");
//! demuxer.open(Path::new("audio.mp3"))?;
//!
//! while let Ok(packet) = demuxer.read_packet() {
//!     // Process decoded PCM audio with ID3 tags parsed
//! }
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## Patent Status
//! MP3 patents expired worldwide:
//! - US patents expired: April 16, 2017
//! - European patents expired: 2012-2015
//!
//! MP3 is now freely usable without licensing fees.
//!
//! ## License
//! - Symphonia MP3 decoder: MPL-2.0
//! - Royalty-free (patents expired)

pub mod decoder;

pub use decoder::{Mp3Decoder, Mp3DecoderConfig};
