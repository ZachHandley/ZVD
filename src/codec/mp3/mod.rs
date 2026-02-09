//! MP3 audio codec implementation using Symphonia
//!
//! MP3 (MPEG-1 Audio Layer III) patents have expired worldwide.
//! It's now freely usable without licensing fees.
//!
//! ## Features
//!
//! - **MPEG versions**: MPEG-1, MPEG-2, MPEG-2.5
//! - **Bitrates**: 8-320 kbps (CBR and VBR)
//! - **Sample rates**: 8000-48000 Hz
//! - **Channels**: Mono and stereo
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd_lib::codec::{Decoder, Mp3Decoder};
//! use zvd_lib::format::Packet;
//!
//! // Create a decoder for 44.1kHz stereo audio
//! let mut decoder = Mp3Decoder::new(44100, 2)?;
//!
//! // Decode packets
//! decoder.send_packet(&packet)?;
//! let frame = decoder.receive_frame()?;
//! ```
//!
//! ## Factory Functions
//!
//! - `create_decoder(sample_rate, channels)` - Basic decoder creation
//! - `create_decoder_with_extradata(sample_rate, channels, extradata)` - With frame header info

pub mod decoder;

pub use decoder::{
    create_decoder, create_decoder_with_extradata, Mp3Decoder, Mp3FrameHeader, MpegVersion,
};
