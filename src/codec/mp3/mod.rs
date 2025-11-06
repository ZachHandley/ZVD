//! MP3 audio codec implementation using Symphonia
//!
//! MP3 (MPEG-1 Audio Layer III) patents have expired worldwide.
//! It's now freely usable without licensing fees.

pub mod decoder;

pub use decoder::Mp3Decoder;
