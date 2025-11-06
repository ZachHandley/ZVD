//! FLAC audio codec implementation using Symphonia
//!
//! FLAC (Free Lossless Audio Codec) is an open-source lossless audio codec.
//! It's patent-free and royalty-free.

pub mod decoder;

pub use decoder::FlacDecoder;
