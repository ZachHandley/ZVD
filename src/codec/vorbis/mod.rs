//! Vorbis audio codec implementation using Symphonia
//!
//! Vorbis is a free, open-source audio codec managed by the Xiph.Org Foundation.
//! It's patent-free and royalty-free.

pub mod decoder;

pub use decoder::VorbisDecoder;
