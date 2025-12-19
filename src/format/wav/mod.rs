//! WAV audio format support
//!
//! This module implements RIFF/WAV file format parsing and writing.
//! WAV is a simple uncompressed audio format widely used for audio interchange.

pub mod demuxer;
pub mod header;
pub mod muxer;

pub use demuxer::WavDemuxer;
pub use header::{FormatTag, WavFormat, WavHeader};
pub use muxer::WavMuxer;

/// WAV format magic numbers
pub const RIFF_MAGIC: &[u8; 4] = b"RIFF";
pub const WAVE_MAGIC: &[u8; 4] = b"WAVE";
pub const FMT_CHUNK: &[u8; 4] = b"fmt ";
pub const DATA_CHUNK: &[u8; 4] = b"data";
pub const LIST_CHUNK: &[u8; 4] = b"LIST";
pub const INFO_CHUNK: &[u8; 4] = b"INFO";

/// Chunk header (4 byte ID + 4 byte size)
#[derive(Debug, Clone, Copy)]
pub struct ChunkHeader {
    pub id: [u8; 4],
    pub size: u32,
}

impl ChunkHeader {
    /// Read a chunk header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }

        let mut id = [0u8; 4];
        id.copy_from_slice(&bytes[0..4]);

        let size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        Some(ChunkHeader { id, size })
    }

    /// Convert chunk header to bytes
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&self.id);
        bytes[4..8].copy_from_slice(&self.size.to_le_bytes());
        bytes
    }
}
