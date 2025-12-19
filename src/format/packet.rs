//! Packet representation for compressed media data

use crate::util::{Buffer, MediaType, Timestamp};
use std::fmt;

/// Packet flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketFlags {
    /// Packet contains a keyframe
    pub keyframe: bool,
    /// Packet is corrupted
    pub corrupt: bool,
    /// Packet is a header/config packet
    pub config: bool,
}

impl Default for PacketFlags {
    fn default() -> Self {
        PacketFlags {
            keyframe: false,
            corrupt: false,
            config: false,
        }
    }
}

/// A packet of compressed media data
#[derive(Debug, Clone)]
pub struct Packet {
    /// Stream index this packet belongs to
    pub stream_index: usize,

    /// Type of media (video, audio, etc.)
    pub codec_type: MediaType,

    /// Compressed data
    pub data: Buffer,

    /// Presentation timestamp
    pub pts: Timestamp,

    /// Decoding timestamp
    pub dts: Timestamp,

    /// Duration of this packet
    pub duration: i64,

    /// Packet flags
    pub flags: PacketFlags,

    /// Byte position in stream (-1 if unknown)
    pub position: i64,
}

impl Packet {
    /// Create a new packet
    pub fn new(stream_index: usize, data: Buffer) -> Self {
        Packet {
            stream_index,
            codec_type: MediaType::Unknown,
            data,
            pts: Timestamp::none(),
            dts: Timestamp::none(),
            duration: 0,
            flags: PacketFlags::default(),
            position: -1,
        }
    }

    /// Create a new video packet
    pub fn new_video(stream_index: usize, data: Buffer) -> Self {
        Packet {
            stream_index,
            codec_type: MediaType::Video,
            data,
            pts: Timestamp::none(),
            dts: Timestamp::none(),
            duration: 0,
            flags: PacketFlags::default(),
            position: -1,
        }
    }

    /// Create a new audio packet
    pub fn new_audio(stream_index: usize, data: Buffer) -> Self {
        Packet {
            stream_index,
            codec_type: MediaType::Audio,
            data,
            pts: Timestamp::none(),
            dts: Timestamp::none(),
            duration: 0,
            flags: PacketFlags::default(),
            position: -1,
        }
    }

    /// Check if this packet is a keyframe
    pub fn is_keyframe(&self) -> bool {
        self.flags.keyframe
    }

    /// Set keyframe flag
    pub fn set_keyframe(&mut self, keyframe: bool) {
        self.flags.keyframe = keyframe;
    }

    /// Get the size of the packet data
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Display for Packet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Packet(stream={}, size={}, pts={}, dts={}, key={})",
            self.stream_index,
            self.size(),
            self.pts,
            self.dts,
            self.is_keyframe()
        )
    }
}
