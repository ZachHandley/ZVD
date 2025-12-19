//! MPEG-TS (MPEG Transport Stream) container format
//!
//! MPEG-TS is a standard format for transmission and storage of audio, video,
//! and data. Widely used in broadcast systems and streaming (HLS).

pub mod demuxer;
pub mod muxer;

pub use demuxer::MpegtsDemuxer;
pub use muxer::MpegtsMuxer;

/// MPEG-TS packet size
pub const TS_PACKET_SIZE: usize = 188;

/// MPEG-TS sync byte (0x47)
pub const TS_SYNC_BYTE: u8 = 0x47;

/// MPEG-TS packet header
#[derive(Debug, Clone)]
pub struct TsPacketHeader {
    pub sync_byte: u8,
    pub transport_error: bool,
    pub payload_unit_start: bool,
    pub transport_priority: bool,
    pub pid: u16,
    pub scrambling_control: u8,
    pub adaptation_field_control: u8,
    pub continuity_counter: u8,
}

impl TsPacketHeader {
    pub fn new(pid: u16, payload_start: bool, counter: u8) -> Self {
        TsPacketHeader {
            sync_byte: TS_SYNC_BYTE,
            transport_error: false,
            payload_unit_start: payload_start,
            transport_priority: false,
            pid,
            scrambling_control: 0,
            adaptation_field_control: 0x01, // Payload only
            continuity_counter: counter & 0x0F,
        }
    }

    /// Encode header to 4 bytes
    pub fn to_bytes(&self) -> [u8; 4] {
        let mut bytes = [0u8; 4];

        bytes[0] = self.sync_byte;

        bytes[1] = if self.transport_error { 0x80 } else { 0 }
            | if self.payload_unit_start { 0x40 } else { 0 }
            | if self.transport_priority { 0x20 } else { 0 }
            | ((self.pid >> 8) & 0x1F) as u8;

        bytes[2] = (self.pid & 0xFF) as u8;

        bytes[3] = (self.scrambling_control << 6)
            | (self.adaptation_field_control << 4)
            | self.continuity_counter;

        bytes
    }

    /// Parse header from 4 bytes
    pub fn from_bytes(bytes: &[u8; 4]) -> Result<Self, &'static str> {
        if bytes[0] != TS_SYNC_BYTE {
            return Err("Invalid sync byte");
        }

        Ok(TsPacketHeader {
            sync_byte: bytes[0],
            transport_error: (bytes[1] & 0x80) != 0,
            payload_unit_start: (bytes[1] & 0x40) != 0,
            transport_priority: (bytes[1] & 0x20) != 0,
            pid: (((bytes[1] & 0x1F) as u16) << 8) | (bytes[2] as u16),
            scrambling_control: (bytes[3] >> 6) & 0x03,
            adaptation_field_control: (bytes[3] >> 4) & 0x03,
            continuity_counter: bytes[3] & 0x0F,
        })
    }
}

/// Standard PIDs
pub mod pids {
    pub const PAT: u16 = 0x0000; // Program Association Table
    pub const CAT: u16 = 0x0001; // Conditional Access Table
    pub const TSDT: u16 = 0x0002; // Transport Stream Description Table
    pub const NULL: u16 = 0x1FFF; // Null packets
}

/// Stream types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    Mpeg1Video = 0x01,
    Mpeg2Video = 0x02,
    Mpeg1Audio = 0x03,
    Mpeg2Audio = 0x04,
    PrivateSection = 0x05,
    PrivateData = 0x06,
    AudioAAC = 0x0F,
    VideoMpeg4 = 0x10,
    AudioAACLatm = 0x11,
    VideoH264 = 0x1B,
    VideoH265 = 0x24,
}

impl StreamType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(StreamType::Mpeg1Video),
            0x02 => Some(StreamType::Mpeg2Video),
            0x03 => Some(StreamType::Mpeg1Audio),
            0x04 => Some(StreamType::Mpeg2Audio),
            0x05 => Some(StreamType::PrivateSection),
            0x06 => Some(StreamType::PrivateData),
            0x0F => Some(StreamType::AudioAAC),
            0x10 => Some(StreamType::VideoMpeg4),
            0x11 => Some(StreamType::AudioAACLatm),
            0x1B => Some(StreamType::VideoH264),
            0x24 => Some(StreamType::VideoH265),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_packet_header() {
        let header = TsPacketHeader::new(256, true, 5);
        assert_eq!(header.sync_byte, TS_SYNC_BYTE);
        assert_eq!(header.pid, 256);
        assert!(header.payload_unit_start);
        assert_eq!(header.continuity_counter, 5);
    }

    #[test]
    fn test_ts_header_encoding_decoding() {
        let header = TsPacketHeader::new(256, true, 5);
        let bytes = header.to_bytes();
        let decoded = TsPacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.pid, 256);
        assert!(decoded.payload_unit_start);
        assert_eq!(decoded.continuity_counter, 5);
    }

    #[test]
    fn test_stream_type() {
        assert_eq!(StreamType::from_u8(0x1B), Some(StreamType::VideoH264));
        assert_eq!(StreamType::from_u8(0x0F), Some(StreamType::AudioAAC));
        assert_eq!(StreamType::from_u8(0xFF), None);
    }
}
