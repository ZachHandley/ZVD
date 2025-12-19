//! FLV (Flash Video) container format
//!
//! FLV is Adobe's container format primarily used for web streaming.
//! Still widely used for RTMP streaming and video distribution.

pub mod demuxer;
pub mod muxer;

pub use demuxer::FlvDemuxer;
pub use muxer::FlvMuxer;

/// FLV file header
#[derive(Debug, Clone)]
pub struct FlvHeader {
    pub signature: [u8; 3], // "FLV"
    pub version: u8,        // Usually 1
    pub has_video: bool,
    pub has_audio: bool,
    pub data_offset: u32, // Offset to first tag (usually 9)
}

impl FlvHeader {
    pub fn new(has_video: bool, has_audio: bool) -> Self {
        FlvHeader {
            signature: *b"FLV",
            version: 1,
            has_video,
            has_audio,
            data_offset: 9,
        }
    }

    /// Encode header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&self.signature);
        bytes.push(self.version);

        let mut flags = 0u8;
        if self.has_video {
            flags |= 0x01;
        }
        if self.has_audio {
            flags |= 0x04;
        }
        bytes.push(flags);

        bytes.extend_from_slice(&self.data_offset.to_be_bytes());
        bytes
    }
}

/// FLV tag type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlvTagType {
    Audio = 8,
    Video = 9,
    ScriptData = 18,
}

impl FlvTagType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            8 => Some(FlvTagType::Audio),
            9 => Some(FlvTagType::Video),
            18 => Some(FlvTagType::ScriptData),
            _ => None,
        }
    }
}

/// FLV tag header
#[derive(Debug, Clone)]
pub struct FlvTagHeader {
    pub tag_type: FlvTagType,
    pub data_size: u32, // Size of tag data (24 bits)
    pub timestamp: u32, // Timestamp in milliseconds (24 bits + 8 bit extension)
    pub stream_id: u32, // Always 0 (24 bits)
}

impl FlvTagHeader {
    pub fn new(tag_type: FlvTagType, data_size: u32, timestamp: u32) -> Self {
        FlvTagHeader {
            tag_type,
            data_size,
            timestamp,
            stream_id: 0,
        }
    }

    /// Encode tag header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(11);

        // Tag type
        bytes.push(self.tag_type as u8);

        // Data size (24 bits)
        bytes.push(((self.data_size >> 16) & 0xFF) as u8);
        bytes.push(((self.data_size >> 8) & 0xFF) as u8);
        bytes.push((self.data_size & 0xFF) as u8);

        // Timestamp (24 bits) + extended (8 bits)
        bytes.push(((self.timestamp >> 16) & 0xFF) as u8);
        bytes.push(((self.timestamp >> 8) & 0xFF) as u8);
        bytes.push((self.timestamp & 0xFF) as u8);
        bytes.push(((self.timestamp >> 24) & 0xFF) as u8);

        // Stream ID (24 bits, always 0)
        bytes.push(0);
        bytes.push(0);
        bytes.push(0);

        bytes
    }
}

/// Video codec IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlvVideoCodec {
    H263 = 2,
    ScreenVideo = 3,
    VP6 = 4,
    VP6Alpha = 5,
    ScreenVideo2 = 6,
    AVC = 7,   // H.264
    HEVC = 12, // H.265
}

/// Audio codec IDs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlvAudioCodec {
    PCM = 0,
    ADPCM = 1,
    MP3 = 2,
    PCMLittleEndian = 3,
    Nellymoser16kHz = 4,
    Nellymoser8kHz = 5,
    Nellymoser = 6,
    G711ALaw = 7,
    G711MuLaw = 8,
    AAC = 10,
    Speex = 11,
    MP3_8kHz = 14,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flv_header() {
        let header = FlvHeader::new(true, true);
        assert_eq!(header.signature, *b"FLV");
        assert_eq!(header.version, 1);
        assert!(header.has_video);
        assert!(header.has_audio);
    }

    #[test]
    fn test_flv_header_encoding() {
        let header = FlvHeader::new(true, true);
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 9);
        assert_eq!(&bytes[0..3], b"FLV");
        assert_eq!(bytes[3], 1); // version
        assert_eq!(bytes[4], 0x05); // flags: video + audio
    }

    #[test]
    fn test_flv_tag_type() {
        assert_eq!(FlvTagType::from_u8(8), Some(FlvTagType::Audio));
        assert_eq!(FlvTagType::from_u8(9), Some(FlvTagType::Video));
        assert_eq!(FlvTagType::from_u8(18), Some(FlvTagType::ScriptData));
        assert_eq!(FlvTagType::from_u8(99), None);
    }
}
