//! AVI (Audio Video Interleave) container format
//!
//! AVI is Microsoft's container format for audio/video data.
//! While older, it's still widely used and supported.

pub mod demuxer;
pub mod muxer;

pub use demuxer::AviDemuxer;
pub use muxer::AviMuxer;

/// RIFF chunk header
#[derive(Debug, Clone)]
pub struct RiffChunk {
    pub fourcc: [u8; 4],
    pub size: u32,
}

/// AVI main header (avih)
#[derive(Debug, Clone)]
pub struct AviMainHeader {
    pub microsec_per_frame: u32,
    pub max_bytes_per_sec: u32,
    pub flags: u32,
    pub total_frames: u32,
    pub initial_frames: u32,
    pub streams: u32,
    pub suggested_buffer_size: u32,
    pub width: u32,
    pub height: u32,
}

impl AviMainHeader {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        AviMainHeader {
            microsec_per_frame: 1_000_000 / fps,
            max_bytes_per_sec: 0,
            flags: 0x10, // AVIF_HASINDEX
            total_frames: 0,
            initial_frames: 0,
            streams: 1,
            suggested_buffer_size: 0,
            width,
            height,
        }
    }
}

/// AVI stream header (strh)
#[derive(Debug, Clone)]
pub struct AviStreamHeader {
    pub fcc_type: [u8; 4],      // 'vids' or 'auds'
    pub fcc_handler: [u8; 4],   // Codec FourCC
    pub flags: u32,
    pub priority: u16,
    pub language: u16,
    pub initial_frames: u32,
    pub scale: u32,
    pub rate: u32,
    pub start: u32,
    pub length: u32,
    pub suggested_buffer_size: u32,
    pub quality: u32,
    pub sample_size: u32,
}

impl AviStreamHeader {
    pub fn video(codec_fourcc: [u8; 4], width: u32, height: u32, fps: u32) -> Self {
        AviStreamHeader {
            fcc_type: *b"vids",
            fcc_handler: codec_fourcc,
            flags: 0,
            priority: 0,
            language: 0,
            initial_frames: 0,
            scale: 1,
            rate: fps,
            start: 0,
            length: 0,
            suggested_buffer_size: width * height * 3,
            quality: 10000,
            sample_size: 0,
        }
    }

    pub fn audio(codec_tag: u16, sample_rate: u32, channels: u16) -> Self {
        AviStreamHeader {
            fcc_type: *b"auds",
            fcc_handler: [0; 4],
            flags: 0,
            priority: 0,
            language: 0,
            initial_frames: 0,
            scale: 1,
            rate: sample_rate,
            start: 0,
            length: 0,
            suggested_buffer_size: sample_rate * channels as u32 * 2,
            quality: 10000,
            sample_size: channels as u32 * 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avi_main_header() {
        let header = AviMainHeader::new(1920, 1080, 30);
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.microsec_per_frame, 33333);
    }

    #[test]
    fn test_avi_stream_header_video() {
        let header = AviStreamHeader::video(*b"H264", 1920, 1080, 30);
        assert_eq!(header.fcc_type, *b"vids");
        assert_eq!(header.fcc_handler, *b"H264");
        assert_eq!(header.rate, 30);
    }
}
