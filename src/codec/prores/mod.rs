//! Apple ProRes codec
//!
//! ProRes is a high-quality video codec designed for professional video editing.
//! It offers excellent image quality with manageable file sizes.
//!
//! ## Implementation Status
//!
//! **Current State**: Header parsing and format structures implemented.
//! **Full Decoding/Encoding**: Requires FFmpeg or similar library.
//!
//! ### What's Implemented
//!
//! - ✅ **ProRes Profile Support**: All variants (Proxy, LT, Standard, HQ, 4444, 4444 XQ)
//! - ✅ **Frame Header Parsing**: Complete ProRes frame header structure
//! - ✅ **FourCC Handling**: Correct identification of all ProRes types
//! - ✅ **Metadata**: Width, height, chroma format, alpha channel detection
//! - ✅ **Bitrate Estimation**: Approximate bitrates for all profiles
//!
//! ### What's NOT Implemented (Requires FFmpeg)
//!
//! - ❌ **Actual Decoding**: Slice parsing, Huffman/VLC decoding, inverse DCT
//! - ❌ **Actual Encoding**: DCT, quantization, slice encoding
//! - ⚠️ The `decode_frame_data()` and `encode_frame_data()` functions are stubs
//!
//! ## Why FFmpeg is Required
//!
//! ProRes encoding/decoding involves:
//! 1. Complex variable-length coding (Huffman/VLC)
//! 2. Inverse DCT transformations
//! 3. Inverse quantization with profile-specific matrices
//! 4. Slice-based encoding with specific block structures
//! 5. Color space conversions
//!
//! Implementing this from scratch would be thousands of lines and require extensive
//! testing against reference implementations. Using FFmpeg's libavcodec is the
//! pragmatic choice for production use.
//!
//! ## Future Work
//!
//! To enable full ProRes support:
//! 1. Add `ffmpeg-next` or `ac-ffmpeg` dependency
//! 2. Create FFmpeg codec adapter
//! 3. Wire up decoder/encoder to use libavcodec
//! 4. Add comprehensive integration tests
//!
//! ## Usage (Current - Header Parsing Only)
//!
//! ```rust
//! use zvd_lib::codec::prores::{ProResProfile, ProResFrameHeader};
//!
//! // Check profile characteristics
//! let profile = ProResProfile::Hq;
//! assert_eq!(profile.fourcc(), *b"apch");
//! assert_eq!(profile.approx_bitrate_mbps(), 220);
//! assert!(!profile.has_alpha());
//!
//! // Create frame header for encoding
//! let header = ProResFrameHeader::new(1920, 1080, profile);
//! assert_eq!(header.width, 1920);
//! ```

pub mod encoder;
pub mod decoder;

pub use encoder::ProResEncoder;
pub use decoder::ProResDecoder;

/// ProRes profile variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProResProfile {
    /// ProRes 422 Proxy - lowest data rate
    Proxy,
    /// ProRes 422 LT - lower data rate
    Lt,
    /// ProRes 422 - standard quality
    Standard,
    /// ProRes 422 HQ - higher data rate for demanding workflows
    Hq,
    /// ProRes 4444 - supports alpha channel
    FourFourFourFour,
    /// ProRes 4444 XQ - highest quality with extended color
    FourFourFourFourXq,
}

impl ProResProfile {
    /// Get FourCC code for the profile
    pub fn fourcc(&self) -> [u8; 4] {
        match self {
            ProResProfile::Proxy => *b"apco",
            ProResProfile::Lt => *b"apcs",
            ProResProfile::Standard => *b"apcn",
            ProResProfile::Hq => *b"apch",
            ProResProfile::FourFourFourFour => *b"ap4h",
            ProResProfile::FourFourFourFourXq => *b"ap4x",
        }
    }

    /// Get approximate bitrate in Mbps for 1920x1080 @ 30fps
    pub fn approx_bitrate_mbps(&self) -> u32 {
        match self {
            ProResProfile::Proxy => 45,
            ProResProfile::Lt => 102,
            ProResProfile::Standard => 147,
            ProResProfile::Hq => 220,
            ProResProfile::FourFourFourFour => 330,
            ProResProfile::FourFourFourFourXq => 500,
        }
    }

    /// Check if profile supports alpha channel
    pub fn has_alpha(&self) -> bool {
        matches!(self, ProResProfile::FourFourFourFour | ProResProfile::FourFourFourFourXq)
    }
}

/// ProRes frame header
#[derive(Debug, Clone)]
pub struct ProResFrameHeader {
    pub frame_size: u32,
    pub frame_identifier: [u8; 4], // "icpf"
    pub header_size: u16,
    pub version: u8,
    pub encoder_id: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub chroma_format: u8,
    pub interlace_mode: u8,
    pub aspect_ratio: u8,
    pub framerate_code: u8,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub alpha_info: u8,
}

impl ProResFrameHeader {
    pub fn new(width: u16, height: u16, profile: ProResProfile) -> Self {
        ProResFrameHeader {
            frame_size: 0, // Will be filled during encoding
            frame_identifier: *b"icpf",
            header_size: 148,
            version: 0,
            encoder_id: *b"zvd0", // Our encoder ID
            width,
            height,
            chroma_format: if profile.has_alpha() { 3 } else { 2 }, // 2=422, 3=444
            interlace_mode: 0, // Progressive
            aspect_ratio: 0,
            framerate_code: 0,
            color_primaries: 1,    // BT.709
            transfer_characteristics: 1, // BT.709
            matrix_coefficients: 1, // BT.709
            alpha_info: if profile.has_alpha() { 1 } else { 0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prores_profile_fourcc() {
        assert_eq!(ProResProfile::Standard.fourcc(), *b"apcn");
        assert_eq!(ProResProfile::Hq.fourcc(), *b"apch");
        assert_eq!(ProResProfile::FourFourFourFour.fourcc(), *b"ap4h");
    }

    #[test]
    fn test_prores_profile_alpha() {
        assert!(!ProResProfile::Standard.has_alpha());
        assert!(ProResProfile::FourFourFourFour.has_alpha());
        assert!(ProResProfile::FourFourFourFourXq.has_alpha());
    }

    #[test]
    fn test_prores_frame_header() {
        let header = ProResFrameHeader::new(1920, 1080, ProResProfile::Standard);
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.encoder_id, *b"zvd0");
    }
}
