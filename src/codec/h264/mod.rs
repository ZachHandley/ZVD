//! H.264/AVC codec implementation using OpenH264
//!
//! This module provides H.264 encoding and decoding using Cisco's OpenH264 library.
//! OpenH264 is licensed under BSD 2-Clause, and Cisco provides a patent license for
//! binary usage.
//!
//! ## Features
//!
//! - **Decoder**: Converts H.264 NAL units to YUV420P video frames
//! - **Encoder**: Converts YUV420P video frames to H.264 NAL units
//! - **NAL Parsing**: Utilities for parsing and manipulating NAL units
//! - **Format Conversion**: Support for both Annex B and AVCC formats
//!
//! ## Security Notes
//!
//! This module requires openh264 >= 0.9.1 which includes fixes for CVE-2025-27091,
//! a heap overflow vulnerability in the decoding functions that could allow
//! remote code execution.
//!
//! ## Patent Notice
//!
//! H.264/AVC is covered by patents. Cisco's OpenH264 binary plugin is covered by
//! Cisco's patent license. Commercial use may require additional patent licensing.
//! See CODEC_LICENSES.md for details.
//!
//! ## Example Usage
//!
//! ```ignore
//! use zvd_lib::codec::h264::{H264Decoder, H264Encoder};
//! use zvd_lib::codec::{Decoder, Encoder, Frame};
//!
//! // Create encoder
//! let mut encoder = H264Encoder::new(1920, 1080).unwrap();
//!
//! // Encode a frame
//! encoder.send_frame(&frame).unwrap();
//! let packet = encoder.receive_packet().unwrap();
//!
//! // Create decoder
//! let mut decoder = H264Decoder::new().unwrap();
//!
//! // Decode the packet
//! decoder.send_packet(&packet).unwrap();
//! let decoded = decoder.receive_frame().unwrap();
//! ```

// NAL unit parsing is always available
pub mod nal;

#[cfg(feature = "h264")]
pub mod decoder;
#[cfg(feature = "h264")]
pub mod encoder;

// Re-export main types when h264 feature is enabled
#[cfg(feature = "h264")]
pub use decoder::H264Decoder;
#[cfg(feature = "h264")]
pub use encoder::{H264Encoder, H264EncoderConfig, RateControlMode};

// Re-export NAL utilities (always available for parsing)
pub use nal::{
    // NAL type constants
    NAL_TYPE_SLICE,
    NAL_TYPE_DPA,
    NAL_TYPE_DPB,
    NAL_TYPE_DPC,
    NAL_TYPE_IDR,
    NAL_TYPE_SEI,
    NAL_TYPE_SPS,
    NAL_TYPE_PPS,
    NAL_TYPE_AUD,
    NAL_TYPE_END_SEQ,
    NAL_TYPE_END_STREAM,
    NAL_TYPE_FILLER,
    // Data structures
    AvcDecoderConfigurationRecord,
    NalUnit,
    OwnedNalUnit,
    // Parsing functions
    parse_avcc,
    build_avcc,
    build_avcc_multi,
    find_nal_units_annex_b,
    find_nal_units_annex_b_owned,
    extract_sps_pps,
    extract_all_sps_pps,
    contains_keyframe,
    // Format conversion
    convert_avcc_to_annex_b,
    convert_annex_b_to_avcc,
    prepend_parameter_sets,
};

/// Codec information for H.264
#[derive(Debug, Clone)]
pub struct H264CodecInfo {
    /// Profile (Baseline, Main, High, etc.)
    pub profile: H264Profile,
    /// Level (e.g., 3.0, 4.0, 5.1)
    pub level: u8,
    /// Whether the codec is for encoding or decoding
    pub is_encoder: bool,
}

/// H.264 Profile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264Profile {
    /// Baseline Profile - simple, low-latency
    Baseline = 66,
    /// Main Profile - standard quality
    Main = 77,
    /// High Profile - high quality, more complex
    High = 100,
    /// High 10 Profile - 10-bit support
    High10 = 110,
    /// High 4:2:2 Profile - 4:2:2 chroma
    High422 = 122,
    /// High 4:4:4 Predictive Profile
    High444 = 244,
    /// Unknown profile
    Unknown = 0,
}

impl From<u8> for H264Profile {
    fn from(value: u8) -> Self {
        match value {
            66 => H264Profile::Baseline,
            77 => H264Profile::Main,
            100 => H264Profile::High,
            110 => H264Profile::High10,
            122 => H264Profile::High422,
            244 => H264Profile::High444,
            _ => H264Profile::Unknown,
        }
    }
}

impl std::fmt::Display for H264Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            H264Profile::Baseline => write!(f, "Baseline"),
            H264Profile::Main => write!(f, "Main"),
            H264Profile::High => write!(f, "High"),
            H264Profile::High10 => write!(f, "High 10"),
            H264Profile::High422 => write!(f, "High 4:2:2"),
            H264Profile::High444 => write!(f, "High 4:4:4"),
            H264Profile::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Parse profile and level from SPS NAL unit
pub fn parse_profile_level(sps: &[u8]) -> Option<(H264Profile, u8)> {
    // SPS NAL unit format:
    // Byte 0: NAL header (forbidden_zero_bit, nal_ref_idc, nal_unit_type)
    // Byte 1: profile_idc
    // Byte 2: constraint_set flags
    // Byte 3: level_idc
    if sps.len() < 4 {
        return None;
    }

    // Verify it's an SPS NAL unit
    if (sps[0] & 0x1F) != NAL_TYPE_SPS {
        return None;
    }

    let profile = H264Profile::from(sps[1]);
    let level = sps[3];

    Some((profile, level))
}

/// Format level as human-readable string (e.g., "4.0", "5.1")
pub fn format_level(level: u8) -> String {
    format!("{}.{}", level / 10, level % 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h264_profile_from_u8() {
        assert_eq!(H264Profile::from(66), H264Profile::Baseline);
        assert_eq!(H264Profile::from(77), H264Profile::Main);
        assert_eq!(H264Profile::from(100), H264Profile::High);
        assert_eq!(H264Profile::from(255), H264Profile::Unknown);
    }

    #[test]
    fn test_h264_profile_display() {
        assert_eq!(format!("{}", H264Profile::Baseline), "Baseline");
        assert_eq!(format!("{}", H264Profile::High), "High");
    }

    #[test]
    fn test_format_level() {
        assert_eq!(format_level(30), "3.0");
        assert_eq!(format_level(40), "4.0");
        assert_eq!(format_level(51), "5.1");
    }

    #[test]
    fn test_parse_profile_level() {
        // SPS with Baseline profile, level 3.0
        let sps = vec![0x67, 66, 0x00, 30]; // NAL header, profile, constraint, level
        let result = parse_profile_level(&sps);
        assert!(result.is_some());
        let (profile, level) = result.unwrap();
        assert_eq!(profile, H264Profile::Baseline);
        assert_eq!(level, 30);
    }

    #[test]
    fn test_parse_profile_level_invalid() {
        // Too short
        let short = vec![0x67, 66];
        assert!(parse_profile_level(&short).is_none());

        // Wrong NAL type (PPS instead of SPS)
        let pps = vec![0x68, 66, 0x00, 30];
        assert!(parse_profile_level(&pps).is_none());
    }
}
