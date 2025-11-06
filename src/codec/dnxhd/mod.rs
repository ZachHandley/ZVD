//! Avid DNxHD/DNxHR codec
//!
//! DNxHD (Digital Nonlinear Extensible High Definition) is Avid's
//! professional video codec for high-quality editing workflows.

pub mod encoder;
pub mod decoder;

pub use encoder::DnxhdEncoder;
pub use decoder::DnxhdDecoder;

/// DNxHD profile (Compression ID)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnxhdProfile {
    /// DNxHD 1080p 36 Mbps
    Dnxhd36,
    /// DNxHD 1080p 45 Mbps
    Dnxhd45,
    /// DNxHD 1080p 75 Mbps
    Dnxhd75,
    /// DNxHD 1080p 115 Mbps
    Dnxhd115,
    /// DNxHD 1080p 120 Mbps
    Dnxhd120,
    /// DNxHD 1080p 145 Mbps
    Dnxhd145,
    /// DNxHD 1080p 175 Mbps (10-bit)
    Dnxhd175,
    /// DNxHD 1080p 185 Mbps (10-bit)
    Dnxhd185,
    /// DNxHD 1080p 220 Mbps (10-bit)
    Dnxhd220,
    /// DNxHR LB (Low Bandwidth)
    DnxhrLb,
    /// DNxHR SQ (Standard Quality)
    DnxhrSq,
    /// DNxHR HQ (High Quality)
    DnxhrHq,
    /// DNxHR HQX (High Quality 10-bit)
    DnxhrHqx,
    /// DNxHR 444 (Full quality with 4:4:4 sampling)
    Dnxhr444,
}

impl DnxhdProfile {
    /// Get compression ID (CID)
    pub fn cid(&self) -> u32 {
        match self {
            DnxhdProfile::Dnxhd36 => 1235,
            DnxhdProfile::Dnxhd45 => 1237,
            DnxhdProfile::Dnxhd75 => 1238,
            DnxhdProfile::Dnxhd115 => 1237,
            DnxhdProfile::Dnxhd120 => 1238,
            DnxhdProfile::Dnxhd145 => 1235,
            DnxhdProfile::Dnxhd175 => 1241,
            DnxhdProfile::Dnxhd185 => 1242,
            DnxhdProfile::Dnxhd220 => 1243,
            DnxhdProfile::DnxhrLb => 1250,
            DnxhdProfile::DnxhrSq => 1251,
            DnxhdProfile::DnxhrHq => 1252,
            DnxhdProfile::DnxhrHqx => 1253,
            DnxhdProfile::Dnxhr444 => 1270,
        }
    }

    /// Get approximate bitrate in Mbps for 1920x1080 @ 30fps
    pub fn approx_bitrate_mbps(&self) -> u32 {
        match self {
            DnxhdProfile::Dnxhd36 => 36,
            DnxhdProfile::Dnxhd45 => 45,
            DnxhdProfile::Dnxhd75 => 75,
            DnxhdProfile::Dnxhd115 => 115,
            DnxhdProfile::Dnxhd120 => 120,
            DnxhdProfile::Dnxhd145 => 145,
            DnxhdProfile::Dnxhd175 => 175,
            DnxhdProfile::Dnxhd185 => 185,
            DnxhdProfile::Dnxhd220 => 220,
            DnxhdProfile::DnxhrLb => 45,
            DnxhdProfile::DnxhrSq => 100,
            DnxhdProfile::DnxhrHq => 185,
            DnxhdProfile::DnxhrHqx => 250,
            DnxhdProfile::Dnxhr444 => 440,
        }
    }

    /// Check if profile is 10-bit
    pub fn is_10bit(&self) -> bool {
        matches!(
            self,
            DnxhdProfile::Dnxhd175
                | DnxhdProfile::Dnxhd185
                | DnxhdProfile::Dnxhd220
                | DnxhdProfile::DnxhrHqx
                | DnxhdProfile::Dnxhr444
        )
    }

    /// Check if profile is DNxHR (resolution independent)
    pub fn is_dnxhr(&self) -> bool {
        matches!(
            self,
            DnxhdProfile::DnxhrLb
                | DnxhdProfile::DnxhrSq
                | DnxhdProfile::DnxhrHq
                | DnxhdProfile::DnxhrHqx
                | DnxhdProfile::Dnxhr444
        )
    }
}

/// DNxHD frame header
#[derive(Debug, Clone)]
pub struct DnxhdFrameHeader {
    pub header_prefix: u32,      // 0x000002800001
    pub compression_id: u32,
    pub width: u16,
    pub height: u16,
    pub is_progressive: bool,
    pub is_422: bool,             // False for 4:4:4
    pub bit_depth: u8,            // 8 or 10
}

impl DnxhdFrameHeader {
    pub fn new(width: u16, height: u16, profile: DnxhdProfile) -> Self {
        DnxhdFrameHeader {
            header_prefix: 0x000002800001,
            compression_id: profile.cid(),
            width,
            height,
            is_progressive: true,
            is_422: profile != DnxhdProfile::Dnxhr444,
            bit_depth: if profile.is_10bit() { 10 } else { 8 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dnxhd_profile_cid() {
        assert_eq!(DnxhdProfile::Dnxhd115.cid(), 1237);
        assert_eq!(DnxhdProfile::DnxhrHq.cid(), 1252);
    }

    #[test]
    fn test_dnxhd_profile_10bit() {
        assert!(!DnxhdProfile::Dnxhd115.is_10bit());
        assert!(DnxhdProfile::Dnxhd220.is_10bit());
        assert!(DnxhdProfile::DnxhrHqx.is_10bit());
    }

    #[test]
    fn test_dnxhd_profile_dnxhr() {
        assert!(!DnxhdProfile::Dnxhd115.is_dnxhr());
        assert!(DnxhdProfile::DnxhrHq.is_dnxhr());
        assert!(DnxhdProfile::Dnxhr444.is_dnxhr());
    }

    #[test]
    fn test_dnxhd_frame_header() {
        let header = DnxhdFrameHeader::new(1920, 1080, DnxhdProfile::DnxhrHq);
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.bit_depth, 8);
    }
}
