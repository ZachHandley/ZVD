//! H.265/HEVC NAL (Network Abstraction Layer) unit parsing
//!
//! NAL units are the fundamental data structures in H.265 bitstreams.
//! Each NAL unit contains a header and payload.
//!
//! ## NAL Unit Structure
//!
//! ```text
//! NAL Unit:
//! +----------------+-------------------+
//! | NAL Header (2) | RBSP Payload (N) |
//! +----------------+-------------------+
//!
//! NAL Header (16 bits):
//! +---+-----+--------+-----+
//! | F | Type | LayerID | TID |
//! +---+-----+--------+-----+
//!   1   6      6        3
//!
//! F: Forbidden zero bit (must be 0)
//! Type: NAL unit type (6 bits)
//! LayerID: Layer ID for scalable extensions (6 bits)
//! TID: Temporal ID (3 bits)
//! ```

use crate::error::{Error, Result};

/// H.265 NAL unit types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    /// Coded slice of a non-TSA, non-STSA trailing picture
    TrailN = 0,
    /// Coded slice of a non-TSA, non-STSA trailing picture
    TrailR = 1,

    /// Coded slice of a TSA picture
    TsaN = 2,
    /// Coded slice of a TSA picture
    TsaR = 3,

    /// Coded slice of an STSA picture
    StsaN = 4,
    /// Coded slice of an STSA picture
    StsaR = 5,

    /// Coded slice of a RADL picture
    RadlN = 6,
    /// Coded slice of a RADL picture
    RadlR = 7,

    /// Coded slice of a RASL picture
    RaslN = 8,
    /// Coded slice of a RASL picture
    RaslR = 9,

    // Reserved (10-15)

    /// Coded slice of a BLA picture
    BlaWLp = 16,
    /// Coded slice of a BLA picture
    BlaWRadl = 17,
    /// Coded slice of a BLA picture
    BlaNLp = 18,

    /// Coded slice of an IDR picture
    IdrWRadl = 19,
    /// Coded slice of an IDR picture
    IdrNLp = 20,

    /// Coded slice of a CRA picture
    CraNut = 21,

    // Reserved (22-31)

    /// Video Parameter Set
    VpsNut = 32,
    /// Sequence Parameter Set
    SpsNut = 33,
    /// Picture Parameter Set
    PpsNut = 34,

    /// Access unit delimiter
    AudNut = 35,
    /// End of sequence
    EosNut = 36,
    /// End of bitstream
    EobNut = 37,
    /// Filler data
    FdNut = 38,

    /// Supplemental enhancement information prefix
    PrefixSeiNut = 39,
    /// Supplemental enhancement information suffix
    SuffixSeiNut = 40,

    // Reserved (41-47)
    // Unspecified (48-63)

    /// Unknown/Invalid NAL unit type
    Unknown = 255,
}

impl NalUnitType {
    /// Create NAL unit type from u8 value
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::TrailN,
            1 => Self::TrailR,
            2 => Self::TsaN,
            3 => Self::TsaR,
            4 => Self::StsaN,
            5 => Self::StsaR,
            6 => Self::RadlN,
            7 => Self::RadlR,
            8 => Self::RaslN,
            9 => Self::RaslR,
            16 => Self::BlaWLp,
            17 => Self::BlaWRadl,
            18 => Self::BlaNLp,
            19 => Self::IdrWRadl,
            20 => Self::IdrNLp,
            21 => Self::CraNut,
            32 => Self::VpsNut,
            33 => Self::SpsNut,
            34 => Self::PpsNut,
            35 => Self::AudNut,
            36 => Self::EosNut,
            37 => Self::EobNut,
            38 => Self::FdNut,
            39 => Self::PrefixSeiNut,
            40 => Self::SuffixSeiNut,
            _ => Self::Unknown,
        }
    }

    /// Check if this NAL unit type is a slice
    pub fn is_slice(&self) -> bool {
        matches!(
            self,
            Self::TrailN | Self::TrailR |
            Self::TsaN | Self::TsaR |
            Self::StsaN | Self::StsaR |
            Self::RadlN | Self::RadlR |
            Self::RaslN | Self::RaslR |
            Self::BlaWLp | Self::BlaWRadl | Self::BlaNLp |
            Self::IdrWRadl | Self::IdrNLp |
            Self::CraNut
        )
    }

    /// Check if this is an IDR (Instantaneous Decoder Refresh) NAL unit
    pub fn is_idr(&self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLp)
    }

    /// Check if this is a parameter set (VPS/SPS/PPS)
    pub fn is_parameter_set(&self) -> bool {
        matches!(self, Self::VpsNut | Self::SpsNut | Self::PpsNut)
    }
}

/// H.265 NAL unit header (2 bytes)
#[derive(Debug, Clone, Copy)]
pub struct NalHeader {
    /// Forbidden zero bit (must be 0)
    pub forbidden_zero_bit: bool,
    /// NAL unit type (6 bits)
    pub nal_unit_type: NalUnitType,
    /// Layer ID for scalable extensions (6 bits)
    pub nuh_layer_id: u8,
    /// Temporal ID minus 1 (3 bits)
    pub nuh_temporal_id_plus1: u8,
}

impl NalHeader {
    /// Parse NAL header from 2 bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::codec("NAL header requires at least 2 bytes"));
        }

        let byte0 = data[0];
        let byte1 = data[1];

        let forbidden_zero_bit = (byte0 & 0x80) != 0;
        let nal_unit_type_raw = (byte0 >> 1) & 0x3F;
        let nuh_layer_id = ((byte0 & 0x01) << 5) | ((byte1 >> 3) & 0x1F);
        let nuh_temporal_id_plus1 = byte1 & 0x07;

        // Forbidden zero bit must be 0
        if forbidden_zero_bit {
            return Err(Error::codec("NAL header forbidden_zero_bit is not 0"));
        }

        // Temporal ID must not be 0
        if nuh_temporal_id_plus1 == 0 {
            return Err(Error::codec("NAL header nuh_temporal_id_plus1 is 0 (invalid)"));
        }

        Ok(NalHeader {
            forbidden_zero_bit,
            nal_unit_type: NalUnitType::from_u8(nal_unit_type_raw),
            nuh_layer_id,
            nuh_temporal_id_plus1,
        })
    }

    /// Get temporal ID (TID)
    pub fn temporal_id(&self) -> u8 {
        self.nuh_temporal_id_plus1.saturating_sub(1)
    }
}

/// H.265 NAL unit (header + RBSP payload)
#[derive(Debug, Clone)]
pub struct NalUnit {
    /// NAL unit header
    pub header: NalHeader,
    /// RBSP (Raw Byte Sequence Payload) data
    pub rbsp: Vec<u8>,
}

impl NalUnit {
    /// Parse NAL unit from bytes
    ///
    /// This function expects data WITHOUT the start code prefix (0x000001 or 0x00000001)
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::codec("NAL unit too short (< 2 bytes)"));
        }

        let header = NalHeader::parse(&data[0..2])?;

        // Extract RBSP by removing emulation prevention bytes (0x03)
        let rbsp = remove_emulation_prevention_bytes(&data[2..])?;

        Ok(NalUnit { header, rbsp })
    }

    /// Get NAL unit type
    pub fn nal_type(&self) -> NalUnitType {
        self.header.nal_unit_type
    }

    /// Check if this is a slice NAL unit
    pub fn is_slice(&self) -> bool {
        self.header.nal_unit_type.is_slice()
    }

    /// Check if this is an IDR NAL unit
    pub fn is_idr(&self) -> bool {
        self.header.nal_unit_type.is_idr()
    }
}

/// Remove emulation prevention bytes (0x03) from RBSP
///
/// H.265 inserts 0x03 bytes to prevent start code emulation:
/// - 0x000000 → 0x00000300
/// - 0x000001 → 0x00000301
/// - 0x000002 → 0x00000302
/// - 0x000003 → 0x00000303
fn remove_emulation_prevention_bytes(data: &[u8]) -> Result<Vec<u8>> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            // Found emulation prevention sequence
            rbsp.push(0x00);
            rbsp.push(0x00);
            i += 3; // Skip the 0x03 byte
        } else {
            rbsp.push(data[i]);
            i += 1;
        }
    }

    Ok(rbsp)
}

/// Find NAL unit start codes in a byte stream
///
/// Returns positions of start codes (either 3-byte 0x000001 or 4-byte 0x00000001)
pub fn find_start_codes(data: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut i = 0;

    while i + 2 < data.len() {
        if data[i] == 0x00 && data[i + 1] == 0x00 {
            if data[i + 2] == 0x01 {
                // Found 3-byte start code
                positions.push(i);
                i += 3;
            } else if i + 3 < data.len() && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                // Found 4-byte start code
                positions.push(i);
                i += 4;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_unit_type_from_u8() {
        assert_eq!(NalUnitType::from_u8(32), NalUnitType::VpsNut);
        assert_eq!(NalUnitType::from_u8(33), NalUnitType::SpsNut);
        assert_eq!(NalUnitType::from_u8(34), NalUnitType::PpsNut);
        assert_eq!(NalUnitType::from_u8(19), NalUnitType::IdrWRadl);
        assert_eq!(NalUnitType::from_u8(99), NalUnitType::Unknown);
    }

    #[test]
    fn test_nal_unit_type_predicates() {
        assert!(NalUnitType::IdrWRadl.is_idr());
        assert!(NalUnitType::IdrWRadl.is_slice());
        assert!(NalUnitType::VpsNut.is_parameter_set());
        assert!(!NalUnitType::VpsNut.is_slice());
    }

    #[test]
    fn test_nal_header_parse() {
        // IDR NAL unit header: type=19, layer_id=0, tid=1
        let data = vec![0x26, 0x01]; // 0010 0110 0000 0001
        let header = NalHeader::parse(&data).unwrap();

        assert!(!header.forbidden_zero_bit);
        assert_eq!(header.nal_unit_type, NalUnitType::IdrWRadl);
        assert_eq!(header.nuh_layer_id, 0);
        assert_eq!(header.nuh_temporal_id_plus1, 1);
        assert_eq!(header.temporal_id(), 0);
    }

    #[test]
    fn test_emulation_prevention_removal() {
        let input = vec![0x00, 0x00, 0x03, 0x01, 0xFF];
        let output = remove_emulation_prevention_bytes(&input).unwrap();
        assert_eq!(output, vec![0x00, 0x00, 0x01, 0xFF]);
    }

    #[test]
    fn test_find_start_codes() {
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // 4-byte start code at 0
            0xFF, 0xFF,
            0x00, 0x00, 0x01,       // 3-byte start code at 6
            0xAA, 0xBB,
        ];
        let positions = find_start_codes(&data);
        assert_eq!(positions, vec![0, 6]);
    }
}
