//! NAL (Network Abstraction Layer) unit parsing for H.264/AVC
//!
//! This module provides utilities for parsing and manipulating H.264 NAL units
//! in both Annex B format (start code delimited) and AVCC format (length prefixed).
//!
//! ## Formats
//!
//! - **Annex B**: Uses start codes (0x000001 or 0x00000001) to delimit NAL units.
//!   This is the format used by OpenH264 and raw H.264 streams.
//!
//! - **AVCC**: Uses length prefixes (typically 4 bytes) before each NAL unit.
//!   This is the format used in MP4 containers.
//!
//! ## NAL Unit Types
//!
//! Common NAL unit types:
//! - Type 1: Non-IDR slice (P/B frame)
//! - Type 5: IDR slice (keyframe)
//! - Type 6: SEI (Supplemental Enhancement Information)
//! - Type 7: SPS (Sequence Parameter Set)
//! - Type 8: PPS (Picture Parameter Set)
//! - Type 9: AUD (Access Unit Delimiter)

use crate::error::{Error, Result};

// NAL unit type constants
pub const NAL_TYPE_SLICE: u8 = 1;
pub const NAL_TYPE_DPA: u8 = 2;
pub const NAL_TYPE_DPB: u8 = 3;
pub const NAL_TYPE_DPC: u8 = 4;
pub const NAL_TYPE_IDR: u8 = 5;
pub const NAL_TYPE_SEI: u8 = 6;
pub const NAL_TYPE_SPS: u8 = 7;
pub const NAL_TYPE_PPS: u8 = 8;
pub const NAL_TYPE_AUD: u8 = 9;
pub const NAL_TYPE_END_SEQ: u8 = 10;
pub const NAL_TYPE_END_STREAM: u8 = 11;
pub const NAL_TYPE_FILLER: u8 = 12;

/// AVC Decoder Configuration Record (avcC box payload in MP4)
///
/// This structure contains the codec-specific configuration for H.264,
/// including the SPS and PPS NAL units needed to initialize a decoder.
#[derive(Debug, Clone, Default)]
pub struct AvcDecoderConfigurationRecord {
    /// Configuration version (always 1)
    pub configuration_version: u8,
    /// AVC profile indication (from SPS)
    pub avc_profile_indication: u8,
    /// Profile compatibility flags
    pub profile_compatibility: u8,
    /// AVC level indication (from SPS)
    pub avc_level_indication: u8,
    /// NAL unit length size minus 1 (typically 3, meaning 4 bytes)
    pub length_size_minus_one: u8,
    /// Sequence Parameter Sets
    pub sps: Vec<Vec<u8>>,
    /// Picture Parameter Sets
    pub pps: Vec<Vec<u8>>,
}

/// A reference to a single NAL unit within a buffer
#[derive(Debug, Clone)]
pub struct NalUnit<'a> {
    /// NAL unit type (lower 5 bits of first byte)
    pub nal_type: u8,
    /// NAL reference IDC (bits 5-6 of first byte)
    pub nal_ref_idc: u8,
    /// Raw NAL unit data including the header byte
    pub data: &'a [u8],
}

impl<'a> NalUnit<'a> {
    /// Check if this NAL unit is a keyframe (IDR slice)
    pub fn is_keyframe(&self) -> bool {
        self.nal_type == NAL_TYPE_IDR
    }

    /// Check if this NAL unit is a parameter set (SPS or PPS)
    pub fn is_parameter_set(&self) -> bool {
        self.nal_type == NAL_TYPE_SPS || self.nal_type == NAL_TYPE_PPS
    }
}

/// An owned NAL unit
#[derive(Debug, Clone)]
pub struct OwnedNalUnit {
    /// NAL unit type (lower 5 bits of first byte)
    pub nal_type: u8,
    /// NAL reference IDC (bits 5-6 of first byte)
    pub nal_ref_idc: u8,
    /// Raw NAL unit data including the header byte
    pub data: Vec<u8>,
}

impl OwnedNalUnit {
    /// Check if this NAL unit is a keyframe (IDR slice)
    pub fn is_keyframe(&self) -> bool {
        self.nal_type == NAL_TYPE_IDR
    }

    /// Check if this NAL unit is a parameter set (SPS or PPS)
    pub fn is_parameter_set(&self) -> bool {
        self.nal_type == NAL_TYPE_SPS || self.nal_type == NAL_TYPE_PPS
    }
}

/// Parse an AVC Decoder Configuration Record (avcC box payload)
///
/// # Arguments
/// * `data` - Raw bytes of the avcC box payload
///
/// # Returns
/// Parsed configuration record with SPS and PPS data
pub fn parse_avcc(data: &[u8]) -> Result<AvcDecoderConfigurationRecord> {
    if data.len() < 7 {
        return Err(Error::invalid_input("avcC data too short"));
    }

    let configuration_version = data[0];
    if configuration_version != 1 {
        return Err(Error::invalid_input(format!(
            "Unsupported avcC version: {}",
            configuration_version
        )));
    }

    let avc_profile_indication = data[1];
    let profile_compatibility = data[2];
    let avc_level_indication = data[3];
    let length_size_minus_one = data[4] & 0x03;

    let mut offset = 5;
    let mut sps_list = Vec::new();
    let mut pps_list = Vec::new();

    // Parse SPS entries
    if offset >= data.len() {
        return Err(Error::invalid_input("avcC truncated before SPS count"));
    }
    let num_sps = (data[offset] & 0x1F) as usize;
    offset += 1;

    for _ in 0..num_sps {
        if offset + 2 > data.len() {
            return Err(Error::invalid_input("avcC truncated in SPS"));
        }
        let sps_len = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
        offset += 2;

        if offset + sps_len > data.len() {
            return Err(Error::invalid_input("avcC SPS data truncated"));
        }
        sps_list.push(data[offset..offset + sps_len].to_vec());
        offset += sps_len;
    }

    // Parse PPS entries
    if offset >= data.len() {
        return Err(Error::invalid_input("avcC truncated before PPS count"));
    }
    let num_pps = data[offset] as usize;
    offset += 1;

    for _ in 0..num_pps {
        if offset + 2 > data.len() {
            return Err(Error::invalid_input("avcC truncated in PPS"));
        }
        let pps_len = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
        offset += 2;

        if offset + pps_len > data.len() {
            return Err(Error::invalid_input("avcC PPS data truncated"));
        }
        pps_list.push(data[offset..offset + pps_len].to_vec());
        offset += pps_len;
    }

    Ok(AvcDecoderConfigurationRecord {
        configuration_version,
        avc_profile_indication,
        profile_compatibility,
        avc_level_indication,
        length_size_minus_one,
        sps: sps_list,
        pps: pps_list,
    })
}

/// Build an AVC Decoder Configuration Record (avcC) from SPS and PPS
///
/// # Arguments
/// * `sps` - Sequence Parameter Set NAL unit data (without start code)
/// * `pps` - Picture Parameter Set NAL unit data (without start code)
///
/// # Returns
/// Complete avcC box payload bytes
pub fn build_avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    build_avcc_multi(&[sps], &[pps])
}

/// Build an AVC Decoder Configuration Record from multiple SPS and PPS
///
/// # Arguments
/// * `sps_list` - List of SPS NAL units (without start codes)
/// * `pps_list` - List of PPS NAL units (without start codes)
///
/// # Returns
/// Complete avcC box payload bytes
pub fn build_avcc_multi(sps_list: &[&[u8]], pps_list: &[&[u8]]) -> Vec<u8> {
    let mut avcc = Vec::new();

    // Get profile/level from first SPS if available
    let (profile, compat, level) = if !sps_list.is_empty() && sps_list[0].len() >= 4 {
        (sps_list[0][1], sps_list[0][2], sps_list[0][3])
    } else {
        (66, 0, 30) // Baseline profile, level 3.0 defaults
    };

    // Configuration version
    avcc.push(1);
    // AVC profile indication
    avcc.push(profile);
    // Profile compatibility
    avcc.push(compat);
    // AVC level indication
    avcc.push(level);
    // Length size minus 1 (0xFF = 4 bytes, with reserved bits set)
    avcc.push(0xFF);

    // Number of SPS (with reserved bits set)
    avcc.push(0xE0 | (sps_list.len() as u8 & 0x1F));

    for sps in sps_list {
        // SPS length (big endian)
        avcc.push((sps.len() >> 8) as u8);
        avcc.push((sps.len() & 0xFF) as u8);
        // SPS data
        avcc.extend_from_slice(sps);
    }

    // Number of PPS
    avcc.push(pps_list.len() as u8);

    for pps in pps_list {
        // PPS length (big endian)
        avcc.push((pps.len() >> 8) as u8);
        avcc.push((pps.len() & 0xFF) as u8);
        // PPS data
        avcc.extend_from_slice(pps);
    }

    avcc
}

/// Find NAL units in Annex B format bitstream
///
/// Scans the data for start codes (0x000001 or 0x00000001) and returns
/// references to each NAL unit found.
///
/// # Arguments
/// * `data` - H.264 Annex B format bitstream
///
/// # Returns
/// Vector of NAL unit references
pub fn find_nal_units_annex_b(data: &[u8]) -> Vec<NalUnit<'_>> {
    let mut nals = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Find start code
        let (start_code_len, nal_start) = if i + 4 <= data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            (4, i + 4)
        } else if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            (3, i + 3)
        } else {
            i += 1;
            continue;
        };

        if nal_start >= data.len() {
            break;
        }

        // Find end of this NAL (next start code or end of data)
        let mut nal_end = data.len();
        for j in nal_start..data.len().saturating_sub(2) {
            if data[j] == 0 && data[j + 1] == 0 {
                if data[j + 2] == 1 {
                    nal_end = j;
                    break;
                } else if j + 3 < data.len() && data[j + 2] == 0 && data[j + 3] == 1 {
                    nal_end = j;
                    break;
                }
            }
        }

        // Remove trailing zeros (emulation prevention)
        while nal_end > nal_start && data[nal_end - 1] == 0 {
            nal_end -= 1;
        }

        if nal_end > nal_start {
            let header = data[nal_start];
            let nal_type = header & 0x1F;
            let nal_ref_idc = (header >> 5) & 0x03;

            nals.push(NalUnit {
                nal_type,
                nal_ref_idc,
                data: &data[nal_start..nal_end],
            });
        }

        i = nal_start + start_code_len;
    }

    nals
}

/// Find NAL units in Annex B format and return owned data
pub fn find_nal_units_annex_b_owned(data: &[u8]) -> Vec<OwnedNalUnit> {
    find_nal_units_annex_b(data)
        .into_iter()
        .map(|nal| OwnedNalUnit {
            nal_type: nal.nal_type,
            nal_ref_idc: nal.nal_ref_idc,
            data: nal.data.to_vec(),
        })
        .collect()
}

/// Extract SPS and PPS from Annex B bitstream
///
/// # Arguments
/// * `data` - H.264 Annex B format bitstream
///
/// # Returns
/// Tuple of (first SPS, first PPS) found, or None if either is missing
pub fn extract_sps_pps(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let nals = find_nal_units_annex_b(data);
    let mut sps = None;
    let mut pps = None;

    for nal in nals {
        match nal.nal_type {
            NAL_TYPE_SPS if sps.is_none() => {
                sps = Some(nal.data.to_vec());
            }
            NAL_TYPE_PPS if pps.is_none() => {
                pps = Some(nal.data.to_vec());
            }
            _ => {}
        }

        // Early exit if we have both
        if sps.is_some() && pps.is_some() {
            break;
        }
    }

    match (sps, pps) {
        (Some(s), Some(p)) => Some((s, p)),
        _ => None,
    }
}

/// Extract all SPS and PPS from Annex B bitstream
///
/// # Arguments
/// * `data` - H.264 Annex B format bitstream
///
/// # Returns
/// Tuple of (all SPS, all PPS) found
pub fn extract_all_sps_pps(data: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let nals = find_nal_units_annex_b(data);
    let mut sps_list = Vec::new();
    let mut pps_list = Vec::new();

    for nal in nals {
        match nal.nal_type {
            NAL_TYPE_SPS => sps_list.push(nal.data.to_vec()),
            NAL_TYPE_PPS => pps_list.push(nal.data.to_vec()),
            _ => {}
        }
    }

    (sps_list, pps_list)
}

/// Check if Annex B data contains a keyframe (IDR NAL unit)
///
/// # Arguments
/// * `data` - H.264 Annex B format bitstream
///
/// # Returns
/// True if an IDR NAL unit is found
pub fn contains_keyframe(data: &[u8]) -> bool {
    find_nal_units_annex_b(data)
        .iter()
        .any(|nal| nal.nal_type == NAL_TYPE_IDR)
}

/// Convert AVCC format (length-prefixed) to Annex B format (start codes)
///
/// # Arguments
/// * `data` - AVCC format data with length-prefixed NAL units
/// * `length_size` - Size of length prefix in bytes (typically 4)
///
/// # Returns
/// Annex B format data with start codes
pub fn convert_avcc_to_annex_b(data: &[u8], length_size: u8) -> Vec<u8> {
    let length_size = length_size as usize;
    let mut result = Vec::with_capacity(data.len() + data.len() / 100); // Estimate slight growth
    let mut offset = 0;

    while offset + length_size <= data.len() {
        // Read NAL unit length
        let mut nal_len = 0usize;
        for i in 0..length_size {
            nal_len = (nal_len << 8) | (data[offset + i] as usize);
        }
        offset += length_size;

        if offset + nal_len > data.len() {
            break;
        }

        // Write start code and NAL data
        result.extend_from_slice(&[0, 0, 0, 1]);
        result.extend_from_slice(&data[offset..offset + nal_len]);
        offset += nal_len;
    }

    result
}

/// Convert Annex B format (start codes) to AVCC format (length-prefixed)
///
/// # Arguments
/// * `data` - Annex B format data with start codes
/// * `length_size` - Desired size of length prefix in bytes (typically 4)
///
/// # Returns
/// AVCC format data with length prefixes
pub fn convert_annex_b_to_avcc(data: &[u8], length_size: u8) -> Vec<u8> {
    let nals = find_nal_units_annex_b(data);
    let length_size = length_size as usize;
    let mut result = Vec::with_capacity(data.len());

    for nal in nals {
        let nal_len = nal.data.len();

        // Write length prefix (big endian)
        for i in (0..length_size).rev() {
            result.push(((nal_len >> (i * 8)) & 0xFF) as u8);
        }

        // Write NAL data
        result.extend_from_slice(nal.data);
    }

    result
}

/// Prepend SPS and PPS to Annex B data if not already present
///
/// This is useful for ensuring keyframes are self-contained.
///
/// # Arguments
/// * `data` - Annex B format data
/// * `sps` - SPS NAL unit data
/// * `pps` - PPS NAL unit data
///
/// # Returns
/// Annex B data with SPS and PPS prepended if they weren't already present
pub fn prepend_parameter_sets(data: &[u8], sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let nals = find_nal_units_annex_b(data);
    let has_sps = nals.iter().any(|n| n.nal_type == NAL_TYPE_SPS);
    let has_pps = nals.iter().any(|n| n.nal_type == NAL_TYPE_PPS);

    if has_sps && has_pps {
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(data.len() + sps.len() + pps.len() + 8);

    if !has_sps {
        result.extend_from_slice(&[0, 0, 0, 1]);
        result.extend_from_slice(sps);
    }

    if !has_pps {
        result.extend_from_slice(&[0, 0, 0, 1]);
        result.extend_from_slice(pps);
    }

    result.extend_from_slice(data);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nal_units_4byte_start_code() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, // SPS
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80, // PPS
        ];

        let nals = find_nal_units_annex_b(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].nal_type, NAL_TYPE_SPS);
        assert_eq!(nals[1].nal_type, NAL_TYPE_PPS);
    }

    #[test]
    fn test_find_nal_units_3byte_start_code() {
        let data = [
            0x00, 0x00, 0x01, 0x67, 0x42, 0x00, // SPS
            0x00, 0x00, 0x01, 0x68, 0xCE, // PPS
        ];

        let nals = find_nal_units_annex_b(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].nal_type, NAL_TYPE_SPS);
        assert_eq!(nals[1].nal_type, NAL_TYPE_PPS);
    }

    #[test]
    fn test_extract_sps_pps() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, // SPS
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80, // PPS
            0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, // IDR
        ];

        let result = extract_sps_pps(&data);
        assert!(result.is_some());

        let (sps, pps) = result.unwrap();
        assert_eq!(sps[0] & 0x1F, NAL_TYPE_SPS);
        assert_eq!(pps[0] & 0x1F, NAL_TYPE_PPS);
    }

    #[test]
    fn test_contains_keyframe() {
        let with_idr = [
            0x00, 0x00, 0x00, 0x01, 0x65, 0x88, // IDR
        ];
        let without_idr = [
            0x00, 0x00, 0x00, 0x01, 0x41, 0x88, // Non-IDR slice
        ];

        assert!(contains_keyframe(&with_idr));
        assert!(!contains_keyframe(&without_idr));
    }

    #[test]
    fn test_build_and_parse_avcc() {
        let sps = vec![0x67, 0x42, 0x00, 0x1E, 0xDA, 0x01];
        let pps = vec![0x68, 0xCE, 0x3C, 0x80];

        let avcc = build_avcc(&sps, &pps);
        assert!(avcc.len() > 6);
        assert_eq!(avcc[0], 1); // Version

        let parsed = parse_avcc(&avcc).unwrap();
        assert_eq!(parsed.configuration_version, 1);
        assert_eq!(parsed.sps.len(), 1);
        assert_eq!(parsed.pps.len(), 1);
        assert_eq!(parsed.sps[0], sps);
        assert_eq!(parsed.pps[0], pps);
    }

    #[test]
    fn test_convert_annex_b_to_avcc_roundtrip() {
        let annex_b = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, // SPS
            0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x3C, 0x80, // PPS
        ];

        let avcc = convert_annex_b_to_avcc(&annex_b, 4);
        let back_to_annex_b = convert_avcc_to_annex_b(&avcc, 4);

        // NAL data should match (ignoring start code differences)
        let orig_nals = find_nal_units_annex_b(&annex_b);
        let roundtrip_nals = find_nal_units_annex_b(&back_to_annex_b);

        assert_eq!(orig_nals.len(), roundtrip_nals.len());
        for (orig, rt) in orig_nals.iter().zip(roundtrip_nals.iter()) {
            assert_eq!(orig.nal_type, rt.nal_type);
            assert_eq!(orig.data, rt.data);
        }
    }

    #[test]
    fn test_prepend_parameter_sets() {
        let sps = vec![0x67, 0x42, 0x00, 0x1E];
        let pps = vec![0x68, 0xCE, 0x3C, 0x80];
        let idr = [0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84];

        let result = prepend_parameter_sets(&idr, &sps, &pps);
        let nals = find_nal_units_annex_b(&result);

        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].nal_type, NAL_TYPE_SPS);
        assert_eq!(nals[1].nal_type, NAL_TYPE_PPS);
        assert_eq!(nals[2].nal_type, NAL_TYPE_IDR);
    }

    #[test]
    fn test_nal_unit_methods() {
        let nal = NalUnit {
            nal_type: NAL_TYPE_IDR,
            nal_ref_idc: 3,
            data: &[0x65, 0x88],
        };
        assert!(nal.is_keyframe());
        assert!(!nal.is_parameter_set());

        let sps_nal = NalUnit {
            nal_type: NAL_TYPE_SPS,
            nal_ref_idc: 3,
            data: &[0x67, 0x42],
        };
        assert!(!sps_nal.is_keyframe());
        assert!(sps_nal.is_parameter_set());
    }
}
