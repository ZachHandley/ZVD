//! DNxHD/DNxHR codec data tables
//!
//! This module contains all the quantization matrices, VLC tables, and CID profiles
//! needed for DNxHD/DNxHR encoding and decoding.

use super::DnxhdProfile;

/// DC Huffman code entry
#[derive(Debug, Clone, Copy)]
pub struct DcCode {
    pub code: u16,
    pub bits: u8,
}

/// AC VLC entry
#[derive(Debug, Clone, Copy)]
pub struct AcVlcEntry {
    pub code: u16,
    pub bits: u8,
    pub run: u8,
    pub level: i16,
}

/// Run-length code entry
#[derive(Debug, Clone, Copy)]
pub struct RunCode {
    pub code: u16,
    pub bits: u8,
}

/// CID-specific codec data
#[derive(Debug, Clone)]
pub struct CidData {
    pub cid: u32,
    pub width: u16,
    pub height: u16,
    pub bit_depth: u8,
    pub is_444: bool,
    pub luma_weight: &'static [u8; 64],
    pub chroma_weight: &'static [u8; 64],
    pub dc_codes: &'static [DcCode],
    pub ac_codes: &'static [AcVlcEntry],
    pub run_codes: &'static [RunCode],
}

// Quantization weight matrices (zigzag order)

/// Luma weights for CID 1235, 1256, 1270
static LUMA_WEIGHT_1235: [u8; 64] = [
    0, 32, 32, 32, 33, 32, 32, 32,
    32, 31, 32, 33, 33, 33, 33, 35,
    36, 36, 34, 34, 36, 37, 37, 36,
    36, 35, 36, 38, 39, 39, 37, 36,
    37, 37, 39, 41, 42, 41, 39, 39,
    40, 41, 42, 43, 42, 42, 41, 41,
    41, 44, 47, 46, 46, 48, 51, 51,
    50, 50, 53, 55, 55, 56, 60, 60,
];

/// Luma weights for CID 1237, 1253, 1259, 1273, 1274
static LUMA_WEIGHT_1237: [u8; 64] = [
    0, 32, 33, 34, 34, 36, 37, 36,
    36, 37, 38, 38, 38, 39, 41, 44,
    43, 41, 40, 41, 46, 49, 47, 46,
    47, 49, 51, 54, 60, 62, 59, 55,
    54, 56, 58, 61, 65, 66, 64, 63,
    66, 73, 78, 79, 80, 79, 78, 78,
    82, 87, 89, 90, 93, 95, 96, 97,
    97, 100, 104, 102, 98, 98, 99, 99,
];

/// Luma weights for CID 1238, 1272
static LUMA_WEIGHT_1238: [u8; 64] = [
    0, 32, 32, 33, 34, 33, 33, 33,
    33, 33, 33, 33, 33, 35, 37, 37,
    36, 36, 35, 36, 38, 38, 36, 35,
    36, 37, 38, 41, 42, 41, 39, 38,
    38, 38, 39, 41, 42, 41, 39, 39,
    40, 41, 43, 44, 44, 44, 44, 44,
    45, 47, 47, 47, 49, 50, 51, 51,
    51, 53, 55, 57, 58, 59, 57, 57,
];

/// Luma weights for CID 1241, 1271
static LUMA_WEIGHT_1241: [u8; 64] = [
    0, 32, 33, 34, 34, 35, 36, 37,
    36, 37, 38, 38, 38, 39, 39, 40,
    40, 38, 38, 39, 38, 37, 39, 41,
    41, 42, 43, 45, 45, 46, 47, 46,
    45, 43, 39, 37, 37, 40, 44, 45,
    45, 46, 46, 46, 47, 47, 46, 44,
    42, 43, 45, 47, 48, 49, 50, 49,
    48, 46, 47, 48, 48, 49, 49, 49,
];

/// Chroma weights for CID 1235, 1256, 1270
static CHROMA_WEIGHT_1235: [u8; 64] = [
    0, 32, 33, 34, 34, 35, 35, 35,
    35, 35, 36, 37, 36, 36, 37, 38,
    38, 37, 37, 38, 39, 38, 38, 39,
    38, 38, 39, 41, 43, 42, 40, 39,
    40, 41, 42, 43, 44, 43, 42, 42,
    43, 45, 48, 49, 49, 48, 47, 47,
    48, 51, 52, 51, 52, 54, 56, 57,
    56, 56, 58, 59, 59, 60, 62, 63,
];

/// Chroma weights for CID 1237, 1253, 1259, 1273, 1274
static CHROMA_WEIGHT_1237: [u8; 64] = [
    0, 32, 36, 38, 38, 40, 41, 40,
    40, 40, 42, 42, 42, 43, 44, 48,
    46, 43, 42, 45, 48, 51, 49, 47,
    48, 51, 54, 58, 64, 66, 62, 59,
    57, 58, 61, 64, 67, 68, 67, 66,
    69, 78, 85, 86, 87, 85, 83, 82,
    86, 91, 93, 93, 96, 97, 99, 100,
    100, 104, 108, 105, 100, 100, 102, 102,
];

/// Chroma weights for CID 1238, 1272
static CHROMA_WEIGHT_1238: [u8; 64] = [
    0, 32, 33, 34, 35, 35, 35, 35,
    34, 35, 35, 36, 37, 37, 38, 40,
    39, 38, 37, 38, 40, 39, 38, 37,
    38, 39, 40, 43, 44, 43, 41, 40,
    40, 40, 41, 43, 44, 43, 41, 41,
    42, 43, 45, 46, 46, 46, 46, 45,
    46, 48, 49, 49, 50, 51, 52, 53,
    53, 55, 57, 58, 59, 60, 59, 58,
];

/// Chroma weights for CID 1241, 1271
static CHROMA_WEIGHT_1241: [u8; 64] = [
    0, 32, 35, 36, 36, 37, 38, 39,
    38, 39, 40, 40, 40, 41, 41, 42,
    42, 40, 40, 41, 40, 39, 41, 43,
    43, 44, 45, 47, 47, 48, 49, 48,
    47, 45, 41, 39, 39, 42, 46, 47,
    47, 48, 48, 48, 49, 49, 48, 46,
    44, 45, 47, 49, 50, 51, 52, 51,
    50, 48, 49, 50, 50, 51, 51, 51,
];

// DC Huffman codes

/// DC codes for CID group 1 (1235, 1236, 1241, 1250, 1256, 1257, 1270, 1271)
static DC_CODES_GROUP1: [DcCode; 14] = [
    DcCode { code: 10, bits: 4 },
    DcCode { code: 62, bits: 6 },
    DcCode { code: 11, bits: 4 },
    DcCode { code: 12, bits: 4 },
    DcCode { code: 13, bits: 4 },
    DcCode { code: 0, bits: 3 },
    DcCode { code: 1, bits: 3 },
    DcCode { code: 2, bits: 3 },
    DcCode { code: 3, bits: 3 },
    DcCode { code: 4, bits: 3 },
    DcCode { code: 14, bits: 4 },
    DcCode { code: 30, bits: 5 },
    DcCode { code: 126, bits: 7 },
    DcCode { code: 127, bits: 7 },
];

/// DC codes for CID group 2 (1237, 1238, 1242, 1243, 1251, 1252, 1253, 1258, 1259, 1260, 1272, 1273, 1274)
static DC_CODES_GROUP2: [DcCode; 12] = [
    DcCode { code: 0, bits: 3 },
    DcCode { code: 12, bits: 4 },
    DcCode { code: 13, bits: 4 },
    DcCode { code: 1, bits: 3 },
    DcCode { code: 2, bits: 3 },
    DcCode { code: 3, bits: 3 },
    DcCode { code: 4, bits: 3 },
    DcCode { code: 5, bits: 3 },
    DcCode { code: 14, bits: 4 },
    DcCode { code: 30, bits: 5 },
    DcCode { code: 62, bits: 6 },
    DcCode { code: 63, bits: 6 },
];

// Run-length codes

/// Run codes for CID group (1235, 1238, 1241, 1243, 1256, 1270, 1271, 1272)
static RUN_CODES_GROUP1: [RunCode; 62] = [
    RunCode { code: 0, bits: 1 },
    RunCode { code: 4, bits: 3 },
    RunCode { code: 10, bits: 4 },
    RunCode { code: 11, bits: 4 },
    RunCode { code: 24, bits: 5 },
    RunCode { code: 25, bits: 5 },
    RunCode { code: 26, bits: 5 },
    RunCode { code: 27, bits: 5 },
    RunCode { code: 56, bits: 6 },
    RunCode { code: 57, bits: 6 },
    RunCode { code: 58, bits: 6 },
    RunCode { code: 59, bits: 6 },
    RunCode { code: 120, bits: 7 },
    RunCode { code: 242, bits: 8 },
    RunCode { code: 486, bits: 9 },
    RunCode { code: 487, bits: 9 },
    RunCode { code: 488, bits: 9 },
    RunCode { code: 489, bits: 9 },
    RunCode { code: 490, bits: 9 },
    RunCode { code: 491, bits: 9 },
    RunCode { code: 492, bits: 9 },
    RunCode { code: 493, bits: 9 },
    RunCode { code: 494, bits: 9 },
    RunCode { code: 495, bits: 9 },
    RunCode { code: 496, bits: 9 },
    RunCode { code: 497, bits: 9 },
    RunCode { code: 498, bits: 9 },
    RunCode { code: 499, bits: 9 },
    RunCode { code: 500, bits: 9 },
    RunCode { code: 501, bits: 9 },
    RunCode { code: 502, bits: 9 },
    RunCode { code: 503, bits: 9 },
    RunCode { code: 504, bits: 9 },
    RunCode { code: 505, bits: 9 },
    RunCode { code: 506, bits: 9 },
    RunCode { code: 507, bits: 9 },
    RunCode { code: 508, bits: 9 },
    RunCode { code: 509, bits: 9 },
    RunCode { code: 510, bits: 9 },
    RunCode { code: 1022, bits: 10 },
    RunCode { code: 1023, bits: 10 },
    RunCode { code: 1024, bits: 10 },
    RunCode { code: 1025, bits: 10 },
    RunCode { code: 1026, bits: 10 },
    RunCode { code: 1027, bits: 10 },
    RunCode { code: 1028, bits: 10 },
    RunCode { code: 1029, bits: 10 },
    RunCode { code: 1030, bits: 10 },
    RunCode { code: 1031, bits: 10 },
    RunCode { code: 1032, bits: 10 },
    RunCode { code: 1033, bits: 10 },
    RunCode { code: 1034, bits: 10 },
    RunCode { code: 1035, bits: 10 },
    RunCode { code: 1036, bits: 10 },
    RunCode { code: 1037, bits: 10 },
    RunCode { code: 1038, bits: 10 },
    RunCode { code: 1039, bits: 10 },
    RunCode { code: 1040, bits: 10 },
    RunCode { code: 1041, bits: 10 },
    RunCode { code: 1042, bits: 10 },
    RunCode { code: 1043, bits: 10 },
    RunCode { code: 1044, bits: 10 },
];

// Simplified AC VLC table - just a starter set for basic functionality
// Full tables would have 257 entries per CID
static AC_CODES_BASIC: [AcVlcEntry; 8] = [
    AcVlcEntry { code: 0, bits: 2, run: 0, level: 0 },  // EOB
    AcVlcEntry { code: 1, bits: 2, run: 0, level: 1 },
    AcVlcEntry { code: 2, bits: 3, run: 0, level: -1 },
    AcVlcEntry { code: 6, bits: 3, run: 1, level: 1 },
    AcVlcEntry { code: 7, bits: 4, run: 0, level: 2 },
    AcVlcEntry { code: 8, bits: 5, run: 0, level: -2 },
    AcVlcEntry { code: 18, bits: 5, run: 2, level: 1 },
    AcVlcEntry { code: 19, bits: 6, run: 1, level: -1 },
];

impl CidData {
    /// Get CID data for a given profile
    pub fn for_profile(profile: DnxhdProfile) -> &'static CidData {
        match profile {
            DnxhdProfile::Dnxhd36 | DnxhdProfile::Dnxhd145 => &CID_1235,
            DnxhdProfile::Dnxhd45 | DnxhdProfile::Dnxhd115 => &CID_1237,
            DnxhdProfile::Dnxhd75 | DnxhdProfile::Dnxhd120 => &CID_1238,
            DnxhdProfile::Dnxhd175 => &CID_1241,
            DnxhdProfile::Dnxhd185 => &CID_1242,
            DnxhdProfile::Dnxhd220 => &CID_1243,
            DnxhdProfile::DnxhrLb => &CID_1250,
            DnxhdProfile::DnxhrSq => &CID_1251,
            DnxhdProfile::DnxhrHq => &CID_1252,
            DnxhdProfile::DnxhrHqx => &CID_1253,
            DnxhdProfile::Dnxhr444 => &CID_1270,
        }
    }
}

// CID profile definitions

static CID_1235: CidData = CidData {
    cid: 1235,
    width: 1920,
    height: 1080,
    bit_depth: 8,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1235,
    chroma_weight: &CHROMA_WEIGHT_1235,
    dc_codes: &DC_CODES_GROUP1,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1237: CidData = CidData {
    cid: 1237,
    width: 1920,
    height: 1080,
    bit_depth: 8,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1237,
    chroma_weight: &CHROMA_WEIGHT_1237,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1238: CidData = CidData {
    cid: 1238,
    width: 1920,
    height: 1080,
    bit_depth: 8,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1238,
    chroma_weight: &CHROMA_WEIGHT_1238,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1241: CidData = CidData {
    cid: 1241,
    width: 1920,
    height: 1080,
    bit_depth: 10,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1241,
    chroma_weight: &CHROMA_WEIGHT_1241,
    dc_codes: &DC_CODES_GROUP1,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1242: CidData = CidData {
    cid: 1242,
    width: 1920,
    height: 1080,
    bit_depth: 10,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1237,
    chroma_weight: &CHROMA_WEIGHT_1237,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1243: CidData = CidData {
    cid: 1243,
    width: 1920,
    height: 1080,
    bit_depth: 10,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1237,
    chroma_weight: &CHROMA_WEIGHT_1237,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1250: CidData = CidData {
    cid: 1250,
    width: 1920,
    height: 1080,
    bit_depth: 8,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1235,
    chroma_weight: &CHROMA_WEIGHT_1235,
    dc_codes: &DC_CODES_GROUP1,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1251: CidData = CidData {
    cid: 1251,
    width: 1920,
    height: 1080,
    bit_depth: 8,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1237,
    chroma_weight: &CHROMA_WEIGHT_1237,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1252: CidData = CidData {
    cid: 1252,
    width: 1920,
    height: 1080,
    bit_depth: 8,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1237,
    chroma_weight: &CHROMA_WEIGHT_1237,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1253: CidData = CidData {
    cid: 1253,
    width: 1920,
    height: 1080,
    bit_depth: 10,
    is_444: false,
    luma_weight: &LUMA_WEIGHT_1237,
    chroma_weight: &CHROMA_WEIGHT_1237,
    dc_codes: &DC_CODES_GROUP2,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

static CID_1270: CidData = CidData {
    cid: 1270,
    width: 1920,
    height: 1080,
    bit_depth: 10,
    is_444: true,
    luma_weight: &LUMA_WEIGHT_1235,
    chroma_weight: &CHROMA_WEIGHT_1235,
    dc_codes: &DC_CODES_GROUP1,
    ac_codes: &AC_CODES_BASIC,
    run_codes: &RUN_CODES_GROUP1,
};

/// Zigzag scan order for 8x8 blocks
pub static ZIGZAG_SCAN: [usize; 64] = [
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cid_data_lookup() {
        let data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        assert_eq!(data.cid, 1252);
        assert_eq!(data.bit_depth, 8);
        assert!(!data.is_444);
    }

    #[test]
    fn test_zigzag_scan() {
        assert_eq!(ZIGZAG_SCAN[0], 0);
        assert_eq!(ZIGZAG_SCAN[1], 1);
        assert_eq!(ZIGZAG_SCAN[63], 63);
    }
}
