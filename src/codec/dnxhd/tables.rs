//! DNxHD/DNxHR quantization and VLC tables
//!
//! These tables are sourced from the public FFmpeg implementation (LGPL).
//! They enable a pure-Rust decoder/encoder without linking against FFmpeg.
//!
//! DNxHD uses different quantization weights for luma and chroma channels,
//! with weights organized in zigzag scan order. The VLC tables use Huffman
//! coding for DC/AC coefficients and run-length encoding.

use super::DnxhdProfile;

// =============================================================================
// ZIGZAG SCAN TABLES
// =============================================================================

/// Standard JPEG/MPEG zigzag scan order (8x8 block)
/// DNxHD uses this standard zigzag pattern for coefficient ordering
pub const DNXHD_ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag scan (position to zigzag index)
pub const DNXHD_ZIGZAG_INVERSE: [u8; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

// =============================================================================
// QUANTIZATION WEIGHT TABLES (LUMA)
// =============================================================================
// These are weight tables, not quantization matrices. The actual quantization
// values are computed from: qmat[i] = weight[i] * scale_factor
// Values are in zigzag order.

/// Luma quantization weights for CID 1235, 1256 (DNxHD 36/145 Mbps, DNxHR variants)
pub const DNXHD_1235_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 32, 32, 33, 32, 32, 32, 32, 31, 32, 33, 33, 33, 33, 35, 36, 36, 34, 34, 36, 37, 37, 36,
    36, 35, 36, 38, 39, 39, 37, 36, 37, 37, 39, 41, 42, 41, 39, 39, 40, 41, 42, 43, 42, 42, 41, 41,
    41, 44, 47, 46, 46, 48, 51, 51, 50, 50, 53, 55, 55, 56, 60, 60,
];

/// Chroma quantization weights for CID 1235, 1256
pub const DNXHD_1235_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 33, 34, 34, 33, 34, 35, 37, 40, 43, 42, 39, 38, 39, 41, 43, 44, 47, 50, 55, 61, 63, 56,
    48, 46, 49, 54, 59, 58, 55, 58, 63, 65, 67, 74, 84, 82, 75, 72, 70, 74, 84, 87, 87, 94, 93, 81,
    75, 78, 83, 89, 91, 86, 82, 85, 90, 90, 85, 79, 73, 73, 73, 73,
];

/// Luma quantization weights for CID 1237, 1253, 1259, 1273, 1274
/// (DNxHD 45/115 Mbps, DNxHR HQX, etc.)
pub const DNXHD_1237_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 33, 34, 34, 36, 37, 36, 36, 37, 38, 38, 38, 39, 41, 44, 43, 41, 40, 41, 46, 49, 47, 46,
    47, 49, 51, 54, 60, 62, 59, 55, 54, 56, 58, 61, 65, 66, 64, 63, 66, 73, 78, 79, 80, 79, 78, 78,
    82, 87, 89, 90, 93, 95, 96, 97, 97, 100, 104, 102, 98, 98, 99, 99,
];

/// Chroma quantization weights for CID 1237, 1253, 1259, 1273, 1274
pub const DNXHD_1237_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 36, 39, 39, 38, 39, 41, 45, 51, 57, 58, 53, 48, 47, 51, 55, 58, 66, 75, 81, 83, 82, 78,
    73, 72, 74, 77, 83, 85, 83, 82, 89, 99, 96, 90, 94, 97, 99, 105, 109, 105, 95, 89, 92, 95, 94,
    93, 92, 88, 89, 90, 93, 95, 96, 97, 97, 100, 104, 102, 98, 98, 99, 99,
];

/// Luma quantization weights for CID 1238, 1272 (DNxHD 75/120 Mbps)
pub const DNXHD_1238_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 32, 33, 34, 33, 33, 33, 33, 33, 33, 33, 33, 35, 37, 37, 36, 36, 35, 36, 38, 38, 36, 35,
    36, 37, 38, 41, 42, 41, 39, 38, 38, 38, 39, 41, 42, 41, 39, 39, 40, 41, 43, 44, 44, 44, 44, 44,
    45, 47, 47, 47, 49, 50, 51, 51, 51, 53, 55, 57, 58, 59, 57, 57,
];

/// Chroma quantization weights for CID 1238, 1272
pub const DNXHD_1238_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 35, 35, 35, 34, 34, 35, 39, 43, 45, 45, 41, 39, 40, 41, 42, 44, 48, 55, 59, 63, 65, 59,
    53, 52, 52, 55, 61, 62, 58, 58, 63, 66, 66, 65, 70, 74, 70, 66, 65, 68, 75, 77, 74, 74, 77, 76,
    73, 73, 73, 73, 76, 80, 89, 90, 82, 77, 80, 86, 84, 82, 82, 82,
];

/// Luma quantization weights for CID 1241, 1271 (DNxHD 175 Mbps 10-bit)
pub const DNXHD_1241_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 33, 34, 34, 35, 36, 37, 36, 37, 38, 38, 38, 39, 39, 40, 40, 38, 38, 39, 38, 37, 39, 41,
    41, 42, 43, 45, 45, 46, 47, 46, 45, 43, 39, 37, 37, 40, 44, 45, 45, 46, 46, 46, 47, 47, 46, 44,
    42, 43, 45, 47, 48, 49, 50, 49, 48, 46, 47, 48, 48, 49, 49, 49,
];

/// Chroma quantization weights for CID 1241, 1271
pub const DNXHD_1241_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 36, 38, 37, 37, 40, 41, 40, 40, 42, 42, 41, 41, 41, 41, 42, 43, 44, 44, 45, 46, 46, 45,
    44, 45, 45, 45, 45, 46, 47, 46, 45, 44, 42, 41, 43, 45, 45, 47, 48, 48, 48, 46, 47, 47, 46, 47,
    46, 45, 45, 47, 48, 49, 50, 49, 48, 46, 48, 49, 48, 49, 49, 49,
];

/// Luma quantization weights for CID 1242 (DNxHD 185 Mbps 10-bit)
pub const DNXHD_1242_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 33, 33, 34, 35, 36, 35, 33, 33, 35, 36, 37, 37, 38, 37, 37, 37, 36, 37, 37, 37, 38, 39,
    37, 36, 37, 40, 42, 45, 46, 44, 41, 42, 44, 45, 47, 49, 50, 48, 46, 48, 49, 50, 52, 52, 50, 49,
    47, 48, 50, 50, 51, 51, 50, 49, 49, 51, 52, 51, 49, 47, 47, 47,
];

/// Chroma quantization weights for CID 1242
pub const DNXHD_1242_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 35, 36, 36, 35, 37, 38, 39, 40, 41, 41, 40, 39, 40, 40, 40, 40, 40, 41, 42, 43, 44, 43,
    42, 41, 42, 44, 46, 48, 49, 48, 46, 47, 47, 48, 50, 51, 52, 51, 50, 51, 52, 52, 53, 53, 52, 51,
    50, 50, 51, 51, 52, 52, 51, 50, 50, 51, 52, 51, 49, 47, 47, 47,
];

/// Luma quantization weights for CID 1243 (DNxHD 220 Mbps 10-bit)
pub const DNXHD_1243_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 32, 33, 33, 35, 35, 35, 35, 35, 35, 35, 34, 35, 38, 40, 39, 37, 37, 37, 36, 35, 36, 38,
    40, 41, 42, 44, 45, 44, 42, 41, 40, 38, 36, 36, 37, 38, 40, 43, 44, 45, 45, 45, 45, 45, 45, 41,
    39, 41, 45, 47, 47, 48, 48, 48, 46, 44, 45, 47, 47, 48, 47, 47,
];

/// Chroma quantization weights for CID 1243
pub const DNXHD_1243_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 36, 37, 36, 37, 39, 39, 41, 43, 43, 42, 41, 41, 41, 42, 43, 43, 43, 44, 44, 44, 46, 47,
    46, 45, 45, 45, 45, 46, 44, 44, 45, 44, 42, 41, 43, 46, 45, 44, 45, 45, 45, 46, 46, 46, 45, 44,
    45, 44, 45, 47, 47, 48, 49, 48, 46, 45, 46, 47, 47, 48, 47, 47,
];

/// Luma quantization weights for CID 1250 (DNxHR LB)
pub const DNXHD_1250_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 32, 33, 34, 35, 35, 35, 34, 34, 35, 36, 36, 36, 36, 36, 37, 38, 38, 38, 38, 38, 39, 39,
    38, 38, 39, 41, 43, 43, 42, 41, 40, 40, 39, 40, 41, 41, 39, 39, 40, 42, 47, 50, 47, 45, 46, 46,
    44, 45, 46, 47, 49, 54, 58, 54, 48, 49, 54, 57, 60, 62, 63, 63,
];

/// Chroma quantization weights for CID 1250
pub const DNXHD_1250_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 35, 35, 35, 34, 35, 36, 39, 43, 46, 45, 42, 40, 41, 41, 43, 45, 49, 55, 60, 64, 65, 60,
    54, 53, 53, 56, 61, 63, 60, 59, 64, 67, 68, 67, 72, 75, 72, 68, 67, 70, 77, 79, 76, 76, 78, 78,
    75, 75, 75, 75, 78, 82, 90, 91, 84, 79, 82, 88, 86, 84, 84, 84,
];

/// Luma quantization weights for CID 1251 (DNxHR SQ)
pub const DNXHD_1251_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 32, 34, 34, 34, 34, 35, 35, 35, 36, 37, 36, 36, 35, 36, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 39, 41, 44, 43, 41, 40, 40, 40, 40, 39, 40, 41, 40, 39, 40, 43, 46, 46, 44, 44, 44, 42,
    41, 43, 46, 48, 50, 55, 58, 53, 48, 50, 55, 58, 61, 62, 62, 62,
];

/// Chroma quantization weights for CID 1251
pub const DNXHD_1251_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 35, 36, 36, 35, 37, 38, 39, 41, 42, 42, 41, 40, 41, 41, 41, 42, 43, 45, 47, 50, 52, 50,
    48, 47, 47, 49, 52, 54, 52, 51, 55, 57, 57, 56, 60, 63, 60, 57, 56, 58, 64, 66, 63, 63, 65, 64,
    62, 62, 62, 62, 65, 68, 75, 76, 70, 66, 68, 73, 72, 70, 70, 70,
];

/// Luma quantization weights for CID 1252 (DNxHR HQ)
pub const DNXHD_1252_LUMA_WEIGHT: [u8; 64] = [
    0, 32, 34, 35, 36, 36, 36, 37, 36, 37, 39, 40, 41, 40, 40, 40, 41, 41, 42, 41, 41, 43, 44, 44,
    45, 46, 48, 55, 60, 57, 52, 50, 49, 49, 52, 52, 53, 55, 58, 62, 65, 73, 82, 82, 80, 78, 73, 68,
    71, 82, 90, 90, 88, 87, 90, 95, 100, 107, 103, 97, 95, 93, 99, 99,
];

/// Chroma quantization weights for CID 1252
pub const DNXHD_1252_CHROMA_WEIGHT: [u8; 64] = [
    0, 32, 37, 39, 40, 39, 41, 43, 45, 49, 54, 54, 51, 48, 49, 51, 53, 57, 64, 73, 80, 84, 86, 80,
    74, 73, 75, 79, 85, 89, 87, 85, 92, 103, 101, 96, 100, 106, 109, 115, 120, 118, 108, 103, 106,
    111, 110, 109, 108, 105, 106, 109, 113, 117, 121, 123, 124, 130, 136, 134, 130, 130, 132, 132,
];

// =============================================================================
// DC VLC TABLES
// =============================================================================

/// DC coefficient codes for CID 1235, 1250, 1270 (14 entries)
pub const DNXHD_1235_DC_CODES: [u16; 14] = [10, 62, 11, 12, 13, 0, 1, 2, 3, 4, 14, 30, 126, 127];

/// DC coefficient bit lengths for CID 1235, 1250, 1270
pub const DNXHD_1235_DC_BITS: [u8; 14] = [4, 6, 4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 7, 7];

/// DC coefficient codes for CID 1237, 1238, 1241, 1242, 1243, 1251, 1252, 1253 (12 entries)
pub const DNXHD_1237_DC_CODES: [u16; 12] = [0, 12, 13, 1, 2, 3, 4, 5, 14, 30, 62, 63];

/// DC coefficient bit lengths for CID 1237, etc.
pub const DNXHD_1237_DC_BITS: [u8; 12] = [3, 4, 4, 3, 3, 3, 3, 3, 4, 5, 6, 6];

// =============================================================================
// RUN LENGTH ENCODING TABLES
// =============================================================================

/// Run-length codes for CID 1235 (62 entries)
pub const DNXHD_1235_RUN_CODES: [u16; 62] = [
    0, 4, 10, 11, 24, 25, 26, 27, 56, 57, 58, 59, 120, 242, 486, 487, 488, 489, 980, 981, 982, 983,
    984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001,
    1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
    1018, 1019, 1020, 1021, 1022, 1023,
];

/// Run-length bit lengths for CID 1235
pub const DNXHD_1235_RUN_BITS: [u8; 62] = [
    1, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
];

/// Run value mapping for CID 1235
pub const DNXHD_1235_RUN: [u8; 62] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 17, 19, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
];

/// Run-length codes for CID 1237 (62 entries)
pub const DNXHD_1237_RUN_CODES: [u16; 62] = [
    0, 4, 10, 11, 24, 25, 26, 54, 55, 56, 57, 58, 118, 119, 240, 482, 483, 484, 485, 486, 487, 488,
    489, 490, 491, 492, 493, 494, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001,
    1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
    1018, 1019, 1020, 1021, 1022, 1023,
];

/// Run-length bit lengths for CID 1237
pub const DNXHD_1237_RUN_BITS: [u8; 62] = [
    1, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10,
];

/// Run value mapping for CID 1237
pub const DNXHD_1237_RUN: [u8; 62] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 53, 57, 58, 59, 60,
    61, 62, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56,
];

/// Run-length codes for CID 1238 (62 entries)
pub const DNXHD_1238_RUN_CODES: [u16; 62] = [
    0, 4, 10, 11, 24, 25, 26, 27, 56, 57, 58, 59, 120, 121, 244, 490, 491, 492, 493, 988, 989, 990,
    991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
    1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023,
    1024, 1025, 1026, 1027, 1028, 1029, 1030,
];

/// Run-length bit lengths for CID 1238
pub const DNXHD_1238_RUN_BITS: [u8; 62] = [
    1, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
];

/// Run value mapping for CID 1238
pub const DNXHD_1238_RUN: [u8; 62] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 17, 18, 19, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
];

/// Run-length codes for CID 1250, 1251, 1252 (62 entries)
pub const DNXHD_1250_RUN_CODES: [u16; 62] = [
    0, 4, 5, 12, 26, 27, 28, 58, 118, 119, 120, 242, 486, 487, 976, 977, 978, 979, 980, 981, 982,
    983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000,
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016,
    1017, 1018, 1019, 1020, 1021, 1022, 1023,
];

/// Run-length bit lengths for CID 1250, 1251, 1252
pub const DNXHD_1250_RUN_BITS: [u8; 62] = [
    1, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
];

/// Run value mapping for CID 1250
pub const DNXHD_1250_RUN: [u8; 62] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 17, 18, 19, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
];

// =============================================================================
// AC VLC TABLES
// =============================================================================
// AC tables are large (257 entries each). These are compact representations.
// The ac_info array contains encoded run-level pairs.

/// Number of AC codes per table
pub const DNXHD_AC_CODES_COUNT: usize = 257;

/// AC codes for CID 1235 (representative subset - first 64 entries)
/// Full table has 257 entries. Values encode Huffman codes.
pub const DNXHD_1235_AC_CODES: [u16; 64] = [
    0, 1, 4, 5, 12, 26, 27, 56, 57, 58, 59, 120, 121, 244, 245, 246, 247, 248, 498, 499, 500, 501,
    502, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033,
    4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 8150, 8151, 8152, 8153, 8154, 8155,
    8156, 8157, 8158, 8159, 16318, 16319, 16320, 16321, 16322, 16323,
];

/// AC bit lengths for CID 1235 (first 64 entries)
pub const DNXHD_1235_AC_BITS: [u8; 64] = [
    2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
    10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
];

/// AC info (run-level encoding) for CID 1235 (first 64 entries)
/// Each entry encodes (run << 4) | level or special markers
pub const DNXHD_1235_AC_INFO: [u8; 64] = [
    0x03, 0x00, 0x03, 0x02, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00, 0x09, 0x00, 0x0B, 0x00, 0x05, 0x02,
    0x0D, 0x00, 0x0F, 0x00, 0x11, 0x00, 0x07, 0x02, 0x13, 0x00, 0x15, 0x00, 0x17, 0x00, 0x09, 0x02,
    0x19, 0x00, 0x1B, 0x00, 0x1D, 0x00, 0x0B, 0x02, 0x1F, 0x00, 0x21, 0x00, 0x23, 0x00, 0x0D, 0x02,
    0x25, 0x00, 0x27, 0x00, 0x29, 0x00, 0x0F, 0x02, 0x2B, 0x00, 0x2D, 0x00, 0x2F, 0x00, 0x11, 0x02,
];

/// AC codes for CID 1237 (first 64 entries)
pub const DNXHD_1237_AC_CODES: [u16; 64] = [
    0, 1, 4, 5, 12, 26, 27, 56, 57, 58, 59, 120, 121, 244, 245, 246, 247, 248, 498, 499, 500, 501,
    502, 1006, 1007, 1008, 1009, 1010, 1011, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 4062,
    4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 8142, 8143, 8144, 8145, 8146, 8147, 8148, 8149,
    8150, 8151, 16292, 16293, 16294, 16295, 16296, 16297, 16298, 16299,
];

/// AC bit lengths for CID 1237 (first 64 entries)
pub const DNXHD_1237_AC_BITS: [u8; 64] = [
    2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
    11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
];

/// AC info for CID 1237 (first 64 entries)
pub const DNXHD_1237_AC_INFO: [u8; 64] = [
    0x03, 0x00, 0x03, 0x02, 0x05, 0x00, 0x00, 0x00, 0x07, 0x00, 0x09, 0x00, 0x05, 0x02, 0x0B, 0x00,
    0x0D, 0x00, 0x0F, 0x00, 0x07, 0x02, 0x11, 0x00, 0x13, 0x00, 0x15, 0x00, 0x09, 0x02, 0x17, 0x00,
    0x19, 0x00, 0x1B, 0x00, 0x0B, 0x02, 0x1D, 0x00, 0x1F, 0x00, 0x21, 0x00, 0x0D, 0x02, 0x23, 0x00,
    0x25, 0x00, 0x27, 0x00, 0x0F, 0x02, 0x29, 0x00, 0x2B, 0x00, 0x2D, 0x00, 0x11, 0x02, 0x2F, 0x00,
];

/// AC codes for CID 1238 (first 64 entries)
pub const DNXHD_1238_AC_CODES: [u16; 64] = [
    0, 1, 4, 5, 12, 26, 27, 56, 57, 58, 59, 120, 121, 244, 245, 246, 247, 248, 498, 499, 500, 501,
    502, 503, 1008, 1009, 1010, 1011, 1012, 1013, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035,
    4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 8164, 8165, 8166, 8167, 8168, 8169,
    8170, 8171, 8172, 8173, 16348, 16349, 16350, 16351, 16352, 16353,
];

/// AC bit lengths for CID 1238 (first 64 entries)
pub const DNXHD_1238_AC_BITS: [u8; 64] = [
    2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
    11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
];

/// AC info for CID 1238 (first 64 entries)
pub const DNXHD_1238_AC_INFO: [u8; 64] = [
    0x03, 0x00, 0x03, 0x02, 0x05, 0x00, 0x00, 0x00, 0x07, 0x00, 0x09, 0x00, 0x05, 0x02, 0x0B, 0x00,
    0x0D, 0x00, 0x0F, 0x00, 0x07, 0x02, 0x11, 0x00, 0x13, 0x00, 0x15, 0x00, 0x09, 0x02, 0x17, 0x00,
    0x19, 0x00, 0x1B, 0x00, 0x0B, 0x02, 0x1D, 0x00, 0x1F, 0x00, 0x21, 0x00, 0x0D, 0x02, 0x23, 0x00,
    0x25, 0x00, 0x27, 0x00, 0x0F, 0x02, 0x29, 0x00, 0x2B, 0x00, 0x2D, 0x00, 0x11, 0x02, 0x2F, 0x00,
];

// =============================================================================
// PROFILE TABLE LOOKUP
// =============================================================================

/// Profile-to-weight table mapping
#[derive(Debug, Clone, Copy)]
pub struct DnxhdProfileTables {
    pub luma_weight: &'static [u8; 64],
    pub chroma_weight: &'static [u8; 64],
    pub dc_codes: &'static [u16],
    pub dc_bits: &'static [u8],
    pub run_codes: &'static [u16; 62],
    pub run_bits: &'static [u8; 62],
    pub run: &'static [u8; 62],
}

impl DnxhdProfileTables {
    /// Get tables for a specific profile
    pub fn for_profile(profile: DnxhdProfile) -> Self {
        match profile {
            DnxhdProfile::Dnxhd36 | DnxhdProfile::Dnxhd145 => DnxhdProfileTables {
                luma_weight: &DNXHD_1235_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1235_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1235_DC_CODES,
                dc_bits: &DNXHD_1235_DC_BITS,
                run_codes: &DNXHD_1235_RUN_CODES,
                run_bits: &DNXHD_1235_RUN_BITS,
                run: &DNXHD_1235_RUN,
            },
            DnxhdProfile::Dnxhd45 | DnxhdProfile::Dnxhd115 | DnxhdProfile::DnxhrHqx => {
                DnxhdProfileTables {
                    luma_weight: &DNXHD_1237_LUMA_WEIGHT,
                    chroma_weight: &DNXHD_1237_CHROMA_WEIGHT,
                    dc_codes: &DNXHD_1237_DC_CODES,
                    dc_bits: &DNXHD_1237_DC_BITS,
                    run_codes: &DNXHD_1237_RUN_CODES,
                    run_bits: &DNXHD_1237_RUN_BITS,
                    run: &DNXHD_1237_RUN,
                }
            }
            DnxhdProfile::Dnxhd75 | DnxhdProfile::Dnxhd120 => DnxhdProfileTables {
                luma_weight: &DNXHD_1238_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1238_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1237_DC_CODES,
                dc_bits: &DNXHD_1237_DC_BITS,
                run_codes: &DNXHD_1238_RUN_CODES,
                run_bits: &DNXHD_1238_RUN_BITS,
                run: &DNXHD_1238_RUN,
            },
            DnxhdProfile::Dnxhd175 => DnxhdProfileTables {
                luma_weight: &DNXHD_1241_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1241_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1237_DC_CODES,
                dc_bits: &DNXHD_1237_DC_BITS,
                run_codes: &DNXHD_1237_RUN_CODES,
                run_bits: &DNXHD_1237_RUN_BITS,
                run: &DNXHD_1237_RUN,
            },
            DnxhdProfile::Dnxhd185 => DnxhdProfileTables {
                luma_weight: &DNXHD_1242_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1242_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1237_DC_CODES,
                dc_bits: &DNXHD_1237_DC_BITS,
                run_codes: &DNXHD_1237_RUN_CODES,
                run_bits: &DNXHD_1237_RUN_BITS,
                run: &DNXHD_1237_RUN,
            },
            DnxhdProfile::Dnxhd220 => DnxhdProfileTables {
                luma_weight: &DNXHD_1243_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1243_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1237_DC_CODES,
                dc_bits: &DNXHD_1237_DC_BITS,
                run_codes: &DNXHD_1237_RUN_CODES,
                run_bits: &DNXHD_1237_RUN_BITS,
                run: &DNXHD_1237_RUN,
            },
            DnxhdProfile::DnxhrLb => DnxhdProfileTables {
                luma_weight: &DNXHD_1250_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1250_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1235_DC_CODES,
                dc_bits: &DNXHD_1235_DC_BITS,
                run_codes: &DNXHD_1250_RUN_CODES,
                run_bits: &DNXHD_1250_RUN_BITS,
                run: &DNXHD_1250_RUN,
            },
            DnxhdProfile::DnxhrSq => DnxhdProfileTables {
                luma_weight: &DNXHD_1251_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1251_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1237_DC_CODES,
                dc_bits: &DNXHD_1237_DC_BITS,
                run_codes: &DNXHD_1250_RUN_CODES,
                run_bits: &DNXHD_1250_RUN_BITS,
                run: &DNXHD_1250_RUN,
            },
            DnxhdProfile::DnxhrHq => DnxhdProfileTables {
                luma_weight: &DNXHD_1252_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1252_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1237_DC_CODES,
                dc_bits: &DNXHD_1237_DC_BITS,
                run_codes: &DNXHD_1250_RUN_CODES,
                run_bits: &DNXHD_1250_RUN_BITS,
                run: &DNXHD_1250_RUN,
            },
            DnxhdProfile::Dnxhr444 => DnxhdProfileTables {
                luma_weight: &DNXHD_1235_LUMA_WEIGHT,
                chroma_weight: &DNXHD_1235_CHROMA_WEIGHT,
                dc_codes: &DNXHD_1235_DC_CODES,
                dc_bits: &DNXHD_1235_DC_BITS,
                run_codes: &DNXHD_1235_RUN_CODES,
                run_bits: &DNXHD_1235_RUN_BITS,
                run: &DNXHD_1235_RUN,
            },
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get the quantization scale factor range for a profile
pub fn get_quantization_scale_range(profile: DnxhdProfile) -> (u8, u8) {
    if profile.is_10bit() {
        (1, 63) // 10-bit profiles use wider range
    } else {
        (1, 31) // 8-bit profiles
    }
}

/// Convert zigzag index to raster index
#[inline]
pub fn zigzag_to_raster(zigzag_idx: usize) -> usize {
    DNXHD_ZIGZAG[zigzag_idx] as usize
}

/// Convert raster index to zigzag index
#[inline]
pub fn raster_to_zigzag(raster_idx: usize) -> usize {
    DNXHD_ZIGZAG_INVERSE[raster_idx] as usize
}

/// Compute quantization matrix from weights and scale factor
pub fn compute_quant_matrix(weights: &[u8; 64], scale: u8) -> [u16; 64] {
    let mut matrix = [0u16; 64];
    for (i, &weight) in weights.iter().enumerate() {
        // Quantization value = weight * scale / 32 (with rounding)
        // First entry (DC) uses special handling
        if i == 0 {
            matrix[i] = scale as u16;
        } else {
            let val = ((weight as u32 * scale as u32) + 16) / 32;
            matrix[i] = val.max(1) as u16;
        }
    }
    matrix
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        for i in 0..64 {
            let zigzag_idx = raster_to_zigzag(i);
            let raster_idx = zigzag_to_raster(zigzag_idx);
            assert_eq!(raster_idx, i);
        }
    }

    #[test]
    fn test_zigzag_first_values() {
        // First row of zigzag should start at (0,0), then (0,1), (1,0)...
        assert_eq!(DNXHD_ZIGZAG[0], 0); // (0,0)
        assert_eq!(DNXHD_ZIGZAG[1], 1); // (0,1)
        assert_eq!(DNXHD_ZIGZAG[2], 8); // (1,0)
        assert_eq!(DNXHD_ZIGZAG[3], 16); // (2,0)
    }

    #[test]
    fn test_profile_tables_lookup() {
        let tables = DnxhdProfileTables::for_profile(DnxhdProfile::Dnxhd115);
        assert_eq!(tables.luma_weight.len(), 64);
        assert_eq!(tables.chroma_weight.len(), 64);
        assert!(tables.dc_codes.len() >= 12);
    }

    #[test]
    fn test_weight_table_first_zero() {
        // All weight tables should have first entry as 0 (DC handled separately)
        assert_eq!(DNXHD_1235_LUMA_WEIGHT[0], 0);
        assert_eq!(DNXHD_1237_LUMA_WEIGHT[0], 0);
        assert_eq!(DNXHD_1238_LUMA_WEIGHT[0], 0);
        assert_eq!(DNXHD_1241_LUMA_WEIGHT[0], 0);
        assert_eq!(DNXHD_1250_LUMA_WEIGHT[0], 0);
    }

    #[test]
    fn test_quant_matrix_computation() {
        let matrix = compute_quant_matrix(&DNXHD_1235_LUMA_WEIGHT, 16);
        // DC coefficient should equal scale
        assert_eq!(matrix[0], 16);
        // Other coefficients should be weight * scale / 32
        assert!(matrix[1] > 0);
    }

    #[test]
    fn test_dc_codes_valid() {
        // DC codes should be reasonably sized
        for &code in &DNXHD_1235_DC_CODES {
            assert!(code < 256);
        }
        for &code in &DNXHD_1237_DC_CODES {
            assert!(code < 256);
        }
    }

    #[test]
    fn test_run_bits_monotonic() {
        // Run bits should generally be non-decreasing
        for i in 1..DNXHD_1235_RUN_BITS.len() {
            assert!(DNXHD_1235_RUN_BITS[i] >= DNXHD_1235_RUN_BITS[i - 1].saturating_sub(1));
        }
    }

    #[test]
    fn test_quantization_scale_range() {
        let (min, max) = get_quantization_scale_range(DnxhdProfile::Dnxhd115);
        assert_eq!(min, 1);
        assert_eq!(max, 31);

        let (min, max) = get_quantization_scale_range(DnxhdProfile::Dnxhd220);
        assert_eq!(min, 1);
        assert_eq!(max, 63);
    }

    #[test]
    fn test_all_profiles_have_tables() {
        let profiles = [
            DnxhdProfile::Dnxhd36,
            DnxhdProfile::Dnxhd45,
            DnxhdProfile::Dnxhd75,
            DnxhdProfile::Dnxhd115,
            DnxhdProfile::Dnxhd120,
            DnxhdProfile::Dnxhd145,
            DnxhdProfile::Dnxhd175,
            DnxhdProfile::Dnxhd185,
            DnxhdProfile::Dnxhd220,
            DnxhdProfile::DnxhrLb,
            DnxhdProfile::DnxhrSq,
            DnxhdProfile::DnxhrHq,
            DnxhdProfile::DnxhrHqx,
            DnxhdProfile::Dnxhr444,
        ];

        for profile in profiles {
            let tables = DnxhdProfileTables::for_profile(profile);
            assert_eq!(tables.luma_weight.len(), 64);
            assert_eq!(tables.chroma_weight.len(), 64);
            assert!(tables.dc_codes.len() >= 12);
            assert!(tables.dc_bits.len() >= 12);
            assert_eq!(tables.run_codes.len(), 62);
            assert_eq!(tables.run_bits.len(), 62);
            assert_eq!(tables.run.len(), 62);
        }
    }
}
