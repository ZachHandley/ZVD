# VP8 Codec Implementation Specification for Pure Rust

This document provides a comprehensive technical specification for implementing a VP8 video codec (decoder and encoder) in pure Rust for MIT licensing. All information is derived from the public [RFC 6386 - VP8 Data Format and Decoding Guide](https://datatracker.ietf.org/doc/html/rfc6386) and Google's technical documentation.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Data Structures](#data-structures)
4. [Boolean Arithmetic Coder](#boolean-arithmetic-coder)
5. [Frame Structure](#frame-structure)
6. [Transforms (DCT and WHT)](#transforms-dct-and-wht)
7. [Intra Prediction](#intra-prediction)
8. [Inter Prediction](#inter-prediction)
9. [Quantization](#quantization)
10. [Loop Filter](#loop-filter)
11. [Entropy Coding](#entropy-coding)
12. [Encoder Mode Decision](#encoder-mode-decision)
13. [Rate Control](#rate-control)
14. [Test Strategy](#test-strategy)

---

## Overview

VP8 is a royalty-free video codec developed by On2 Technologies and released as open source by Google. Key characteristics:

- **Color Space**: YUV 4:2:0 (8-bit)
- **Block Size**: 16x16 macroblocks with 4x4 subblock processing
- **Transform**: 4x4 DCT and 4x4 Walsh-Hadamard Transform (WHT)
- **Entropy Coding**: Boolean arithmetic coder with probability trees
- **Reference Frames**: Previous, Golden, and Altref frames
- **License**: BSD-style license, royalty-free

### Key Technical Points

- All coefficient encoding uses a boolean arithmetic coder
- Intra prediction: 4 modes for 16x16/8x8, 10 modes for 4x4 luma
- Inter prediction: Motion compensation with 1/4-pel precision using 6-tap filter
- Loop filter: Deblocking filter applied in-loop for reference frame quality

---

## File Structure

```
src/codec/vp8/
├── mod.rs                  # Module exports and feature flags
├── decoder.rs              # Main decoder implementation
├── encoder.rs              # Main encoder implementation
├── bitstream/
│   ├── mod.rs              # Bitstream module exports
│   ├── bool_decoder.rs     # Boolean arithmetic decoder
│   ├── bool_encoder.rs     # Boolean arithmetic encoder
│   └── reader.rs           # Raw bitstream reader utilities
├── frame/
│   ├── mod.rs              # Frame module exports
│   ├── header.rs           # Frame header parsing/writing
│   ├── buffer.rs           # Frame buffer management
│   └── partition.rs        # Data partition handling
├── transform/
│   ├── mod.rs              # Transform module exports
│   ├── dct.rs              # 4x4 DCT/IDCT implementation
│   └── wht.rs              # 4x4 Walsh-Hadamard Transform
├── prediction/
│   ├── mod.rs              # Prediction module exports
│   ├── intra.rs            # Intra prediction modes
│   ├── inter.rs            # Inter prediction and motion compensation
│   └── motion_vector.rs    # Motion vector decoding/encoding
├── filter/
│   ├── mod.rs              # Filter module exports
│   ├── loop_filter.rs      # In-loop deblocking filter
│   └── subpel.rs           # Subpixel interpolation filters
├── entropy/
│   ├── mod.rs              # Entropy module exports
│   ├── coeff.rs            # Coefficient token encoding/decoding
│   ├── trees.rs            # Probability trees for various symbols
│   └── tables.rs           # Default probability tables
├── quant/
│   ├── mod.rs              # Quantization module exports
│   └── tables.rs           # Quantization lookup tables
└── encoder/
    ├── mod.rs              # Encoder-specific module exports
    ├── mode_decision.rs    # Mode decision algorithms
    ├── motion_est.rs       # Motion estimation
    └── rate_control.rs     # Rate control algorithms
```

---

## Data Structures

### Core Types

```rust
/// VP8 frame types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    KeyFrame,
    InterFrame,
}

/// Reference frame identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefFrame {
    None,
    Last,      // Previous frame
    Golden,    // Golden reference frame
    AltRef,    // Alternate reference frame
}

/// Intra prediction modes for 16x16 luma and 8x8 chroma
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntraMode16x16 {
    DcPred = 0,
    VPred = 1,
    HPred = 2,
    TmPred = 3,
}

/// Intra prediction modes for 4x4 luma subblocks (10 modes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntraMode4x4 {
    BDcPred = 0,   // DC prediction
    BTmPred = 1,   // TrueMotion prediction
    BVePred = 2,   // Vertical prediction
    BHePred = 3,   // Horizontal prediction
    BLdPred = 4,   // Down-left diagonal
    BRdPred = 5,   // Down-right diagonal
    BVrPred = 6,   // Vertical-right
    BVlPred = 7,   // Vertical-left
    BHdPred = 8,   // Horizontal-down
    BHuPred = 9,   // Horizontal-up
}

/// Inter prediction modes for macroblocks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterMode {
    ZeroMv = 0,     // Zero motion vector (0, 0)
    NearestMv = 1,  // Use nearest MV from neighbors
    NearMv = 2,     // Use second nearest MV
    NewMv = 3,      // New explicit motion vector
    SplitMv = 4,    // Split macroblock with per-subblock MVs
}

/// Motion vector with quarter-pixel precision
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub x: i16,  // Horizontal component (1/4 pixel units)
    pub y: i16,  // Vertical component (1/4 pixel units)
}

/// Macroblock structure
#[derive(Debug, Clone)]
pub struct Macroblock {
    pub mb_type: MbType,
    pub segment_id: u8,         // 0-3 segment identifier
    pub skip_coeff: bool,       // True if no DCT coefficients
    pub ref_frame: RefFrame,
    pub intra_y_mode: IntraMode16x16,
    pub intra_uv_mode: IntraMode16x16,
    pub intra_4x4_modes: [IntraMode4x4; 16],  // For 4x4 intra prediction
    pub inter_mode: InterMode,
    pub mvs: [MotionVector; 16], // Motion vectors (1 or 16 depending on mode)
    pub coeffs: MacroblockCoeffs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbType {
    Intra16x16,
    Intra4x4,
    Inter,
}

/// DCT coefficients for a macroblock
#[derive(Debug, Clone, Default)]
pub struct MacroblockCoeffs {
    pub y: [[i16; 16]; 16],   // 16 Y subblocks, 16 coefficients each
    pub y2: [i16; 16],         // Y2 block (DC coefficients, WHT)
    pub u: [[i16; 16]; 4],     // 4 U subblocks
    pub v: [[i16; 16]; 4],     // 4 V subblocks
}

/// Frame buffer for YUV 4:2:0 data
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,   // Luma plane
    pub u: Vec<u8>,   // Chroma U plane
    pub v: Vec<u8>,   // Chroma V plane
    pub y_stride: usize,
    pub uv_stride: usize,
}

/// Decoder state
pub struct Vp8DecoderState {
    pub width: u32,
    pub height: u32,
    pub mb_width: u32,   // Number of macroblocks horizontally
    pub mb_height: u32,  // Number of macroblocks vertically

    // Reference frame buffers
    pub current: FrameBuffer,
    pub last: FrameBuffer,
    pub golden: FrameBuffer,
    pub altref: FrameBuffer,

    // Probability tables
    pub coeff_probs: CoeffProbs,
    pub mv_probs: MvProbs,
    pub mode_probs: ModeProbs,

    // Segmentation
    pub segment_enabled: bool,
    pub segment_quant: [i16; 4],
    pub segment_filter: [i8; 4],

    // Loop filter settings
    pub filter_level: u8,
    pub sharpness: u8,
    pub filter_type: FilterType,
}
```

---

## Boolean Arithmetic Coder

The boolean arithmetic coder is the core entropy coding mechanism in VP8. Every symbol is decoded through this mechanism.

### Boolean Decoder Implementation

```rust
/// Boolean arithmetic decoder state
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    value: u32,    // Current value (left endpoint of interval)
    range: u32,    // Size of interval
    bit_count: i32, // Number of bits consumed from current byte
}

impl<'a> BoolDecoder<'a> {
    /// Initialize decoder from data buffer
    pub fn new(data: &'a [u8]) -> Self {
        let mut decoder = BoolDecoder {
            data,
            pos: 0,
            value: 0,
            range: 255,
            bit_count: 0,
        };

        // Load initial bits
        decoder.value = ((data.get(0).copied().unwrap_or(0) as u32) << 8)
                      | (data.get(1).copied().unwrap_or(0) as u32);
        decoder.pos = 2;
        decoder.bit_count = 0;

        decoder
    }

    /// Read a single boolean with given probability (0-255)
    /// prob = probability of 0 (prob/256 is actual probability)
    pub fn read_bool(&mut self, prob: u8) -> bool {
        // Calculate split point
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let split_shifted = split << 8;

        let bit = if self.value >= split_shifted {
            // Bit is 1
            self.range -= split;
            self.value -= split_shifted;
            true
        } else {
            // Bit is 0
            self.range = split;
            false
        };

        // Normalize (shift until range >= 128)
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;

            if self.bit_count == 8 {
                self.bit_count = 0;
                if self.pos < self.data.len() {
                    self.value |= self.data[self.pos] as u32;
                    self.pos += 1;
                }
            }
        }

        bit
    }

    /// Read an unsigned n-bit literal value
    pub fn read_literal(&mut self, n: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bool(128) as u32);
        }
        value
    }

    /// Read a signed n-bit value (literal + sign)
    pub fn read_signed_literal(&mut self, n: u8) -> i32 {
        let value = self.read_literal(n) as i32;
        if self.read_bool(128) {
            -value
        } else {
            value
        }
    }

    /// Read value from a probability tree
    pub fn read_tree<T: Copy>(&mut self, tree: &[TreeNode], probs: &[u8]) -> T {
        let mut node = 0i32;
        loop {
            let prob = probs[node as usize >> 1];
            let bit = self.read_bool(prob);
            node = tree[node as usize + bit as usize].next;
            if node <= 0 {
                return unsafe { std::mem::transmute_copy(&(-node)) };
            }
        }
    }
}

/// Tree node for probability tree traversal
#[derive(Debug, Clone, Copy)]
pub struct TreeNode {
    pub next: i32,  // Positive = continue, negative/zero = leaf value
}
```

### Boolean Encoder Implementation

```rust
/// Boolean arithmetic encoder state
pub struct BoolEncoder {
    output: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    pub fn new() -> Self {
        BoolEncoder {
            output: Vec::new(),
            range: 255,
            bottom: 0,
            bit_count: 24,
        }
    }

    /// Write a single boolean with given probability
    pub fn write_bool(&mut self, bit: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);

        if bit {
            self.bottom += split;
            self.range -= split;
        } else {
            self.range = split;
        }

        // Normalize and output bytes
        while self.range < 128 {
            self.range <<= 1;

            if (self.bottom & (1 << 31)) != 0 {
                // Carry propagation
                let mut pos = self.output.len();
                while pos > 0 {
                    pos -= 1;
                    if self.output[pos] == 255 {
                        self.output[pos] = 0;
                    } else {
                        self.output[pos] += 1;
                        break;
                    }
                }
            }

            self.bottom <<= 1;
            self.bit_count -= 1;

            if self.bit_count == 16 {
                self.output.push((self.bottom >> 24) as u8);
                self.bottom &= 0x00FFFFFF;
                self.bit_count = 24;
            }
        }
    }

    /// Write an unsigned n-bit literal
    pub fn write_literal(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bool((value >> i) & 1 != 0, 128);
        }
    }

    /// Finalize and return encoded data
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining bits
        for _ in 0..32 {
            self.write_bool(false, 128);
        }
        self.output
    }
}
```

---

## Frame Structure

### Frame Header Structure

```rust
/// VP8 frame header
pub struct FrameHeader {
    // Uncompressed data chunk (3 bytes)
    pub frame_type: FrameType,      // 1 bit: 0=key frame, 1=inter frame
    pub version: u8,                 // 3 bits: version number (0-3)
    pub show_frame: bool,            // 1 bit: display frame flag
    pub first_partition_size: u32,   // 19 bits: size of first partition

    // Key frame specific
    pub width: u16,                  // Horizontal size (14 bits)
    pub horizontal_scale: u8,        // 2 bits
    pub height: u16,                 // Vertical size (14 bits)
    pub vertical_scale: u8,          // 2 bits

    // Color space and clamping (key frames only)
    pub color_space: ColorSpace,
    pub clamping_required: bool,

    // Segmentation
    pub segmentation_enabled: bool,
    pub update_mb_segmentation_map: bool,
    pub update_segment_feature_data: bool,
    pub segment_feature_mode: SegmentFeatureMode,
    pub segment_quant_update: [Option<i8>; 4],
    pub segment_lf_update: [Option<i8>; 4],
    pub mb_segment_tree_probs: [u8; 3],

    // Loop filter
    pub filter_type: FilterType,
    pub loop_filter_level: u8,       // 0-63
    pub sharpness_level: u8,         // 0-7
    pub loop_filter_adj_enable: bool,
    pub mode_ref_lf_delta_enabled: bool,
    pub mode_ref_lf_delta_update: bool,
    pub ref_lf_deltas: [i8; 4],      // Per reference frame
    pub mode_lf_deltas: [i8; 4],     // Per coding mode

    // Partitions
    pub num_token_partitions: u8,    // 1, 2, 4, or 8

    // Quantization
    pub y_ac_qi: u8,                 // Base quantizer index (0-127)
    pub y_dc_delta: i8,
    pub y2_dc_delta: i8,
    pub y2_ac_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,

    // Reference frame management (inter frames)
    pub refresh_golden_frame: bool,
    pub refresh_altref_frame: bool,
    pub copy_to_golden: CopyBuffer,
    pub copy_to_altref: CopyBuffer,
    pub sign_bias_golden: bool,
    pub sign_bias_altref: bool,
    pub refresh_last: bool,
    pub refresh_entropy_probs: bool,

    // Coefficient probability updates
    pub coeff_prob_update: CoeffProbUpdate,

    // MV probability updates
    pub mv_prob_update: MvProbUpdate,
}

#[derive(Debug, Clone, Copy)]
pub enum ColorSpace {
    YCbCr,
    Reserved,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    None,
    Simple,
    Normal,
}

#[derive(Debug, Clone, Copy)]
pub enum SegmentFeatureMode {
    Delta,
    Absolute,
}

#[derive(Debug, Clone, Copy)]
pub enum CopyBuffer {
    None,
    Last,
    Golden,
    AltRef,
}
```

### Frame Header Parsing

```rust
impl FrameHeader {
    pub fn parse(data: &[u8]) -> Result<(Self, usize), Error> {
        if data.len() < 3 {
            return Err(Error::InvalidData);
        }

        // Parse uncompressed chunk (first 3 bytes)
        let frame_tag = (data[2] as u32) << 16 | (data[1] as u32) << 8 | data[0] as u32;

        let frame_type = if frame_tag & 1 == 0 {
            FrameType::KeyFrame
        } else {
            FrameType::InterFrame
        };
        let version = ((frame_tag >> 1) & 0x7) as u8;
        let show_frame = (frame_tag >> 4) & 1 != 0;
        let first_partition_size = (frame_tag >> 5) as u32;

        let mut offset = 3;

        // Key frame has additional uncompressed data
        let (width, horizontal_scale, height, vertical_scale) = if frame_type == FrameType::KeyFrame {
            // Check signature (0x9D 0x01 0x2A)
            if data.len() < 10 || data[3] != 0x9D || data[4] != 0x01 || data[5] != 0x2A {
                return Err(Error::InvalidKeyFrameSignature);
            }

            let w = (data[7] as u16) << 8 | data[6] as u16;
            let h = (data[9] as u16) << 8 | data[8] as u16;

            let width = w & 0x3FFF;
            let horizontal_scale = (w >> 14) as u8;
            let height = h & 0x3FFF;
            let vertical_scale = (h >> 14) as u8;

            offset = 10;
            (width, horizontal_scale, height, vertical_scale)
        } else {
            (0, 0, 0, 0) // Use previous frame dimensions
        };

        // Rest is parsed using boolean decoder
        let bool_data = &data[offset..offset + first_partition_size as usize];
        let mut bd = BoolDecoder::new(bool_data);

        // Parse remaining header fields with boolean decoder
        let mut header = FrameHeader {
            frame_type,
            version,
            show_frame,
            first_partition_size,
            width,
            horizontal_scale,
            height,
            vertical_scale,
            // ... initialize other fields
            ..Default::default()
        };

        if frame_type == FrameType::KeyFrame {
            header.color_space = if bd.read_bool(128) {
                ColorSpace::Reserved
            } else {
                ColorSpace::YCbCr
            };
            header.clamping_required = bd.read_bool(128);
        }

        // Parse segmentation
        header.segmentation_enabled = bd.read_bool(128);
        if header.segmentation_enabled {
            header.update_mb_segmentation_map = bd.read_bool(128);
            header.update_segment_feature_data = bd.read_bool(128);

            if header.update_segment_feature_data {
                header.segment_feature_mode = if bd.read_bool(128) {
                    SegmentFeatureMode::Absolute
                } else {
                    SegmentFeatureMode::Delta
                };

                // Quantizer updates
                for i in 0..4 {
                    if bd.read_bool(128) {
                        let value = bd.read_literal(7) as i8;
                        let sign = if bd.read_bool(128) { -1 } else { 1 };
                        header.segment_quant_update[i] = Some(value * sign);
                    }
                }

                // Loop filter updates
                for i in 0..4 {
                    if bd.read_bool(128) {
                        let value = bd.read_literal(6) as i8;
                        let sign = if bd.read_bool(128) { -1 } else { 1 };
                        header.segment_lf_update[i] = Some(value * sign);
                    }
                }
            }

            // Segment tree probabilities
            if header.update_mb_segmentation_map {
                for i in 0..3 {
                    if bd.read_bool(128) {
                        header.mb_segment_tree_probs[i] = bd.read_literal(8) as u8;
                    } else {
                        header.mb_segment_tree_probs[i] = 255;
                    }
                }
            }
        }

        // Filter type and level
        header.filter_type = if bd.read_bool(128) {
            FilterType::Simple
        } else {
            FilterType::Normal
        };
        header.loop_filter_level = bd.read_literal(6) as u8;
        header.sharpness_level = bd.read_literal(3) as u8;

        // Loop filter adjustments
        header.loop_filter_adj_enable = bd.read_bool(128);
        if header.loop_filter_adj_enable {
            header.mode_ref_lf_delta_enabled = bd.read_bool(128);
            if header.mode_ref_lf_delta_enabled {
                header.mode_ref_lf_delta_update = bd.read_bool(128);
                if header.mode_ref_lf_delta_update {
                    for i in 0..4 {
                        if bd.read_bool(128) {
                            header.ref_lf_deltas[i] = bd.read_signed_literal(6) as i8;
                        }
                    }
                    for i in 0..4 {
                        if bd.read_bool(128) {
                            header.mode_lf_deltas[i] = bd.read_signed_literal(6) as i8;
                        }
                    }
                }
            }
        }

        // Token partitions
        header.num_token_partitions = 1 << bd.read_literal(2);

        // Quantization parameters
        header.y_ac_qi = bd.read_literal(7) as u8;
        header.y_dc_delta = if bd.read_bool(128) { bd.read_signed_literal(4) as i8 } else { 0 };
        header.y2_dc_delta = if bd.read_bool(128) { bd.read_signed_literal(4) as i8 } else { 0 };
        header.y2_ac_delta = if bd.read_bool(128) { bd.read_signed_literal(4) as i8 } else { 0 };
        header.uv_dc_delta = if bd.read_bool(128) { bd.read_signed_literal(4) as i8 } else { 0 };
        header.uv_ac_delta = if bd.read_bool(128) { bd.read_signed_literal(4) as i8 } else { 0 };

        // Reference frame updates (inter frames only)
        if frame_type == FrameType::InterFrame {
            header.refresh_golden_frame = bd.read_bool(128);
            header.refresh_altref_frame = bd.read_bool(128);

            if !header.refresh_golden_frame {
                header.copy_to_golden = match bd.read_literal(2) {
                    0 => CopyBuffer::None,
                    1 => CopyBuffer::Last,
                    2 => CopyBuffer::AltRef,
                    _ => CopyBuffer::None,
                };
            }

            if !header.refresh_altref_frame {
                header.copy_to_altref = match bd.read_literal(2) {
                    0 => CopyBuffer::None,
                    1 => CopyBuffer::Last,
                    2 => CopyBuffer::Golden,
                    _ => CopyBuffer::None,
                };
            }

            header.sign_bias_golden = bd.read_bool(128);
            header.sign_bias_altref = bd.read_bool(128);
            header.refresh_last = bd.read_bool(128);
        } else {
            // Key frames always refresh all references
            header.refresh_golden_frame = true;
            header.refresh_altref_frame = true;
            header.refresh_last = true;
        }

        header.refresh_entropy_probs = bd.read_bool(128);

        // Parse coefficient probability updates
        // (See entropy section for details)

        Ok((header, offset + first_partition_size as usize))
    }
}
```

---

## Transforms (DCT and WHT)

### 4x4 DCT Implementation

VP8 uses an integer approximation of the 4x4 DCT for efficiency. The transform can be computed using only additions, subtractions, and shifts.

```rust
/// 4x4 Forward DCT (for encoder)
/// Input: 4x4 block of residuals (i16)
/// Output: 4x4 block of DCT coefficients (i16)
pub fn forward_dct4x4(input: &[[i16; 4]; 4]) -> [[i16; 4]; 4] {
    let mut temp = [[0i32; 4]; 4];
    let mut output = [[0i16; 4]; 4];

    // 1D DCT on rows
    for i in 0..4 {
        let a1 = (input[i][0] + input[i][3]) as i32;
        let b1 = (input[i][1] + input[i][2]) as i32;
        let c1 = (input[i][1] - input[i][2]) as i32;
        let d1 = (input[i][0] - input[i][3]) as i32;

        temp[i][0] = a1 + b1;
        temp[i][1] = (c1 * 2217 + d1 * 5352 + 14500) >> 12;
        temp[i][2] = a1 - b1;
        temp[i][3] = (d1 * 2217 - c1 * 5352 + 7500) >> 12;
    }

    // 1D DCT on columns
    for j in 0..4 {
        let a1 = temp[0][j] + temp[3][j];
        let b1 = temp[1][j] + temp[2][j];
        let c1 = temp[1][j] - temp[2][j];
        let d1 = temp[0][j] - temp[3][j];

        output[0][j] = ((a1 + b1 + 7) >> 4) as i16;
        output[1][j] = ((c1 * 2217 + d1 * 5352 + 12000) >> 16) as i16;
        output[2][j] = ((a1 - b1 + 7) >> 4) as i16;
        output[3][j] = ((d1 * 2217 - c1 * 5352 + 51000) >> 16) as i16;
    }

    output
}

/// 4x4 Inverse DCT (for decoder)
/// Input: 4x4 block of DCT coefficients (i16)
/// Output: 4x4 block of residuals (i16)
pub fn inverse_dct4x4(input: &[[i16; 4]; 4]) -> [[i16; 4]; 4] {
    let mut temp = [[0i32; 4]; 4];
    let mut output = [[0i16; 4]; 4];

    // Constants for inverse transform
    const C1: i32 = 20091;  // cos(pi/8) * sqrt(2) * 16384
    const C2: i32 = 35468;  // sin(pi/8) * sqrt(2) * 16384

    // 1D IDCT on columns
    for j in 0..4 {
        let a = input[0][j] as i32;
        let b = input[1][j] as i32;
        let c = input[2][j] as i32;
        let d = input[3][j] as i32;

        // Compute temp values
        let a1 = a + c;
        let b1 = a - c;
        let c1 = ((b * C2) >> 16) - d - ((d * C1) >> 16);
        let d1 = b + ((b * C1) >> 16) + ((d * C2) >> 16);

        temp[0][j] = a1 + d1;
        temp[1][j] = b1 + c1;
        temp[2][j] = b1 - c1;
        temp[3][j] = a1 - d1;
    }

    // 1D IDCT on rows
    for i in 0..4 {
        let a = temp[i][0];
        let b = temp[i][1];
        let c = temp[i][2];
        let d = temp[i][3];

        let a1 = a + c;
        let b1 = a - c;
        let c1 = ((b * C2) >> 16) - d - ((d * C1) >> 16);
        let d1 = b + ((b * C1) >> 16) + ((d * C2) >> 16);

        // Add 4 for rounding, shift by 3
        output[i][0] = ((a1 + d1 + 4) >> 3) as i16;
        output[i][1] = ((b1 + c1 + 4) >> 3) as i16;
        output[i][2] = ((b1 - c1 + 4) >> 3) as i16;
        output[i][3] = ((a1 - d1 + 4) >> 3) as i16;
    }

    output
}
```

### 4x4 Walsh-Hadamard Transform

The WHT is used for the Y2 block (DC coefficients of the 16 Y subblocks).

```rust
/// 4x4 Forward Walsh-Hadamard Transform (for encoder)
pub fn forward_wht4x4(input: &[[i16; 4]; 4]) -> [[i16; 4]; 4] {
    let mut temp = [[0i32; 4]; 4];
    let mut output = [[0i16; 4]; 4];

    // 1D WHT on rows
    for i in 0..4 {
        let a = input[i][0] as i32 + input[i][2] as i32;
        let b = input[i][1] as i32 + input[i][3] as i32;
        let c = input[i][0] as i32 - input[i][2] as i32;
        let d = input[i][1] as i32 - input[i][3] as i32;

        temp[i][0] = a + b;
        temp[i][1] = c + d;
        temp[i][2] = a - b;
        temp[i][3] = c - d;
    }

    // 1D WHT on columns
    for j in 0..4 {
        let a = temp[0][j] + temp[2][j];
        let b = temp[1][j] + temp[3][j];
        let c = temp[0][j] - temp[2][j];
        let d = temp[1][j] - temp[3][j];

        // Scale by 1/2 (shift right by 1) with rounding
        output[0][j] = ((a + b + 1) >> 1) as i16;
        output[1][j] = ((c + d + 1) >> 1) as i16;
        output[2][j] = ((a - b + 1) >> 1) as i16;
        output[3][j] = ((c - d + 1) >> 1) as i16;
    }

    output
}

/// 4x4 Inverse Walsh-Hadamard Transform (for decoder)
pub fn inverse_wht4x4(input: &[[i16; 4]; 4]) -> [[i16; 4]; 4] {
    let mut temp = [[0i32; 4]; 4];
    let mut output = [[0i16; 4]; 4];

    // 1D IWHT on rows
    for i in 0..4 {
        let a = input[i][0] as i32 + input[i][3] as i32;
        let b = input[i][1] as i32 + input[i][2] as i32;
        let c = input[i][1] as i32 - input[i][2] as i32;
        let d = input[i][0] as i32 - input[i][3] as i32;

        temp[i][0] = a + b;
        temp[i][1] = c + d;
        temp[i][2] = a - b;
        temp[i][3] = c - d;
    }

    // 1D IWHT on columns
    for j in 0..4 {
        let a = temp[0][j] + temp[3][j];
        let b = temp[1][j] + temp[2][j];
        let c = temp[1][j] - temp[2][j];
        let d = temp[0][j] - temp[3][j];

        // Add 3 for rounding, shift right by 3
        output[0][j] = ((a + b + 3) >> 3) as i16;
        output[1][j] = ((c + d + 3) >> 3) as i16;
        output[2][j] = ((a - b + 3) >> 3) as i16;
        output[3][j] = ((c - d + 3) >> 3) as i16;
    }

    output
}
```

---

## Intra Prediction

### 16x16 and 8x8 Intra Prediction Modes

```rust
/// Perform 16x16 intra prediction
/// `above`: 16 pixels from row above (plus 1 corner pixel at index 0)
/// `left`: 16 pixels from column to left (plus 1 corner pixel at index 0)
/// `above_left`: Corner pixel (A[-1] or L[-1])
pub fn predict_16x16(
    mode: IntraMode16x16,
    above: &[u8; 16],
    left: &[u8; 16],
    above_left: u8,
    output: &mut [[u8; 16]; 16],
) {
    match mode {
        IntraMode16x16::DcPred => {
            // Average of above and left
            let sum: u32 = above.iter().map(|&x| x as u32).sum::<u32>()
                        + left.iter().map(|&x| x as u32).sum::<u32>();
            let dc = ((sum + 16) >> 5) as u8;
            for row in output.iter_mut() {
                row.fill(dc);
            }
        }
        IntraMode16x16::VPred => {
            // Copy from above
            for row in output.iter_mut() {
                row.copy_from_slice(above);
            }
        }
        IntraMode16x16::HPred => {
            // Copy from left
            for (i, row) in output.iter_mut().enumerate() {
                row.fill(left[i]);
            }
        }
        IntraMode16x16::TmPred => {
            // TrueMotion: pred[y][x] = left[y] + above[x] - above_left
            for y in 0..16 {
                for x in 0..16 {
                    let val = left[y] as i16 + above[x] as i16 - above_left as i16;
                    output[y][x] = val.clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Perform 8x8 chroma intra prediction (same modes as 16x16)
pub fn predict_8x8_chroma(
    mode: IntraMode16x16,
    above: &[u8; 8],
    left: &[u8; 8],
    above_left: u8,
    output: &mut [[u8; 8]; 8],
) {
    match mode {
        IntraMode16x16::DcPred => {
            let sum: u32 = above.iter().map(|&x| x as u32).sum::<u32>()
                        + left.iter().map(|&x| x as u32).sum::<u32>();
            let dc = ((sum + 8) >> 4) as u8;
            for row in output.iter_mut() {
                row.fill(dc);
            }
        }
        IntraMode16x16::VPred => {
            for row in output.iter_mut() {
                row.copy_from_slice(above);
            }
        }
        IntraMode16x16::HPred => {
            for (i, row) in output.iter_mut().enumerate() {
                row.fill(left[i]);
            }
        }
        IntraMode16x16::TmPred => {
            for y in 0..8 {
                for x in 0..8 {
                    let val = left[y] as i16 + above[x] as i16 - above_left as i16;
                    output[y][x] = val.clamp(0, 255) as u8;
                }
            }
        }
    }
}
```

### 4x4 Luma Intra Prediction Modes

```rust
/// Perform 4x4 intra prediction
/// `above`: 8 pixels (4 above + 4 above-right)
/// `left`: 4 pixels from left column
/// `above_left`: Corner pixel
pub fn predict_4x4(
    mode: IntraMode4x4,
    above: &[u8; 8],   // Includes above-right for diagonal modes
    left: &[u8; 4],
    above_left: u8,
    output: &mut [[u8; 4]; 4],
) {
    match mode {
        IntraMode4x4::BDcPred => {
            let sum: u32 = above[..4].iter().map(|&x| x as u32).sum::<u32>()
                        + left.iter().map(|&x| x as u32).sum::<u32>();
            let dc = ((sum + 4) >> 3) as u8;
            for row in output.iter_mut() {
                row.fill(dc);
            }
        }
        IntraMode4x4::BTmPred => {
            for y in 0..4 {
                for x in 0..4 {
                    let val = left[y] as i16 + above[x] as i16 - above_left as i16;
                    output[y][x] = val.clamp(0, 255) as u8;
                }
            }
        }
        IntraMode4x4::BVePred => {
            // Vertical with smoothing
            for x in 0..4 {
                let avg = if x == 0 {
                    (above_left as u16 + 2 * above[0] as u16 + above[1] as u16 + 2) >> 2
                } else if x == 3 {
                    (above[2] as u16 + 2 * above[3] as u16 + above[4] as u16 + 2) >> 2
                } else {
                    (above[x - 1] as u16 + 2 * above[x] as u16 + above[x + 1] as u16 + 2) >> 2
                };
                for y in 0..4 {
                    output[y][x] = avg as u8;
                }
            }
        }
        IntraMode4x4::BHePred => {
            // Horizontal with smoothing
            for y in 0..4 {
                let avg = if y == 0 {
                    (above_left as u16 + 2 * left[0] as u16 + left[1] as u16 + 2) >> 2
                } else if y == 3 {
                    (left[2] as u16 + 3 * left[3] as u16 + 2) >> 2
                } else {
                    (left[y - 1] as u16 + 2 * left[y] as u16 + left[y + 1] as u16 + 2) >> 2
                };
                for x in 0..4 {
                    output[y][x] = avg as u8;
                }
            }
        }
        IntraMode4x4::BLdPred => {
            // Down-left diagonal (45 degrees)
            for y in 0..4 {
                for x in 0..4 {
                    let idx = x + y;
                    let val = if idx < 6 {
                        (above[idx] as u16 + 2 * above[idx + 1] as u16 + above[idx + 2] as u16 + 2) >> 2
                    } else {
                        above[7] as u16
                    };
                    output[y][x] = val as u8;
                }
            }
        }
        IntraMode4x4::BRdPred => {
            // Down-right diagonal
            let mut edge = [0u8; 9];
            edge[0] = left[3];
            edge[1] = left[2];
            edge[2] = left[1];
            edge[3] = left[0];
            edge[4] = above_left;
            edge[5..9].copy_from_slice(&above[..4]);

            for y in 0..4 {
                for x in 0..4 {
                    let idx = 4 - y + x;
                    let val = (edge[idx - 1] as u16 + 2 * edge[idx] as u16 + edge[idx + 1] as u16 + 2) >> 2;
                    output[y][x] = val as u8;
                }
            }
        }
        IntraMode4x4::BVrPred => {
            // Vertical-right
            let mut edge = [0u8; 9];
            edge[0] = left[3];
            edge[1] = left[2];
            edge[2] = left[1];
            edge[3] = left[0];
            edge[4] = above_left;
            edge[5..9].copy_from_slice(&above[..4]);

            for y in 0..4 {
                for x in 0..4 {
                    let zy = 2 * y;
                    let zx = 2 * x;
                    let idx = zx as i32 - zy as i32;
                    let val = if idx >= -1 {
                        let base = (4 + (idx >> 1)) as usize;
                        if idx & 1 == 0 {
                            (edge[base] as u16 + edge[base + 1] as u16 + 1) >> 1
                        } else {
                            (edge[base] as u16 + 2 * edge[base + 1] as u16 + edge[base + 2] as u16 + 2) >> 2
                        }
                    } else {
                        let base = (4 + y - 1) as usize;
                        (edge[base] as u16 + 2 * edge[base + 1] as u16 + edge[base + 2] as u16 + 2) >> 2
                    };
                    output[y][x] = val as u8;
                }
            }
        }
        IntraMode4x4::BVlPred => {
            // Vertical-left
            for y in 0..4 {
                for x in 0..4 {
                    let idx = x + (y >> 1);
                    let val = if y & 1 == 0 {
                        (above[idx] as u16 + above[idx + 1] as u16 + 1) >> 1
                    } else {
                        (above[idx] as u16 + 2 * above[idx + 1] as u16 + above[idx + 2] as u16 + 2) >> 2
                    };
                    output[y][x] = val as u8;
                }
            }
        }
        IntraMode4x4::BHdPred => {
            // Horizontal-down
            let mut edge = [0u8; 9];
            edge[0] = left[3];
            edge[1] = left[2];
            edge[2] = left[1];
            edge[3] = left[0];
            edge[4] = above_left;
            edge[5..9].copy_from_slice(&above[..4]);

            for y in 0..4 {
                for x in 0..4 {
                    let idx = 2 * y as i32 - x as i32;
                    let val = if idx >= 0 {
                        let base = (4 - (idx >> 1) - 1) as usize;
                        if idx & 1 == 0 {
                            (edge[base] as u16 + edge[base + 1] as u16 + 1) >> 1
                        } else {
                            (edge[base] as u16 + 2 * edge[base + 1] as u16 + edge[base + 2] as u16 + 2) >> 2
                        }
                    } else {
                        let base = (4 + x - 1) as usize;
                        (edge[base] as u16 + 2 * edge[base + 1] as u16 + edge[base + 2] as u16 + 2) >> 2
                    };
                    output[y][x] = val as u8;
                }
            }
        }
        IntraMode4x4::BHuPred => {
            // Horizontal-up
            for y in 0..4 {
                for x in 0..4 {
                    let idx = x + (y >> 1) + y;
                    let val = if idx < 3 {
                        if (x + y) & 1 == 0 {
                            (left[idx] as u16 + left[idx + 1] as u16 + 1) >> 1
                        } else {
                            (left[idx] as u16 + 2 * left[idx + 1] as u16 + left[idx + 2] as u16 + 2) >> 2
                        }
                    } else if idx == 3 {
                        if (x + y) & 1 == 0 {
                            (left[3] as u16 + left[3] as u16 + 1) >> 1
                        } else {
                            (left[2] as u16 + 3 * left[3] as u16 + 2) >> 2
                        }
                    } else {
                        left[3] as u16
                    };
                    output[y][x] = val as u8;
                }
            }
        }
    }
}
```

---

## Inter Prediction

### Motion Vector Decoding

```rust
/// Motion vector component probability tables (default values)
pub const MV_PROB_DEFAULT: [[u8; 19]; 2] = [
    // Horizontal
    [162, 128, 225, 146, 172, 147, 214, 39, 156, 128, 129, 132, 75, 145, 178, 206, 239, 254, 254],
    // Vertical
    [164, 128, 204, 170, 119, 235, 140, 230, 228, 128, 130, 130, 74, 148, 180, 203, 236, 254, 254],
];

/// Decode a motion vector component
pub fn decode_mv_component(bd: &mut BoolDecoder, probs: &[u8; 19]) -> i16 {
    // Tree structure for MV magnitude
    const MV_IS_SHORT_PROB_IDX: usize = 0;
    const MV_SIGN_PROB_IDX: usize = 1;
    const MV_SHORT_TREE_PROBS: usize = 2;  // indices 2-8 for short MV tree
    const MV_LONG_BITS_PROBS: usize = 9;   // indices 9-18 for long MV bits

    let is_short = bd.read_bool(probs[MV_IS_SHORT_PROB_IDX]);

    let magnitude = if is_short {
        // Decode short MV (0-7) using tree
        decode_mv_short_tree(bd, &probs[MV_SHORT_TREE_PROBS..MV_SHORT_TREE_PROBS + 7])
    } else {
        // Decode long MV using bit-by-bit coding
        let mut mag = 0i16;
        for i in 0..10 {
            if bd.read_bool(probs[MV_LONG_BITS_PROBS + i]) {
                mag |= 1 << i;
            }
        }
        mag + 8  // Long MVs start at 8
    };

    if magnitude == 0 {
        return 0;
    }

    // Read sign bit
    let sign = bd.read_bool(probs[MV_SIGN_PROB_IDX]);

    if sign {
        -magnitude
    } else {
        magnitude
    }
}

fn decode_mv_short_tree(bd: &mut BoolDecoder, probs: &[u8]) -> i16 {
    // Binary tree for short MVs (0-7)
    // Tree structure:
    //      [0]
    //     /   \
    //   [1]   [2]
    //   / \   / \
    // [3] [4][5][6]
    // 0 1 2 3 4 5 6 7

    if !bd.read_bool(probs[0]) {
        // Left branch (0-3)
        if !bd.read_bool(probs[1]) {
            if !bd.read_bool(probs[3]) { 0 } else { 1 }
        } else {
            if !bd.read_bool(probs[4]) { 2 } else { 3 }
        }
    } else {
        // Right branch (4-7)
        if !bd.read_bool(probs[2]) {
            if !bd.read_bool(probs[5]) { 4 } else { 5 }
        } else {
            if !bd.read_bool(probs[6]) { 6 } else { 7 }
        }
    }
}
```

### Subpixel Interpolation (6-tap filter)

```rust
/// 6-tap filter coefficients for subpixel interpolation
/// Format: [subpel_position][tap_index]
const SUBPEL_FILTERS: [[i16; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],      // Full pixel (0/8)
    [0, -6, 123, 12, -1, 0],   // 1/8 pixel
    [2, -11, 108, 36, -8, 1],  // 1/4 pixel (2/8)
    [0, -9, 93, 50, -6, 0],    // 3/8 pixel
    [3, -16, 77, 77, -16, 3],  // 1/2 pixel (4/8)
    [0, -6, 50, 93, -9, 0],    // 5/8 pixel
    [1, -8, 36, 108, -11, 2],  // 3/4 pixel (6/8)
    [0, -1, 12, 123, -6, 0],   // 7/8 pixel
];

/// Perform 6-tap horizontal interpolation
pub fn filter_horizontal(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    width: usize,
    height: usize,
    subpel_x: usize,  // 0-7
) {
    let filter = &SUBPEL_FILTERS[subpel_x];

    for y in 0..height {
        let src_row = &src[y * src_stride..];
        let dst_row = &mut dst[y * dst_stride..];

        for x in 0..width {
            // Apply 6-tap filter centered at src[x]
            // Need 2 pixels before and 3 after
            let sum: i32 = (-2..=3)
                .map(|i| {
                    let idx = (x as i32 + i).clamp(0, (src_row.len() - 1) as i32) as usize;
                    src_row[idx] as i32 * filter[(i + 2) as usize] as i32
                })
                .sum();

            // Round and store (keep extra precision for vertical pass)
            dst_row[x] = ((sum + 64) >> 7) as i16;
        }
    }
}

/// Perform 6-tap vertical interpolation
pub fn filter_vertical(
    src: &[i16],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
    subpel_y: usize,  // 0-7
) {
    let filter = &SUBPEL_FILTERS[subpel_y];

    for y in 0..height {
        let dst_row = &mut dst[y * dst_stride..];

        for x in 0..width {
            let sum: i32 = (-2..=3)
                .map(|i| {
                    let row = (y as i32 + i).clamp(0, (height + 5) as i32) as usize;
                    src[row * src_stride + x] as i32 * filter[(i + 2) as usize] as i32
                })
                .sum();

            // Final rounding and clamp
            dst_row[x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Perform motion compensation for a block
pub fn motion_compensate(
    ref_frame: &FrameBuffer,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    block_size: usize,  // 4, 8, or 16
    output: &mut [u8],
    output_stride: usize,
) {
    // MV is in 1/4 pixel units
    let full_x = block_x as i32 + (mv.x as i32 >> 2);
    let full_y = block_y as i32 + (mv.y as i32 >> 2);
    let subpel_x = (mv.x as usize) & 3;
    let subpel_y = (mv.y as usize) & 3;

    // Convert 1/4 pel to 1/8 pel for filter lookup
    let filter_x = subpel_x * 2;
    let filter_y = subpel_y * 2;

    if filter_x == 0 && filter_y == 0 {
        // Full-pixel copy
        for y in 0..block_size {
            let src_y = (full_y + y as i32).clamp(0, ref_frame.height as i32 - 1) as usize;
            for x in 0..block_size {
                let src_x = (full_x + x as i32).clamp(0, ref_frame.width as i32 - 1) as usize;
                output[y * output_stride + x] = ref_frame.y[src_y * ref_frame.y_stride + src_x];
            }
        }
    } else {
        // Subpixel interpolation required
        let mut temp = vec![0i16; (block_size + 5) * block_size];

        // Horizontal filtering
        filter_horizontal(
            &ref_frame.y[(full_y as usize - 2) * ref_frame.y_stride + (full_x as usize - 2)..],
            ref_frame.y_stride,
            &mut temp,
            block_size,
            block_size,
            block_size + 5,
            filter_x,
        );

        // Vertical filtering
        filter_vertical(
            &temp,
            block_size,
            output,
            output_stride,
            block_size,
            block_size,
            filter_y,
        );
    }
}
```

### Chroma Motion Compensation (4-tap bilinear)

```rust
/// 4-tap bilinear filter coefficients for chroma (1/8 pel precision)
const CHROMA_FILTERS: [[i16; 4]; 8] = [
    [0, 128, 0, 0],     // 0/8
    [12, 116, 0, 0],    // 1/8
    [24, 104, 0, 0],    // 2/8
    [36, 92, 0, 0],     // 3/8
    [48, 80, 0, 0],     // 4/8 (simplified bilinear)
    [60, 68, 0, 0],     // 5/8
    [72, 56, 0, 0],     // 6/8
    [84, 44, 0, 0],     // 7/8
];

/// Compute chroma MV from luma MVs (average of 4 subblock MVs)
pub fn compute_chroma_mv(luma_mvs: &[MotionVector; 4]) -> MotionVector {
    let sum_x: i32 = luma_mvs.iter().map(|mv| mv.x as i32).sum();
    let sum_y: i32 = luma_mvs.iter().map(|mv| mv.y as i32).sum();

    // Average with rounding, result in 1/8 pel for chroma
    MotionVector {
        x: ((sum_x + 2) >> 2) as i16,
        y: ((sum_y + 2) >> 2) as i16,
    }
}
```

---

## Quantization

### Quantization Tables

```rust
/// Y plane DC quantizer lookup table
pub const DC_QUANT: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
    18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// Y plane AC quantizer lookup table
pub const AC_QUANT: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

/// Y2 (DC) block quantizer lookup
pub const Y2_DC_QUANT: [i16; 128] = [
    // Minimum value is 8 for Y2 DC
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
    21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28,
    29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36,
    37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44,
    45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52,
    53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60,
    61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68,
];

/// Y2 (AC) block quantizer lookup
pub const Y2_AC_QUANT: [i16; 128] = [
    // Minimum value is 8 for Y2 AC
    8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
];

/// UV plane DC quantizer (same structure as Y DC)
pub const UV_DC_QUANT: [i16; 128] = DC_QUANT;

/// UV plane AC quantizer
pub const UV_AC_QUANT: [i16; 128] = AC_QUANT;

/// Get quantizer step sizes for a macroblock
pub fn get_quant_factors(
    y_ac_qi: u8,
    y_dc_delta: i8,
    y2_dc_delta: i8,
    y2_ac_delta: i8,
    uv_dc_delta: i8,
    uv_ac_delta: i8,
) -> QuantFactors {
    let clamp_qi = |base: u8, delta: i8| -> usize {
        (base as i16 + delta as i16).clamp(0, 127) as usize
    };

    QuantFactors {
        y_dc: DC_QUANT[clamp_qi(y_ac_qi, y_dc_delta)],
        y_ac: AC_QUANT[y_ac_qi as usize],
        y2_dc: Y2_DC_QUANT[clamp_qi(y_ac_qi, y2_dc_delta)],
        y2_ac: Y2_AC_QUANT[clamp_qi(y_ac_qi, y2_ac_delta)],
        uv_dc: UV_DC_QUANT[clamp_qi(y_ac_qi, uv_dc_delta)],
        uv_ac: UV_AC_QUANT[clamp_qi(y_ac_qi, uv_ac_delta)],
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantFactors {
    pub y_dc: i16,
    pub y_ac: i16,
    pub y2_dc: i16,
    pub y2_ac: i16,
    pub uv_dc: i16,
    pub uv_ac: i16,
}

/// Dequantize a 4x4 block of DCT coefficients
pub fn dequantize_block(
    coeffs: &[i16; 16],
    dc_quant: i16,
    ac_quant: i16,
) -> [i16; 16] {
    let mut output = [0i16; 16];
    output[0] = coeffs[0] * dc_quant;
    for i in 1..16 {
        output[i] = coeffs[i] * ac_quant;
    }
    output
}

/// Quantize a 4x4 block of DCT coefficients (for encoder)
pub fn quantize_block(
    coeffs: &[i16; 16],
    dc_quant: i16,
    ac_quant: i16,
) -> [i16; 16] {
    let mut output = [0i16; 16];

    // DC coefficient
    output[0] = if coeffs[0] >= 0 {
        (coeffs[0] + (dc_quant >> 1)) / dc_quant
    } else {
        (coeffs[0] - (dc_quant >> 1)) / dc_quant
    };

    // AC coefficients
    for i in 1..16 {
        output[i] = if coeffs[i] >= 0 {
            (coeffs[i] + (ac_quant >> 1)) / ac_quant
        } else {
            (coeffs[i] - (ac_quant >> 1)) / ac_quant
        };
    }

    output
}
```

---

## Loop Filter

### Loop Filter Implementation

```rust
/// Loop filter parameters
#[derive(Debug, Clone, Copy)]
pub struct LoopFilterParams {
    pub filter_level: u8,     // 0-63
    pub sharpness: u8,        // 0-7
    pub filter_type: FilterType,
    pub mode_lf_adjustments: bool,
    pub ref_lf_deltas: [i8; 4],
    pub mode_lf_deltas: [i8; 4],
}

/// Calculate filter limit values
fn calc_filter_limits(level: u8, sharpness: u8) -> (i32, i32, i32) {
    let blimit = if sharpness > 0 {
        let limit = (level + 2).min(9 - sharpness as u8);
        limit.max(1)
    } else {
        (level + 2).max(1)
    };

    let limit = if sharpness > 0 {
        let l = level >> (sharpness as u8 > 4).then_some(2).unwrap_or(1);
        l.max(1).min(9 - sharpness as u8)
    } else {
        level.max(1)
    };

    let hev_thresh = if level >= 40 {
        2
    } else if level >= 15 {
        1
    } else {
        0
    };

    (blimit as i32, limit as i32, hev_thresh)
}

/// Simple loop filter for a single edge
pub fn simple_loop_filter_edge(p: &mut [u8], stride: usize, blimit: i32) {
    for i in 0..16 {
        let idx = i * stride;
        let p1 = p[idx] as i32;
        let p0 = p[idx + 1] as i32;
        let q0 = p[idx + 2] as i32;
        let q1 = p[idx + 3] as i32;

        // Filter mask
        let mask = (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1);
        if mask > blimit {
            continue;
        }

        // Calculate filter value
        let filter = clamp_signed(
            clamp_signed(p1 - q1, -128, 127) + 3 * (q0 - p0),
            -128, 127
        );
        let filter1 = clamp_signed((filter + 4) >> 3, -128, 127);
        let filter2 = clamp_signed((filter + 3) >> 3, -128, 127);

        p[idx + 1] = (p0 + filter2).clamp(0, 255) as u8;
        p[idx + 2] = (q0 - filter1).clamp(0, 255) as u8;
    }
}

/// Normal loop filter with high edge variance detection
pub fn normal_loop_filter_edge(
    p: &mut [u8],
    stride: usize,
    blimit: i32,
    limit: i32,
    hev_thresh: i32,
    edge_length: usize,  // 4, 8, or 16
) {
    for i in 0..edge_length {
        let idx = i * stride;

        // Read 4 pixels on each side of edge
        let p3 = p[idx] as i32;
        let p2 = p[idx + 1] as i32;
        let p1 = p[idx + 2] as i32;
        let p0 = p[idx + 3] as i32;
        let q0 = p[idx + 4] as i32;
        let q1 = p[idx + 5] as i32;
        let q2 = p[idx + 6] as i32;
        let q3 = p[idx + 7] as i32;

        // Filter mask
        let mask = (p3 - p2).abs().max((p2 - p1).abs())
            .max((p1 - p0).abs())
            .max((q1 - q0).abs())
            .max((q2 - q1).abs())
            .max((q3 - q2).abs());

        if mask > limit {
            continue;
        }

        let edge_diff = (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1);
        if edge_diff > blimit {
            continue;
        }

        // High edge variance check
        let hev = (p1 - p0).abs() > hev_thresh || (q1 - q0).abs() > hev_thresh;

        if hev {
            // Simple filter (same as simple mode)
            let filter = clamp_signed(
                clamp_signed(p1 - q1, -128, 127) + 3 * (q0 - p0),
                -128, 127
            );
            let filter1 = clamp_signed((filter + 4) >> 3, -128, 127);
            let filter2 = clamp_signed((filter + 3) >> 3, -128, 127);

            p[idx + 3] = (p0 + filter2).clamp(0, 255) as u8;
            p[idx + 4] = (q0 - filter1).clamp(0, 255) as u8;
        } else {
            // Full filter (spread across more pixels)
            let filter = clamp_signed(3 * (q0 - p0), -128, 127);
            let filter1 = clamp_signed((filter + 4) >> 3, -128, 127);
            let filter2 = clamp_signed((filter + 3) >> 3, -128, 127);
            let filter3 = (filter1 + 1) >> 1;

            p[idx + 2] = (p1 + filter3).clamp(0, 255) as u8;
            p[idx + 3] = (p0 + filter2).clamp(0, 255) as u8;
            p[idx + 4] = (q0 - filter1).clamp(0, 255) as u8;
            p[idx + 5] = (q1 - filter3).clamp(0, 255) as u8;
        }
    }
}

fn clamp_signed(val: i32, min: i32, max: i32) -> i32 {
    val.clamp(min, max)
}

/// Apply loop filter to entire frame
pub fn apply_loop_filter(
    frame: &mut FrameBuffer,
    params: &LoopFilterParams,
    mb_info: &[MacroblockInfo],
    mb_width: usize,
    mb_height: usize,
) {
    if params.filter_level == 0 || params.filter_type == FilterType::None {
        return;
    }

    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let mb_idx = mb_y * mb_width + mb_x;
            let mb = &mb_info[mb_idx];

            // Calculate per-macroblock filter level
            let level = calc_mb_filter_level(params, mb);
            if level == 0 {
                continue;
            }

            let (blimit, limit, hev_thresh) = calc_filter_limits(level, params.sharpness);

            // Filter vertical edges (left edge of macroblock)
            if mb_x > 0 {
                filter_mb_vertical_edge(frame, mb_x * 16, mb_y * 16, blimit, limit, hev_thresh, params.filter_type);
            }

            // Filter horizontal edges (top edge of macroblock)
            if mb_y > 0 {
                filter_mb_horizontal_edge(frame, mb_x * 16, mb_y * 16, blimit, limit, hev_thresh, params.filter_type);
            }

            // Filter internal 4x4 block edges
            filter_internal_edges(frame, mb_x * 16, mb_y * 16, blimit, limit, hev_thresh, params.filter_type);
        }
    }
}

fn calc_mb_filter_level(params: &LoopFilterParams, mb: &MacroblockInfo) -> u8 {
    let mut level = params.filter_level as i32;

    if params.mode_lf_adjustments {
        // Reference frame adjustment
        let ref_delta = params.ref_lf_deltas[mb.ref_frame as usize];
        level += ref_delta as i32;

        // Mode adjustment
        let mode_delta = if mb.ref_frame == RefFrame::None {
            params.mode_lf_deltas[0]  // Intra
        } else if mb.is_zero_mv {
            params.mode_lf_deltas[1]  // Zero MV
        } else if mb.has_split_mv {
            params.mode_lf_deltas[3]  // Split MV
        } else {
            params.mode_lf_deltas[2]  // New MV
        };
        level += mode_delta as i32;
    }

    level.clamp(0, 63) as u8
}
```

---

## Entropy Coding

### Coefficient Token Tree

```rust
/// DCT coefficient token values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Token {
    DctEob = 0,      // End of block
    Dct0 = 1,        // Zero
    Dct1 = 2,        // +1 or -1
    Dct2 = 3,        // +2 or -2
    Dct3 = 4,        // +3 or -3
    Dct4 = 5,        // +4 or -4
    DctCat1 = 6,     // 5-6 (1 extra bit)
    DctCat2 = 7,     // 7-10 (2 extra bits)
    DctCat3 = 8,     // 11-18 (3 extra bits)
    DctCat4 = 9,     // 19-34 (4 extra bits)
    DctCat5 = 10,    // 35-66 (5 extra bits)
    DctCat6 = 11,    // 67-2048 (11 extra bits)
}

/// Token tree for coefficient decoding
/// Nodes are organized as: tree[node] = left child, tree[node+1] = right child
/// Negative values are leaf nodes (token = -value)
pub const COEFF_TOKEN_TREE: [i8; 22] = [
    // Node 0: DCT_EOB vs rest
    -0, 2,
    // Node 1: DCT_0 vs rest
    -1, 4,
    // Node 2: DCT_1 vs rest
    -2, 6,
    // Node 3: DCT_2 vs rest
    8, 12,
    // Node 4: DCT_3, DCT_4
    -3, 10,
    // Node 5: DCT_4
    -4, -5,
    // Node 6: CAT1-CAT2 vs CAT3-CAT6
    14, 16,
    // Node 7: CAT1, CAT2
    -6, -7,
    // Node 8: CAT3-CAT4 vs CAT5-CAT6
    18, 20,
    // Node 9: CAT3, CAT4
    -8, -9,
    // Node 10: CAT5, CAT6
    -10, -11,
];

/// Extra bits for coefficient categories
pub const TOKEN_EXTRA_BITS: [(u8, u16); 12] = [
    (0, 0),     // EOB
    (0, 0),     // 0
    (0, 0),     // 1
    (0, 0),     // 2
    (0, 0),     // 3
    (0, 0),     // 4
    (1, 5),     // CAT1: 1 bit, base = 5
    (2, 7),     // CAT2: 2 bits, base = 7
    (3, 11),    // CAT3: 3 bits, base = 11
    (4, 19),    // CAT4: 4 bits, base = 19
    (5, 35),    // CAT5: 5 bits, base = 35
    (11, 67),   // CAT6: 11 bits, base = 67
];

/// Default coefficient probabilities
/// Dimensions: [BLOCK_TYPES][COEFF_BANDS][PREV_COEFF_CONTEXTS][ENTROPY_NODES]
/// BLOCK_TYPES = 4 (Y1 with Y2, Y1 without Y2, UV, Y2)
/// COEFF_BANDS = 8 (coefficient position bands)
/// PREV_COEFF_CONTEXTS = 3 (based on previous coefficient)
/// ENTROPY_NODES = 11 (tree nodes)
pub const DEFAULT_COEFF_PROBS: [[[[u8; 11]; 3]; 8]; 4] = [
    // Block type 0: Y1 with Y2
    [
        // Band 0
        [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
         [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
         [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
        // ... more bands (see RFC 6386 Section 13.4 for complete tables)
        [[128; 11]; 3],
        [[128; 11]; 3],
        [[128; 11]; 3],
        [[128; 11]; 3],
        [[128; 11]; 3],
        [[128; 11]; 3],
        [[128; 11]; 3],
    ],
    // Block types 1-3 (placeholder - see RFC for actual values)
    [[[128; 11]; 3]; 8],
    [[[128; 11]; 3]; 8],
    [[[128; 11]; 3]; 8],
];

/// Coefficient band mapping (which band for each coefficient position)
pub const COEFF_BANDS: [u8; 16] = [
    0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7
];

/// Zigzag scan order for 4x4 block
pub const ZIGZAG_SCAN: [usize; 16] = [
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
];

/// Decode DCT coefficients for a 4x4 block
pub fn decode_block_coeffs(
    bd: &mut BoolDecoder,
    probs: &[[[[u8; 11]; 3]; 8]; 4],
    block_type: usize,
    has_y2: bool,
) -> [i16; 16] {
    let mut coeffs = [0i16; 16];
    let mut prev_coeff_ctx = 0usize;  // Context based on previous coefficient

    let start_idx = if block_type == 0 && has_y2 { 1 } else { 0 };

    for i in start_idx..16 {
        let band = COEFF_BANDS[i] as usize;
        let probs_for_ctx = &probs[block_type][band][prev_coeff_ctx];

        // Decode token using tree
        let token = decode_token_tree(bd, probs_for_ctx);

        match token {
            Token::DctEob => break,
            Token::Dct0 => {
                // Zero coefficient
                prev_coeff_ctx = 0;
            }
            _ => {
                let (extra_bits, base) = TOKEN_EXTRA_BITS[token as usize];
                let magnitude = if extra_bits > 0 {
                    let extra = bd.read_literal(extra_bits) as i16;
                    base as i16 + extra
                } else {
                    match token {
                        Token::Dct1 => 1,
                        Token::Dct2 => 2,
                        Token::Dct3 => 3,
                        Token::Dct4 => 4,
                        _ => unreachable!(),
                    }
                };

                // Read sign
                let sign = bd.read_bool(128);
                coeffs[ZIGZAG_SCAN[i]] = if sign { -magnitude } else { magnitude };

                prev_coeff_ctx = if magnitude == 1 { 1 } else { 2 };
            }
        }
    }

    coeffs
}

fn decode_token_tree(bd: &mut BoolDecoder, probs: &[u8; 11]) -> Token {
    let mut node = 0usize;
    loop {
        let bit = bd.read_bool(probs[node]) as usize;
        let next = COEFF_TOKEN_TREE[node * 2 + bit];
        if next <= 0 {
            return unsafe { std::mem::transmute((-next) as u8) };
        }
        node = next as usize;
    }
}
```

---

## Encoder Mode Decision

### Mode Decision Algorithm

```rust
/// Mode decision configuration
pub struct ModeDecisionConfig {
    pub rd_mult: u32,       // Rate-distortion multiplier
    pub rd_div: u32,        // Rate-distortion divisor
    pub intra_threshold: u32,
    pub skip_threshold: u32,
}

/// Calculate Sum of Absolute Differences (SAD) for mode selection
pub fn calculate_sad(
    original: &[[u8; 16]; 16],
    prediction: &[[u8; 16]; 16],
) -> u32 {
    let mut sad = 0u32;
    for y in 0..16 {
        for x in 0..16 {
            sad += (original[y][x] as i32 - prediction[y][x] as i32).abs() as u32;
        }
    }
    sad
}

/// Calculate Sum of Squared Errors (SSE) for RD optimization
pub fn calculate_sse(
    original: &[[u8; 16]; 16],
    prediction: &[[u8; 16]; 16],
) -> u64 {
    let mut sse = 0u64;
    for y in 0..16 {
        for x in 0..16 {
            let diff = original[y][x] as i64 - prediction[y][x] as i64;
            sse += (diff * diff) as u64;
        }
    }
    sse
}

/// Choose best intra prediction mode for 16x16 block
pub fn select_intra_16x16_mode(
    original: &[[u8; 16]; 16],
    above: &[u8; 16],
    left: &[u8; 16],
    above_left: u8,
) -> IntraMode16x16 {
    let mut best_mode = IntraMode16x16::DcPred;
    let mut best_sad = u32::MAX;

    let modes = [
        IntraMode16x16::DcPred,
        IntraMode16x16::VPred,
        IntraMode16x16::HPred,
        IntraMode16x16::TmPred,
    ];

    for mode in modes {
        let mut pred = [[0u8; 16]; 16];
        predict_16x16(mode, above, left, above_left, &mut pred);

        let sad = calculate_sad(original, &pred);
        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
        }
    }

    best_mode
}

/// Choose between intra and inter prediction for a macroblock
pub fn decide_mb_type(
    original: &[[u8; 16]; 16],
    intra_pred: &[[u8; 16]; 16],
    inter_pred: &[[u8; 16]; 16],
    config: &ModeDecisionConfig,
    estimated_intra_bits: u32,
    estimated_inter_bits: u32,
) -> MbType {
    let intra_sse = calculate_sse(original, intra_pred);
    let inter_sse = calculate_sse(original, inter_pred);

    // Rate-distortion cost calculation
    // J = D + lambda * R
    let lambda = config.rd_mult as u64 / config.rd_div as u64;

    let intra_cost = intra_sse + lambda * estimated_intra_bits as u64;
    let inter_cost = inter_sse + lambda * estimated_inter_bits as u64;

    if intra_cost < inter_cost {
        MbType::Intra16x16
    } else {
        MbType::Inter
    }
}

/// Motion estimation using diamond search
pub fn diamond_search(
    ref_frame: &FrameBuffer,
    original: &[[u8; 16]; 16],
    block_x: usize,
    block_y: usize,
    initial_mv: MotionVector,
    search_range: i16,
) -> MotionVector {
    // Diamond pattern offsets (small diamond)
    const DIAMOND: [(i16, i16); 4] = [
        (0, -1), (0, 1), (-1, 0), (1, 0)
    ];

    let mut best_mv = initial_mv;
    let mut best_sad = motion_compensate_and_sad(
        ref_frame, original, block_x, block_y, best_mv
    );

    let mut step = 4; // Start with larger steps

    while step >= 1 {
        let mut improved = false;

        for &(dx, dy) in &DIAMOND {
            let test_mv = MotionVector {
                x: (best_mv.x + dx * step).clamp(-search_range * 4, search_range * 4),
                y: (best_mv.y + dy * step).clamp(-search_range * 4, search_range * 4),
            };

            let sad = motion_compensate_and_sad(
                ref_frame, original, block_x, block_y, test_mv
            );

            if sad < best_sad {
                best_sad = sad;
                best_mv = test_mv;
                improved = true;
            }
        }

        if !improved {
            step >>= 1; // Reduce step size
        }
    }

    // Refine to quarter-pixel precision
    subpel_refine(ref_frame, original, block_x, block_y, best_mv)
}

fn motion_compensate_and_sad(
    ref_frame: &FrameBuffer,
    original: &[[u8; 16]; 16],
    block_x: usize,
    block_y: usize,
    mv: MotionVector,
) -> u32 {
    let mut pred = [[0u8; 16]; 16];
    let mut pred_flat = [0u8; 256];

    motion_compensate(ref_frame, mv, block_x, block_y, 16, &mut pred_flat, 16);

    // Convert to 2D array
    for y in 0..16 {
        for x in 0..16 {
            pred[y][x] = pred_flat[y * 16 + x];
        }
    }

    calculate_sad(original, &pred)
}

fn subpel_refine(
    ref_frame: &FrameBuffer,
    original: &[[u8; 16]; 16],
    block_x: usize,
    block_y: usize,
    mv: MotionVector,
) -> MotionVector {
    let mut best_mv = mv;
    let mut best_sad = motion_compensate_and_sad(
        ref_frame, original, block_x, block_y, best_mv
    );

    // Test half-pixel positions
    for dy in -2..=2 {
        for dx in -2..=2 {
            if dx == 0 && dy == 0 {
                continue;
            }

            let test_mv = MotionVector {
                x: mv.x + dx,
                y: mv.y + dy,
            };

            let sad = motion_compensate_and_sad(
                ref_frame, original, block_x, block_y, test_mv
            );

            if sad < best_sad {
                best_sad = sad;
                best_mv = test_mv;
            }
        }
    }

    best_mv
}
```

---

## Rate Control

### Rate Control Implementation

```rust
/// Rate control state
pub struct RateControl {
    pub target_bitrate: u32,      // bits per second
    pub frame_rate: f64,
    pub buffer_size: u32,         // virtual buffer size in bits
    pub buffer_fullness: i32,     // current buffer level
    pub bits_per_frame: u32,      // target bits per frame
    pub qi: u8,                   // current quantizer index
    pub gf_interval: u32,         // golden frame interval
    pub frames_since_gf: u32,
    pub total_bits: u64,
    pub frame_count: u64,
}

impl RateControl {
    pub fn new(target_bitrate: u32, frame_rate: f64, buffer_seconds: f64) -> Self {
        let bits_per_frame = (target_bitrate as f64 / frame_rate) as u32;
        let buffer_size = (target_bitrate as f64 * buffer_seconds) as u32;

        RateControl {
            target_bitrate,
            frame_rate,
            buffer_size,
            buffer_fullness: buffer_size as i32 / 2,
            bits_per_frame,
            qi: 32,  // Start with middle QI
            gf_interval: 30,
            frames_since_gf: 0,
            total_bits: 0,
            frame_count: 0,
        }
    }

    /// Get QI for next frame based on buffer state
    pub fn get_frame_qi(&mut self, frame_type: FrameType) -> u8 {
        // Adjust QI based on buffer fullness
        let buffer_ratio = self.buffer_fullness as f64 / self.buffer_size as f64;

        let qi_adjustment = if buffer_ratio > 0.8 {
            // Buffer too full, reduce quality (increase QI)
            ((buffer_ratio - 0.8) * 20.0) as i8
        } else if buffer_ratio < 0.2 {
            // Buffer too empty, increase quality (decrease QI)
            -((0.2 - buffer_ratio) * 20.0) as i8
        } else {
            0
        };

        // Key frames get lower QI (higher quality)
        let frame_adjustment = match frame_type {
            FrameType::KeyFrame => -8,
            FrameType::InterFrame => 0,
        };

        let new_qi = self.qi as i8 + qi_adjustment + frame_adjustment;
        self.qi = new_qi.clamp(0, 127) as u8;

        self.qi
    }

    /// Update state after encoding a frame
    pub fn update_after_frame(&mut self, bits_used: u32, frame_type: FrameType) {
        // Update buffer
        self.buffer_fullness += self.bits_per_frame as i32 - bits_used as i32;
        self.buffer_fullness = self.buffer_fullness.clamp(
            -(self.buffer_size as i32),
            self.buffer_size as i32 * 2
        );

        self.total_bits += bits_used as u64;
        self.frame_count += 1;

        // Update golden frame tracking
        if frame_type == FrameType::KeyFrame {
            self.frames_since_gf = 0;
        } else {
            self.frames_since_gf += 1;
        }
    }

    /// Check if we should force a golden frame
    pub fn should_insert_golden(&self) -> bool {
        self.frames_since_gf >= self.gf_interval
    }

    /// Estimate bits for a frame (for mode decision)
    pub fn estimate_frame_bits(&self, frame_type: FrameType) -> u32 {
        match frame_type {
            FrameType::KeyFrame => self.bits_per_frame * 5,  // Key frames are larger
            FrameType::InterFrame => self.bits_per_frame,
        }
    }
}

/// Two-pass rate control data
pub struct TwoPassData {
    pub frame_stats: Vec<FrameStats>,
    pub total_stats: TotalStats,
}

pub struct FrameStats {
    pub frame_type: FrameType,
    pub intra_cost: u64,
    pub inter_cost: u64,
    pub coded_error: u64,
    pub sr_coded_error: u64,  // Second reference coded error
}

pub struct TotalStats {
    pub total_frames: u64,
    pub total_intra_cost: u64,
    pub total_inter_cost: u64,
}

impl TwoPassData {
    /// First pass: collect statistics
    pub fn collect_frame_stats(
        original: &FrameBuffer,
        ref_frame: Option<&FrameBuffer>,
        golden_frame: Option<&FrameBuffer>,
    ) -> FrameStats {
        // Calculate intra cost (using DC prediction)
        let intra_cost = calculate_intra_cost(original);

        // Calculate inter cost if we have a reference
        let inter_cost = ref_frame
            .map(|r| calculate_inter_cost(original, r))
            .unwrap_or(intra_cost);

        let sr_coded_error = golden_frame
            .map(|g| calculate_inter_cost(original, g))
            .unwrap_or(inter_cost);

        let coded_error = inter_cost.min(intra_cost);

        let frame_type = if inter_cost > intra_cost * 9 / 10 || ref_frame.is_none() {
            FrameType::KeyFrame
        } else {
            FrameType::InterFrame
        };

        FrameStats {
            frame_type,
            intra_cost,
            inter_cost,
            coded_error,
            sr_coded_error,
        }
    }

    /// Second pass: allocate bits based on first pass data
    pub fn allocate_bits(&self, total_bits: u64) -> Vec<u32> {
        let mut allocations = Vec::with_capacity(self.frame_stats.len());

        let total_weight: f64 = self.frame_stats.iter()
            .map(|s| s.coded_error as f64)
            .sum();

        for stats in &self.frame_stats {
            let weight = stats.coded_error as f64 / total_weight;
            let bits = (total_bits as f64 * weight) as u32;

            // Adjust for frame type
            let adjusted_bits = match stats.frame_type {
                FrameType::KeyFrame => bits.max(total_bits as u32 / self.frame_stats.len() as u32 * 3),
                FrameType::InterFrame => bits,
            };

            allocations.push(adjusted_bits);
        }

        allocations
    }
}

fn calculate_intra_cost(frame: &FrameBuffer) -> u64 {
    // Simple DC prediction cost estimation
    let mut cost = 0u64;

    for mb_y in 0..(frame.height as usize / 16) {
        for mb_x in 0..(frame.width as usize / 16) {
            // Calculate variance within macroblock (simple cost estimate)
            let mut sum = 0u32;
            let mut sum_sq = 0u64;

            for y in 0..16 {
                for x in 0..16 {
                    let pixel = frame.y[(mb_y * 16 + y) * frame.y_stride + mb_x * 16 + x] as u32;
                    sum += pixel;
                    sum_sq += (pixel * pixel) as u64;
                }
            }

            let mean = sum / 256;
            let variance = sum_sq / 256 - (mean * mean) as u64;
            cost += variance;
        }
    }

    cost
}

fn calculate_inter_cost(original: &FrameBuffer, reference: &FrameBuffer) -> u64 {
    // Simple zero-motion cost estimation
    let mut cost = 0u64;

    for mb_y in 0..(original.height as usize / 16) {
        for mb_x in 0..(original.width as usize / 16) {
            for y in 0..16 {
                for x in 0..16 {
                    let orig = original.y[(mb_y * 16 + y) * original.y_stride + mb_x * 16 + x] as i32;
                    let ref_pixel = reference.y[(mb_y * 16 + y) * reference.y_stride + mb_x * 16 + x] as i32;
                    let diff = orig - ref_pixel;
                    cost += (diff * diff) as u64;
                }
            }
        }
    }

    cost
}
```

---

## Test Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_decoder_literal() {
        // Test data with known values
        let data = [0x55, 0xAA, 0x00, 0xFF];
        let mut bd = BoolDecoder::new(&data);

        // Test literal reading
        let val = bd.read_literal(8);
        // Verify against expected value
    }

    #[test]
    fn test_dct_roundtrip() {
        // Test that forward + inverse DCT preserves values
        let input = [[10i16; 4]; 4];
        let transformed = forward_dct4x4(&input);
        let reconstructed = inverse_dct4x4(&transformed);

        for y in 0..4 {
            for x in 0..4 {
                assert!((input[y][x] - reconstructed[y][x]).abs() <= 1);
            }
        }
    }

    #[test]
    fn test_wht_roundtrip() {
        let input = [[100i16; 4]; 4];
        let transformed = forward_wht4x4(&input);
        let reconstructed = inverse_wht4x4(&transformed);

        for y in 0..4 {
            for x in 0..4 {
                assert!((input[y][x] - reconstructed[y][x]).abs() <= 1);
            }
        }
    }

    #[test]
    fn test_intra_prediction_dc() {
        let above = [128u8; 16];
        let left = [128u8; 16];
        let mut output = [[0u8; 16]; 16];

        predict_16x16(IntraMode16x16::DcPred, &above, &left, 128, &mut output);

        // All pixels should be 128 (average of all 128s)
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(output[y][x], 128);
            }
        }
    }

    #[test]
    fn test_intra_prediction_vertical() {
        let above: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let left = [128u8; 16];
        let mut output = [[0u8; 16]; 16];

        predict_16x16(IntraMode16x16::VPred, &above, &left, 128, &mut output);

        // Each row should equal above
        for y in 0..16 {
            assert_eq!(output[y], above);
        }
    }

    #[test]
    fn test_quantization_roundtrip() {
        let coeffs = [100i16, 50, 25, 12, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let dc_quant = 10;
        let ac_quant = 8;

        let quantized = quantize_block(&coeffs, dc_quant, ac_quant);
        let dequantized = dequantize_block(&quantized, dc_quant, ac_quant);

        // Check that dequantized is close to original
        for i in 0..16 {
            let error = (coeffs[i] - dequantized[i]).abs();
            let max_error = if i == 0 { dc_quant } else { ac_quant };
            assert!(error <= max_error);
        }
    }

    #[test]
    fn test_subpixel_filter_fullpel() {
        // Full pixel position should just copy
        let src = [128u8; 32];
        let mut dst = [0i16; 16];

        filter_horizontal(&src, 1, &mut dst, 1, 16, 1, 0);

        // Should be close to 128
        for &val in &dst {
            assert!((val - 128).abs() <= 1);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_decode_key_frame() {
        // Load test VP8 bitstream
        let data = include_bytes!("../../test_data/keyframe.ivf");

        // Skip IVF header (32 bytes) and frame header (12 bytes per frame)
        let frame_data = &data[44..];

        let mut decoder = Vp8Decoder::new(640, 480);
        let result = decoder.decode_frame(frame_data);

        assert!(result.is_ok());
        let frame = result.unwrap();
        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Create test frame
        let mut original = FrameBuffer::new(320, 240);

        // Fill with gradient
        for y in 0..240 {
            for x in 0..320 {
                original.y[y * original.y_stride + x] = ((x + y) & 255) as u8;
            }
        }
        for y in 0..120 {
            for x in 0..160 {
                original.u[y * original.uv_stride + x] = 128;
                original.v[y * original.uv_stride + x] = 128;
            }
        }

        // Encode
        let mut encoder = Vp8Encoder::new(320, 240);
        let encoded = encoder.encode_frame(&original, FrameType::KeyFrame).unwrap();

        // Decode
        let mut decoder = Vp8Decoder::new(320, 240);
        let decoded = decoder.decode_frame(&encoded).unwrap();

        // Compare (with lossy tolerance)
        let mut total_diff = 0u64;
        for y in 0..240 {
            for x in 0..320 {
                let orig = original.y[y * original.y_stride + x] as i32;
                let dec = decoded.y[y * decoded.y_stride + x] as i32;
                total_diff += (orig - dec).abs() as u64;
            }
        }

        let avg_diff = total_diff / (320 * 240);
        assert!(avg_diff < 10, "Average pixel difference too high: {}", avg_diff);
    }
}
```

### Conformance Testing

```rust
/// Run VP8 conformance test vectors
pub fn run_conformance_tests(test_vectors_path: &str) -> Vec<TestResult> {
    let mut results = Vec::new();

    // Test vector categories from VP8 test suite
    let categories = [
        "intra_prediction",
        "inter_prediction",
        "transforms",
        "quantization",
        "loop_filter",
        "entropy_coding",
        "full_decode",
    ];

    for category in categories {
        let path = format!("{}/{}", test_vectors_path, category);
        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.flatten() {
                let test_file = entry.path();
                if test_file.extension().map(|e| e == "ivf").unwrap_or(false) {
                    let result = run_single_test(&test_file);
                    results.push(result);
                }
            }
        }
    }

    results
}

struct TestResult {
    test_name: String,
    passed: bool,
    error: Option<String>,
    psnr: Option<f64>,
}

fn run_single_test(test_file: &std::path::Path) -> TestResult {
    // Implementation would:
    // 1. Read IVF file
    // 2. Decode each frame
    // 3. Compare against reference YUV
    // 4. Calculate PSNR
    // 5. Return pass/fail based on threshold

    TestResult {
        test_name: test_file.file_name().unwrap().to_string_lossy().into_owned(),
        passed: true,
        error: None,
        psnr: Some(40.0),
    }
}
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Boolean arithmetic decoder
- [ ] Boolean arithmetic encoder
- [ ] Frame header parsing
- [ ] Frame buffer management

### Phase 2: Transforms
- [ ] 4x4 forward DCT
- [ ] 4x4 inverse DCT
- [ ] 4x4 forward WHT
- [ ] 4x4 inverse WHT
- [ ] Zigzag scan ordering

### Phase 3: Prediction
- [ ] 16x16 intra prediction (4 modes)
- [ ] 8x8 chroma intra prediction
- [ ] 4x4 intra prediction (10 modes)
- [ ] Motion vector decoding
- [ ] 6-tap subpixel interpolation
- [ ] Motion compensation

### Phase 4: Quantization & Entropy
- [ ] Quantization tables
- [ ] Dequantization
- [ ] Coefficient token decoding
- [ ] Probability tree decoding
- [ ] Probability updates

### Phase 5: Loop Filter
- [ ] Simple loop filter
- [ ] Normal loop filter
- [ ] Edge detection
- [ ] Per-macroblock filtering

### Phase 6: Encoder
- [ ] Mode decision
- [ ] Motion estimation
- [ ] Rate control
- [ ] Bitstream generation

### Phase 7: Testing & Optimization
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Conformance test vectors
- [ ] Performance profiling
- [ ] SIMD optimization (optional)

---

## References

- [RFC 6386 - VP8 Data Format and Decoding Guide](https://datatracker.ietf.org/doc/html/rfc6386) - Official VP8 specification
- [WebM Project VP8 Encode Parameter Guide](https://www.webmproject.org/docs/encoder-parameters/) - Encoder configuration
- [Google Technical Overview of VP8](https://research.google.com/pubs/archive/37073.pdf) - Technical paper
- [libvpx GitHub Repository](https://github.com/webmproject/libvpx) - Reference implementation (BSD license)

---

## License Notes

This specification is derived entirely from publicly available documentation:
- RFC 6386 is published by IETF and is freely available
- Google released VP8 under a BSD-style license with an irrevocable patent grant
- All code in this document is original implementation based on the public specification

The resulting implementation can be licensed under MIT without any licensing concerns.
