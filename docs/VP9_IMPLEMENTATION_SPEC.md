# VP9 Codec Implementation Specification for Pure Rust

**Version:** 1.0
**Date:** December 2025
**License Target:** MIT
**Reference Specification:** VP9 Bitstream & Decoding Process Specification v0.6/v0.7

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Structure](#2-file-structure)
3. [Binary Arithmetic Range Coder](#3-binary-arithmetic-range-coder)
4. [Frame Structure and Headers](#4-frame-structure-and-headers)
5. [Superblock Partitioning](#5-superblock-partitioning)
6. [Transform System](#6-transform-system)
7. [Quantization](#7-quantization)
8. [Intra Prediction](#8-intra-prediction)
9. [Inter Prediction](#9-inter-prediction)
10. [Loop Filter](#10-loop-filter)
11. [Segmentation](#11-segmentation)
12. [Tile Structure](#12-tile-structure)
13. [Probability Tables](#13-probability-tables)
14. [Encoder Rate-Distortion Optimization](#14-encoder-rate-distortion-optimization)
15. [Data Structures](#15-data-structures)
16. [Test Strategy](#16-test-strategy)

---

## 1. Overview

### 1.1 What is VP9?

VP9 is a royalty-free video codec developed by Google as the successor to VP8. It achieves approximately 50% better compression than VP8 while maintaining similar computational complexity for decoding.

### 1.2 Key Differences from VP8

| Feature | VP8 | VP9 |
|---------|-----|-----|
| Max block size | 16x16 | 64x64 (superblocks) |
| Transform sizes | 4x4, 8x8, 16x16 | 4x4, 8x8, 16x16, 32x32 |
| Transform types | DCT, WHT | DCT, ADST, WHT |
| Intra modes | 4 basic + 6 directional | 10 modes with angular prediction |
| Motion precision | 1/4 pel | 1/8 pel |
| Reference frames | 3 | 3 (with compound prediction) |
| Entropy coding | Boolean coder | Range coder with backward adaptation |

### 1.3 VP9 Profiles

| Profile | Bit Depth | Chroma Subsampling |
|---------|-----------|-------------------|
| 0 | 8 bit | 4:2:0 |
| 1 | 8 bit | 4:2:0, 4:2:2, 4:4:4 |
| 2 | 10-12 bit | 4:2:0 |
| 3 | 10-12 bit | 4:2:0, 4:2:2, 4:4:4 |

### 1.4 Reference Implementations (MIT/Apache/BSD Licensed)

- **vp9-parser** (Rust, MIT/Apache/Zlib): https://github.com/hasenbanck/vp9-parser
- **cros-codecs** (Rust, BSD-3-Clause): https://github.com/chromeos/cros-codecs
- **libvpx** (C, BSD-3-Clause): https://github.com/webmproject/libvpx

---

## 2. File Structure

### 2.1 Proposed Directory Layout

```
src/codec/vp9/
├── mod.rs                    # Module exports and feature flags
├── decoder.rs                # Main decoder implementation
├── encoder.rs                # Main encoder implementation
├── bitstream/
│   ├── mod.rs               # Bitstream reading/writing
│   ├── reader.rs            # Boolean/range decoder
│   └── writer.rs            # Boolean/range encoder
├── entropy/
│   ├── mod.rs               # Entropy coding module
│   ├── range_coder.rs       # Range coder implementation
│   ├── probabilities.rs     # Probability tables and updates
│   └── trees.rs             # Tree structures for symbol coding
├── frame/
│   ├── mod.rs               # Frame-level structures
│   ├── header.rs            # Uncompressed + compressed headers
│   ├── superblock.rs        # Superblock processing
│   └── tile.rs              # Tile structure and parallel decode
├── transform/
│   ├── mod.rs               # Transform module
│   ├── dct.rs               # DCT transforms (4x4 to 32x32)
│   ├── adst.rs              # ADST transforms
│   ├── wht.rs               # Walsh-Hadamard (lossless)
│   └── inverse.rs           # Inverse transform dispatcher
├── prediction/
│   ├── mod.rs               # Prediction module
│   ├── intra.rs             # Intra prediction modes
│   ├── inter.rs             # Inter prediction and motion comp
│   ├── motion.rs            # Motion vector handling
│   └── filter.rs            # Interpolation filters
├── quantization/
│   ├── mod.rs               # Quantization module
│   ├── tables.rs            # Quantization tables
│   └── dequant.rs           # Dequantization
├── loop_filter/
│   ├── mod.rs               # Loop filter module
│   └── deblock.rs           # Deblocking filter
├── segmentation/
│   ├── mod.rs               # Segmentation features
│   └── map.rs               # Segment map handling
├── tables/
│   ├── mod.rs               # Static tables
│   ├── probability.rs       # Default probability tables
│   ├── scan.rs              # Coefficient scan tables
│   └── quant.rs             # Quantization lookup tables
└── util/
    ├── mod.rs               # Utilities
    └── math.rs              # Fixed-point math helpers
```

---

## 3. Binary Arithmetic Range Coder

### 3.1 Overview

VP9 uses a binary arithmetic range coder for entropy coding. Unlike VP8's simple boolean coder, VP9's range coder operates with a 16-bit range and uses backward adaptation of probabilities.

### 3.2 Range Coder State

```rust
/// Range coder state for decoding
pub struct RangeCoder {
    /// Current range value (16-bit)
    range: u16,
    /// Current coded value being decoded
    value: u32,
    /// Number of bits consumed
    bits_consumed: u32,
    /// Input buffer position
    position: usize,
}

impl RangeCoder {
    /// Initialize with minimum 16-bit range
    pub const MIN_RANGE: u16 = 128;

    /// Create new range coder from buffer
    pub fn new(data: &[u8]) -> Self {
        let mut coder = Self {
            range: 255,
            value: 0,
            bits_consumed: 0,
            position: 0,
        };
        // Initialize value from first bytes
        coder.value = ((data[0] as u32) << 8) | (data[1] as u32);
        coder.position = 2;
        coder
    }
}
```

### 3.3 Boolean Decoding Algorithm

```rust
impl RangeCoder {
    /// Read a single boolean with given probability (0-255 scale)
    /// probability represents P(bit == 0)
    pub fn read_bool(&mut self, data: &[u8], probability: u8) -> bool {
        // Split calculation
        let split = 1 + (((self.range as u32 - 1) * probability as u32) >> 8);
        let split = split as u16;

        let big_split = (split as u32) << 8;

        if self.value < big_split {
            // Bit is 0
            self.range = split;
            self.renormalize(data);
            false
        } else {
            // Bit is 1
            self.range -= split;
            self.value -= big_split;
            self.renormalize(data);
            true
        }
    }

    /// Renormalize after each symbol
    fn renormalize(&mut self, data: &[u8]) {
        while self.range < Self::MIN_RANGE {
            self.range <<= 1;
            self.value = (self.value << 1) | self.read_bit(data);
        }
    }

    /// Read a literal value of n bits
    pub fn read_literal(&mut self, data: &[u8], n: usize) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bool(data, 128) as u32;
        }
        value
    }
}
```

### 3.4 Range Encoder for Output

```rust
/// Range coder for encoding
pub struct RangeEncoder {
    /// Low end of range
    low: u64,
    /// Range size
    range: u32,
    /// Output buffer
    output: Vec<u8>,
    /// Pending carry count
    outstanding_bytes: u32,
    /// First byte flag
    first_byte: bool,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0xFF00,
            output: Vec::new(),
            outstanding_bytes: 0,
            first_byte: true,
        }
    }

    /// Encode a boolean with given probability
    pub fn write_bool(&mut self, bit: bool, probability: u8) {
        let split = 1 + (((self.range - 1) * probability as u32) >> 8);

        if !bit {
            self.range = split;
        } else {
            self.low += split as u64;
            self.range -= split;
        }

        self.renormalize();
    }

    fn renormalize(&mut self) {
        while self.range < 0x80 {
            // Output byte and handle carry propagation
            self.shift_out();
            self.range <<= 1;
        }
    }
}
```

### 3.5 Tree-Based Symbol Decoding

VP9 uses binary trees to represent multi-symbol alphabets:

```rust
/// Decode a symbol using a probability tree
pub fn read_tree<const N: usize>(
    coder: &mut RangeCoder,
    data: &[u8],
    tree: &[(i8, i8)],  // (left_child, right_child), negative = leaf
    probs: &[u8; N],
) -> u8 {
    let mut node = 0usize;

    loop {
        let bit = coder.read_bool(data, probs[node]);
        let child = if bit { tree[node].1 } else { tree[node].0 };

        if child < 0 {
            return (-child - 1) as u8;
        }
        node = child as usize;
    }
}
```

---

## 4. Frame Structure and Headers

### 4.1 Frame Types

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameType {
    KeyFrame = 0,
    InterFrame = 1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020 = 5,
    Reserved = 6,
    Srgb = 7,
}
```

### 4.2 Uncompressed Header Structure

```rust
/// VP9 uncompressed frame header
pub struct UncompressedHeader {
    /// Frame marker (must be 2)
    pub frame_marker: u8,
    /// Profile (0-3)
    pub profile: u8,
    /// Show existing frame flag
    pub show_existing_frame: bool,
    /// Frame to show (if show_existing_frame)
    pub frame_to_show_map_idx: u8,
    /// Frame type (key/inter)
    pub frame_type: FrameType,
    /// Show frame flag
    pub show_frame: bool,
    /// Error resilient mode
    pub error_resilient_mode: bool,

    // Key frame specific
    /// Color space
    pub color_space: ColorSpace,
    /// Color range (studio/full)
    pub color_range: bool,
    /// Bit depth (8/10/12)
    pub bit_depth: u8,
    /// Subsampling X
    pub subsampling_x: bool,
    /// Subsampling Y
    pub subsampling_y: bool,

    /// Frame width
    pub width: u16,
    /// Frame height
    pub height: u16,
    /// Render width (may differ from coded)
    pub render_width: u16,
    /// Render height
    pub render_height: u16,

    // Inter frame specific
    /// Intra-only flag (for non-key frames)
    pub intra_only: bool,
    /// Reset frame context
    pub reset_frame_context: u8,
    /// Refresh frame flags (8 bits for 8 reference slots)
    pub refresh_frame_flags: u8,
    /// Reference frame indices (3 refs)
    pub ref_frame_idx: [u8; 3],
    /// Reference frame sign bias
    pub ref_frame_sign_bias: [bool; 4],
    /// Allow high precision motion vectors
    pub allow_high_precision_mv: bool,
    /// Interpolation filter type
    pub interpolation_filter: InterpolationFilter,

    /// Refresh frame context flag
    pub refresh_frame_context: bool,
    /// Frame parallel decoding mode
    pub frame_parallel_decoding_mode: bool,
    /// Frame context index
    pub frame_context_idx: u8,

    // Loop filter params
    pub loop_filter: LoopFilterParams,

    // Quantization params
    pub quantization: QuantizationParams,

    // Segmentation
    pub segmentation: SegmentationParams,

    // Tile info
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,

    /// Header size in bytes (uncompressed portion)
    pub header_size_in_bytes: u16,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationFilter {
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}
```

### 4.3 Header Parsing Pseudocode

```rust
impl UncompressedHeader {
    pub fn parse(data: &[u8]) -> Result<(Self, usize), Error> {
        let mut reader = BitReader::new(data);

        // Frame marker (2 bits, must be 2)
        let frame_marker = reader.read_bits(2)? as u8;
        if frame_marker != 2 {
            return Err(Error::InvalidFrameMarker);
        }

        // Profile (bits 0 and 2)
        let profile_low = reader.read_bit()?;
        let profile_high = reader.read_bit()?;
        let profile = (profile_high << 1) | profile_low;

        // For profile 3, check reserved bit
        if profile == 3 {
            let reserved = reader.read_bit()?;
            if reserved {
                return Err(Error::InvalidProfile);
            }
        }

        // Show existing frame
        let show_existing_frame = reader.read_bit()?;
        if show_existing_frame {
            let frame_to_show_map_idx = reader.read_bits(3)? as u8;
            return Ok((Self::show_existing(frame_to_show_map_idx), reader.position()));
        }

        // Frame type
        let frame_type = if reader.read_bit()? {
            FrameType::InterFrame
        } else {
            FrameType::KeyFrame
        };

        let show_frame = reader.read_bit()?;
        let error_resilient_mode = reader.read_bit()?;

        // ... continue parsing based on frame type

        Ok((header, reader.position()))
    }
}
```

### 4.4 Compressed Header

The compressed header is encoded using the range coder and contains:

```rust
/// Compressed header contains probability updates and mode info
pub struct CompressedHeader {
    /// Transform mode (if not lossless)
    pub tx_mode: TxMode,
    /// Probability updates for various syntax elements
    pub prob_updates: ProbabilityUpdates,
}

#[derive(Debug, Clone, Copy)]
pub enum TxMode {
    Only4x4 = 0,
    Allow8x8 = 1,
    Allow16x16 = 2,
    Allow32x32 = 3,
    TxModeSelect = 4,  // Per-block selection
}
```

---

## 5. Superblock Partitioning

### 5.1 Partition Types

VP9 uses a recursive quadtree partitioning starting from 64x64 superblocks:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Partition {
    None = 0,       // No split, use full block
    Horizontal = 1, // Split into top and bottom halves
    Vertical = 2,   // Split into left and right halves
    Split = 3,      // Split into 4 quadrants (recursive)
}

/// Block sizes in VP9
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockSize {
    Block4x4 = 0,
    Block4x8 = 1,
    Block8x4 = 2,
    Block8x8 = 3,
    Block8x16 = 4,
    Block16x8 = 5,
    Block16x16 = 6,
    Block16x32 = 7,
    Block32x16 = 8,
    Block32x32 = 9,
    Block32x64 = 10,
    Block64x32 = 11,
    Block64x64 = 12,
}
```

### 5.2 Partition Tree

```
64x64 Superblock
├── None -> 64x64 block
├── Horizontal -> 64x32 + 64x32
├── Vertical -> 32x64 + 32x64
└── Split -> 4x 32x32
    ├── None -> 32x32 block
    ├── Horizontal -> 32x16 + 32x16
    ├── Vertical -> 16x32 + 16x32
    └── Split -> 4x 16x16
        ├── None -> 16x16 block
        ├── Horizontal -> 16x8 + 16x8
        ├── Vertical -> 8x16 + 8x16
        └── Split -> 4x 8x8
            ├── None -> 8x8 block
            ├── Horizontal -> 8x4 + 8x4
            ├── Vertical -> 4x8 + 4x8
            └── Split -> 4x 4x4
```

### 5.3 Partition Decoding

```rust
/// Superblock decoder
pub struct SuperblockDecoder<'a> {
    coder: &'a mut RangeCoder,
    frame_context: &'a FrameContext,
}

impl<'a> SuperblockDecoder<'a> {
    /// Decode partition tree recursively
    pub fn decode_partition(
        &mut self,
        data: &[u8],
        mi_row: usize,
        mi_col: usize,
        block_size: BlockSize,
        above_context: &[u8],
        left_context: &[u8],
    ) -> Result<PartitionTree, Error> {
        // Get context for partition probability
        let ctx = self.get_partition_context(above_context, left_context, block_size);

        // Read partition type
        let partition = if block_size >= BlockSize::Block8x8 {
            let probs = &self.frame_context.partition_probs[ctx];
            read_partition_tree(self.coder, data, probs)?
        } else {
            Partition::None
        };

        let sub_size = get_subsize(block_size, partition);

        match partition {
            Partition::None => {
                self.decode_block(data, mi_row, mi_col, block_size)
            }
            Partition::Horizontal => {
                let half_height = block_size.height_mi() / 2;
                self.decode_block(data, mi_row, mi_col, sub_size)?;
                self.decode_block(data, mi_row + half_height, mi_col, sub_size)
            }
            Partition::Vertical => {
                let half_width = block_size.width_mi() / 2;
                self.decode_block(data, mi_row, mi_col, sub_size)?;
                self.decode_block(data, mi_row, mi_col + half_width, sub_size)
            }
            Partition::Split => {
                let half = block_size.size_mi() / 2;
                for i in 0..2 {
                    for j in 0..2 {
                        self.decode_partition(
                            data,
                            mi_row + i * half,
                            mi_col + j * half,
                            sub_size,
                            above_context,
                            left_context,
                        )?;
                    }
                }
                Ok(())
            }
        }
    }

    fn get_partition_context(
        &self,
        above: &[u8],
        left: &[u8],
        block_size: BlockSize,
    ) -> usize {
        // Context based on neighboring partitions
        let above_partition = above.iter().any(|&x| x != 0) as usize;
        let left_partition = left.iter().any(|&x| x != 0) as usize;
        let bsl = block_size.log2() - 1;

        (left_partition * 2 + above_partition) + bsl * 4
    }
}
```

---

## 6. Transform System

### 6.1 Transform Types

VP9 uses three transform types with sizes 4x4, 8x8, 16x16, and 32x32:

1. **DCT (Discrete Cosine Transform)**: Used for inter blocks and some intra modes
2. **ADST (Asymmetric Discrete Sine Transform)**: Used for intra blocks with edge prediction
3. **WHT (Walsh-Hadamard Transform)**: Used for lossless mode (4x4 only)

### 6.2 Transform Mode Selection

```rust
/// Transform type for rows and columns
#[derive(Debug, Clone, Copy)]
pub enum TxType {
    DctDct = 0,    // DCT rows, DCT columns
    AdstDct = 1,   // DCT rows, ADST columns
    DctAdst = 2,   // ADST rows, DCT columns
    AdstAdst = 3,  // ADST rows, ADST columns
}

/// Select transform type based on intra mode
pub fn get_tx_type(prediction_mode: IntraMode, tx_size: TxSize) -> TxType {
    // Only 4x4 blocks use ADST for intra modes
    if tx_size != TxSize::Tx4x4 {
        return TxType::DctDct;
    }

    match prediction_mode {
        // Vertical modes: DCT horizontal, ADST vertical
        IntraMode::V | IntraMode::D63 | IntraMode::D117 => TxType::AdstDct,

        // Horizontal modes: ADST horizontal, DCT vertical
        IntraMode::H | IntraMode::D153 | IntraMode::D207 => TxType::DctAdst,

        // Diagonal modes: ADST both
        IntraMode::D135 | IntraMode::D45 | IntraMode::TM => TxType::AdstAdst,

        // DC mode: DCT both
        IntraMode::DC => TxType::DctDct,
    }
}
```

### 6.3 DCT Implementation (Butterfly Structure)

```rust
/// 4x4 DCT butterfly implementation
pub fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    // Constants for 4x4 DCT
    const COS_PI_8_64: i32 = 15137;   // cos(pi/8) * (1 << 14)
    const SIN_PI_8_64: i32 = 6270;    // sin(pi/8) * (1 << 14)
    const COS_PI_16_64: i32 = 16069;  // cos(pi/16) * (1 << 14)
    const SIN_PI_16_64: i32 = 3196;   // sin(pi/16) * (1 << 14)

    // Stage 1: Butterfly
    let a = input[0] + input[2];
    let b = input[0] - input[2];

    // Stage 2: Rotation
    let c = ((input[1] * COS_PI_8_64) - (input[3] * SIN_PI_8_64) + (1 << 13)) >> 14;
    let d = ((input[1] * SIN_PI_8_64) + (input[3] * COS_PI_8_64) + (1 << 13)) >> 14;

    // Stage 3: Output
    output[0] = a + d;
    output[1] = b + c;
    output[2] = b - c;
    output[3] = a - d;
}

/// 8x8 DCT using 4x4 as building block
pub fn idct8(input: &[i32; 8], output: &mut [i32; 8]) {
    // First stage: Process even and odd indices
    let mut even = [0i32; 4];
    let mut odd = [0i32; 4];

    for i in 0..4 {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    // Apply 4x4 transforms
    let mut even_out = [0i32; 4];
    let mut odd_out = [0i32; 4];
    idct4(&even, &mut even_out);
    idct4_odd(&odd, &mut odd_out);

    // Combine
    for i in 0..4 {
        output[i] = even_out[i] + odd_out[3 - i];
        output[7 - i] = even_out[i] - odd_out[3 - i];
    }
}

/// 2D inverse transform (rows then columns)
pub fn inverse_transform_2d<const N: usize>(
    coeffs: &[[i32; N]; N],
    output: &mut [[i32; N]; N],
    tx_type: TxType,
) {
    let mut intermediate = [[0i32; N]; N];

    // Transform rows
    for i in 0..N {
        let row_fn = match tx_type {
            TxType::DctDct | TxType::DctAdst => idct_row::<N>,
            TxType::AdstDct | TxType::AdstAdst => iadst_row::<N>,
        };
        row_fn(&coeffs[i], &mut intermediate[i]);
    }

    // Transform columns
    for j in 0..N {
        let col_fn = match tx_type {
            TxType::DctDct | TxType::AdstDct => idct_col::<N>,
            TxType::DctAdst | TxType::AdstAdst => iadst_col::<N>,
        };

        let mut col_in = [0i32; N];
        let mut col_out = [0i32; N];
        for i in 0..N {
            col_in[i] = intermediate[i][j];
        }
        col_fn(&col_in, &mut col_out);
        for i in 0..N {
            output[i][j] = col_out[i];
        }
    }
}
```

### 6.4 ADST Implementation

```rust
/// 4x4 ADST (Asymmetric Discrete Sine Transform)
pub fn iadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    // ADST constants
    const SIN_PI_9: i32 = 5283;
    const SIN_2PI_9: i32 = 9929;
    const SIN_PI_3: i32 = 13377;
    const SIN_4PI_9: i32 = 15212;

    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // Intermediate calculations
    let s0 = SIN_PI_9 * x0;
    let s1 = SIN_2PI_9 * x0;
    let s2 = SIN_PI_3 * x1;
    let s3 = SIN_4PI_9 * x2;
    let s4 = SIN_PI_9 * x2;
    let s5 = SIN_2PI_9 * x3;
    let s6 = SIN_4PI_9 * x3;

    // Output with rounding
    let round_shift = |x: i32| -> i32 { (x + (1 << 13)) >> 14 };

    output[0] = round_shift(s0 + s3 + s5);
    output[1] = round_shift(s2);
    output[2] = round_shift(s1 - s4 - s6);
    output[3] = round_shift(s0 + s1 - s3 - s4 + s5 + s6);
}

/// 8x8 ADST
pub fn iadst8(input: &[i32; 8], output: &mut [i32; 8]) {
    // Constants derived from sin/cos tables
    const SINPI_1_9: i32 = 5283;
    const SINPI_2_9: i32 = 9929;
    // ... additional constants

    // Implementation follows butterfly structure similar to DCT
    // but with sine-based rotations

    // Stage 1: Input permutation and first rotations
    // Stage 2: Cross-additions
    // Stage 3: Final rotations and output
}
```

### 6.5 Walsh-Hadamard Transform (Lossless Mode)

```rust
/// 4x4 Walsh-Hadamard Transform (lossless)
pub fn iwht4(input: &[i32; 4], output: &mut [i32; 4]) {
    // WHT uses only additions/subtractions (no multiplies)
    let a = input[0];
    let b = input[1];
    let c = input[2];
    let d = input[3];

    // First stage
    let e = a + b;
    let f = a - b;
    let g = c + d;
    let h = c - d;

    // Second stage
    output[0] = e + g;
    output[1] = f + h;
    output[2] = e - g;
    output[3] = f - h;
}

/// 2D 4x4 WHT for lossless mode
pub fn iwht4x4_2d(coeffs: &[[i32; 4]; 4], output: &mut [[i32; 4]; 4]) {
    let mut temp = [[0i32; 4]; 4];

    // Rows
    for i in 0..4 {
        iwht4(&coeffs[i], &mut temp[i]);
    }

    // Columns
    for j in 0..4 {
        let mut col = [0i32; 4];
        for i in 0..4 {
            col[i] = temp[i][j];
        }
        let mut out_col = [0i32; 4];
        iwht4(&col, &mut out_col);
        for i in 0..4 {
            output[i][j] = out_col[i];
        }
    }
}
```

---

## 7. Quantization

### 7.1 Quantization Parameters

VP9 uses separate quantizers for DC and AC coefficients of luma and chroma:

```rust
/// Quantization parameters from frame header
pub struct QuantizationParams {
    /// Base QIndex (0-255)
    pub base_q_idx: u8,
    /// Delta for Y DC (signed)
    pub y_dc_delta: i8,
    /// Delta for UV DC (signed)
    pub uv_dc_delta: i8,
    /// Delta for UV AC (signed)
    pub uv_ac_delta: i8,
}

/// Dequantization lookup tables (spec section 10.2)
pub struct DequantTables {
    /// DC dequant values indexed by QIndex
    pub dc_quant: [i16; 256],
    /// AC dequant values indexed by QIndex
    pub ac_quant: [i16; 256],
}
```

### 7.2 Dequantization Tables

```rust
/// DC dequantization table for 8-bit depth
pub const DC_QUANT_8BIT: [i16; 256] = [
    4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19,
    20, 21, 22, 23, 24, 25, 26, 26, 27, 28, 29, 30, 31, 32, 32, 33,
    34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47,
    48, 48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60,
    // ... complete table from specification
    // Values continue up to index 255
];

/// AC dequantization table for 8-bit depth
pub const AC_QUANT_8BIT: [i16; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    // ... complete table from specification
];

impl DequantTables {
    /// Get dequantization value
    pub fn get_dequant(
        &self,
        q_index: u8,
        is_dc: bool,
        plane: Plane,
        params: &QuantizationParams,
    ) -> i16 {
        let delta = match (plane, is_dc) {
            (Plane::Y, true) => params.y_dc_delta,
            (Plane::Y, false) => 0,
            (_, true) => params.uv_dc_delta,
            (_, false) => params.uv_ac_delta,
        };

        let adjusted = (q_index as i16 + delta as i16).clamp(0, 255) as usize;

        if is_dc {
            self.dc_quant[adjusted]
        } else {
            self.ac_quant[adjusted]
        }
    }
}
```

### 7.3 Coefficient Dequantization

```rust
/// Dequantize a block of coefficients
pub fn dequantize_coeffs<const N: usize>(
    coeffs: &[i16; N * N],
    dequant_dc: i16,
    dequant_ac: i16,
    output: &mut [i32; N * N],
) {
    // DC coefficient uses DC dequant
    output[0] = coeffs[0] as i32 * dequant_dc as i32;

    // AC coefficients use AC dequant
    for i in 1..N * N {
        output[i] = coeffs[i] as i32 * dequant_ac as i32;
    }
}
```

---

## 8. Intra Prediction

### 8.1 Intra Prediction Modes

VP9 supports 10 intra prediction modes:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntraMode {
    DC = 0,       // Average of above and left
    V = 1,        // Vertical (copy from above)
    H = 2,        // Horizontal (copy from left)
    D45 = 3,      // 45-degree diagonal (down-right)
    D135 = 4,     // 135-degree diagonal (down-left)
    D117 = 5,     // 117-degree
    D153 = 6,     // 153-degree
    D207 = 7,     // 207-degree (horizontal-up)
    D63 = 8,      // 63-degree (vertical-right)
    TM = 9,       // True Motion (gradient)
}
```

### 8.2 Prediction Algorithms

```rust
/// Intra predictor
pub struct IntraPredictor {
    /// Above samples (reference row)
    above: Vec<u8>,
    /// Left samples (reference column)
    left: Vec<u8>,
    /// Top-left corner sample
    top_left: u8,
}

impl IntraPredictor {
    /// DC prediction: average of above and left
    pub fn predict_dc(&self, block: &mut [u8], stride: usize, size: usize,
                      have_above: bool, have_left: bool) {
        let sum = if have_above && have_left {
            let above_sum: u32 = self.above[..size].iter().map(|&x| x as u32).sum();
            let left_sum: u32 = self.left[..size].iter().map(|&x| x as u32).sum();
            (above_sum + left_sum + size as u32) / (2 * size as u32)
        } else if have_above {
            let above_sum: u32 = self.above[..size].iter().map(|&x| x as u32).sum();
            (above_sum + (size / 2) as u32) / size as u32
        } else if have_left {
            let left_sum: u32 = self.left[..size].iter().map(|&x| x as u32).sum();
            (left_sum + (size / 2) as u32) / size as u32
        } else {
            128 // Default value
        };

        let dc = sum as u8;
        for row in 0..size {
            for col in 0..size {
                block[row * stride + col] = dc;
            }
        }
    }

    /// Vertical prediction: copy from above row
    pub fn predict_v(&self, block: &mut [u8], stride: usize, size: usize) {
        for row in 0..size {
            for col in 0..size {
                block[row * stride + col] = self.above[col];
            }
        }
    }

    /// Horizontal prediction: copy from left column
    pub fn predict_h(&self, block: &mut [u8], stride: usize, size: usize) {
        for row in 0..size {
            for col in 0..size {
                block[row * stride + col] = self.left[row];
            }
        }
    }

    /// True Motion prediction (gradient)
    pub fn predict_tm(&self, block: &mut [u8], stride: usize, size: usize) {
        for row in 0..size {
            for col in 0..size {
                // TM = left + above - top_left, clamped to [0, 255]
                let pred = self.left[row] as i16 + self.above[col] as i16
                           - self.top_left as i16;
                block[row * stride + col] = pred.clamp(0, 255) as u8;
            }
        }
    }

    /// D45 diagonal prediction (45-degree down-right)
    pub fn predict_d45(&self, block: &mut [u8], stride: usize, size: usize) {
        for row in 0..size {
            for col in 0..size {
                let idx = row + col + 1;
                if idx < 2 * size - 1 {
                    // Average of above[idx] and above[idx+1]
                    let pred = (self.above[idx] as u16 + self.above[idx + 1] as u16 + 1) / 2;
                    block[row * stride + col] = pred as u8;
                } else {
                    block[row * stride + col] = self.above[2 * size - 1];
                }
            }
        }
    }

    /// D135 diagonal prediction (135-degree down-left)
    pub fn predict_d135(&self, block: &mut [u8], stride: usize, size: usize) {
        // Build extended reference samples
        let mut ref_samples = vec![0u8; 2 * size + 1];
        ref_samples[0] = self.top_left;
        ref_samples[1..=size].copy_from_slice(&self.above[..size]);

        for row in 0..size {
            for col in 0..size {
                let idx = (size as i32 - 1 - row as i32 + col as i32) as usize;
                if col > row {
                    block[row * stride + col] = ref_samples[idx];
                } else if col < row {
                    block[row * stride + col] = self.left[row - col - 1];
                } else {
                    block[row * stride + col] = self.top_left;
                }
            }
        }
    }

    // Additional angular modes (D63, D117, D153, D207) follow similar patterns
    // with different angle calculations and reference sample interpolation
}
```

### 8.3 Smooth Prediction Filter

VP9 can apply a low-pass filter to predictions for smoother results:

```rust
/// Apply smoothing filter to prediction
pub fn apply_intra_filter(pred: &mut [u8], stride: usize, size: usize,
                          filter_type: FilterType) {
    if filter_type == FilterType::None {
        return;
    }

    // Apply appropriate filter based on type
    match filter_type {
        FilterType::Vertical => apply_vertical_filter(pred, stride, size),
        FilterType::Horizontal => apply_horizontal_filter(pred, stride, size),
        FilterType::Both => {
            apply_vertical_filter(pred, stride, size);
            apply_horizontal_filter(pred, stride, size);
        }
        FilterType::None => {}
    }
}
```

---

## 9. Inter Prediction

### 9.1 Reference Frame Management

VP9 maintains 8 reference frame slots with 3 active references per frame:

```rust
/// Reference frame indices
#[derive(Debug, Clone, Copy)]
pub enum RefFrame {
    None = 0,
    Last = 1,      // Most recent frame
    Golden = 2,    // Temporal anchor (less frequently updated)
    AltRef = 3,    // Alternate reference (can be forward/backward)
}

/// Reference frame buffer
pub struct RefFrameBuffer {
    /// 8 reference frame slots
    pub slots: [Option<Frame>; 8],
    /// Mapping of ref types to slot indices
    pub ref_frame_idx: [usize; 3],
    /// Sign bias for motion vectors
    pub ref_frame_sign_bias: [bool; 4],
}
```

### 9.2 Motion Vector Structure

```rust
/// Motion vector with 1/8-pel precision
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// Horizontal component (1/8-pel units)
    pub row: i16,
    /// Vertical component (1/8-pel units)
    pub col: i16,
}

impl MotionVector {
    /// Full-pel component
    pub fn row_int(&self) -> i16 { self.row >> 3 }
    pub fn col_int(&self) -> i16 { self.col >> 3 }

    /// Sub-pel component (0-7)
    pub fn row_frac(&self) -> u8 { (self.row & 7) as u8 }
    pub fn col_frac(&self) -> u8 { (self.col & 7) as u8 }
}

/// Inter prediction mode
#[derive(Debug, Clone, Copy)]
pub enum InterMode {
    NearestMv = 0,  // Use nearest MV from neighbors
    NearMv = 1,     // Use near MV from neighbors
    ZeroMv = 2,     // Zero motion vector
    NewMv = 3,      // Explicitly coded new MV
}
```

### 9.3 Motion Compensation

```rust
/// 8-tap subpixel filter kernels
pub const SUBPEL_FILTERS_8TAP: [[i8; 8]; 16] = [
    // Regular filter
    [0, 0, 0, 128, 0, 0, 0, 0],      // Offset 0 (full-pel)
    [0, 1, -5, 126, 8, -3, 1, 0],    // Offset 1
    [0, 1, -10, 122, 18, -6, 2, -1], // Offset 2
    [-1, 2, -13, 116, 29, -9, 3, -1],// Offset 3
    [-1, 3, -16, 108, 41, -12, 4, -1],// Offset 4
    [-1, 4, -17, 98, 53, -14, 5, -2],// Offset 5
    [-1, 4, -18, 87, 65, -16, 6, -1],// Offset 6
    [-1, 4, -18, 76, 76, -18, 4, -1],// Offset 7 (half-pel)
    // ... symmetric for 8-15
];

/// Perform motion compensation
pub fn motion_compensate(
    ref_frame: &Frame,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    width: usize,
    height: usize,
    filter_type: InterpolationFilter,
    output: &mut [u8],
    stride: usize,
) {
    let ref_x = block_x as i32 + mv.col_int() as i32;
    let ref_y = block_y as i32 + mv.row_int() as i32;
    let frac_x = mv.col_frac();
    let frac_y = mv.row_frac();

    let filter = get_filter_kernel(filter_type, frac_x, frac_y);

    // Apply separable filter (horizontal then vertical)
    let mut temp = vec![0i16; (width + 7) * (height + 7)];

    // Horizontal filter
    for y in 0..height + 7 {
        for x in 0..width {
            let mut sum = 0i32;
            for k in 0..8 {
                let src_x = (ref_x + x as i32 - 3 + k as i32).clamp(0, ref_frame.width as i32 - 1);
                let src_y = (ref_y + y as i32 - 3).clamp(0, ref_frame.height as i32 - 1);
                sum += ref_frame.get_pixel(src_x as usize, src_y as usize) as i32
                       * filter.horizontal[k] as i32;
            }
            temp[y * width + x] = ((sum + 64) >> 7) as i16;
        }
    }

    // Vertical filter
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += temp[(y + k) * width + x] as i32 * filter.vertical[k] as i32;
            }
            output[y * stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}
```

### 9.4 Compound Prediction

VP9's compound prediction averages predictions from two reference frames:

```rust
/// Compound prediction (two references)
pub fn compound_predict(
    ref1_frame: &Frame,
    ref2_frame: &Frame,
    mv1: MotionVector,
    mv2: MotionVector,
    block_x: usize,
    block_y: usize,
    width: usize,
    height: usize,
    filter_type: InterpolationFilter,
    output: &mut [u8],
    stride: usize,
) {
    let mut pred1 = vec![0u8; width * height];
    let mut pred2 = vec![0u8; width * height];

    // Get predictions from both references
    motion_compensate(ref1_frame, mv1, block_x, block_y, width, height,
                      filter_type, &mut pred1, width);
    motion_compensate(ref2_frame, mv2, block_x, block_y, width, height,
                      filter_type, &mut pred2, width);

    // Average the predictions (VP9 uses simple 50/50 blend)
    for y in 0..height {
        for x in 0..width {
            let p1 = pred1[y * width + x] as u16;
            let p2 = pred2[y * width + x] as u16;
            output[y * stride + x] = ((p1 + p2 + 1) >> 1) as u8;
        }
    }
}
```

---

## 10. Loop Filter

### 10.1 Loop Filter Parameters

```rust
/// Loop filter parameters from frame header
pub struct LoopFilterParams {
    /// Filter level (0-63)
    pub level: u8,
    /// Sharpness (0-7)
    pub sharpness: u8,
    /// Delta enabled flag
    pub delta_enabled: bool,
    /// Delta update flag
    pub delta_update: bool,
    /// Reference frame deltas
    pub ref_deltas: [i8; 4],
    /// Mode deltas
    pub mode_deltas: [i8; 2],
}
```

### 10.2 Filter Thresholds

```rust
/// Calculate loop filter thresholds
pub fn calc_filter_thresholds(level: u8, sharpness: u8) -> (u8, u8, u8) {
    // Interior limit
    let mut interior_limit = level;
    if sharpness > 0 {
        interior_limit >>= if sharpness > 4 { 2 } else { 1 };
        interior_limit = interior_limit.min(9 - sharpness);
    }
    interior_limit = interior_limit.max(1);

    // HEV threshold (high edge variance)
    let hev_threshold = if level >= 40 {
        2
    } else if level >= 20 {
        1
    } else {
        0
    };

    // Edge limit
    let edge_limit = (level * 2 + interior_limit) as u8;

    (interior_limit, hev_threshold, edge_limit)
}
```

### 10.3 Deblocking Filter Algorithm

```rust
/// Apply loop filter to block edge
pub fn filter_block_edge(
    pixels: &mut [u8],
    stride: usize,
    is_vertical: bool,
    filter_length: usize,  // 4, 8, or 16
    interior_limit: u8,
    hev_threshold: u8,
    edge_limit: u8,
) {
    // Get samples around edge
    let (p, q) = if is_vertical {
        get_vertical_samples(pixels, stride, filter_length)
    } else {
        get_horizontal_samples(pixels, stride, filter_length)
    };

    // Check if filtering should be applied
    if !should_filter(&p, &q, interior_limit, edge_limit) {
        return;
    }

    // Check high edge variance
    let is_hev = is_high_edge_variance(&p, &q, hev_threshold);

    // Apply appropriate filter
    if is_hev {
        // Use 4-tap filter (affects p1, p0, q0, q1)
        filter4(&mut p[..2], &mut q[..2], interior_limit);
    } else {
        match filter_length {
            4 => filter4(&mut p[..2], &mut q[..2], interior_limit),
            8 => filter8(&mut p[..4], &mut q[..4], interior_limit),
            16 => filter16(&mut p[..8], &mut q[..8], interior_limit),
            _ => {}
        }
    }

    // Write filtered samples back
    write_samples(pixels, stride, is_vertical, &p, &q);
}

/// Check if samples should be filtered
fn should_filter(p: &[u8], q: &[u8], interior: u8, edge: u8) -> bool {
    // Check flatness condition
    let p0 = p[0] as i16;
    let p1 = p[1] as i16;
    let q0 = q[0] as i16;
    let q1 = q[1] as i16;

    (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= edge as i16
        && (p1 - p0).abs() <= interior as i16
        && (q1 - q0).abs() <= interior as i16
}

/// 4-tap loop filter
fn filter4(p: &mut [u8], q: &mut [u8], limit: u8) {
    let p0 = p[0] as i16;
    let p1 = p[1] as i16;
    let q0 = q[0] as i16;
    let q1 = q[1] as i16;

    // Calculate filter adjustment
    let filter_value = clamp_filter(3 * (q0 - p0) + (p1 - q1));

    // Apply filter
    let filter1 = (filter_value + 4) >> 3;
    let filter2 = (filter_value + 3) >> 3;

    p[0] = clamp_pixel(p0 + filter2);
    q[0] = clamp_pixel(q0 - filter1);

    // Adjust outer pixels
    let outer = (filter1 + 1) >> 1;
    p[1] = clamp_pixel(p1 + outer);
    q[1] = clamp_pixel(q1 - outer);
}

fn clamp_filter(value: i16) -> i16 {
    value.clamp(-128, 127)
}

fn clamp_pixel(value: i16) -> u8 {
    value.clamp(0, 255) as u8
}
```

---

## 11. Segmentation

### 11.1 Segmentation Features

VP9 supports up to 8 segments with per-segment adjustments:

```rust
/// Segmentation parameters
pub struct SegmentationParams {
    /// Segmentation enabled
    pub enabled: bool,
    /// Update map flag
    pub update_map: bool,
    /// Temporal update flag
    pub temporal_update: bool,
    /// Update data flag
    pub update_data: bool,
    /// Absolute vs delta values
    pub abs_or_delta_update: bool,
    /// Per-segment features
    pub features: [SegmentFeatures; 8],
    /// Segment prediction probabilities
    pub pred_probs: [u8; 3],
}

/// Features that can be adjusted per segment
pub struct SegmentFeatures {
    /// Alt quantizer
    pub alt_q: Option<i16>,
    /// Alt loop filter
    pub alt_lf: Option<i8>,
    /// Reference frame override
    pub ref_frame: Option<RefFrame>,
    /// Skip flag
    pub skip: bool,
}
```

### 11.2 Segment Map Coding

```rust
/// Decode segment ID for a block
pub fn decode_segment_id(
    coder: &mut RangeCoder,
    data: &[u8],
    params: &SegmentationParams,
    above_seg_id: Option<u8>,
    left_seg_id: Option<u8>,
    prev_seg_id: Option<u8>,
) -> u8 {
    if params.temporal_update {
        // Use temporal prediction
        let pred = prev_seg_id.unwrap_or(0);
        let ctx = get_temporal_context(above_seg_id, left_seg_id);

        if coder.read_bool(data, params.pred_probs[ctx]) {
            // Prediction is correct
            return pred;
        }
    }

    // Decode segment ID directly using tree
    let tree_probs = get_segment_tree_probs(params);
    read_segment_tree(coder, data, &tree_probs)
}
```

---

## 12. Tile Structure

### 12.1 Tile Parameters

```rust
/// Tile configuration
pub struct TileConfig {
    /// Log2 of tile columns (0-6, meaning 1-64 tiles)
    pub tile_cols_log2: u8,
    /// Log2 of tile rows (0-2, meaning 1-4 tiles)
    pub tile_rows_log2: u8,
    /// Calculated tile column boundaries
    pub col_starts: Vec<usize>,
    /// Calculated tile row boundaries
    pub row_starts: Vec<usize>,
}

impl TileConfig {
    /// Calculate tile boundaries from frame dimensions
    pub fn new(width_sb: usize, height_sb: usize,
               tile_cols_log2: u8, tile_rows_log2: u8) -> Self {
        let num_cols = 1usize << tile_cols_log2;
        let num_rows = 1usize << tile_rows_log2;

        let mut col_starts = Vec::with_capacity(num_cols + 1);
        let mut row_starts = Vec::with_capacity(num_rows + 1);

        // Calculate evenly spaced tile boundaries
        for i in 0..=num_cols {
            col_starts.push(i * width_sb / num_cols);
        }
        for i in 0..=num_rows {
            row_starts.push(i * height_sb / num_rows);
        }

        Self {
            tile_cols_log2,
            tile_rows_log2,
            col_starts,
            row_starts,
        }
    }

    /// Get number of tiles
    pub fn num_tiles(&self) -> usize {
        (1 << self.tile_cols_log2) * (1 << self.tile_rows_log2)
    }
}
```

### 12.2 Parallel Tile Decoding

```rust
/// Tile decoder for parallel processing
pub struct TileDecoder {
    /// Tile index
    pub index: usize,
    /// Tile data slice
    pub data: Vec<u8>,
    /// Tile boundaries in superblocks
    pub sb_col_start: usize,
    pub sb_col_end: usize,
    pub sb_row_start: usize,
    pub sb_row_end: usize,
}

impl TileDecoder {
    /// Decode tile independently
    pub fn decode(&self, frame_context: &FrameContext) -> Result<TileData, Error> {
        let mut coder = RangeCoder::new(&self.data);
        let mut tile_data = TileData::new(
            self.sb_col_end - self.sb_col_start,
            self.sb_row_end - self.sb_row_start,
        );

        // Decode superblocks in raster order within tile
        for sb_row in self.sb_row_start..self.sb_row_end {
            for sb_col in self.sb_col_start..self.sb_col_end {
                decode_superblock(&mut coder, &self.data, frame_context,
                                  sb_row, sb_col, &mut tile_data)?;
            }
        }

        Ok(tile_data)
    }
}

/// Parallel tile decoding using threads
pub fn decode_tiles_parallel(
    tiles: Vec<TileDecoder>,
    frame_context: &FrameContext,
) -> Result<Vec<TileData>, Error> {
    use std::thread;

    let handles: Vec<_> = tiles.into_iter().map(|tile| {
        let ctx = frame_context.clone();
        thread::spawn(move || tile.decode(&ctx))
    }).collect();

    handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect()
}
```

---

## 13. Probability Tables

### 13.1 Probability Update Mechanism

VP9 uses both forward (explicit) and backward (adaptive) probability updates:

```rust
/// Frame probability context
pub struct FrameContext {
    /// Partition probabilities [ctx][partition]
    pub partition_probs: [[u8; 3]; 16],
    /// Skip probabilities [ctx]
    pub skip_probs: [u8; 3],
    /// Inter mode probabilities [ctx][mode]
    pub inter_mode_probs: [[u8; 3]; 7],
    /// Intra mode probabilities (Y) [size][mode]
    pub y_mode_probs: [[u8; 9]; 4],
    /// Intra mode probabilities (UV) [mode]
    pub uv_mode_probs: [[u8; 9]; 10],
    /// Reference frame probabilities
    pub comp_ref_prob: [u8; 5],
    pub single_ref_prob: [[u8; 2]; 5],
    /// Motion vector probabilities
    pub mv_probs: MvProbabilities,
    /// Coefficient probabilities
    pub coef_probs: CoefProbabilities,
    /// Transform size probabilities
    pub tx_probs: TxProbabilities,
}

impl FrameContext {
    /// Reset to default probabilities
    pub fn reset_to_defaults(&mut self) {
        self.partition_probs = DEFAULT_PARTITION_PROBS;
        self.skip_probs = DEFAULT_SKIP_PROBS;
        self.inter_mode_probs = DEFAULT_INTER_MODE_PROBS;
        // ... reset all probability tables
    }

    /// Apply backward adaptation based on decoded symbols
    pub fn adapt(&mut self, counts: &SymbolCounts) {
        // Merge counts with existing probabilities
        for ctx in 0..16 {
            for i in 0..3 {
                self.partition_probs[ctx][i] = merge_prob(
                    self.partition_probs[ctx][i],
                    counts.partition[ctx][i],
                    counts.partition[ctx][3], // Total count
                );
            }
        }
        // ... adapt all probability tables
    }
}
```

### 13.2 Default Probability Tables

```rust
/// Default partition probabilities
pub const DEFAULT_PARTITION_PROBS: [[u8; 3]; 16] = [
    // [above_partition][left_partition] indexed contexts
    [199, 122, 141],  // No neighbors partitioned
    [147, 63, 159],   // Above partitioned
    [148, 133, 118],  // Left partitioned
    [121, 104, 114],  // Both partitioned
    // ... for different block size levels
    [174, 73, 87],
    [92, 41, 83],
    [82, 99, 50],
    [53, 39, 39],
    [177, 58, 59],
    [68, 26, 63],
    [52, 79, 25],
    [17, 14, 12],
    [222, 34, 30],
    [72, 16, 44],
    [58, 32, 12],
    [10, 7, 6],
];

/// Default intra mode probabilities (Y plane, 4x4 blocks)
pub const DEFAULT_Y_MODE_PROBS_4X4: [u8; 9] = [
    65, 32, 18, 144, 162, 194, 41, 51, 98
];

/// Default coefficient probabilities structure
pub struct DefaultCoefProbs;

impl DefaultCoefProbs {
    pub const TX_4X4: [[[u8; 3]; 6]; 3] = [
        // Band 0
        [[128, 128, 128], [128, 128, 128], [128, 128, 128],
         [128, 128, 128], [128, 128, 128], [128, 128, 128]],
        // ... additional bands
    ];
}
```

### 13.3 Probability Merging

```rust
/// Merge observed counts with existing probability
pub fn merge_prob(existing: u8, count0: u32, total: u32) -> u8 {
    if total == 0 {
        return existing;
    }

    // Weight for adaptation (higher = faster adaptation)
    const ADAPTATION_WEIGHT: u32 = 24;

    let existing_weight = 256 - ADAPTATION_WEIGHT;
    let count_prob = if total > 0 {
        ((count0 * 256 + total / 2) / total) as u8
    } else {
        128
    };

    let merged = (existing as u32 * existing_weight +
                  count_prob as u32 * ADAPTATION_WEIGHT + 128) / 256;
    merged.clamp(1, 255) as u8
}
```

---

## 14. Encoder Rate-Distortion Optimization

### 14.1 RDO Framework

```rust
/// Rate-Distortion Optimization controller
pub struct RdoController {
    /// Lambda parameter (rate-distortion tradeoff)
    pub lambda: f64,
    /// Quantization parameter
    pub qp: u8,
}

impl RdoController {
    /// Calculate lambda from QP
    pub fn lambda_from_qp(qp: u8) -> f64 {
        let qp_f = qp as f64;
        // Empirical formula: lambda = 0.85 * pow(2, (qp - 12) / 3)
        0.85 * (2.0_f64).powf((qp_f - 12.0) / 3.0)
    }

    /// Calculate RD cost: Distortion + lambda * Rate
    pub fn rd_cost(&self, distortion: u64, bits: u32) -> f64 {
        distortion as f64 + self.lambda * bits as f64
    }
}

/// Mode decision using RDO
pub fn select_best_mode(
    block: &Block,
    rdo: &RdoController,
    available_modes: &[CodingMode],
) -> CodingMode {
    let mut best_cost = f64::MAX;
    let mut best_mode = available_modes[0];

    for mode in available_modes {
        // Calculate distortion (SSD/SAD)
        let distortion = calculate_distortion(block, mode);

        // Estimate bits for this mode
        let bits = estimate_bits(mode);

        // Calculate RD cost
        let cost = rdo.rd_cost(distortion, bits);

        if cost < best_cost {
            best_cost = cost;
            best_mode = *mode;
        }
    }

    best_mode
}
```

### 14.2 Partition Decision

```rust
/// Recursive partition decision with RDO
pub fn select_partition(
    rdo: &RdoController,
    pixels: &[u8],
    stride: usize,
    block_size: BlockSize,
    mi_row: usize,
    mi_col: usize,
) -> PartitionDecision {
    if block_size == BlockSize::Block4x4 {
        // Minimum size, must use this block
        let mode = select_best_intra_mode(rdo, pixels, stride, block_size);
        return PartitionDecision::None(mode);
    }

    let mut best_cost = f64::MAX;
    let mut best_partition = PartitionDecision::None(IntraMode::DC);

    // Try no partition
    let none_cost = evaluate_none_partition(rdo, pixels, stride, block_size);
    if none_cost < best_cost {
        best_cost = none_cost;
        best_partition = PartitionDecision::None(
            select_best_intra_mode(rdo, pixels, stride, block_size)
        );
    }

    // Try horizontal partition
    let horiz_cost = evaluate_horizontal_partition(rdo, pixels, stride, block_size);
    if horiz_cost < best_cost {
        best_cost = horiz_cost;
        let sub_size = get_subsize(block_size, Partition::Horizontal);
        best_partition = PartitionDecision::Horizontal(
            Box::new([
                select_partition(rdo, pixels, stride, sub_size, mi_row, mi_col),
                select_partition(rdo, &pixels[sub_size.height() * stride..],
                                 stride, sub_size, mi_row + sub_size.height_mi(), mi_col),
            ])
        );
    }

    // Try vertical partition
    let vert_cost = evaluate_vertical_partition(rdo, pixels, stride, block_size);
    if vert_cost < best_cost {
        best_cost = vert_cost;
        // ... similar to horizontal
    }

    // Try split partition (recursive)
    if block_size >= BlockSize::Block8x8 {
        let split_cost = evaluate_split_partition(rdo, pixels, stride, block_size);
        if split_cost < best_cost {
            best_cost = split_cost;
            // Recursively select partitions for 4 sub-blocks
        }
    }

    best_partition
}
```

### 14.3 Motion Estimation

```rust
/// Motion estimation for inter prediction
pub struct MotionEstimator {
    /// Search range in pixels
    pub search_range: i32,
    /// Subpel refinement enabled
    pub subpel_search: bool,
}

impl MotionEstimator {
    /// Full-pel motion search using diamond pattern
    pub fn diamond_search(
        &self,
        current: &Block,
        reference: &Frame,
        initial_mv: MotionVector,
    ) -> MotionVector {
        let mut best_mv = initial_mv;
        let mut best_cost = u64::MAX;

        // Diamond search pattern (step size starts large, decreases)
        let diamond_offsets = [
            [0, -1], [1, 0], [0, 1], [-1, 0],  // Small diamond
            [0, -2], [2, 0], [0, 2], [-2, 0],  // Large diamond
        ];

        let mut step = 8i32;
        while step >= 1 {
            let mut improved = true;

            while improved {
                improved = false;

                for offset in &diamond_offsets[..4] {
                    let test_mv = MotionVector {
                        row: (best_mv.row as i32 + offset[1] * step) as i16,
                        col: (best_mv.col as i32 + offset[0] * step) as i16,
                    };

                    let cost = self.evaluate_mv(current, reference, test_mv);
                    if cost < best_cost {
                        best_cost = cost;
                        best_mv = test_mv;
                        improved = true;
                    }
                }
            }

            step /= 2;
        }

        // Sub-pel refinement
        if self.subpel_search {
            best_mv = self.subpel_refine(current, reference, best_mv);
        }

        best_mv
    }

    /// Sub-pixel motion refinement
    fn subpel_refine(&self, current: &Block, reference: &Frame,
                      mv: MotionVector) -> MotionVector {
        let mut best_mv = MotionVector {
            row: mv.row * 8, // Convert to 1/8-pel
            col: mv.col * 8,
        };
        let mut best_cost = u64::MAX;

        // Test 1/2-pel positions
        for dy in -4i16..=4 {
            for dx in -4i16..=4 {
                let test_mv = MotionVector {
                    row: best_mv.row + dy,
                    col: best_mv.col + dx,
                };

                let cost = self.evaluate_subpel_mv(current, reference, test_mv);
                if cost < best_cost {
                    best_cost = cost;
                    best_mv = test_mv;
                }
            }
        }

        best_mv
    }
}
```

---

## 15. Data Structures

### 15.1 Frame Buffer

```rust
/// Decoded frame buffer
pub struct Frame {
    /// Frame dimensions
    pub width: u32,
    pub height: u32,
    /// Y plane data
    pub y_plane: Vec<u8>,
    /// U plane data
    pub u_plane: Vec<u8>,
    /// V plane data
    pub v_plane: Vec<u8>,
    /// Y plane stride
    pub y_stride: usize,
    /// UV plane stride
    pub uv_stride: usize,
    /// Bit depth
    pub bit_depth: u8,
    /// Frame type
    pub frame_type: FrameType,
}

impl Frame {
    pub fn new(width: u32, height: u32, bit_depth: u8) -> Self {
        let y_stride = ((width as usize + 63) / 64) * 64; // Align to 64
        let uv_stride = y_stride / 2;
        let height_aligned = ((height as usize + 63) / 64) * 64;

        Self {
            width,
            height,
            y_plane: vec![0u8; y_stride * height_aligned],
            u_plane: vec![0u8; uv_stride * (height_aligned / 2)],
            v_plane: vec![0u8; uv_stride * (height_aligned / 2)],
            y_stride,
            uv_stride,
            bit_depth,
            frame_type: FrameType::InterFrame,
        }
    }
}
```

### 15.2 Mode Info

```rust
/// Per-block mode information
pub struct ModeInfo {
    /// Block size
    pub block_size: BlockSize,
    /// Is intra predicted
    pub is_intra: bool,
    /// Intra mode (Y plane)
    pub y_mode: IntraMode,
    /// Intra mode (UV planes)
    pub uv_mode: IntraMode,
    /// Inter mode
    pub inter_mode: InterMode,
    /// Reference frames (up to 2 for compound)
    pub ref_frames: [RefFrame; 2],
    /// Motion vectors
    pub mv: [MotionVector; 2],
    /// Transform size
    pub tx_size: TxSize,
    /// Segment ID
    pub segment_id: u8,
    /// Skip residual flag
    pub skip: bool,
}

/// Block of mode info covering superblock
pub struct ModeInfoBlock {
    /// 64x64 superblock's mode info (8x8 grid)
    pub mi: [[ModeInfo; 8]; 8],
}
```

### 15.3 Coefficient Buffer

```rust
/// Coefficient buffer for a block
pub struct CoeffBuffer {
    /// Y coefficients
    pub y_coeffs: Vec<i16>,
    /// U coefficients
    pub u_coeffs: Vec<i16>,
    /// V coefficients
    pub v_coeffs: Vec<i16>,
    /// End of block positions
    pub eob_y: Vec<u16>,
    pub eob_u: Vec<u16>,
    pub eob_v: Vec<u16>,
}
```

---

## 16. Test Strategy

### 16.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Test range coder round-trip
    #[test]
    fn test_range_coder_roundtrip() {
        let mut encoder = RangeEncoder::new();
        let test_bits = [true, false, true, true, false, false, true, false];
        let prob = 128; // 50% probability

        for &bit in &test_bits {
            encoder.write_bool(bit, prob);
        }
        encoder.flush();

        let encoded = encoder.finish();
        let mut decoder = RangeCoder::new(&encoded);

        for &expected in &test_bits {
            let decoded = decoder.read_bool(&encoded, prob);
            assert_eq!(decoded, expected);
        }
    }

    /// Test 4x4 DCT
    #[test]
    fn test_dct4_inverse() {
        let input = [100i32, 50, -25, 10];
        let mut output = [0i32; 4];
        let mut reconstructed = [0i32; 4];

        dct4(&input, &mut output);
        idct4(&output, &mut reconstructed);

        for i in 0..4 {
            // Allow small rounding error
            assert!((input[i] - reconstructed[i]).abs() <= 1);
        }
    }

    /// Test intra DC prediction
    #[test]
    fn test_intra_dc_prediction() {
        let above = vec![100u8; 8];
        let left = vec![100u8; 8];
        let predictor = IntraPredictor {
            above,
            left,
            top_left: 100,
        };

        let mut block = vec![0u8; 64];
        predictor.predict_dc(&mut block, 8, 8, true, true);

        // All pixels should be 100 (average of above and left)
        for pixel in block {
            assert_eq!(pixel, 100);
        }
    }

    /// Test partition tree parsing
    #[test]
    fn test_partition_sizes() {
        assert_eq!(get_subsize(BlockSize::Block64x64, Partition::Split), BlockSize::Block32x32);
        assert_eq!(get_subsize(BlockSize::Block32x32, Partition::Horizontal), BlockSize::Block32x16);
        assert_eq!(get_subsize(BlockSize::Block16x16, Partition::Vertical), BlockSize::Block8x16);
    }
}
```

### 16.2 Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    /// Test decoding known VP9 keyframe
    #[test]
    fn test_decode_keyframe() {
        let data = include_bytes!("../tests/data/keyframe.ivf");
        let decoder = Vp9Decoder::new().unwrap();

        let frame = decoder.decode_frame(&data[32..]).unwrap();

        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
        assert_eq!(frame.frame_type, FrameType::KeyFrame);
    }

    /// Test encoding and decoding round-trip
    #[test]
    fn test_encode_decode_roundtrip() {
        let original = create_test_frame(320, 240);

        let encoder = Vp9Encoder::new(320, 240).unwrap();
        let encoded = encoder.encode_frame(&original).unwrap();

        let decoder = Vp9Decoder::new().unwrap();
        let decoded = decoder.decode_frame(&encoded).unwrap();

        // Check PSNR > 30dB for lossy compression
        let psnr = calculate_psnr(&original, &decoded);
        assert!(psnr > 30.0);
    }

    /// Test lossless mode
    #[test]
    fn test_lossless_roundtrip() {
        let original = create_test_frame(64, 64);

        let mut encoder = Vp9Encoder::new(64, 64).unwrap();
        encoder.set_lossless(true);
        let encoded = encoder.encode_frame(&original).unwrap();

        let decoder = Vp9Decoder::new().unwrap();
        let decoded = decoder.decode_frame(&encoded).unwrap();

        // Lossless should be exact
        assert_eq!(original.y_plane, decoded.y_plane);
        assert_eq!(original.u_plane, decoded.u_plane);
        assert_eq!(original.v_plane, decoded.v_plane);
    }
}
```

### 16.3 Conformance Testing

```rust
/// VP9 conformance test suite
pub struct ConformanceTests {
    /// Test vector files
    test_vectors: Vec<TestVector>,
}

impl ConformanceTests {
    /// Run all conformance tests
    pub fn run_all(&self) -> TestResults {
        let mut results = TestResults::new();

        for vector in &self.test_vectors {
            let result = self.run_test(vector);
            results.add(vector.name.clone(), result);
        }

        results
    }

    fn run_test(&self, vector: &TestVector) -> TestResult {
        let decoder = Vp9Decoder::new().unwrap();
        let mut decoded_frames = Vec::new();

        // Decode all frames
        for packet in &vector.packets {
            match decoder.decode_frame(packet) {
                Ok(frame) => decoded_frames.push(frame),
                Err(e) => return TestResult::Fail(e.to_string()),
            }
        }

        // Compare against reference
        for (i, (decoded, reference)) in
            decoded_frames.iter().zip(vector.reference_frames.iter()).enumerate()
        {
            if !frames_match(decoded, reference) {
                return TestResult::Fail(format!("Frame {} mismatch", i));
            }
        }

        TestResult::Pass
    }
}
```

### 16.4 Performance Benchmarks

```rust
#[cfg(test)]
mod benchmarks {
    use test::Bencher;

    #[bench]
    fn bench_decode_1080p_keyframe(b: &mut Bencher) {
        let data = include_bytes!("../tests/data/1080p_keyframe.ivf");
        let decoder = Vp9Decoder::new().unwrap();

        b.iter(|| {
            decoder.decode_frame(&data[32..]).unwrap()
        });
    }

    #[bench]
    fn bench_dct32x32(b: &mut Bencher) {
        let input = [0i32; 1024];
        let mut output = [0i32; 1024];

        b.iter(|| {
            dct32x32(&input, &mut output)
        });
    }

    #[bench]
    fn bench_motion_compensation_8tap(b: &mut Bencher) {
        let ref_frame = create_test_frame(1920, 1080);
        let mv = MotionVector { row: 13, col: 27 }; // 1.625, 3.375 pixels
        let mut output = vec![0u8; 16 * 16];

        b.iter(|| {
            motion_compensate(&ref_frame, mv, 100, 100, 16, 16,
                             InterpolationFilter::EightTap, &mut output, 16)
        });
    }
}
```

---

## Summary

This specification provides a comprehensive guide for implementing a VP9 codec in pure Rust. Key points:

1. **Pure Rust Implementation**: All algorithms derived from public VP9 specification, no GPL code dependencies

2. **MIT/Apache Reference Libraries**:
   - [vp9-parser](https://github.com/hasenbanck/vp9-parser) - MIT/Apache/Zlib VP9 parser
   - [cros-codecs](https://github.com/chromeos/cros-codecs) - BSD VP9 parser/decoder
   - [libvpx](https://github.com/webmproject/libvpx) - BSD reference implementation

3. **Critical Components**:
   - Binary arithmetic range coder with backward adaptation
   - Recursive 64x64 superblock partitioning
   - DCT/ADST/WHT transforms up to 32x32
   - 10 intra prediction modes with angular directions
   - 1/8-pel motion compensation with compound prediction
   - Loop filter for deblocking artifacts

4. **Encoder Considerations**:
   - Rate-distortion optimization for mode decisions
   - Diamond search for motion estimation
   - Probability table management for efficient coding

---

## References

- [VP9 Bitstream Specification v0.6](https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.6-20160331-draft.pdf)
- [WebM Project VP9 Summary](https://www.webmproject.org/vp9/)
- [Overview of the VP9 video codec - Ronald Bultje](https://blogs.gnome.org/rbultje/2016/12/13/overview-of-the-vp9-video-codec/)
- [VP9 Coding Profiles](https://www.webmproject.org/vp9/profiles/)
- [A Butterfly Structured Design of The Hybrid Transform Coding Scheme](https://research.google.com/pubs/archive/41418.pdf)
