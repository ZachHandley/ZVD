# H.265/HEVC Pure Rust Implementation - Research & Development Guide

**Status**: Phase 8.3 - Full Intra Decoder COMPLETE! (2025-11-19)
**Goal**: Implement H.265/HEVC decoder and encoder in pure Rust
**Impact**: Break the MPEG-LA licensing monopoly ($0.20-0.40 per device royalty)

---

## Quick Links

- **Official Spec**: ITU-T H.265 (ISO/IEC 23008-2) - https://www.itu.int/rec/T-REC-H.265
- **Reference Implementation**: HM (HEVC Test Model) - https://hevc.hhi.fraunhofer.de/
- **Test Vectors**: JCT-VC Conformance - https://www.itu.int/wftp3/av-arch/jctvc-site/
- **Patent Info**: MPEG-LA HEVC - https://www.mpegla.com/programs/hevc/

---

## Table of Contents

1. [H.265 Overview](#h265-overview)
2. [Implementation Roadmap](#implementation-roadmap)
3. [NAL Unit Structure](#nal-unit-structure)
4. [Parameter Sets](#parameter-sets)
5. [Coding Tree Units (CTU)](#coding-tree-units-ctu)
6. [Intra Prediction](#intra-prediction)
7. [Transform Coding](#transform-coding)
8. [Entropy Coding (CABAC)](#entropy-coding-cabac)
9. [In-Loop Filters](#in-loop-filters)
10. [Inter Prediction](#inter-prediction)
11. [Rate Control](#rate-control)
12. [SIMD Optimization](#simd-optimization)
13. [Testing Strategy](#testing-strategy)

---

## H.265 Overview

### What is H.265/HEVC?

**H.265 (High Efficiency Video Coding)**, also known as **HEVC**, is a video compression standard that achieves:
- **~50% bitrate savings** vs H.264 at same quality
- Support for up to **8K resolution** (8192×4320)
- **10-bit and 12-bit** color depth
- Used in: 4K Blu-ray, Netflix, YouTube, phones, cameras

### Key Improvements over H.264

1. **Larger coding blocks**: Up to 64×64 (vs 16×16 in H.264)
2. **More intra prediction modes**: 35 (vs 9 in H.264)
3. **Better transform coding**: Multiple sizes (4×4 to 32×32)
4. **Improved entropy coding**: Context-adaptive CABAC
5. **Advanced filters**: Deblocking + SAO (Sample Adaptive Offset)

### Block Structure

```text
H.265 Block Hierarchy:

CTU (Coding Tree Unit)
  ↓
CU (Coding Unit) - 8×8 to 64×64
  ↓
PU (Prediction Unit) - Various sizes
  ↓
TU (Transform Unit) - 4×4 to 32×32
```

---

## Implementation Roadmap

### Phase 8.1: Decoder Foundation ✅ COMPLETE

**Goal**: Parse H.265 bitstreams and understand structure

**Status**: ✅ **COMPLETE** (2025-11-19)

**Tasks**:
- [x] Create module structure (`src/codec/h265/`)
- [x] NAL unit parser (nal.rs) - 340 lines, 6 tests
- [x] Bitstream reader (for Exp-Golomb codes) - 380 lines, 15 tests
- [x] VPS/SPS/PPS parsing (headers.rs) - Complete
- [x] Slice header parsing - Complete
- [x] CTU structure definitions - Complete

**Success Criteria**: ✅ All met
- ✅ Can parse all parameter sets (VPS/SPS/PPS)
- ✅ Can identify slice types (I/P/B)
- ✅ Can extract basic video info (resolution, profile, etc.)

**Files**:
- `src/codec/h265/nal.rs` - NAL unit parsing ✅ (340 lines, 6 tests)
- `src/codec/h265/headers.rs` - VPS/SPS/PPS parsing ✅ (1200+ lines, 13 tests)
- `src/codec/h265/decoder.rs` - Main decoder structure ✅
- `src/codec/h265/bitstream.rs` - Exp-Golomb reader ✅ (380 lines, 15 tests)
- `tests/h265_parser_test.rs` - Integration tests ✅ (505 lines, 16 tests)

**Total**: ~2,500 lines, 54 tests

---

### Phase 8.2: Basic Intra Decoder ✅ COMPLETE

**Goal**: Implement basic intra prediction and transforms

**Status**: ✅ **COMPLETE** (2025-11-19)

**Tasks**:
- [x] CTU (Coding Tree Unit) structure (380 lines, 14 tests)
- [x] Quadtree partitioning
- [x] Frame buffer management
- [x] Planar intra prediction (mode 0)
- [x] DC intra prediction (mode 1)
- [x] All 35 angular intra prediction modes (modes 2-34)
- [x] 4×4 DCT inverse transform
- [x] 4×4 DST inverse transform
- [x] 8×8 DCT inverse transform
- [x] 16×16 DCT inverse transform
- [x] 32×32 DCT inverse transform
- [x] Residual reconstruction with bit-depth clamping

**Success Criteria**: ✅ All met
- ✅ All 35 intra prediction modes implemented
- ✅ All transform sizes (4×4 to 32×32)
- ✅ Proper reference sample handling
- ✅ Comprehensive test coverage

**Components**:
- `src/codec/h265/ctu.rs` - CTU structures ✅ (380 lines, 14 tests)
- `src/codec/h265/intra.rs` - All 35 intra modes ✅ (600 lines, 16 tests)
- `src/codec/h265/transform.rs` - All DCT/DST sizes ✅ (630 lines, 15 tests)

**Total**: ~1,610 lines, 45 tests

---

### Phase 8.3: Full Intra Decoder ✅ COMPLETE

**Completion Date**: 2025-11-19

**Goal**: Complete H.265 intra decoding infrastructure

**Status**: ✅ **COMPLETE**

**Implementation Summary**:
- **CABAC Decoder**: 470 lines, 14 tests
- **Quantization**: 290 lines, 18 tests
- **Deblocking Filter**: 360 lines, 18 tests
- **SAO Filter**: 240 lines, 16 tests
- **Coefficient Scanning**: 430 lines, 20 tests
- **Total**: ~1,790 lines, 86 tests

**Features Implemented**:
- [x] CABAC arithmetic decoder with context modeling
- [x] Quantization/dequantization (QP 0-51)
- [x] Deblocking filter (vertical/horizontal edges)
- [x] SAO filter (band offset + edge offset)
- [x] Coefficient scanning (diagonal/horizontal/vertical)
- [x] 8/10/12-bit support throughout

**Files Created**:
- `src/codec/h265/cabac.rs` - CABAC decoder
- `src/codec/h265/quant.rs` - Quantization
- `src/codec/h265/filter.rs` - Deblocking + SAO
- `src/codec/h265/scan.rs` - Coefficient scanning

---

### Phase 8.4: Inter Prediction (8-12 weeks)

**Goal**: Decode P-frames and B-frames with motion compensation

**Tasks**:
- [ ] Motion vector prediction
- [ ] Fractional-pel interpolation (1/4-pixel precision)
- [ ] Bi-directional prediction (B-frames)
- [ ] Weighted prediction
- [ ] Reference picture list management
- [ ] Temporal motion vector prediction (TMVP)
- [ ] Merge mode
- [ ] Advanced motion vector prediction (AMVP)

**Success Criteria**:
- Can decode full video sequences (I/P/B frames)
- Proper motion compensation
- Quality competitive with HM decoder
- Can decode JCT-VC conformance streams

**Files**:
- `src/codec/h265/inter.rs` - Inter prediction
- `src/codec/h265/mc.rs` - Motion compensation
- `src/codec/h265/mv.rs` - Motion vector prediction
- `src/codec/h265/refs.rs` - Reference picture management

---

### Phase 8.5: Encoder Foundation (12-16 weeks)

**Goal**: Encode basic I-frames

**Tasks**:
- [ ] Intra mode decision (RD optimization)
- [ ] Transform selection
- [ ] Quantization parameter (QP) selection
- [ ] CABAC encoding
- [ ] Rate control basics
- [ ] Bitstream writing

**Success Criteria**:
- Can encode I-frames
- Files decodable by HM reference decoder
- PSNR within 2 dB of HM encoder (acceptable for v1.0)

---

### Phase 8.6: Full Encoder (16-24 weeks)

**Goal**: Production-ready encoder with motion estimation

**Tasks**:
- [ ] Motion estimation (block matching)
- [ ] Mode decision for all block sizes
- [ ] Rate-distortion optimization (RDO)
- [ ] Advanced rate control
- [ ] Multi-threading
- [ ] SIMD optimization
- [ ] Encoding speed optimizations

**Success Criteria**:
- Can encode full sequences (I/P/B)
- PSNR within 1 dB of HM/x265
- Encoding speed ≥ 50% of x265 (acceptable for v1.0)
- Files compatible with all H.265 decoders

---

## NAL Unit Structure

### NAL Unit Header (2 bytes)

```text
 0                   1
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|F|   Type    | LayerID |  TID  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

F: Forbidden zero bit (1 bit) = 0
Type: NAL unit type (6 bits) - See NalUnitType enum
LayerID: Layer ID (6 bits) - For scalable extensions
TID: Temporal ID + 1 (3 bits) - For temporal scalability
```

### Important NAL Unit Types

| Value | Name | Description |
|-------|------|-------------|
| 0-9   | VCL  | Video Coding Layer (slices) |
| 16-21 | IDR/CRA | Random access points |
| 32    | VPS  | Video Parameter Set |
| 33    | SPS  | Sequence Parameter Set |
| 34    | PPS  | Picture Parameter Set |
| 39-40 | SEI  | Supplemental Enhancement Information |

### Emulation Prevention

H.265 prevents start code emulation by inserting 0x03:
- `0x000000` → `0x00000300`
- `0x000001` → `0x00000301`
- `0x000002` → `0x00000302`
- `0x000003` → `0x00000303`

**Implemented in**: `src/codec/h265/nal.rs::remove_emulation_prevention_bytes()`

---

## Parameter Sets

### VPS (Video Parameter Set)

Highest-level parameters:
- Multiple sequence support
- Layer information (for scalability)
- Sub-layer information (temporal scalability)

**Fields**:
- `vps_video_parameter_set_id` (0-15)
- `vps_max_layers_minus1`
- `vps_max_sub_layers_minus1` (0-6)
- Temporal ID nesting flag

### SPS (Sequence Parameter Set)

Sequence-level parameters:
- Picture dimensions
- Chroma format (4:2:0, 4:2:2, 4:4:4)
- Bit depth (8-bit, 10-bit, 12-bit)
- Profile/level information

**Key Fields**:
- `pic_width_in_luma_samples`
- `pic_height_in_luma_samples`
- `chroma_format_idc` (1=4:2:0, 2=4:2:2, 3=4:4:4)
- `bit_depth_luma_minus8`
- `bit_depth_chroma_minus8`

### PPS (Picture Parameter Set)

Picture-level parameters:
- Quantization parameters
- Tile configuration
- Loop filter settings

**Key Fields**:
- `pps_pic_parameter_set_id` (0-63)
- `pps_seq_parameter_set_id` (references SPS)
- Deblocking filter parameters
- Tile boundaries

---

## Coding Tree Units (CTU)

### CTU Size

- **Maximum**: 64×64 pixels
- **Typical**: 32×32 or 64×64 for HD/4K
- Divided into **CUs (Coding Units)** via quadtree

### Quadtree Split

```text
64×64 CTU can split into:
  → Four 32×32 CUs
    → Four 16×16 CUs
      → Four 8×8 CUs
```

**Minimum CU size**: Usually 8×8

### Coding Unit (CU)

Each CU has:
- **Prediction mode**: Intra or Inter
- **Skip flag**: For inter-predicted blocks
- **Partition into PUs**: Prediction Units
- **Transform tree**: Division into TUs

### Prediction Unit (PU)

Partition types for intra:
- **2N×2N**: Entire CU (most common)
- **N×N**: Four sub-blocks (8×8 CUs only)

Partition types for inter:
- 2N×2N, 2N×N, N×2N, N×N
- Asymmetric: 2N×nU, 2N×nD, nL×2N, nR×2N

### Transform Unit (TU)

- Sizes: 4×4, 8×8, 16×16, 32×32
- Can be split via quadtree (Residual Quadtree - RQT)

---

## Intra Prediction

### 35 Prediction Modes

```text
Mode 0: Planar
Mode 1: DC
Modes 2-34: Angular (33 directional modes)
```

### Planar Mode (Mode 0)

Smoothest prediction - uses bilinear interpolation:
```rust
pred[x][y] = (
    (H - 1 - x) * left[y] + (x + 1) * top_right +
    (W - 1 - y) * top[x] + (y + 1) * bottom_left
) / (W + H)
```

### DC Mode (Mode 1)

Average of reference samples:
```rust
dc = (sum(top) + sum(left)) / (W + H)
```

### Angular Modes (Modes 2-34)

Directional prediction with angles:
- **Horizontal** (Mode 10)
- **Vertical** (Mode 26)
- **Diagonal** (Mode 18: 45°)
- Various angles in between

**Reference samples**:
- Top: `top[0..2*W]`
- Left: `left[0..2*H]`
- Top-left corner: `top_left`

---

## Transform Coding

### Integer DCT (Discrete Cosine Transform)

H.265 uses **integer approximations** of DCT for:
- 4×4, 8×8, 16×16, 32×32 sizes
- Avoids floating-point arithmetic
- Exact integer inverse

**Formula (conceptual)**:
```
DCT[u][v] = Σ Σ pixel[x][y] * cos(...) * cos(...)
```

**Implementation**: Matrix multiplication with pre-defined integer matrices

### Integer DST (Discrete Sine Transform)

Used for **4×4 intra residuals** only:
- Better for intra prediction residuals
- Similar to DCT but with sine basis

### Quantization

```rust
coeff_q = (coeff * scale) >> shift
```

- **QP (Quantization Parameter)**: 0-51
  - QP=0: Highest quality (almost lossless)
  - QP=51: Lowest quality (high compression)
- **Scale matrix**: Depends on QP and transform size

### Inverse Transform (IDCT)

```rust
pixel = (coeff_q * scale) >> shift
```

Then apply inverse DCT matrix multiplication

---

## Entropy Coding (CABAC)

### CABAC Overview

**Context-Adaptive Binary Arithmetic Coding**:
- Adaptive probability estimation
- Context modeling
- Binary arithmetic coding

### Binarization

Convert syntax elements to binary:
- **Unary**: 0, 10, 110, 1110, ...
- **Truncated Unary**: Limited length
- **Fixed-Length**: Direct binary
- **Exponential-Golomb**: For large values

### Context Modeling

Context = probability model for each bit
- Depends on:
  - Neighboring blocks
  - Syntax element type
  - Previous bits

**Example contexts**:
- `cu_skip_flag`: 4 contexts
- `coeff_abs_level_greater1_flag`: Many contexts

### Arithmetic Coding

- Maintains probability range `[low, high)`
- Narrows range based on symbol probability
- Output bits when range narrows enough

---

## In-Loop Filters

### Deblocking Filter

Applied to **block boundaries** to reduce blocking artifacts:

1. **Boundary Strength (BS)**: 0-2
   - BS=2: Intra blocks or large motion diff
   - BS=1: Inter blocks with different refs
   - BS=0: No filtering

2. **Filter Decision**:
   ```rust
   if (|p0 - q0| < β && |p1 - p0| < tc && |q1 - q0| < tc) {
       apply_filter()
   }
   ```

3. **Strong/Weak Filter**:
   - Strong: Modifies 3 pixels each side
   - Weak: Modifies 1-2 pixels each side

### SAO (Sample Adaptive Offset)

Reduces ringing artifacts:

**Two types**:
1. **Edge Offset** (EO):
   - Classifies pixels by edge category
   - Adds offset based on category
   - 4 edge types (0°, 90°, 45°, 135°)

2. **Band Offset** (BO):
   - Divides sample values into bands
   - Adds offset per band

**Application**:
```rust
reconstructed_pixel += sao_offset[category]
```

---

## Inter Prediction

### Motion Vectors

- **Precision**: 1/4-pixel
- **Range**: ±128 pixels (typical)
- **Storage**: Integer + fractional part

### Motion Vector Prediction (MVP)

Predict MV from **spatial** and **temporal** neighbors:

**Spatial candidates**:
- Left (A)
- Above (B)
- Above-right (C)
- Below-left (D)
- Above-left (E)

**Temporal candidate**:
- Co-located block in reference frame

### Fractional-Pel Interpolation

Generate sub-pixel samples using **8-tap filters**:

**1/2-pixel**:
```
sample = (Σ coeff[i] * ref[i+offset]) >> shift
```

**1/4-pixel**: Interpolate between integer and 1/2-pixel

### Merge Mode

Copy MV and ref from neighbor without signaling:
- Up to 5 merge candidates
- Saves bits (no MV difference coded)

---

## Rate Control

### Rate-Distortion Optimization (RDO)

Choose mode minimizing:
```
Cost = Distortion + λ * Rate
```

- **Distortion**: SAD or SSD between original and reconstructed
- **Rate**: Bits needed to encode
- **λ (Lambda)**: Lagrange multiplier (depends on QP)

### QP Selection

- **Frame-level**: Based on target bitrate
- **CTU-level**: Adaptive quantization
- **λ calculation**:
  ```rust
  lambda = 0.85 * 2^((QP - 12) / 3)
  ```

---

## SIMD Optimization

### Priority SIMD Functions

1. **Transform (DCT/IDCT)**: AVX2 or AVX-512
2. **Intra prediction**: SSE4.2 / AVX2
3. **Motion compensation**: AVX2 (8-tap interpolation)
4. **SAD/SSD**: AVX2 (motion estimation)
5. **Deblocking filter**: SSE4.2

### Rust SIMD Options

- **`std::arch`**: Direct SIMD intrinsics
- **`safe_arch`**: Safe wrappers around intrinsics
- **`packed_simd`**: Portable SIMD (experimental)

**Example** (AVX2 4×4 DCT):
```rust
use std::arch::x86_64::*;

unsafe fn dct_4x4_avx2(block: &[i16; 16]) -> [i16; 16] {
    // Load 4×4 block into SIMD registers
    let row0 = _mm_loadu_si128(block.as_ptr() as *const __m128i);
    // ... DCT matrix multiplication using _mm_madd_epi16 ...
}
```

---

## Testing Strategy

### Unit Tests

- **NAL parsing**: ✅ Already started in `nal.rs`
- **Bitstream reader**: Exp-Golomb decoding
- **Intra prediction**: Each mode separately
- **Transform**: DCT/IDCT correctness
- **CABAC**: Context updates

### Integration Tests

- **Parse parameter sets**: From real H.265 files
- **Decode I-frames**: Simple test vectors
- **Decode sequences**: Full I/P/B streams

### Conformance Tests

Download **JCT-VC test vectors**:
- https://www.itu.int/wftp3/av-arch/jctvc-site/

**Test bitstreams**:
- `STRUCT_A`: All intra
- `DBLK_A`: Deblocking
- `SAO_A`: SAO filter
- `MVCLIP_A`: Motion vectors

### Quality Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
  ```
  PSNR = 10 * log10(255² / MSE)
  ```
- **SSIM**: Structural Similarity Index
- **BD-Rate**: Bjøntegaard Delta Rate (vs reference)

### Performance Benchmarks

- **Decoding speed**: FPS at 1080p, 4K
- **Encoding speed**: FPS at 1080p
- **Memory usage**: Peak RAM
- **Comparison**: vs HM, x265, FFmpeg

---

## Resources

### Specifications

- **ITU-T H.265**: https://www.itu.int/rec/T-REC-H.265
- **ISO/IEC 23008-2**: Same as H.265
- **Profiles/Levels**: Annex A of spec

### Reference Software

- **HM (HEVC Test Model)**: https://hevc.hhi.fraunhofer.de/
- **x265**: https://www.x265.org/ (Fast encoder)
- **FFmpeg libx265**: Part of FFmpeg

### Books & Papers

- "High Efficiency Video Coding (HEVC): Algorithms and Architectures" (Springer)
- "The H.264 Advanced Video Compression Standard" (Wiley) - Background
- JCTVC meeting documents: https://www.itu.int/wftp3/av-arch/jctvc-site/

### Tools

- **YUView**: YUV video viewer - https://github.com/IENT/YUView
- **Elecard StreamEye**: H.265 analyzer
- **FFmpeg**: Decode/encode H.265 for comparison

---

## Implementation Notes

### Clean-Room Development

**CRITICAL**: Never copy code from HM, x265, or FFmpeg
- **Read** specification and understand algorithm
- **Close** all reference code
- **Implement** independently in Rust
- **Test** against conformance streams

### Patent Considerations

- Implementing the **specification** ≠ infringing patents
- Many patents cover **optimizations**, not standard
- Users responsible for licensing
- Document sources of knowledge

### Code Quality

- **No unsafe** except for SIMD (and mark clearly)
- Comprehensive **error handling**
- **Unit tests** for every function
- **Documentation** for all public APIs
- **Benchmarks** for hot paths

---

## Phase 8.1 + 8.2 + 8.3 Complete! 🎉

### Completed ✅
1. ✅ Phase 8.1: Parser foundation (~2,500 lines, 54 tests)
2. ✅ Phase 8.2: Basic intra decoder (~1,610 lines, 45 tests)
3. ✅ Phase 8.3: Full intra decoder (~1,790 lines, 86 tests)
4. ✅ **Total H.265 Code**: ~5,900 lines, 185 tests! 🚀

### Next: Phase 8.4 - Inter Prediction

**Priority Tasks**:
1. **Motion vector prediction (AMVP)** - Predict motion vectors from neighbors
2. **Merge mode** - Copy motion from neighbors without signaling
3. **Motion compensation** - Luma and chroma interpolation
4. **Fractional pixel interpolation** - 1/4-pixel precision with 8-tap filters
5. **Weighted prediction** - Adaptive weighting for B-frames
6. **Reference picture management** - DPB (Decoded Picture Buffer)
7. **P-frames and B-frames** - Full temporal prediction

**Target**: 8-12 weeks to complete Phase 8.4

**Then**: Complete H.265 decoder capable of decoding full video sequences!

---

**Let's break the MPEG-LA monopoly! 🚀**
