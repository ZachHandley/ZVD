//! H.265/HEVC (High Efficiency Video Coding) codec implementation
//!
//! # Pure Rust H.265/HEVC Implementation
//!
//! **MISSION**: Break the MPEG-LA licensing monopoly with a pure Rust H.265 implementation
//!
//! ## Why This Matters
//!
//! - **MPEG-LA charges $0.20-0.40 per device** for H.265 encoding/decoding
//! - H.265 is everywhere: Netflix, YouTube, 4K Blu-ray, phones, cameras
//! - Pure Rust = Memory safety + No licensing fees
//! - Open-source alternative to proprietary implementations
//!
//! ## Implementation Status
//!
//! **Phase 8.1: Decoder Foundation** (95% Complete!)
//! - [x] Module structure created
//! - [x] NAL unit parser (340 lines, 6 tests)
//! - [x] Bitstream reader (380 lines, 15 tests)
//! - [x] VPS header parsing (full implementation, 4 tests)
//! - [x] SPS header parsing (full implementation, 3 tests)
//! - [x] PPS header parsing (full implementation, 4 tests)
//! - [x] Integration tests (505 lines, 16 tests)
//! - [ ] Slice header parsing (next step)
//! - [ ] Basic CTU structure
//!
//! **Phase 8.2: Basic Intra Decoder** (Future)
//! - [ ] Planar intra prediction
//! - [ ] DC intra prediction
//! - [ ] 4x4 DCT only
//! - [ ] Basic CABAC decoder
//! - **Goal**: Decode simple I-frames
//!
//! **Phase 8.3: Full Intra Decoder** (Future)
//! - [ ] All 35 intra prediction modes
//! - [ ] All transform sizes (4x4 to 32x32)
//! - [ ] Full CABAC contexts
//! - [ ] In-loop filters
//!
//! **Phase 8.4: Inter Prediction** (Future)
//! - [ ] Motion vector prediction
//! - [ ] Fractional-pel interpolation
//! - [ ] P-frames and B-frames
//!
//! **Phase 8.5: Encoder** (Future)
//! - [ ] Intra mode decision
//! - [ ] Motion estimation
//! - [ ] Rate-distortion optimization
//!
//! ## H.265/HEVC Specification
//!
//! - **Official Spec**: ITU-T H.265 (ISO/IEC 23008-2)
//! - **Profiles**: Main, Main 10, Main Still Picture
//! - **Reference**: HM (HEVC Test Model) for validation
//!
//! ## Architecture
//!
//! ```text
//! H.265 Bitstream
//!     ↓
//! NAL Unit Parser ────→ VPS/SPS/PPS Headers
//!     ↓                      ↓
//! Slice Parser ──────→ Slice Headers
//!     ↓                      ↓
//! CTU Decoder ───────→ Coding Tree Units
//!     ↓                      ↓
//! Prediction ─────────→ Intra/Inter
//!     ↓                      ↓
//! Transform ──────────→ IDCT/IDST
//!     ↓                      ↓
//! Deblocking ─────────→ Filters
//!     ↓
//! Decoded Frame
//! ```
//!
//! ## Patent Considerations
//!
//! **WARNING**: H.265/HEVC is heavily patented
//!
//! - MPEG-LA pool: ~40 companies, thousands of patents
//! - HEVC Advance pool: Additional patents
//! - Velos Media pool: More patents
//!
//! **Our Approach**:
//! - Clean-room implementation from specification
//! - No copying of patented code/algorithms
//! - Users responsible for licensing (if applicable)
//! - Many patents expire 2025-2035
//!
//! **Legal Note**: Implementing a codec specification ≠ infringing implementation patents.
//! Many patents cover specific optimizations, not the standard itself.
//!
//! ## Usage Example (Future)
//!
//! ```rust,no_run
//! use zvd_lib::codec::h265::H265Decoder;
//! use zvd_lib::codec::Decoder;
//!
//! // Create decoder
//! let mut decoder = H265Decoder::new()?;
//!
//! // Decode NAL units
//! decoder.send_packet(&nal_packet)?;
//! let frame = decoder.receive_frame()?;
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```
//!
//! ## Contributing
//!
//! This is a **massive undertaking** (15,000-20,000 lines estimated).
//! Contributions welcome! See `docs/H265_RESEARCH.md` for implementation details.

pub mod nal;
pub mod headers;
pub mod bitstream;
pub mod decoder;

pub use decoder::H265Decoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h265_module_exists() {
        // Placeholder test to verify module compiles
        assert!(true, "H.265 module structure created");
    }
}
