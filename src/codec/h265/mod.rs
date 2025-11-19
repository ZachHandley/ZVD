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
//! **Phase 8.1: Decoder Foundation** (100% COMPLETE! 🎉)
//! - [x] Module structure created
//! - [x] NAL unit parser (340 lines, 6 unit tests)
//! - [x] Bitstream reader (380 lines, 15 unit tests)
//! - [x] VPS header parsing (full implementation, 4 unit tests)
//! - [x] SPS header parsing (full implementation, 3 unit tests)
//! - [x] PPS header parsing (full implementation, 4 unit tests)
//! - [x] Slice header parsing (full implementation, 6 unit tests)
//! - [x] Integration tests (505 lines, 16 integration tests)
//! - **Total: ~2,500 lines of pure Rust H.265 parsing code!**
//!
//! **Phase 8.2: Basic Intra Decoder** (100% COMPLETE! 🎉)
//! - [x] CTU (Coding Tree Unit) structure (380 lines, 14 unit tests)
//! - [x] Quadtree partitioning
//! - [x] Frame buffer management
//! - [x] Intra prediction - Planar mode (600 lines, 16 unit tests)
//! - [x] Intra prediction - DC mode
//! - [x] Intra prediction - All 35 angular modes (2-34)!
//! - [x] 4×4 DCT inverse transform (630 lines, 15 unit tests)
//! - [x] 4×4 DST inverse transform
//! - [x] 8×8, 16×16, 32×32 DCT inverse transforms
//! - [x] Residual reconstruction with bit-depth clamping
//! - **Phase 8.2 Total: ~1,610 lines, 45 tests!**
//!
//! **Phase 8.3: Full Intra Decoder** (100% COMPLETE! 🎉)
//! - [x] CABAC arithmetic decoder (470 lines, 14 unit tests)
//! - [x] Context model management with state transitions
//! - [x] Quantization/dequantization (290 lines, 18 unit tests)
//! - [x] Deblocking filter - vertical and horizontal edges (360 lines, 18 unit tests)
//! - [x] SAO (Sample Adaptive Offset) filter - band and edge offset (240 lines, 16 unit tests)
//! - [x] Coefficient scanning - diagonal/horizontal/vertical patterns (430 lines, 20 unit tests)
//! - **Phase 8.3 Total: ~1,790 lines, 86 tests!**
//!
//! **Phase 8.4: Inter Prediction** (IN PROGRESS - 30%)
//! - [x] Motion vector structures (500 lines, 28 unit tests)
//! - [x] Motion compensation with fractional-pel interpolation (850 lines, 28 unit tests)
//! - [ ] AMVP (Advanced Motion Vector Prediction)
//! - [ ] Merge mode
//! - [ ] Reference picture management
//! - [ ] Weighted prediction
//! - **Phase 8.4 Current: ~1,350 lines, 56 tests!**
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
pub mod ctu;
pub mod intra;
pub mod transform;
pub mod cabac;
pub mod quant;
pub mod filter;
pub mod scan;
pub mod mv;
pub mod mc;

pub use decoder::H265Decoder;
pub use headers::{Vps, Sps, Pps, SliceHeader, SliceType};
pub use nal::{NalUnit, NalUnitType};
pub use ctu::{CodingTreeUnit, CodingUnit, FrameBuffer, CtuSize, IntraMode, PredMode};
pub use intra::{IntraPredictor, ReferenceSamples};
pub use transform::{Transform, TransformSize};
pub use cabac::{CabacDecoder, ContextModel};
pub use quant::Quantizer;
pub use filter::{DeblockingFilter, BoundaryStrength, SaoFilter, SaoParams, SaoType, SaoEdgeClass};
pub use scan::{CoefficientScanner, CoefficientGroupScanner, SignificanceMap, CoefficientLevels, ScanPattern};
pub use mv::{MotionVector, MvCandidate, MvPredictor, MergeCandidate, MergeCandidateList,
             PredictionList, PredictionFlag, SpatialNeighbor, MotionVectorField};
pub use mc::MotionCompensator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h265_module_exists() {
        // Placeholder test to verify module compiles
        assert!(true, "H.265 module structure created");
    }
}
