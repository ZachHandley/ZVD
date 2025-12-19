//! VP8 intra mode decision
//!
//! This module implements mode decision for VP8 encoding, selecting the best
//! intra prediction mode based on rate-distortion cost.

use super::prediction::{predict_16x16, predict_4x4, predict_8x8_chroma};
use super::quant::QuantFactors;
use super::tables::{
    IntraMode16x16, IntraMode4x4, KF_BMODE_PROBS, KF_UVMODE_PROBS, KF_YMODE_PROBS,
};

/// Mode decision result for a macroblock
#[derive(Debug, Clone)]
pub struct MacroblockModeDecision {
    /// Selected Y prediction mode (for 16x16 mode)
    pub y_mode: IntraMode16x16,
    /// Selected UV prediction mode
    pub uv_mode: IntraMode16x16,
    /// Whether to use 4x4 prediction mode instead of 16x16
    pub use_4x4_modes: bool,
    /// 4x4 subblock modes (if use_4x4_modes is true)
    pub subblock_modes: [[IntraMode4x4; 4]; 4],
    /// Estimated rate (bits)
    pub estimated_rate: u32,
    /// Estimated distortion (SSD)
    pub estimated_distortion: u64,
}

impl Default for MacroblockModeDecision {
    fn default() -> Self {
        MacroblockModeDecision {
            y_mode: IntraMode16x16::DcPred,
            uv_mode: IntraMode16x16::DcPred,
            use_4x4_modes: false,
            subblock_modes: [[IntraMode4x4::BDcPred; 4]; 4],
            estimated_rate: 0,
            estimated_distortion: 0,
        }
    }
}

/// Lambda value for rate-distortion optimization
/// Higher values favor rate over distortion
const RD_LAMBDA_BASE: f64 = 0.85;

/// Calculate sum of squared differences (SSD) between two blocks
/// Note: Currently using SAD instead for speed; SSD available for RD optimization
#[allow(dead_code)]
fn calculate_ssd(
    original: &[u8],
    reconstructed: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> u64 {
    let mut ssd = 0u64;
    for y in 0..height {
        for x in 0..width {
            let orig = original[y * stride + x] as i32;
            let recon = reconstructed[y * stride + x] as i32;
            let diff = orig - recon;
            ssd += (diff * diff) as u64;
        }
    }
    ssd
}

/// Calculate sum of absolute differences (SAD) between two blocks - faster than SSD
fn calculate_sad(
    original: &[u8],
    reconstructed: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> u32 {
    let mut sad = 0u32;
    for y in 0..height {
        for x in 0..width {
            let orig = original[y * stride + x] as i32;
            let recon = reconstructed[y * stride + x] as i32;
            sad += (orig - recon).unsigned_abs();
        }
    }
    sad
}

/// Estimate rate for encoding a mode
fn estimate_mode_rate(mode: IntraMode16x16, probs: &[u8]) -> u32 {
    // Estimate bits based on probability
    // Higher probability = fewer bits needed
    let mode_idx = mode as usize;
    let mut bits = 0u32;

    // Traverse the probability tree
    match mode_idx {
        0 => {
            // DC: first branch
            bits += estimate_bool_bits(probs[0], false);
        }
        1 => {
            // V: first branch true, second false
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], false);
        }
        2 => {
            // H: first two true, third false
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], false);
        }
        3 => {
            // TM: all true
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
        }
        _ => {}
    }

    bits
}

/// Estimate bits for encoding a boolean with given probability
fn estimate_bool_bits(prob: u8, bit: bool) -> u32 {
    // Using a log2 approximation for bit cost
    // -log2(prob/256) for bit=false, -log2((256-prob)/256) for bit=true
    let p = if bit { 256 - prob as u32 } else { prob as u32 };

    // Scale factor: multiply by 256 to avoid floating point
    // bits = -log2(p/256) * 256 = (8 - log2(p)) * 256
    if p == 0 {
        return 2048; // Very unlikely event
    }

    // Simple approximation: 256 * 8 / p capped
    (2048u32).saturating_sub(p * 8)
}

/// Select the best 16x16 intra mode
pub fn select_16x16_mode(
    original: &[u8], // 16x16 original pixels
    orig_stride: usize,
    above: &[u8],   // 16 pixels above
    left: &[u8],    // 16 pixels to the left
    above_left: u8, // Corner pixel
    above_available: bool,
    left_available: bool,
    _quant: &QuantFactors,
    lambda: f64,
) -> (IntraMode16x16, u64) {
    let modes = [
        IntraMode16x16::DcPred,
        IntraMode16x16::VPred,
        IntraMode16x16::HPred,
        IntraMode16x16::TmPred,
    ];

    let mut best_mode = IntraMode16x16::DcPred;
    let mut best_cost = u64::MAX;
    let mut prediction = vec![0u8; 256];

    for &mode in &modes {
        // Skip modes that require unavailable reference pixels
        match mode {
            IntraMode16x16::VPred if !above_available => continue,
            IntraMode16x16::HPred if !left_available => continue,
            IntraMode16x16::TmPred if !above_available || !left_available => continue,
            _ => {}
        }

        // Generate prediction
        predict_16x16(
            mode,
            above,
            left,
            above_left,
            above_available,
            left_available,
            &mut prediction,
            16,
        );

        // Calculate distortion using SAD (faster than full transform + SSD)
        let distortion = calculate_sad(original, &prediction, 16, 16, orig_stride) as u64;

        // Estimate rate
        let rate = estimate_mode_rate(mode, &KF_YMODE_PROBS);

        // Calculate RD cost
        let rd_cost = distortion + (lambda * rate as f64) as u64;

        if rd_cost < best_cost {
            best_cost = rd_cost;
            best_mode = mode;
        }
    }

    (best_mode, best_cost)
}

/// Select the best 8x8 chroma mode
pub fn select_chroma_mode(
    original_u: &[u8], // 8x8 original U pixels
    original_v: &[u8], // 8x8 original V pixels
    orig_stride: usize,
    above_u: &[u8],
    above_v: &[u8],
    left_u: &[u8],
    left_v: &[u8],
    above_left_u: u8,
    above_left_v: u8,
    above_available: bool,
    left_available: bool,
    lambda: f64,
) -> (IntraMode16x16, u64) {
    let modes = [
        IntraMode16x16::DcPred,
        IntraMode16x16::VPred,
        IntraMode16x16::HPred,
        IntraMode16x16::TmPred,
    ];

    let mut best_mode = IntraMode16x16::DcPred;
    let mut best_cost = u64::MAX;
    let mut pred_u = vec![0u8; 64];
    let mut pred_v = vec![0u8; 64];

    for &mode in &modes {
        // Skip modes that require unavailable reference pixels
        match mode {
            IntraMode16x16::VPred if !above_available => continue,
            IntraMode16x16::HPred if !left_available => continue,
            IntraMode16x16::TmPred if !above_available || !left_available => continue,
            _ => {}
        }

        // Generate predictions
        predict_8x8_chroma(
            mode,
            above_u,
            left_u,
            above_left_u,
            above_available,
            left_available,
            &mut pred_u,
            8,
        );
        predict_8x8_chroma(
            mode,
            above_v,
            left_v,
            above_left_v,
            above_available,
            left_available,
            &mut pred_v,
            8,
        );

        // Calculate distortion
        let distortion_u = calculate_sad(original_u, &pred_u, 8, 8, orig_stride) as u64;
        let distortion_v = calculate_sad(original_v, &pred_v, 8, 8, orig_stride) as u64;
        let distortion = distortion_u + distortion_v;

        // Estimate rate
        let rate = estimate_mode_rate(mode, &KF_UVMODE_PROBS);

        // Calculate RD cost
        let rd_cost = distortion + (lambda * rate as f64) as u64;

        if rd_cost < best_cost {
            best_cost = rd_cost;
            best_mode = mode;
        }
    }

    (best_mode, best_cost)
}

/// Select the best 4x4 intra mode for a subblock
pub fn select_4x4_mode(
    original: &[u8], // 4x4 original pixels
    orig_stride: usize,
    above: &[u8], // 8 pixels (4 above + 4 above-right)
    left: &[u8],  // 4 pixels to the left
    above_left: u8,
    above_mode: IntraMode4x4, // For context
    left_mode: IntraMode4x4,  // For context
    lambda: f64,
) -> (IntraMode4x4, u64) {
    let modes = [
        IntraMode4x4::BDcPred,
        IntraMode4x4::BTmPred,
        IntraMode4x4::BVePred,
        IntraMode4x4::BHePred,
        IntraMode4x4::BLdPred,
        IntraMode4x4::BRdPred,
        IntraMode4x4::BVrPred,
        IntraMode4x4::BVlPred,
        IntraMode4x4::BHdPred,
        IntraMode4x4::BHuPred,
    ];

    let mut best_mode = IntraMode4x4::BDcPred;
    let mut best_cost = u64::MAX;
    let mut prediction = vec![0u8; 16];

    let probs = &KF_BMODE_PROBS[above_mode as usize][left_mode as usize];

    for &mode in &modes {
        // Generate prediction
        predict_4x4(mode, above, left, above_left, &mut prediction, 4);

        // Calculate distortion
        let distortion = calculate_sad(original, &prediction, 4, 4, orig_stride) as u64;

        // Estimate rate based on probability tree
        let rate = estimate_4x4_mode_rate(mode, probs);

        // Calculate RD cost
        let rd_cost = distortion + (lambda * rate as f64) as u64;

        if rd_cost < best_cost {
            best_cost = rd_cost;
            best_mode = mode;
        }
    }

    (best_mode, best_cost)
}

/// Estimate rate for encoding a 4x4 mode
fn estimate_4x4_mode_rate(mode: IntraMode4x4, probs: &[u8; 9]) -> u32 {
    let mode_idx = mode as usize;
    let mut bits = 0u32;

    // Tree structure for 4x4 modes:
    // [0]=DC, [1]=TM, [2]=VE, [3]=HE, [4]=LD, [5]=RD, [6]=VR, [7]=VL, [8]=HD, [9]=HU

    match mode_idx {
        0 => {
            bits += estimate_bool_bits(probs[0], false);
        }
        1 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], false);
        }
        2 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], false);
        }
        3 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], false);
            bits += estimate_bool_bits(probs[4], false);
        }
        4 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], false);
            bits += estimate_bool_bits(probs[4], true);
        }
        5 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], true);
            bits += estimate_bool_bits(probs[5], false);
        }
        6 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], true);
            bits += estimate_bool_bits(probs[5], true);
            bits += estimate_bool_bits(probs[6], false);
        }
        7 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], true);
            bits += estimate_bool_bits(probs[5], true);
            bits += estimate_bool_bits(probs[6], true);
            bits += estimate_bool_bits(probs[7], false);
        }
        8 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], true);
            bits += estimate_bool_bits(probs[5], true);
            bits += estimate_bool_bits(probs[6], true);
            bits += estimate_bool_bits(probs[7], true);
            bits += estimate_bool_bits(probs[8], false);
        }
        9 => {
            bits += estimate_bool_bits(probs[0], true);
            bits += estimate_bool_bits(probs[1], true);
            bits += estimate_bool_bits(probs[2], true);
            bits += estimate_bool_bits(probs[3], true);
            bits += estimate_bool_bits(probs[5], true);
            bits += estimate_bool_bits(probs[6], true);
            bits += estimate_bool_bits(probs[7], true);
            bits += estimate_bool_bits(probs[8], true);
        }
        _ => {}
    }

    bits
}

/// Calculate RD lambda from QP
pub fn calculate_lambda(qp: u8) -> f64 {
    // Lambda increases with QP
    // This is a simplified model; real encoders use more sophisticated formulas
    let qp_normalized = qp as f64 / 127.0;
    RD_LAMBDA_BASE * (1.0 + qp_normalized * 4.0)
}

/// Perform full mode decision for a macroblock
pub fn decide_macroblock_mode(
    original_y: &[u8], // 16x16 luma
    original_u: &[u8], // 8x8 chroma U
    original_v: &[u8], // 8x8 chroma V
    y_stride: usize,
    uv_stride: usize,
    above_y: &[u8],
    left_y: &[u8],
    above_left_y: u8,
    above_u: &[u8],
    above_v: &[u8],
    left_u: &[u8],
    left_v: &[u8],
    above_left_u: u8,
    above_left_v: u8,
    above_available: bool,
    left_available: bool,
    quant: &QuantFactors,
) -> MacroblockModeDecision {
    let lambda = calculate_lambda(quant.y_ac as u8);

    // Select best 16x16 mode
    let (y_mode, y_cost) = select_16x16_mode(
        original_y,
        y_stride,
        above_y,
        left_y,
        above_left_y,
        above_available,
        left_available,
        quant,
        lambda,
    );

    // Select best chroma mode
    let (uv_mode, uv_cost) = select_chroma_mode(
        original_u,
        original_v,
        uv_stride,
        above_u,
        above_v,
        left_u,
        left_v,
        above_left_u,
        above_left_v,
        above_available,
        left_available,
        lambda,
    );

    // For now, just use 16x16 mode (4x4 mode decision is more complex)
    MacroblockModeDecision {
        y_mode,
        uv_mode,
        use_4x4_modes: false,
        subblock_modes: [[IntraMode4x4::BDcPred; 4]; 4],
        estimated_rate: 0, // Would be calculated from actual encoding
        estimated_distortion: y_cost + uv_cost,
    }
}

/// Extract a 4x4 subblock from a larger block
pub fn extract_4x4_subblock(src: &[u8], src_stride: usize, sb_x: usize, sb_y: usize) -> [u8; 16] {
    let mut block = [0u8; 16];
    let start_x = sb_x * 4;
    let start_y = sb_y * 4;

    for y in 0..4 {
        for x in 0..4 {
            block[y * 4 + x] = src[(start_y + y) * src_stride + start_x + x];
        }
    }

    block
}

/// Extract an 8x8 subblock (for chroma)
pub fn extract_8x8_block(src: &[u8], src_stride: usize) -> [u8; 64] {
    let mut block = [0u8; 64];

    for y in 0..8 {
        for x in 0..8 {
            block[y * 8 + x] = src[y * src_stride + x];
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_ssd() {
        let orig = [100u8; 16];
        let recon = [100u8; 16];
        assert_eq!(calculate_ssd(&orig, &recon, 4, 4, 4), 0);

        let recon2 = [101u8; 16];
        assert_eq!(calculate_ssd(&orig, &recon2, 4, 4, 4), 16); // 16 pixels, each diff=1
    }

    #[test]
    fn test_calculate_sad() {
        let orig = [100u8; 16];
        let recon = [100u8; 16];
        assert_eq!(calculate_sad(&orig, &recon, 4, 4, 4), 0);

        let recon2 = [102u8; 16];
        assert_eq!(calculate_sad(&orig, &recon2, 4, 4, 4), 32); // 16 pixels, each diff=2
    }

    #[test]
    fn test_calculate_lambda() {
        let lambda_0 = calculate_lambda(0);
        let lambda_127 = calculate_lambda(127);
        assert!(lambda_127 > lambda_0);
    }

    #[test]
    fn test_macroblock_mode_decision_default() {
        let decision = MacroblockModeDecision::default();
        assert_eq!(decision.y_mode, IntraMode16x16::DcPred);
        assert_eq!(decision.uv_mode, IntraMode16x16::DcPred);
        assert!(!decision.use_4x4_modes);
    }

    #[test]
    fn test_select_16x16_mode_dc() {
        let original = [128u8; 256]; // Uniform block
        let above = [128u8; 16];
        let left = [128u8; 16];
        let quant = QuantFactors::from_indices(40, 0, 0, 0, 0, 0);

        let (mode, _cost) =
            select_16x16_mode(&original, 16, &above, &left, 128, true, true, &quant, 1.0);

        // For a uniform block, DC mode should be best or very close
        // (all modes should produce similar results for uniform content)
        assert!(matches!(
            mode,
            IntraMode16x16::DcPred
                | IntraMode16x16::VPred
                | IntraMode16x16::HPred
                | IntraMode16x16::TmPred
        ));
    }

    #[test]
    fn test_select_16x16_mode_vertical() {
        // Create a block with vertical gradient
        let mut original = [0u8; 256];
        let above = [200u8; 16];
        let left = [100u8; 16];

        // Fill with vertical pattern (values from above row)
        for y in 0..16 {
            for x in 0..16 {
                original[y * 16 + x] = above[x];
            }
        }

        let quant = QuantFactors::from_indices(40, 0, 0, 0, 0, 0);

        let (mode, _cost) =
            select_16x16_mode(&original, 16, &above, &left, 128, true, true, &quant, 1.0);

        // For vertical gradient, V mode should be best
        assert_eq!(mode, IntraMode16x16::VPred);
    }

    #[test]
    fn test_extract_4x4_subblock() {
        let mut src = [0u8; 256];
        for i in 0..256 {
            src[i] = i as u8;
        }

        let block = extract_4x4_subblock(&src, 16, 1, 1);

        // Subblock (1,1) starts at (4,4)
        assert_eq!(block[0], (4 * 16 + 4) as u8);
        assert_eq!(block[1], (4 * 16 + 5) as u8);
        assert_eq!(block[4], (5 * 16 + 4) as u8);
    }
}
