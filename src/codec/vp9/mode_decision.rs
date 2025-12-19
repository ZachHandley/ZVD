//! VP9 Mode Decision for Encoding
//!
//! This module handles intra mode selection for VP9 encoding.
//! It evaluates all 10 intra prediction modes and selects the best one
//! based on rate-distortion optimization.

use super::prediction::IntraPredictor;
use super::quant::{quantize, PlaneQuantFactors};
use super::tables::{BlockSize, IntraMode, TxSize, TxType};
use super::transform::{compute_residual, forward_transform};

/// Mode decision result
#[derive(Debug, Clone)]
pub struct ModeDecision {
    /// Selected Y mode
    pub y_mode: IntraMode,
    /// Selected UV mode
    pub uv_mode: IntraMode,
    /// Transform size to use
    pub tx_size: TxSize,
    /// Transform type to use
    pub tx_type: TxType,
    /// Rate-distortion cost
    pub rd_cost: u64,
    /// Skip flag (no residual)
    pub skip: bool,
}

impl Default for ModeDecision {
    fn default() -> Self {
        ModeDecision {
            y_mode: IntraMode::DcPred,
            uv_mode: IntraMode::DcPred,
            tx_size: TxSize::Tx4x4,
            tx_type: TxType::DctDct,
            rd_cost: u64::MAX,
            skip: false,
        }
    }
}

/// Mode decision context
pub struct ModeDecisionContext {
    /// Lambda for rate-distortion optimization
    pub lambda: u64,
    /// Quantization factors for Y plane
    pub y_quant: PlaneQuantFactors,
    /// Quantization factors for UV planes
    pub uv_quant: PlaneQuantFactors,
    /// Speed setting (0-9, higher = faster but lower quality)
    pub speed: u8,
}

impl ModeDecisionContext {
    /// Create a new mode decision context
    pub fn new(base_q_idx: u8, speed: u8) -> Self {
        use super::tables::{AC_QUANT_8BIT, DC_QUANT_8BIT};

        let q_idx = base_q_idx as usize;

        // Lambda based on Q index (simplified)
        let lambda = (base_q_idx as u64).saturating_mul(base_q_idx as u64) / 4;

        ModeDecisionContext {
            lambda,
            y_quant: PlaneQuantFactors {
                dc: DC_QUANT_8BIT[q_idx],
                ac: AC_QUANT_8BIT[q_idx],
            },
            uv_quant: PlaneQuantFactors {
                dc: DC_QUANT_8BIT[q_idx],
                ac: AC_QUANT_8BIT[q_idx],
            },
            speed,
        }
    }

    /// Get modes to evaluate based on speed setting
    fn get_modes_to_test(&self) -> &[IntraMode] {
        static ALL_MODES: [IntraMode; 10] = [
            IntraMode::DcPred,
            IntraMode::VPred,
            IntraMode::HPred,
            IntraMode::D45Pred,
            IntraMode::D135Pred,
            IntraMode::D117Pred,
            IntraMode::D153Pred,
            IntraMode::D207Pred,
            IntraMode::D63Pred,
            IntraMode::TmPred,
        ];

        static FAST_MODES: [IntraMode; 4] = [
            IntraMode::DcPred,
            IntraMode::VPred,
            IntraMode::HPred,
            IntraMode::TmPred,
        ];

        static MEDIUM_MODES: [IntraMode; 6] = [
            IntraMode::DcPred,
            IntraMode::VPred,
            IntraMode::HPred,
            IntraMode::D45Pred,
            IntraMode::D135Pred,
            IntraMode::TmPred,
        ];

        if self.speed >= 8 {
            &FAST_MODES
        } else if self.speed >= 5 {
            &MEDIUM_MODES
        } else {
            &ALL_MODES
        }
    }
}

/// Calculate Sum of Absolute Differences (SAD)
pub fn calc_sad(
    original: &[u8],
    prediction: &[u8],
    width: usize,
    height: usize,
    orig_stride: usize,
    pred_stride: usize,
) -> u64 {
    let mut sad = 0u64;
    for y in 0..height {
        for x in 0..width {
            let orig = original[y * orig_stride + x] as i32;
            let pred = prediction[y * pred_stride + x] as i32;
            sad += (orig - pred).unsigned_abs() as u64;
        }
    }
    sad
}

/// Calculate Sum of Squared Errors (SSE)
pub fn calc_sse(
    original: &[u8],
    prediction: &[u8],
    width: usize,
    height: usize,
    orig_stride: usize,
    pred_stride: usize,
) -> u64 {
    let mut sse = 0u64;
    for y in 0..height {
        for x in 0..width {
            let orig = original[y * orig_stride + x] as i32;
            let pred = prediction[y * pred_stride + x] as i32;
            let diff = orig - pred;
            sse += (diff * diff) as u64;
        }
    }
    sse
}

/// Calculate SATD (Sum of Absolute Transformed Differences)
/// Uses Hadamard transform for better correlation with actual rate
pub fn calc_satd_4x4(
    original: &[u8],
    prediction: &[u8],
    orig_stride: usize,
    pred_stride: usize,
) -> u64 {
    // Compute difference
    let mut diff = [0i32; 16];
    for y in 0..4 {
        for x in 0..4 {
            let orig = original[y * orig_stride + x] as i32;
            let pred = prediction[y * pred_stride + x] as i32;
            diff[y * 4 + x] = orig - pred;
        }
    }

    // Apply 4x4 Hadamard transform
    let mut temp = [0i32; 16];

    // Row transform
    for i in 0..4 {
        let row_offset = i * 4;
        let a = diff[row_offset] + diff[row_offset + 3];
        let b = diff[row_offset + 1] + diff[row_offset + 2];
        let c = diff[row_offset + 1] - diff[row_offset + 2];
        let d = diff[row_offset] - diff[row_offset + 3];

        temp[row_offset] = a + b;
        temp[row_offset + 1] = c + d;
        temp[row_offset + 2] = a - b;
        temp[row_offset + 3] = d - c;
    }

    // Column transform
    let mut hadamard = [0i32; 16];
    for j in 0..4 {
        let a = temp[j] + temp[12 + j];
        let b = temp[4 + j] + temp[8 + j];
        let c = temp[4 + j] - temp[8 + j];
        let d = temp[j] - temp[12 + j];

        hadamard[j] = a + b;
        hadamard[4 + j] = c + d;
        hadamard[8 + j] = a - b;
        hadamard[12 + j] = d - c;
    }

    // Sum absolute values
    let mut satd = 0u64;
    for &v in &hadamard {
        satd += v.unsigned_abs() as u64;
    }

    satd
}

/// Select best intra mode for a block
pub fn select_intra_mode(
    ctx: &ModeDecisionContext,
    original: &[u8],
    orig_stride: usize,
    above: &[u8],
    left: &[u8],
    top_left: u8,
    have_above: bool,
    have_left: bool,
    block_size: BlockSize,
) -> ModeDecision {
    let width = block_size.width();
    let height = block_size.height();
    let size = width; // Assume square for simplicity

    let mut best = ModeDecision::default();
    let mut prediction = vec![128u8; width * height];

    let predictor = IntraPredictor::new(above, left, top_left, have_above, have_left);

    let modes_to_test = ctx.get_modes_to_test();

    for &mode in modes_to_test {
        // Generate prediction
        predictor.predict(mode, &mut prediction, width, size);

        // Calculate distortion
        let distortion = calc_sse(original, &prediction, width, height, orig_stride, width);

        // Estimate rate (simplified - just use mode cost)
        let mode_cost = estimate_mode_cost(mode);

        // Calculate RD cost
        let rd_cost = distortion + ctx.lambda * mode_cost;

        if rd_cost < best.rd_cost {
            best.y_mode = mode;
            best.rd_cost = rd_cost;
        }
    }

    // Select transform size based on block size
    best.tx_size = select_tx_size(block_size);

    // Select transform type based on mode
    best.tx_type = get_tx_type_for_mode(best.y_mode, best.tx_size);

    // UV mode - typically same as Y or DC
    best.uv_mode = if ctx.speed >= 5 {
        IntraMode::DcPred
    } else {
        best.y_mode
    };

    best
}

/// Estimate mode signaling cost (in bits * 256)
fn estimate_mode_cost(mode: IntraMode) -> u64 {
    // Approximate bit costs for each mode based on typical probabilities
    match mode {
        IntraMode::DcPred => 512,   // ~2 bits - most common
        IntraMode::VPred => 768,    // ~3 bits
        IntraMode::HPred => 768,    // ~3 bits
        IntraMode::TmPred => 1024,  // ~4 bits
        IntraMode::D45Pred => 1280, // ~5 bits
        IntraMode::D135Pred => 1280,
        IntraMode::D117Pred => 1536, // ~6 bits
        IntraMode::D153Pred => 1536,
        IntraMode::D207Pred => 1536,
        IntraMode::D63Pred => 1536,
    }
}

/// Select transform size based on block size
fn select_tx_size(block_size: BlockSize) -> TxSize {
    match block_size {
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4 => TxSize::Tx4x4,
        BlockSize::Block8x8 | BlockSize::Block8x16 | BlockSize::Block16x8 => TxSize::Tx8x8,
        BlockSize::Block16x16 | BlockSize::Block16x32 | BlockSize::Block32x16 => TxSize::Tx16x16,
        _ => TxSize::Tx32x32,
    }
}

/// Get transform type based on intra mode
fn get_tx_type_for_mode(mode: IntraMode, tx_size: TxSize) -> TxType {
    // ADST is only used for 4x4 and 8x8
    if tx_size == TxSize::Tx16x16 || tx_size == TxSize::Tx32x32 {
        return TxType::DctDct;
    }

    match mode {
        IntraMode::VPred | IntraMode::D63Pred | IntraMode::D117Pred => TxType::AdstDct,
        IntraMode::HPred | IntraMode::D153Pred | IntraMode::D207Pred => TxType::DctAdst,
        IntraMode::D45Pred | IntraMode::D135Pred | IntraMode::TmPred => TxType::AdstAdst,
        IntraMode::DcPred => TxType::DctDct,
    }
}

/// Partition decision for superblocks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionDecision {
    None,       // Use full block
    Split,      // Split into 4 sub-blocks
    Horizontal, // Split horizontally
    Vertical,   // Split vertically
}

/// Select partition for a block
pub fn select_partition(
    ctx: &ModeDecisionContext,
    original: &[u8],
    orig_stride: usize,
    x: usize,
    y: usize,
    block_size: BlockSize,
    frame_width: usize,
    frame_height: usize,
) -> PartitionDecision {
    let width = block_size.width();
    let height = block_size.height();

    // Check if block fits in frame
    let x_overflow = x + width > frame_width;
    let y_overflow = y + height > frame_height;

    if x_overflow && y_overflow {
        return PartitionDecision::Split;
    } else if x_overflow {
        return PartitionDecision::Vertical;
    } else if y_overflow {
        return PartitionDecision::Horizontal;
    }

    // For smallest blocks, don't split
    if block_size == BlockSize::Block8x8 || block_size == BlockSize::Block4x4 {
        return PartitionDecision::None;
    }

    // Use variance to decide on splitting
    let variance = calculate_variance(original, orig_stride, width, height);

    // Threshold based on Q and block size
    let threshold = (ctx.y_quant.ac as u64) * (width * height) as u64 / 16;

    if variance > threshold && ctx.speed < 7 {
        PartitionDecision::Split
    } else {
        PartitionDecision::None
    }
}

/// Calculate block variance
fn calculate_variance(block: &[u8], stride: usize, width: usize, height: usize) -> u64 {
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    let n = (width * height) as u64;

    for y in 0..height {
        for x in 0..width {
            let val = block[y * stride + x] as u64;
            sum += val;
            sum_sq += val * val;
        }
    }

    // variance = E[x^2] - E[x]^2
    let mean_sq = (sum * sum) / n;
    sum_sq.saturating_sub(mean_sq)
}

/// Encode block decision
#[derive(Debug, Clone)]
pub struct BlockEncodeInfo {
    /// Mode decision
    pub mode: ModeDecision,
    /// Quantized Y coefficients (variable size based on transform)
    pub y_coeffs: Vec<i16>,
    /// Quantized U coefficients
    pub u_coeffs: Vec<i16>,
    /// Quantized V coefficients
    pub v_coeffs: Vec<i16>,
    /// Y end of block
    pub y_eob: usize,
    /// U end of block
    pub u_eob: usize,
    /// V end of block
    pub v_eob: usize,
}

impl BlockEncodeInfo {
    pub fn new(mode: ModeDecision, tx_size: TxSize) -> Self {
        let num_coeffs = tx_size.num_coeffs();
        BlockEncodeInfo {
            mode,
            y_coeffs: vec![0; num_coeffs],
            u_coeffs: vec![0; num_coeffs / 4], // Chroma is half size
            v_coeffs: vec![0; num_coeffs / 4],
            y_eob: 0,
            u_eob: 0,
            v_eob: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_decision_default() {
        let decision = ModeDecision::default();
        assert_eq!(decision.y_mode, IntraMode::DcPred);
        assert_eq!(decision.rd_cost, u64::MAX);
    }

    #[test]
    fn test_calc_sad() {
        let original = vec![100u8; 64];
        let prediction = vec![90u8; 64];

        let sad = calc_sad(&original, &prediction, 8, 8, 8, 8);
        assert_eq!(sad, 64 * 10); // Each pixel differs by 10
    }

    #[test]
    fn test_calc_sse() {
        let original = vec![100u8; 64];
        let prediction = vec![90u8; 64];

        let sse = calc_sse(&original, &prediction, 8, 8, 8, 8);
        assert_eq!(sse, 64 * 100); // Each pixel differs by 10, squared = 100
    }

    #[test]
    fn test_calculate_variance() {
        // Uniform block should have low variance
        let uniform = vec![128u8; 64];
        let var1 = calculate_variance(&uniform, 8, 8, 8);
        assert_eq!(var1, 0);

        // Block with variation
        let mut varied = vec![0u8; 64];
        for i in 0..64 {
            varied[i] = (i * 4) as u8;
        }
        let var2 = calculate_variance(&varied, 8, 8, 8);
        assert!(var2 > 0);
    }

    #[test]
    fn test_select_tx_size() {
        assert_eq!(select_tx_size(BlockSize::Block4x4), TxSize::Tx4x4);
        assert_eq!(select_tx_size(BlockSize::Block8x8), TxSize::Tx8x8);
        assert_eq!(select_tx_size(BlockSize::Block16x16), TxSize::Tx16x16);
        assert_eq!(select_tx_size(BlockSize::Block64x64), TxSize::Tx32x32);
    }

    #[test]
    fn test_mode_decision_context() {
        let ctx = ModeDecisionContext::new(100, 5);
        assert!(ctx.lambda > 0);
        assert_eq!(ctx.speed, 5);
    }
}
