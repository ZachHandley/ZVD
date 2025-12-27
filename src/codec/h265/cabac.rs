//! CABAC (Context-Adaptive Binary Arithmetic Coding) for H.265/HEVC
//!
//! This module implements the entropy decoder used in H.265 to decode
//! syntax elements from the bitstream. CABAC is a highly efficient
//! arithmetic coding method that adapts to the statistics of the data.
//!
//! # Components
//!
//! - **Arithmetic Decoder**: Binary arithmetic coding engine
//! - **Context Models**: Adaptive probability models
//! - **Binarization**: Conversion of syntax elements to binary decisions
//! - **Coefficient Decoding**: Transform coefficient parsing

use crate::error::{Error, Result};

/// CABAC context state
///
/// Represents the probability state for a context model.
/// H.265 uses 64 possible states (0-63) where:
/// - 0-31: MPS (Most Probable Symbol) = 0
/// - 32-63: MPS = 1
#[derive(Debug, Clone, Copy)]
pub struct ContextModel {
    /// State value (0-63)
    state: u8,
}

impl ContextModel {
    /// Create a new context model with given state
    pub fn new(state: u8) -> Self {
        Self {
            state: state.min(63),
        }
    }

    /// Initialize context from table
    pub fn init(init_value: u8, qp: u8) -> Self {
        // H.265 context initialization formula:
        // preCtxState = Clip3(1, 126, ((m * Clip3(0, 51, QP)) >> 4) + n)
        // where init_value encodes m and n
        let m = ((init_value >> 4) as i32) * 5 - 45;
        let n = (init_value & 15) as i32;

        let qp_clamped = qp.min(51) as i32;
        let pre_ctx_state = ((m * qp_clamped) >> 4) + n;
        let pre_ctx_state = pre_ctx_state.clamp(1, 126);

        // Convert to state index (0-63)
        let state = if pre_ctx_state <= 63 {
            63 - pre_ctx_state
        } else {
            pre_ctx_state - 64
        } as u8;

        Self { state }
    }

    /// Get the MPS (Most Probable Symbol) value
    #[inline]
    pub fn mps(&self) -> u8 {
        (self.state >> 6) & 1
    }

    /// Get the LPS (Least Probable Symbol) value
    #[inline]
    pub fn lps(&self) -> u8 {
        1 - self.mps()
    }

    /// Get the probability state index (0-63)
    #[inline]
    pub fn p_state_idx(&self) -> u8 {
        self.state & 63
    }

    /// Update context after decoding MPS
    pub fn update_mps(&mut self) {
        self.state = NEXT_STATE_MPS[self.state as usize];
    }

    /// Update context after decoding LPS
    pub fn update_lps(&mut self) {
        self.state = NEXT_STATE_LPS[self.state as usize];
    }
}

/// State transition table for MPS (Most Probable Symbol)
const NEXT_STATE_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// State transition table for LPS (Least Probable Symbol)
const NEXT_STATE_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// Range table for LPS (Least Probable Symbol) probability
/// Indexed by [p_state_idx][range_idx]
const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [95, 116, 137, 158], [90, 110, 130, 150], [85, 104, 123, 142], [81, 99, 117, 135],
    [77, 94, 111, 128], [73, 89, 105, 122], [69, 85, 100, 116], [66, 80, 95, 110],
    [62, 76, 90, 104], [59, 72, 86, 99], [56, 69, 81, 94], [53, 65, 77, 89],
    [51, 62, 73, 85], [48, 59, 69, 80], [46, 56, 66, 76], [43, 53, 63, 72],
    [41, 50, 59, 69], [39, 48, 56, 65], [37, 45, 54, 62], [35, 43, 51, 59],
    [33, 41, 48, 56], [32, 39, 46, 53], [30, 37, 43, 50], [29, 35, 41, 48],
    [27, 33, 39, 45], [26, 31, 37, 43], [24, 30, 35, 41], [23, 28, 33, 39],
    [22, 27, 32, 37], [21, 26, 30, 35], [20, 24, 29, 33], [19, 23, 27, 31],
    [18, 22, 26, 30], [17, 21, 25, 28], [16, 20, 23, 27], [15, 19, 22, 25],
    [14, 18, 21, 24], [14, 17, 20, 23], [13, 16, 19, 22], [12, 15, 18, 21],
    [12, 14, 17, 20], [11, 14, 16, 19], [11, 13, 15, 18], [10, 12, 15, 17],
    [10, 12, 14, 16], [9, 11, 13, 15], [9, 11, 12, 14], [8, 10, 12, 14],
    [8, 9, 11, 13], [7, 9, 11, 12], [7, 9, 10, 12], [7, 8, 10, 11],
    [6, 8, 9, 11], [6, 7, 9, 10], [6, 7, 8, 9], [2, 2, 2, 2],
];

/// CABAC arithmetic decoder
pub struct CabacDecoder<'a> {
    /// Input bitstream
    data: &'a [u8],
    /// Current byte position
    pos: usize,
    /// Coding range (codIRange in spec)
    range: u32,
    /// Coding offset (codIOffset in spec)
    offset: u32,
    /// Bits needed counter
    bits_needed: u8,
}

impl<'a> CabacDecoder<'a> {
    /// Create a new CABAC decoder
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Codec("Empty CABAC data".to_string()));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            range: 510,
            offset: 0,
            bits_needed: 0,
        };

        // Initialize offset with first 9 bits
        decoder.init_offset()?;

        Ok(decoder)
    }

    /// Initialize the offset register
    fn init_offset(&mut self) -> Result<()> {
        // Read first 9 bits into offset
        for _ in 0..9 {
            self.offset <<= 1;
            if let Some(bit) = self.read_bit()? {
                self.offset |= bit as u32;
            }
        }
        Ok(())
    }

    /// Read a single bit from the bitstream
    fn read_bit(&mut self) -> Result<Option<u8>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }

        let byte = self.data[self.pos];
        let bit_offset = self.bits_needed as usize;
        let bit = (byte >> (7 - bit_offset)) & 1;

        self.bits_needed += 1;
        if self.bits_needed == 8 {
            self.bits_needed = 0;
            self.pos += 1;
        }

        Ok(Some(bit))
    }

    /// Decode a single binary decision with context
    pub fn decode_decision(&mut self, ctx: &mut ContextModel) -> Result<u8> {
        // Get range index (top 2 bits of range >> 6)
        let range_idx = ((self.range >> 6) & 3) as usize;

        // Get LPS range from table
        let p_state_idx = ctx.p_state_idx() as usize;
        let range_lps = RANGE_TAB_LPS[p_state_idx][range_idx] as u32;

        // Update range
        self.range -= range_lps;

        let bin_val;

        // Check if LPS or MPS
        if self.offset >= self.range {
            // LPS path
            bin_val = ctx.lps();
            self.offset -= self.range;
            self.range = range_lps;
            ctx.update_lps();
        } else {
            // MPS path
            bin_val = ctx.mps();
            ctx.update_mps();
        }

        // Renormalization
        self.renormalize()?;

        Ok(bin_val)
    }

    /// Decode a binary decision in bypass mode (equal probability)
    pub fn decode_bypass(&mut self) -> Result<u8> {
        self.offset <<= 1;

        if let Some(bit) = self.read_bit()? {
            self.offset |= bit as u32;
        }

        let bin_val = if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        };

        Ok(bin_val)
    }

    /// Decode multiple bypass bins
    pub fn decode_bypass_bins(&mut self, num_bins: usize) -> Result<u32> {
        let mut value = 0;
        for _ in 0..num_bins {
            value = (value << 1) | (self.decode_bypass()? as u32);
        }
        Ok(value)
    }

    /// Decode a terminate decision (end of slice)
    pub fn decode_terminate(&mut self) -> Result<bool> {
        self.range -= 2;

        let terminate = if self.offset >= self.range {
            true
        } else {
            // Renormalize if not terminating
            self.renormalize()?;
            false
        };

        Ok(terminate)
    }

    /// Renormalization - maintain range in [256, 510]
    fn renormalize(&mut self) -> Result<()> {
        while self.range < 256 {
            self.range <<= 1;
            self.offset <<= 1;

            if let Some(bit) = self.read_bit()? {
                self.offset |= bit as u32;
            }
        }
        Ok(())
    }

    /// Get current byte position in stream
    pub fn byte_pos(&self) -> usize {
        self.pos
    }
}

/// H.265 CABAC context table indices for various syntax elements
/// These define the starting context index for each syntax element
pub mod ctx_idx {
    /// split_cu_flag contexts (3 contexts based on depth)
    pub const SPLIT_CU_FLAG: usize = 0;
    /// cu_skip_flag contexts (3 contexts)
    pub const CU_SKIP_FLAG: usize = 3;
    /// pred_mode_flag context (1 context)
    pub const PRED_MODE_FLAG: usize = 6;
    /// part_mode contexts (4 contexts)
    pub const PART_MODE: usize = 7;
    /// prev_intra_luma_pred_flag context (1 context)
    pub const PREV_INTRA_LUMA_PRED_FLAG: usize = 11;
    /// intra_chroma_pred_mode context (1 context)
    pub const INTRA_CHROMA_PRED_MODE: usize = 12;
    /// cbf_luma contexts (2 contexts)
    pub const CBF_LUMA: usize = 13;
    /// cbf_chroma contexts (4 contexts)
    pub const CBF_CHROMA: usize = 15;
    /// split_transform_flag contexts (3 contexts)
    pub const SPLIT_TRANSFORM_FLAG: usize = 19;
    /// sig_coeff_group_flag contexts (4 contexts)
    pub const SIG_COEFF_GROUP_FLAG: usize = 22;
    /// sig_coeff_flag contexts (44 contexts)
    pub const SIG_COEFF_FLAG: usize = 26;
    /// coeff_abs_level_greater1_flag contexts (24 contexts)
    pub const COEFF_ABS_LEVEL_GREATER1_FLAG: usize = 70;
    /// coeff_abs_level_greater2_flag contexts (6 contexts)
    pub const COEFF_ABS_LEVEL_GREATER2_FLAG: usize = 94;
    /// Total number of contexts
    pub const NUM_CONTEXTS: usize = 100;
}

/// H.265 CABAC context table with all context models
#[derive(Debug, Clone)]
pub struct CabacContexts {
    /// Context models for all syntax elements
    pub contexts: Vec<ContextModel>,
}

impl CabacContexts {
    /// Create and initialize all CABAC contexts for given QP
    pub fn new(qp: u8) -> Self {
        let mut contexts = Vec::with_capacity(ctx_idx::NUM_CONTEXTS);

        // Initialize contexts with default init values
        // These init values are from H.265 specification tables

        // split_cu_flag (3 contexts)
        contexts.push(ContextModel::init(139, qp)); // ctxIdx 0
        contexts.push(ContextModel::init(141, qp)); // ctxIdx 1
        contexts.push(ContextModel::init(157, qp)); // ctxIdx 2

        // cu_skip_flag (3 contexts)
        contexts.push(ContextModel::init(197, qp)); // ctxIdx 3
        contexts.push(ContextModel::init(185, qp)); // ctxIdx 4
        contexts.push(ContextModel::init(201, qp)); // ctxIdx 5

        // pred_mode_flag (1 context)
        contexts.push(ContextModel::init(149, qp)); // ctxIdx 6

        // part_mode (4 contexts)
        contexts.push(ContextModel::init(154, qp)); // ctxIdx 7
        contexts.push(ContextModel::init(139, qp)); // ctxIdx 8
        contexts.push(ContextModel::init(154, qp)); // ctxIdx 9
        contexts.push(ContextModel::init(154, qp)); // ctxIdx 10

        // prev_intra_luma_pred_flag (1 context)
        contexts.push(ContextModel::init(183, qp)); // ctxIdx 11

        // intra_chroma_pred_mode (1 context)
        contexts.push(ContextModel::init(152, qp)); // ctxIdx 12

        // cbf_luma (2 contexts)
        contexts.push(ContextModel::init(111, qp)); // ctxIdx 13
        contexts.push(ContextModel::init(141, qp)); // ctxIdx 14

        // cbf_chroma (4 contexts)
        contexts.push(ContextModel::init(94, qp));  // ctxIdx 15
        contexts.push(ContextModel::init(138, qp)); // ctxIdx 16
        contexts.push(ContextModel::init(182, qp)); // ctxIdx 17
        contexts.push(ContextModel::init(154, qp)); // ctxIdx 18

        // split_transform_flag (3 contexts)
        contexts.push(ContextModel::init(153, qp)); // ctxIdx 19
        contexts.push(ContextModel::init(138, qp)); // ctxIdx 20
        contexts.push(ContextModel::init(138, qp)); // ctxIdx 21

        // sig_coeff_group_flag (4 contexts)
        for _ in 0..4 {
            contexts.push(ContextModel::init(91, qp));
        }

        // sig_coeff_flag (44 contexts)
        for _ in 0..44 {
            contexts.push(ContextModel::init(110, qp));
        }

        // coeff_abs_level_greater1_flag (24 contexts)
        for _ in 0..24 {
            contexts.push(ContextModel::init(140, qp));
        }

        // coeff_abs_level_greater2_flag (6 contexts)
        for _ in 0..6 {
            contexts.push(ContextModel::init(140, qp));
        }

        Self { contexts }
    }

    /// Get mutable reference to context at given index
    pub fn get_mut(&mut self, idx: usize) -> &mut ContextModel {
        &mut self.contexts[idx]
    }

    /// Reset all contexts for new slice with given QP
    pub fn reset(&mut self, qp: u8) {
        *self = Self::new(qp);
    }
}

/// Extended CABAC decoder with context tables
pub struct CabacReader<'a> {
    /// Base CABAC decoder
    decoder: CabacDecoder<'a>,
    /// Context models
    pub contexts: CabacContexts,
}

impl<'a> CabacReader<'a> {
    /// Create a new CABAC reader with initialized contexts
    pub fn new(data: &'a [u8], qp: u8) -> Result<Self> {
        let decoder = CabacDecoder::new(data)?;
        let contexts = CabacContexts::new(qp);
        Ok(Self { decoder, contexts })
    }

    /// Decode split_cu_flag
    ///
    /// Returns true if the CU should be split into 4 sub-CUs
    pub fn decode_split_cu_flag(&mut self, depth: u8, left_split: bool, above_split: bool) -> Result<bool> {
        // Context selection based on neighbors
        let ctx_inc = (left_split as usize) + (above_split as usize);
        let ctx_idx = ctx_idx::SPLIT_CU_FLAG + ctx_inc;

        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode cu_skip_flag (for inter prediction)
    pub fn decode_cu_skip_flag(&mut self, ctx_inc: usize) -> Result<bool> {
        let ctx_idx = ctx_idx::CU_SKIP_FLAG + ctx_inc.min(2);
        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode pred_mode_flag
    ///
    /// Returns true for INTRA mode, false for INTER mode
    pub fn decode_pred_mode_flag(&mut self) -> Result<bool> {
        let ctx_idx = ctx_idx::PRED_MODE_FLAG;
        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode prev_intra_luma_pred_flag
    ///
    /// If true, the intra mode is one of the 3 MPM (Most Probable Modes)
    pub fn decode_prev_intra_luma_pred_flag(&mut self) -> Result<bool> {
        let ctx_idx = ctx_idx::PREV_INTRA_LUMA_PRED_FLAG;
        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode mpm_idx (0, 1, or 2)
    ///
    /// Indicates which of the 3 MPMs to use
    pub fn decode_mpm_idx(&mut self) -> Result<u8> {
        // First bin: 0 means mpm_idx=0
        let bin0 = self.decoder.decode_bypass()?;
        if bin0 == 0 {
            return Ok(0);
        }

        // Second bin: 0 means mpm_idx=1, 1 means mpm_idx=2
        let bin1 = self.decoder.decode_bypass()?;
        Ok(if bin1 == 0 { 1 } else { 2 })
    }

    /// Decode rem_intra_luma_pred_mode (5 bits in bypass mode)
    ///
    /// Used when not using MPM, encodes mode index with exclusion
    pub fn decode_rem_intra_luma_pred_mode(&mut self) -> Result<u8> {
        let value = self.decoder.decode_bypass_bins(5)? as u8;
        Ok(value)
    }

    /// Decode intra_chroma_pred_mode
    ///
    /// Returns 0-4: 0 = planar, 1 = vertical, 2 = horizontal, 3 = DC, 4 = luma mode
    pub fn decode_intra_chroma_pred_mode(&mut self) -> Result<u8> {
        let ctx_idx = ctx_idx::INTRA_CHROMA_PRED_MODE;
        let bin0 = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;

        if bin0 == 0 {
            // Derive from luma mode (mode 4 in our encoding)
            return Ok(4);
        }

        // Read 2 bypass bins for mode 0-3
        let mode = self.decoder.decode_bypass_bins(2)? as u8;
        Ok(mode)
    }

    /// Decode cbf_luma (coded block flag for luma)
    pub fn decode_cbf_luma(&mut self, depth: u8) -> Result<bool> {
        let ctx_idx = ctx_idx::CBF_LUMA + (depth > 0) as usize;
        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode cbf_chroma (coded block flag for chroma)
    pub fn decode_cbf_chroma(&mut self, depth: u8) -> Result<bool> {
        let ctx_idx = ctx_idx::CBF_CHROMA + (depth.min(3) as usize);
        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode split_transform_flag
    pub fn decode_split_transform_flag(&mut self, depth: u8) -> Result<bool> {
        let ctx_idx = ctx_idx::SPLIT_TRANSFORM_FLAG + (depth.min(2) as usize);
        let bin = self.decoder.decode_decision(self.contexts.get_mut(ctx_idx))?;
        Ok(bin == 1)
    }

    /// Decode a truncated unary value (used for various syntax elements)
    pub fn decode_truncated_unary(&mut self, max_val: u32) -> Result<u32> {
        let mut value = 0;
        while value < max_val {
            let bin = self.decoder.decode_bypass()?;
            if bin == 0 {
                break;
            }
            value += 1;
        }
        Ok(value)
    }

    /// Decode an exponential-golomb coded value (bypass mode)
    pub fn decode_exp_golomb(&mut self, k: u8) -> Result<u32> {
        // Prefix: count zeros until we hit a 1
        let mut prefix = 0;
        loop {
            let bin = self.decoder.decode_bypass()?;
            if bin == 1 {
                break;
            }
            prefix += 1;
        }

        // Suffix: read (prefix + k) bits
        let suffix_len = prefix + k as u32;
        let suffix = if suffix_len > 0 {
            self.decoder.decode_bypass_bins(suffix_len as usize)?
        } else {
            0
        };

        // Compute value
        let value = ((1 << prefix) - 1 + (1 << k)) + suffix - (1 << k);
        Ok(value)
    }

    /// Check if we've reached the end of the slice
    pub fn decode_end_of_slice(&mut self) -> Result<bool> {
        self.decoder.decode_terminate()
    }

    /// Get current byte position
    pub fn byte_pos(&self) -> usize {
        self.decoder.byte_pos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_model_init() {
        let ctx = ContextModel::init(154, 26);
        assert!(ctx.state <= 63);
    }

    #[test]
    fn test_context_model_mps_lps() {
        let ctx = ContextModel::new(10);
        assert_eq!(ctx.mps(), 0);
        assert_eq!(ctx.lps(), 1);

        let ctx = ContextModel::new(70);
        assert_eq!(ctx.mps(), 1);
        assert_eq!(ctx.lps(), 0);
    }

    #[test]
    fn test_context_update() {
        let mut ctx = ContextModel::new(0);
        let initial = ctx.state;
        ctx.update_mps();
        assert!(ctx.state >= initial);

        let mut ctx = ContextModel::new(10);
        ctx.update_lps();
        assert!(ctx.state <= 10);
    }

    #[test]
    fn test_cabac_decoder_creation() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let decoder = CabacDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_cabac_decoder_empty_data() {
        let data: Vec<u8> = vec![];
        let decoder = CabacDecoder::new(&data);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_cabac_bypass_mode() {
        let data = vec![0b10101010, 0xFF, 0xFF];
        let mut decoder = CabacDecoder::new(&data).unwrap();

        // Bypass mode should decode bits without context
        let result = decoder.decode_bypass();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cabac_bypass_bins() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = CabacDecoder::new(&data).unwrap();

        let result = decoder.decode_bypass_bins(4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_context_state_transitions() {
        // Test that state transitions stay in valid range
        for state in 0..64 {
            let mps_next = NEXT_STATE_MPS[state];
            let lps_next = NEXT_STATE_LPS[state];
            assert!(mps_next <= 63);
            assert!(lps_next <= 63);
        }
    }

    #[test]
    fn test_range_table_values() {
        // Verify range table has valid probability ranges
        for row in RANGE_TAB_LPS.iter() {
            for &val in row {
                assert!(val <= 240);
            }
        }
    }

    #[test]
    fn test_cabac_terminate() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = CabacDecoder::new(&data).unwrap();

        let result = decoder.decode_terminate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_context_initialization_qp_range() {
        // Test context initialization with different QP values
        for qp in 0..52 {
            let ctx = ContextModel::init(154, qp);
            assert!(ctx.state <= 63);
        }
    }

    #[test]
    fn test_cabac_renormalization() {
        let data = vec![0xAA, 0xAA, 0xAA, 0xAA];
        let decoder = CabacDecoder::new(&data).unwrap();

        // After initialization, range should be in valid range
        assert!(decoder.range >= 256 && decoder.range <= 510);
    }
}
