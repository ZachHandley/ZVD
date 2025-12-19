//! VP9 Binary Arithmetic Range Coder
//!
//! VP9 uses a binary arithmetic range coder similar to VP8's boolean coder
//! but with 16-bit range operations. This module implements both the decoder
//! (for reading compressed bitstreams) and can be extended for encoding.

/// Range coder state for VP9 decoding
///
/// The VP9 range coder maintains:
/// - A 16-bit range value (starts at 255)
/// - A multi-byte value being decoded
/// - Position tracking in the input buffer
pub struct RangeCoder<'a> {
    /// Input data buffer
    data: &'a [u8],
    /// Current position in data
    pos: usize,
    /// Current range (8-bit, but stored as u32 for calculations)
    range: u32,
    /// Current value being decoded
    value: u32,
    /// Number of bits consumed in current byte
    bit_count: i32,
}

impl<'a> RangeCoder<'a> {
    /// Minimum range value before renormalization
    pub const MIN_RANGE: u32 = 128;

    /// Initialize a new range coder from input data
    pub fn new(data: &'a [u8]) -> Self {
        let mut coder = RangeCoder {
            data,
            pos: 0,
            range: 255,
            value: 0,
            bit_count: 0,
        };

        // Initialize value from first two bytes
        if data.len() >= 2 {
            coder.value = ((data[0] as u32) << 8) | (data[1] as u32);
            coder.pos = 2;
        } else if data.len() == 1 {
            coder.value = (data[0] as u32) << 8;
            coder.pos = 1;
        }

        coder
    }

    /// Read a single boolean with given probability
    ///
    /// The probability is on a scale of 0-255 where prob represents P(bit == 0).
    /// A probability of 128 means 50/50 chance.
    #[inline]
    pub fn read_bool(&mut self, prob: u8) -> bool {
        // Calculate split point: where 0/1 boundary lies in the range
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let split_shifted = split << 8;

        let bit = if self.value >= split_shifted {
            // Bit is 1 - value is in upper part
            self.range -= split;
            self.value -= split_shifted;
            true
        } else {
            // Bit is 0 - value is in lower part
            self.range = split;
            false
        };

        // Renormalize: ensure range >= 128
        self.renormalize();

        bit
    }

    /// Read a single bit with probability 128 (uniform distribution)
    #[inline]
    pub fn read_bit(&mut self) -> bool {
        self.read_bool(128)
    }

    /// Read n bits as an unsigned literal value (MSB first, uniform distribution)
    #[inline]
    pub fn read_literal(&mut self, n: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit() as u32);
        }
        value
    }

    /// Read n bits as a signed literal (magnitude + sign)
    #[inline]
    pub fn read_signed_literal(&mut self, n: u8) -> i32 {
        let value = self.read_literal(n) as i32;
        if value != 0 && self.read_bit() {
            -value
        } else {
            value
        }
    }

    /// Read an unsigned value up to n bits, with sign bit if non-zero
    /// Used for delta values in VP9
    #[inline]
    pub fn read_delta(&mut self, n: u8) -> i32 {
        if self.read_bit() {
            let magnitude = self.read_literal(n) as i32;
            if self.read_bit() {
                -magnitude
            } else {
                magnitude
            }
        } else {
            0
        }
    }

    /// Read a symbol using a binary tree structure
    ///
    /// Tree is encoded as pairs of (left_child, right_child) where:
    /// - Positive values are indices to next node
    /// - Negative values are leaf symbols (negate and subtract 1 to get symbol)
    #[inline]
    pub fn read_tree(&mut self, tree: &[i8], probs: &[u8]) -> i8 {
        let mut node = 0usize;
        loop {
            let prob = probs[node >> 1];
            let bit = self.read_bool(prob) as usize;
            let next = tree[node + bit];
            if next <= 0 {
                return -next;
            }
            node = next as usize;
        }
    }

    /// Read a symbol using a tree with context-dependent probabilities
    #[inline]
    pub fn read_tree_with_context(&mut self, tree: &[i8], probs: &[u8], prob_offset: usize) -> i8 {
        let mut node = 0usize;
        loop {
            let prob = probs[prob_offset + (node >> 1)];
            let bit = self.read_bool(prob) as usize;
            let next = tree[node + bit];
            if next <= 0 {
                return -next;
            }
            node = next as usize;
        }
    }

    /// Read a probability update value
    /// VP9 uses a specific format for updating probabilities
    #[inline]
    pub fn read_prob_update(&mut self, prob: u8) -> u8 {
        if self.read_bool(prob) {
            // Read new probability as 7-bit value
            let new_prob = self.read_literal(7) as u8;
            (new_prob << 1) | 1 // VP9 stores probs as (value << 1) | 1
        } else {
            0
        }
    }

    /// Read a probability value (1-255 range)
    #[inline]
    pub fn read_probability(&mut self) -> u8 {
        let v = self.read_literal(8) as u8;
        if v == 0 {
            1
        } else {
            v
        }
    }

    /// Renormalize the range coder state
    #[inline]
    fn renormalize(&mut self) {
        while self.range < Self::MIN_RANGE {
            self.range <<= 1;
            self.value <<= 1;
            self.bit_count += 1;

            if self.bit_count == 8 {
                self.bit_count = 0;
                if self.pos < self.data.len() {
                    self.value |= self.data[self.pos] as u32;
                    self.pos += 1;
                }
            }
        }
    }

    /// Get current position in bytes (approximate)
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Check if all data has been consumed
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Get remaining bytes in buffer
    pub fn remaining(&self) -> usize {
        if self.pos >= self.data.len() {
            0
        } else {
            self.data.len() - self.pos
        }
    }

    /// Exit the arithmetic coder (called at end of tile/frame)
    /// Returns remaining unprocessed bits
    pub fn exit_and_get_position(&self) -> usize {
        // Account for any unprocessed bits in the buffer
        self.pos - ((8 - self.bit_count) / 8) as usize
    }
}

/// Bit reader for uncompressed header data
/// VP9 uncompressed headers use simple bit-packing, not arithmetic coding
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit
    #[inline]
    pub fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.data.len() {
            return None;
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;

        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Some(bit != 0)
    }

    /// Read n bits as unsigned value (MSB first)
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Some(value)
    }

    /// Read n bits as signed value (2's complement)
    #[inline]
    pub fn read_signed_bits(&mut self, n: u8) -> Option<i32> {
        let value = self.read_bits(n)? as i32;
        // Sign extend if high bit is set
        if value & (1 << (n - 1)) != 0 {
            Some(value | (!0i32 << n))
        } else {
            Some(value)
        }
    }

    /// Get current byte position
    pub fn position(&self) -> usize {
        self.byte_pos
    }

    /// Get position in bits
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Align to byte boundary
    pub fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Skip n bits
    pub fn skip(&mut self, n: usize) {
        let total_bits = self.byte_pos * 8 + self.bit_pos as usize + n;
        self.byte_pos = total_bits / 8;
        self.bit_pos = (total_bits % 8) as u8;
    }

    /// Check if more data is available
    pub fn has_more(&self) -> bool {
        self.byte_pos < self.data.len()
    }

    /// Get remaining bytes (after byte alignment)
    pub fn remaining_bytes(&self) -> usize {
        let current = if self.bit_pos > 0 {
            self.byte_pos + 1
        } else {
            self.byte_pos
        };
        if current >= self.data.len() {
            0
        } else {
            self.data.len() - current
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_coder_creation() {
        let data = [0x9D, 0x01, 0x2A, 0x00, 0x00];
        let coder = RangeCoder::new(&data);
        assert_eq!(coder.position(), 2);
        assert_eq!(coder.range, 255);
    }

    #[test]
    fn test_read_literal() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut coder = RangeCoder::new(&data);

        // Read 4 bits
        let val = coder.read_literal(4);
        assert!(val <= 15); // 4-bit max
    }

    #[test]
    fn test_read_bool_sequence() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut coder = RangeCoder::new(&data);

        // Read several bools
        for _ in 0..8 {
            let _ = coder.read_bool(128);
        }
    }

    #[test]
    fn test_bit_reader() {
        // 0b10110100 0b11001010
        let data = [0xB4, 0xCA];
        let mut reader = BitReader::new(&data);

        // Read individual bits
        assert_eq!(reader.read_bit(), Some(true)); // 1
        assert_eq!(reader.read_bit(), Some(false)); // 0
        assert_eq!(reader.read_bit(), Some(true)); // 1
        assert_eq!(reader.read_bit(), Some(true)); // 1
    }

    #[test]
    fn test_bit_reader_multi_bit() {
        let data = [0xB4, 0xCA]; // 10110100 11001010
        let mut reader = BitReader::new(&data);

        // Read 4 bits: should be 1011 = 11
        assert_eq!(reader.read_bits(4), Some(11));

        // Read 4 more: should be 0100 = 4
        assert_eq!(reader.read_bits(4), Some(4));
    }

    #[test]
    fn test_signed_literal() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut coder = RangeCoder::new(&data);

        // Just verify it doesn't panic and returns reasonable range
        let val = coder.read_signed_literal(4);
        assert!(val >= -15 && val <= 15);
    }
}
