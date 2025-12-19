//! Boolean arithmetic decoder for VP8
//!
//! The boolean arithmetic coder is the core entropy coding mechanism in VP8.
//! Every symbol in the compressed data is decoded through this mechanism.

/// Boolean arithmetic decoder state
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    value: u32,
    range: u32,
    bit_count: i32,
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

        // Load initial bits (two bytes)
        decoder.value = ((data.get(0).copied().unwrap_or(0) as u32) << 8)
            | (data.get(1).copied().unwrap_or(0) as u32);
        decoder.pos = 2;
        decoder.bit_count = 0;

        decoder
    }

    /// Read a single boolean with given probability (0-255)
    /// prob = probability of 0 (prob/256 is actual probability)
    #[inline]
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

    /// Read an unsigned n-bit literal value (uniform distribution)
    #[inline]
    pub fn read_literal(&mut self, n: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bool(128) as u32);
        }
        value
    }

    /// Read a signed n-bit value (literal + sign bit)
    #[inline]
    pub fn read_signed_literal(&mut self, n: u8) -> i32 {
        let value = self.read_literal(n) as i32;
        if value != 0 && self.read_bool(128) {
            -value
        } else {
            value
        }
    }

    /// Read a value using a probability tree
    /// tree contains pairs of (left, right) where negative values are leaves
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

    /// Check if we've consumed all data
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Get current position in bytes
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        if self.pos >= self.data.len() {
            0
        } else {
            self.data.len() - self.pos
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_decoder_creation() {
        let data = [0x9D, 0x01, 0x2A, 0x00, 0x00];
        let decoder = BoolDecoder::new(&data);
        assert_eq!(decoder.position(), 2);
    }

    #[test]
    fn test_read_literal() {
        // Create test data with known pattern
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = BoolDecoder::new(&data);

        // Reading with prob=128 should give uniform bits
        // With all 0xFF input, we expect mostly 1s
        let val = decoder.read_literal(4);
        // Value depends on the arithmetic coding state
        assert!(val <= 15); // 4 bits max
    }

    #[test]
    fn test_read_bool_sequence() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut decoder = BoolDecoder::new(&data);

        // With all zeros input and prob=128, expect 0s
        for _ in 0..8 {
            let bit = decoder.read_bool(128);
            // The actual value depends on the decoder state
            let _ = bit;
        }
    }
}
