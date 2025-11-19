//! H.265/HEVC bitstream reader
//!
//! Provides utilities for reading bits and Exp-Golomb coded values from H.265 bitstreams.
//!
//! ## Exp-Golomb Codes
//!
//! H.265 uses Exponential-Golomb codes (Exp-Golomb) to encode many syntax elements.
//!
//! ### Unsigned Exp-Golomb (ue(v))
//!
//! Format: [M zeros][1][M bits for value]
//!
//! Examples:
//! - 0: `1` (1 bit)
//! - 1: `010` (3 bits)
//! - 2: `011` (3 bits)
//! - 3: `00100` (5 bits)
//! - 4: `00101` (5 bits)
//!
//! Decoding:
//! 1. Count leading zeros (M)
//! 2. Read M+1 bits total
//! 3. Value = (1 << M) - 1 + remaining M bits
//!
//! ### Signed Exp-Golomb (se(v))
//!
//! Maps unsigned values to signed:
//! - 0 → 0
//! - 1 → 1
//! - 2 → -1
//! - 3 → 2
//! - 4 → -2
//! - ...
//!
//! Formula: `(-1)^(k+1) * ceil(k/2)` where k is the unsigned value

use crate::error::{Error, Result};

/// Bitstream reader for H.265 RBSP data
///
/// Reads bits sequentially from a byte slice.
pub struct BitstreamReader<'a> {
    /// RBSP data (emulation prevention bytes already removed)
    data: &'a [u8],
    /// Current byte position
    byte_pos: usize,
    /// Current bit position within byte (0-7, where 0 is MSB)
    bit_pos: u8,
}

impl<'a> BitstreamReader<'a> {
    /// Create a new bitstream reader
    pub fn new(data: &'a [u8]) -> Self {
        BitstreamReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Get current bit position in the stream
    pub fn position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Check if more data is available
    pub fn has_more_data(&self) -> bool {
        self.byte_pos < self.data.len()
    }

    /// Read a single bit (returns 0 or 1)
    pub fn read_bit(&mut self) -> Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(Error::codec("Bitstream read past end"));
        }

        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit)
    }

    /// Read N bits as u32 (N <= 32)
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(Error::codec("Cannot read more than 32 bits at once"));
        }

        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }

        Ok(value)
    }

    /// Read N bits as u64 (N <= 64)
    pub fn read_bits_64(&mut self, n: u32) -> Result<u64> {
        if n > 64 {
            return Err(Error::codec("Cannot read more than 64 bits at once"));
        }

        let mut value = 0u64;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u64);
        }

        Ok(value)
    }

    /// Read boolean (1 bit)
    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bit()? == 1)
    }

    /// Read unsigned Exp-Golomb coded value ue(v)
    ///
    /// Algorithm:
    /// 1. Count leading zeros
    /// 2. Read that many additional bits
    /// 3. Compute value from formula
    pub fn read_ue(&mut self) -> Result<u32> {
        // Count leading zeros
        let mut leading_zeros = 0;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(Error::codec("Exp-Golomb code too long (>31 leading zeros)"));
            }
        }

        // If no leading zeros, value is 0
        if leading_zeros == 0 {
            return Ok(0);
        }

        // Read leading_zeros bits for the value part
        let value_part = self.read_bits(leading_zeros)?;

        // Compute final value: 2^leading_zeros - 1 + value_part
        let value = (1u32 << leading_zeros) - 1 + value_part;

        Ok(value)
    }

    /// Read signed Exp-Golomb coded value se(v)
    ///
    /// Maps unsigned to signed:
    /// - 0 → 0
    /// - 1 → 1
    /// - 2 → -1
    /// - 3 → 2
    /// - 4 → -2
    pub fn read_se(&mut self) -> Result<i32> {
        let unsigned = self.read_ue()?;

        // Map unsigned to signed
        let signed = if unsigned == 0 {
            0
        } else if unsigned % 2 == 1 {
            // Odd values map to positive: 1→1, 3→2, 5→3, ...
            ((unsigned + 1) / 2) as i32
        } else {
            // Even values map to negative: 2→-1, 4→-2, 6→-3, ...
            -((unsigned / 2) as i32)
        };

        Ok(signed)
    }

    /// Skip N bits
    pub fn skip_bits(&mut self, n: u32) -> Result<()> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Ok(())
    }

    /// Align to byte boundary (skip to next byte)
    pub fn byte_align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Read remaining bits in current byte (for RBSP trailing bits)
    pub fn read_rbsp_trailing_bits(&mut self) -> Result<()> {
        // Read rbsp_stop_one_bit (should be 1)
        let stop_bit = self.read_bit()?;
        if stop_bit != 1 {
            return Err(Error::codec("RBSP trailing: stop bit is not 1"));
        }

        // Skip remaining bits to byte boundary (should all be 0)
        while self.bit_pos != 0 {
            let zero_bit = self.read_bit()?;
            if zero_bit != 0 {
                return Err(Error::codec("RBSP trailing: alignment bits are not 0"));
            }
        }

        Ok(())
    }

    /// Check if we're at the end of RBSP data
    ///
    /// Returns true if only RBSP trailing bits remain
    pub fn more_rbsp_data(&self) -> bool {
        if self.byte_pos >= self.data.len() {
            return false;
        }

        // Simple check: if we have more than 1 byte left, there's more data
        if self.byte_pos < self.data.len() - 1 {
            return true;
        }

        // Check if current byte has more than just trailing bits
        // This is a simplified check; proper implementation would verify stop bit
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bit() {
        let data = vec![0b10110010];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
    }

    #[test]
    fn test_read_bits() {
        let data = vec![0b10110010, 0b11010101];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(8).unwrap(), 0b00101101);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0101);
    }

    #[test]
    fn test_read_bool() {
        let data = vec![0b10000000];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_bool().unwrap(), true);
        assert_eq!(reader.read_bool().unwrap(), false);
    }

    #[test]
    fn test_read_ue_basic() {
        // Test cases: bit pattern → value
        // 1 → 0
        // 010 → 1
        // 011 → 2
        // 00100 → 3
        // 00101 → 4

        // Value 0: "1"
        let data = vec![0b10000000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 0);

        // Value 1: "010"
        let data = vec![0b01000000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 1);

        // Value 2: "011"
        let data = vec![0b01100000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 2);

        // Value 3: "00100"
        let data = vec![0b00100000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 3);

        // Value 4: "00101"
        let data = vec![0b00101000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 4);
    }

    #[test]
    fn test_read_ue_larger() {
        // Value 7: "0001000" (3 leading zeros + 1 + 3 bits = 111)
        // 2^3 - 1 + 0 = 7
        let data = vec![0b00010000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 7);

        // Value 10: "0001011" (3 leading zeros + 1 + 3 bits = 011)
        // 2^3 - 1 + 3 = 10
        let data = vec![0b00010110];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 10);
    }

    #[test]
    fn test_read_se() {
        // se(v) mapping: ue(v) → se(v)
        // 0 → 0
        // 1 → 1
        // 2 → -1
        // 3 → 2
        // 4 → -2

        // Value 0: "1" → 0
        let data = vec![0b10000000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), 0);

        // Value 1: "010" → 1
        let data = vec![0b01000000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), 1);

        // Value 2: "011" → -1
        let data = vec![0b01100000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), -1);

        // Value 3: "00100" → 2
        let data = vec![0b00100000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), 2);

        // Value 4: "00101" → -2
        let data = vec![0b00101000];
        let mut reader = BitstreamReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), -2);
    }

    #[test]
    fn test_byte_align() {
        let data = vec![0b10110010, 0b11010101];
        let mut reader = BitstreamReader::new(&data);

        reader.read_bits(3).unwrap(); // Read 3 bits
        assert_eq!(reader.bit_pos, 3);

        reader.byte_align();
        assert_eq!(reader.bit_pos, 0);
        assert_eq!(reader.byte_pos, 1);
    }

    #[test]
    fn test_position() {
        let data = vec![0b10110010, 0b11010101];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.position(), 0);
        reader.read_bit().unwrap();
        assert_eq!(reader.position(), 1);
        reader.read_bits(7).unwrap();
        assert_eq!(reader.position(), 8);
        reader.read_bits(4).unwrap();
        assert_eq!(reader.position(), 12);
    }
}
