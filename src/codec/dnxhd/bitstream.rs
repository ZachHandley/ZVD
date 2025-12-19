//! Bitstream reading and writing utilities for DNxHD encoding/decoding.
//!
//! This module provides BitReader for reading bits from encoded data and
//! BitWriter for writing bits during encoding. It also includes VLC
//! (Variable Length Coding) functions using a hybrid Rice/ExpGolomb scheme.
//!
//! # Bit Ordering
//!
//! Both BitReader and BitWriter use MSB-first (big-endian) bit ordering,
//! which is the standard for most video codecs including DNxHD.

use crate::error::{Error, Result};

/// Bitstream reader for decoding DNxHD data.
///
/// Reads bits from a byte slice in MSB-first order. Maintains an internal
/// position tracking the current bit offset within the data.
pub struct BitReader<'a> {
    /// The underlying byte data
    data: &'a [u8],
    /// Current bit position (0-based from start of data)
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new BitReader from a byte slice.
    ///
    /// # Arguments
    /// * `data` - The byte slice to read bits from
    ///
    /// # Example
    /// ```ignore
    /// let data = &[0b10110001, 0b01010101];
    /// let mut br = BitReader::new(data);
    /// assert_eq!(br.read_bit().unwrap(), true);  // First bit is 1
    /// ```
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, bit_pos: 0 }
    }

    /// Read a single bit from the stream.
    ///
    /// Returns `true` for 1, `false` for 0.
    ///
    /// # Errors
    /// Returns `Error::EndOfStream` if no bits remain.
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool> {
        if self.bit_pos >= self.data.len() * 8 {
            return Err(Error::EndOfStream);
        }

        let byte_idx = self.bit_pos / 8;
        let bit_idx = 7 - (self.bit_pos % 8); // MSB first
        let bit = (self.data[byte_idx] >> bit_idx) & 1;
        self.bit_pos += 1;
        Ok(bit != 0)
    }

    /// Read multiple bits from the stream (up to 32 bits).
    ///
    /// Bits are returned in MSB-first order packed into a u32.
    ///
    /// # Arguments
    /// * `n` - Number of bits to read (1-32)
    ///
    /// # Errors
    /// Returns `Error::EndOfStream` if insufficient bits remain.
    /// Returns `Error::InvalidInput` if n > 32 or n == 0.
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::invalid_input(
                "Cannot read more than 32 bits at once",
            ));
        }

        let remaining = self.remaining();
        if (n as usize) > remaining {
            return Err(Error::EndOfStream);
        }

        let mut result: u32 = 0;
        for _ in 0..n {
            result = (result << 1) | (self.read_bit()? as u32);
        }
        Ok(result)
    }

    /// Peek at the next n bits without advancing the position.
    ///
    /// This is useful for lookahead parsing where you need to check
    /// bits before deciding how to decode them.
    ///
    /// # Arguments
    /// * `n` - Number of bits to peek (1-32)
    ///
    /// # Errors
    /// Returns `Error::EndOfStream` if insufficient bits remain.
    #[inline]
    pub fn peek_bits(&self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::invalid_input(
                "Cannot peek more than 32 bits at once",
            ));
        }

        let remaining = self.remaining();
        if (n as usize) > remaining {
            return Err(Error::EndOfStream);
        }

        let mut result: u32 = 0;
        let mut pos = self.bit_pos;

        for _ in 0..n {
            let byte_idx = pos / 8;
            let bit_idx = 7 - (pos % 8);
            let bit = (self.data[byte_idx] >> bit_idx) & 1;
            result = (result << 1) | (bit as u32);
            pos += 1;
        }

        Ok(result)
    }

    /// Skip n bits without reading them.
    ///
    /// If n exceeds remaining bits, position is set to end of stream.
    ///
    /// # Arguments
    /// * `n` - Number of bits to skip
    #[inline]
    pub fn skip_bits(&mut self, n: u32) {
        let total_bits = self.data.len() * 8;
        self.bit_pos = (self.bit_pos + n as usize).min(total_bits);
    }

    /// Get the current bit position (0-based).
    #[inline]
    pub fn position(&self) -> usize {
        self.bit_pos
    }

    /// Get the number of bits remaining to read.
    #[inline]
    pub fn remaining(&self) -> usize {
        let total_bits = self.data.len() * 8;
        total_bits.saturating_sub(self.bit_pos)
    }

    /// Check if there are any bits remaining.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Align to the next byte boundary.
    ///
    /// Skips bits until the position is at a byte boundary (multiple of 8).
    #[inline]
    pub fn align_to_byte(&mut self) {
        let remainder = self.bit_pos % 8;
        if remainder != 0 {
            self.bit_pos += 8 - remainder;
        }
    }

    /// Get the current byte position (rounded down).
    #[inline]
    pub fn byte_position(&self) -> usize {
        self.bit_pos / 8
    }

    /// Read an unsigned ExpGolomb code (ue(v) in H.264 terminology).
    ///
    /// Format: leading zeros, 1-bit terminator, suffix bits
    /// The value encoded is: (1 << num_zeros) - 1 + suffix
    pub fn read_exp_golomb(&mut self) -> Result<u32> {
        let mut leading_zeros: u32 = 0;

        // Count leading zeros
        while !self.read_bit()? {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(Error::codec(
                    "Invalid ExpGolomb code: too many leading zeros",
                ));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        // Read suffix bits
        let suffix = self.read_bits(leading_zeros as u8)?;
        Ok((1 << leading_zeros) - 1 + suffix)
    }

    /// Read a signed ExpGolomb code (se(v) in H.264 terminology).
    ///
    /// Maps unsigned values to signed: 0->0, 1->1, 2->-1, 3->2, 4->-2, ...
    pub fn read_signed_exp_golomb(&mut self) -> Result<i32> {
        let unsigned = self.read_exp_golomb()?;
        let value = ((unsigned + 1) >> 1) as i32;
        if unsigned & 1 == 0 {
            Ok(-value)
        } else {
            Ok(value)
        }
    }
}

/// Bitstream writer for encoding DNxHD data.
///
/// Writes bits to an internal buffer in MSB-first order. The buffer
/// grows automatically as needed.
pub struct BitWriter {
    /// Output byte buffer
    data: Vec<u8>,
    /// Current byte being assembled
    current_byte: u8,
    /// Number of bits written to current_byte (0-7)
    bit_count: usize,
}

impl BitWriter {
    /// Create a new BitWriter with default capacity.
    pub fn new() -> Self {
        BitWriter {
            data: Vec::with_capacity(4096),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Create a new BitWriter with specified initial capacity.
    ///
    /// # Arguments
    /// * `capacity` - Initial capacity in bytes
    pub fn with_capacity(capacity: usize) -> Self {
        BitWriter {
            data: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Write a single bit to the stream.
    ///
    /// # Arguments
    /// * `bit` - The bit value (true for 1, false for 0)
    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.current_byte = (self.current_byte << 1) | (bit as u8);
        self.bit_count += 1;

        if self.bit_count == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Write multiple bits to the stream (up to 32 bits).
    ///
    /// Bits are written in MSB-first order from the value.
    ///
    /// # Arguments
    /// * `value` - The bits to write (only lower n bits are used)
    /// * `n` - Number of bits to write (0-32)
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32, "Cannot write more than 32 bits at once");

        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1 != 0);
        }
    }

    /// Flush any remaining bits, padding with zeros.
    ///
    /// This should be called before retrieving the final bytes to ensure
    /// all bits are properly aligned to a byte boundary.
    pub fn flush(&mut self) {
        if self.bit_count > 0 {
            self.current_byte <<= 8 - self.bit_count;
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Consume the writer and return the written bytes.
    ///
    /// This flushes any remaining partial byte first.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.data
    }

    /// Get the current bit position (total bits written).
    #[inline]
    pub fn bit_position(&self) -> usize {
        self.data.len() * 8 + self.bit_count
    }

    /// Get a reference to the written data (excluding partial byte).
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Align to byte boundary by writing zero bits.
    pub fn align_to_byte(&mut self) {
        if self.bit_count > 0 {
            let padding = 8 - self.bit_count;
            for _ in 0..padding {
                self.write_bit(false);
            }
        }
    }

    /// Write an unsigned ExpGolomb code.
    ///
    /// Format: (value+1).leading_zeros() zeros, 1, suffix bits
    pub fn write_exp_golomb(&mut self, value: u32) {
        if value == 0 {
            self.write_bit(true);
            return;
        }

        // Find number of bits needed for (value + 1)
        let val_plus_one = value + 1;
        let num_bits = 32 - val_plus_one.leading_zeros();
        let leading_zeros = num_bits - 1;

        // Write leading zeros
        for _ in 0..leading_zeros {
            self.write_bit(false);
        }

        // Write the value + 1 (which starts with a 1 bit)
        self.write_bits(val_plus_one, num_bits as u8);
    }

    /// Write a signed ExpGolomb code.
    ///
    /// Maps signed to unsigned: 0->0, 1->1, -1->2, 2->3, -2->4, ...
    pub fn write_signed_exp_golomb(&mut self, value: i32) {
        let unsigned = if value == 0 {
            0
        } else if value > 0 {
            (value * 2 - 1) as u32
        } else {
            (-value * 2) as u32
        };
        self.write_exp_golomb(unsigned);
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// VLC (Variable Length Coding) Functions
// ============================================================================

/// Convert a signed value to unsigned for VLC encoding.
///
/// Uses interleaved sign encoding: 0->0, -1->1, 1->2, -2->3, 2->4, ...
/// This ensures small magnitude values (positive or negative) have small codes.
#[inline]
pub fn to_unsigned(v: i32) -> u32 {
    if v == 0 {
        0
    } else if v < 0 {
        // Negative: -1->1, -2->3, -3->5 => (-v)*2 - 1
        ((-v) * 2 - 1) as u32
    } else {
        // Positive: 1->2, 2->4, 3->6 => v*2
        (v * 2) as u32
    }
}

/// Convert an unsigned VLC value back to signed.
///
/// Inverse of `to_unsigned`: 0->0, 1->-1, 2->1, 3->-2, 4->2, ...
#[inline]
pub fn to_signed(v: u32) -> i32 {
    // Even values (except 0) are positive: v/2
    // Odd values are negative: -(v+1)/2
    ((v >> 1) as i32) ^ -((v as i32) & 1)
}

/// Decode a VLC codeword using the hybrid Rice/ExpGolomb scheme.
///
/// # Codebook Format
/// The codebook byte encodes three parameters:
/// - bits 0-1: switch_bits - threshold for switching between Rice and ExpGolomb
/// - bits 2-4: exp_order - exponential order for ExpGolomb coding
/// - bits 5-7: rice_order - Rice parameter (number of suffix bits for small values)
///
/// # Decoding Procedure
/// 1. Count leading zeros (q)
/// 2. Read the terminating 1 bit
/// 3. If q <= switch_bits: Rice mode - read rice_order suffix bits
/// 4. If q > switch_bits: ExpGolomb mode - read variable suffix bits
///
/// # Arguments
/// * `br` - BitReader positioned at the start of the codeword
/// * `codebook` - The codebook byte encoding the VLC parameters
///
/// # Returns
/// The decoded signed coefficient value.
pub fn decode_vlc_codeword(br: &mut BitReader, codebook: u8) -> Result<i32> {
    let switch_bits = (codebook & 0x03) as u32;
    let rice_order = ((codebook >> 5) & 0x07) as u32;
    let exp_order = ((codebook >> 2) & 0x07) as u32;

    // Count leading zeros (quotient)
    let mut q: u32 = 0;
    while !br.read_bit()? {
        q += 1;
        if q > 32 {
            return Err(Error::codec("Invalid VLC codeword: too many leading zeros"));
        }
    }

    let value = if q <= switch_bits {
        // Rice coding mode
        // Read rice_order bits for the remainder
        let r = if rice_order > 0 {
            br.read_bits(rice_order as u8)?
        } else {
            0
        };
        (q << rice_order) | r
    } else {
        // ExpGolomb coding mode
        // Calculate number of suffix bits
        let suffix_bits = exp_order.saturating_sub(switch_bits) + ((q - 1) << 1);
        let suffix = if suffix_bits > 0 {
            br.read_bits(suffix_bits.min(32) as u8)?
        } else {
            0
        };

        // Calculate the value
        // Base value from Rice portion
        let base = (switch_bits + 1) << rice_order;
        // ExpGolomb portion contribution
        let exp_contribution = if suffix_bits > 0 {
            suffix + (1 << (suffix_bits - exp_order.saturating_sub(switch_bits)))
        } else {
            1
        };
        base + exp_contribution - (1 << exp_order)
    };

    Ok(to_signed(value))
}

/// Encode a VLC codeword using the hybrid Rice/ExpGolomb scheme.
///
/// This is the inverse of `decode_vlc_codeword`.
///
/// # Codebook Format
/// The codebook byte encodes three parameters:
/// - bits 0-1: switch_bits - threshold for switching between Rice and ExpGolomb
/// - bits 2-4: exp_order - exponential order for ExpGolomb coding
/// - bits 5-7: rice_order - Rice parameter (number of suffix bits for small values)
///
/// # Arguments
/// * `bw` - BitWriter to write the encoded codeword to
/// * `value` - The signed coefficient value to encode
/// * `codebook` - The codebook byte encoding the VLC parameters
pub fn encode_vlc_codeword(bw: &mut BitWriter, value: i32, codebook: u8) {
    let switch_bits = (codebook & 0x03) as u32;
    let rice_order = ((codebook >> 5) & 0x07) as u32;
    let exp_order = ((codebook >> 2) & 0x07) as u32;

    // Convert signed to unsigned
    let unsigned_value = to_unsigned(value);

    // Calculate Rice threshold: values below this use Rice coding
    let rice_threshold = (switch_bits + 1) << rice_order;

    if unsigned_value < rice_threshold {
        // Rice coding mode
        let q = unsigned_value >> rice_order;
        let r = unsigned_value & ((1 << rice_order) - 1);

        // Write q zeros followed by 1
        for _ in 0..q {
            bw.write_bit(false);
        }
        bw.write_bit(true);

        // Write rice_order bits of remainder
        if rice_order > 0 {
            bw.write_bits(r, rice_order as u8);
        }
    } else {
        // ExpGolomb coding mode
        // Adjust value for ExpGolomb range
        let adjusted = unsigned_value - rice_threshold + (1 << exp_order);

        // Find the number of bits needed
        let mut bits = exp_order;
        while (1u64 << bits) <= adjusted as u64 {
            bits += 1;
        }

        // Calculate q from the number of bits
        // bits = exp_order - switch_bits + 2*(q-1-switch_bits) + 2
        //      = exp_order - switch_bits + 2*q - 2*switch_bits
        // Solving for q: q = (bits - exp_order + switch_bits) / 2 + switch_bits + 1
        let q = if bits >= exp_order {
            let extra_bits = bits - exp_order;
            switch_bits + 1 + extra_bits.div_ceil(2)
        } else {
            switch_bits + 1
        };

        // Write q zeros followed by 1
        for _ in 0..q {
            bw.write_bit(false);
        }
        bw.write_bit(true);

        // Calculate suffix bits and write suffix
        let suffix_bits =
            exp_order.saturating_sub(switch_bits) + ((q - 1).saturating_sub(switch_bits) << 1);
        if suffix_bits > 0 {
            // The suffix encodes the value within the ExpGolomb range
            let suffix_value = adjusted & ((1 << suffix_bits) - 1);
            bw.write_bits(suffix_value, suffix_bits.min(32) as u8);
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BitReader Tests
    // ========================================================================

    #[test]
    fn test_bitreader_read_single_bits() {
        let data = [0b10110001];
        let mut br = BitReader::new(&data);

        assert_eq!(br.read_bit().unwrap(), true); // 1
        assert_eq!(br.read_bit().unwrap(), false); // 0
        assert_eq!(br.read_bit().unwrap(), true); // 1
        assert_eq!(br.read_bit().unwrap(), true); // 1
        assert_eq!(br.read_bit().unwrap(), false); // 0
        assert_eq!(br.read_bit().unwrap(), false); // 0
        assert_eq!(br.read_bit().unwrap(), false); // 0
        assert_eq!(br.read_bit().unwrap(), true); // 1
    }

    #[test]
    fn test_bitreader_read_multiple_bits() {
        let data = [0b10110001, 0b11110000];
        let mut br = BitReader::new(&data);

        assert_eq!(br.read_bits(4).unwrap(), 0b1011);
        assert_eq!(br.read_bits(4).unwrap(), 0b0001);
        assert_eq!(br.read_bits(8).unwrap(), 0b11110000);
    }

    #[test]
    fn test_bitreader_read_across_bytes() {
        let data = [0b10110001, 0b11110000];
        let mut br = BitReader::new(&data);

        assert_eq!(br.read_bits(6).unwrap(), 0b101100);
        assert_eq!(br.read_bits(6).unwrap(), 0b011111);
        assert_eq!(br.read_bits(4).unwrap(), 0b0000);
    }

    #[test]
    fn test_bitreader_peek_bits() {
        let data = [0b10110001];
        let br = BitReader::new(&data);

        assert_eq!(br.peek_bits(4).unwrap(), 0b1011);
        assert_eq!(br.peek_bits(8).unwrap(), 0b10110001);
        // Position should not have changed
        assert_eq!(br.position(), 0);
    }

    #[test]
    fn test_bitreader_skip_bits() {
        let data = [0b10110001, 0b11110000];
        let mut br = BitReader::new(&data);

        br.skip_bits(4);
        assert_eq!(br.position(), 4);
        assert_eq!(br.read_bits(4).unwrap(), 0b0001);

        br.skip_bits(4);
        assert_eq!(br.read_bits(4).unwrap(), 0b0000);
    }

    #[test]
    fn test_bitreader_remaining() {
        let data = [0xFF, 0xFF];
        let mut br = BitReader::new(&data);

        assert_eq!(br.remaining(), 16);
        br.read_bits(5).unwrap();
        assert_eq!(br.remaining(), 11);
        br.skip_bits(11);
        assert_eq!(br.remaining(), 0);
        assert!(br.is_empty());
    }

    #[test]
    fn test_bitreader_end_of_stream() {
        let data = [0xFF];
        let mut br = BitReader::new(&data);

        br.read_bits(8).unwrap();
        assert!(br.read_bit().is_err());
        assert!(br.read_bits(1).is_err());
    }

    #[test]
    fn test_bitreader_align_to_byte() {
        let data = [0xFF, 0x00];
        let mut br = BitReader::new(&data);

        br.read_bits(3).unwrap();
        assert_eq!(br.position(), 3);
        br.align_to_byte();
        assert_eq!(br.position(), 8);
        assert_eq!(br.read_bits(8).unwrap(), 0x00);
    }

    #[test]
    fn test_bitreader_exp_golomb() {
        // Test unsigned ExpGolomb decoding
        // 0 -> 1 (single 1 bit)
        let data = [0b10000000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_exp_golomb().unwrap(), 0);

        // 1 -> 010 (one zero, 1, one suffix bit)
        let data = [0b01000000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_exp_golomb().unwrap(), 1);

        // 2 -> 011
        let data = [0b01100000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_exp_golomb().unwrap(), 2);

        // 3 -> 00100
        let data = [0b00100000];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_exp_golomb().unwrap(), 3);
    }

    // ========================================================================
    // BitWriter Tests
    // ========================================================================

    #[test]
    fn test_bitwriter_write_single_bits() {
        let mut bw = BitWriter::new();
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(true);
        bw.write_bit(true);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(false);
        bw.write_bit(true);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 0b10110001);
    }

    #[test]
    fn test_bitwriter_write_multiple_bits() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b11010, 5);
        bw.write_bits(0b110, 3);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 0b11010110);
    }

    #[test]
    fn test_bitwriter_partial_byte_flush() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 1);
        // Should be padded with zeros: 10100000
        assert_eq!(data[0], 0b10100000);
    }

    #[test]
    fn test_bitwriter_multi_byte() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b10110001, 8);
        bw.write_bits(0b11110000, 8);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], 0b10110001);
        assert_eq!(data[1], 0b11110000);
    }

    #[test]
    fn test_bitwriter_cross_byte_boundary() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101100, 6);
        bw.write_bits(0b011111, 6);
        bw.write_bits(0b0000, 4);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], 0b10110001);
        assert_eq!(data[1], 0b11110000);
    }

    #[test]
    fn test_bitwriter_bit_position() {
        let mut bw = BitWriter::new();
        assert_eq!(bw.bit_position(), 0);
        bw.write_bits(0xFF, 8);
        assert_eq!(bw.bit_position(), 8);
        bw.write_bits(0b111, 3);
        assert_eq!(bw.bit_position(), 11);
    }

    #[test]
    fn test_bitwriter_exp_golomb() {
        // Test that ExpGolomb encoding matches expected bit patterns
        let mut bw = BitWriter::new();
        bw.write_exp_golomb(0); // Should produce: 1
        bw.write_exp_golomb(1); // Should produce: 010
        bw.write_exp_golomb(2); // Should produce: 011
        bw.write_exp_golomb(3); // Should produce: 00100
        bw.flush();

        let data = bw.into_bytes();
        // 1 010 011 00100 -> 10100110 0100xxxx
        assert_eq!(data[0], 0b10100110);
    }

    // ========================================================================
    // Round-trip Tests
    // ========================================================================

    #[test]
    fn test_roundtrip_bits() {
        let test_values = [0u32, 1, 7, 15, 127, 255, 1023, 65535];

        for &value in &test_values {
            let bits_needed = if value == 0 {
                1
            } else {
                32 - value.leading_zeros()
            };

            let mut bw = BitWriter::new();
            bw.write_bits(value, bits_needed as u8);
            let data = bw.into_bytes();

            let mut br = BitReader::new(&data);
            let read_value = br.read_bits(bits_needed as u8).unwrap();
            assert_eq!(read_value, value, "Roundtrip failed for value {}", value);
        }
    }

    #[test]
    fn test_roundtrip_exp_golomb() {
        for value in 0u32..100 {
            let mut bw = BitWriter::new();
            bw.write_exp_golomb(value);
            let data = bw.into_bytes();

            let mut br = BitReader::new(&data);
            let decoded = br.read_exp_golomb().unwrap();
            assert_eq!(decoded, value, "ExpGolomb roundtrip failed for {}", value);
        }
    }

    #[test]
    fn test_roundtrip_signed_exp_golomb() {
        for value in -50i32..=50 {
            let mut bw = BitWriter::new();
            bw.write_signed_exp_golomb(value);
            let data = bw.into_bytes();

            let mut br = BitReader::new(&data);
            let decoded = br.read_signed_exp_golomb().unwrap();
            assert_eq!(
                decoded, value,
                "Signed ExpGolomb roundtrip failed for {}",
                value
            );
        }
    }

    // ========================================================================
    // Signed Conversion Tests
    // ========================================================================

    #[test]
    fn test_to_unsigned_to_signed_inverse() {
        for i in -100i32..=100 {
            let unsigned = to_unsigned(i);
            let back = to_signed(unsigned);
            assert_eq!(back, i, "to_unsigned/to_signed roundtrip failed for {}", i);
        }
    }

    #[test]
    fn test_to_signed_values() {
        assert_eq!(to_signed(0), 0);
        assert_eq!(to_signed(1), -1);
        assert_eq!(to_signed(2), 1);
        assert_eq!(to_signed(3), -2);
        assert_eq!(to_signed(4), 2);
        assert_eq!(to_signed(5), -3);
        assert_eq!(to_signed(6), 3);
    }

    #[test]
    fn test_to_unsigned_values() {
        assert_eq!(to_unsigned(0), 0);
        assert_eq!(to_unsigned(-1), 1);
        assert_eq!(to_unsigned(1), 2);
        assert_eq!(to_unsigned(-2), 3);
        assert_eq!(to_unsigned(2), 4);
        assert_eq!(to_unsigned(-3), 5);
        assert_eq!(to_unsigned(3), 6);
    }

    // ========================================================================
    // VLC Encode/Decode Tests
    // ========================================================================

    #[test]
    fn test_vlc_simple_rice_coding() {
        // Simple codebook: rice_order=2, exp_order=2, switch_bits=1
        // bits 5-7: rice_order=2 -> 010xxxxx = 0x40
        // bits 2-4: exp_order=2  -> xxx010xx = 0x08
        // bits 0-1: switch_bits=1 -> xxxxxx01 = 0x01
        let codebook: u8 = 0x40 | 0x08 | 0x01; // 0x49

        // Test encoding value 0
        let mut bw = BitWriter::new();
        encode_vlc_codeword(&mut bw, 0, codebook);
        let data = bw.into_bytes();

        // Value 0 with rice_order=2, switch_bits=1:
        // q=0 (since 0 >> 2 = 0), r=0
        // Output: 1 (terminator) + 00 (2 suffix bits) = 100
        assert!(!data.is_empty());
    }

    #[test]
    fn test_vlc_small_values() {
        // Codebook: rice_order=0, exp_order=1, switch_bits=0
        let codebook: u8 = 0x04; // 0b00000100

        for value in 0i32..=10 {
            let mut bw = BitWriter::new();
            encode_vlc_codeword(&mut bw, value, codebook);
            // Just verify it doesn't panic
            let _data = bw.into_bytes();
        }
    }

    #[test]
    fn test_vlc_negative_values() {
        let codebook: u8 = 0x04;

        // Test negative values
        for value in [-1i32, -5, -10, -50] {
            let mut bw = BitWriter::new();
            encode_vlc_codeword(&mut bw, value, codebook);
            let data = bw.into_bytes();
            assert!(!data.is_empty());
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_data() {
        let data: [u8; 0] = [];
        let br = BitReader::new(&data);
        assert!(br.is_empty());
        assert_eq!(br.remaining(), 0);
    }

    #[test]
    fn test_read_zero_bits() {
        let data = [0xFF];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(0).unwrap(), 0);
        assert_eq!(br.position(), 0);
    }

    #[test]
    fn test_write_zero_bits() {
        let mut bw = BitWriter::new();
        bw.write_bits(0xFFFFFFFF, 0);
        assert_eq!(bw.bit_position(), 0);
    }

    #[test]
    fn test_large_bit_values() {
        let mut bw = BitWriter::new();
        bw.write_bits(0xFFFFFFFF, 32);
        let data = bw.into_bytes();

        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(32).unwrap(), 0xFFFFFFFF);
    }

    #[test]
    fn test_skip_past_end() {
        let data = [0xFF];
        let mut br = BitReader::new(&data);
        br.skip_bits(100);
        assert_eq!(br.remaining(), 0);
        assert!(br.is_empty());
    }
}
