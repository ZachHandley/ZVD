//! DNxHD bitstream reader and writer
//!
//! Handles bit-level I/O for DNxHD VLC decoding and encoding

use crate::error::{Error, Result};

/// Bitstream reader for DNxHD
pub struct DnxhdBitstreamReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> DnxhdBitstreamReader<'a> {
    /// Create a new bitstream reader
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Check if more data is available
    pub fn has_more(&self) -> bool {
        self.byte_pos < self.data.len()
    }

    /// Get current bit position
    pub fn tell(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Read a single bit
    pub fn read_bit(&mut self) -> Result<u8> {
        if self.byte_pos >= self.data.len() {
            return Err(Error::invalid_input("Bitstream exhausted"));
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

    /// Read n bits (up to 32)
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n > 32 {
            return Err(Error::invalid_input("Cannot read more than 32 bits"));
        }

        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }

        Ok(value)
    }

    /// Read a byte-aligned u8
    pub fn read_u8(&mut self) -> Result<u8> {
        self.align();
        if self.byte_pos >= self.data.len() {
            return Err(Error::invalid_input("Bitstream exhausted"));
        }
        let value = self.data[self.byte_pos];
        self.byte_pos += 1;
        Ok(value)
    }

    /// Read a byte-aligned u16 (big-endian)
    pub fn read_u16(&mut self) -> Result<u16> {
        self.align();
        if self.byte_pos + 1 >= self.data.len() {
            return Err(Error::invalid_input("Bitstream exhausted"));
        }
        let value = u16::from_be_bytes([
            self.data[self.byte_pos],
            self.data[self.byte_pos + 1],
        ]);
        self.byte_pos += 2;
        Ok(value)
    }

    /// Read a byte-aligned u32 (big-endian)
    pub fn read_u32(&mut self) -> Result<u32> {
        self.align();
        if self.byte_pos + 3 >= self.data.len() {
            return Err(Error::invalid_input("Bitstream exhausted"));
        }
        let value = u32::from_be_bytes([
            self.data[self.byte_pos],
            self.data[self.byte_pos + 1],
            self.data[self.byte_pos + 2],
            self.data[self.byte_pos + 3],
        ]);
        self.byte_pos += 4;
        Ok(value)
    }

    /// Align to next byte boundary
    pub fn align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Skip n bytes
    pub fn skip_bytes(&mut self, n: usize) {
        self.align();
        self.byte_pos += n;
    }
}

/// Bitstream writer for DNxHD
pub struct DnxhdBitstreamWriter {
    data: Vec<u8>,
    bit_pos: u8,
}

impl DnxhdBitstreamWriter {
    /// Create a new bitstream writer
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_pos: 0,
        }
    }

    /// Write a single bit
    pub fn write_bit(&mut self, bit: u8) {
        if self.bit_pos == 0 {
            self.data.push(0);
        }

        if bit != 0 {
            let len = self.data.len();
            self.data[len - 1] |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
    }

    /// Write n bits from value
    pub fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// Write a byte-aligned u8
    pub fn write_u8(&mut self, value: u8) {
        self.align();
        self.data.push(value);
    }

    /// Write a byte-aligned u16 (big-endian)
    pub fn write_u16(&mut self, value: u16) {
        self.align();
        self.data.extend_from_slice(&value.to_be_bytes());
    }

    /// Write a byte-aligned u32 (big-endian)
    pub fn write_u32(&mut self, value: u32) {
        self.align();
        self.data.extend_from_slice(&value.to_be_bytes());
    }

    /// Align to next byte boundary (pad with zeros)
    pub fn align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
        }
    }

    /// Finish writing and return the data
    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.data
    }

    /// Get current length in bytes (including partial byte)
    pub fn len(&self) -> usize {
        if self.bit_pos > 0 {
            self.data.len() + 1
        } else {
            self.data.len()
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.bit_pos == 0
    }
}

impl Default for DnxhdBitstreamWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstream_read_write_bits() {
        let mut writer = DnxhdBitstreamWriter::new();
        writer.write_bits(0b1010, 4);
        writer.write_bits(0b1100, 4);

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1100);
    }

    #[test]
    fn test_bitstream_read_write_u8() {
        let mut writer = DnxhdBitstreamWriter::new();
        writer.write_u8(0x42);
        writer.write_u8(0xFF);

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        assert_eq!(reader.read_u8().unwrap(), 0x42);
        assert_eq!(reader.read_u8().unwrap(), 0xFF);
    }

    #[test]
    fn test_bitstream_read_write_u16() {
        let mut writer = DnxhdBitstreamWriter::new();
        writer.write_u16(0x1234);

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        assert_eq!(reader.read_u16().unwrap(), 0x1234);
    }

    #[test]
    fn test_bitstream_alignment() {
        let mut writer = DnxhdBitstreamWriter::new();
        writer.write_bits(0b101, 3);
        writer.align();
        writer.write_u8(0x42);

        let data = writer.finish();
        assert_eq!(data.len(), 2);
    }
}
