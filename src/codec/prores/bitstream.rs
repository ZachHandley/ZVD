//! ProRes Bitstream Reading and Writing
//!
//! Utilities for reading and writing ProRes bitstreams with bit-level precision.

use crate::error::{Error, Result};

/// Bitstream reader for ProRes
pub struct ProResBitstreamReader<'a> {
    /// Input data
    data: &'a [u8],
    /// Current byte position
    pos: usize,
    /// Current bit position within byte (0-7)
    bit_pos: u8,
}

impl<'a> ProResBitstreamReader<'a> {
    /// Create new bitstream reader
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit
    pub fn read_bit(&mut self) -> Result<bool> {
        if self.pos >= self.data.len() {
            return Err(Error::InvalidData("End of bitstream".to_string()));
        }

        let byte = self.data[self.pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.pos += 1;
        }

        Ok(bit != 0)
    }

    /// Read multiple bits as u32
    pub fn read_bits(&mut self, num_bits: u8) -> Result<u32> {
        if num_bits > 32 {
            return Err(Error::InvalidData("Cannot read more than 32 bits".to_string()));
        }

        let mut value = 0u32;
        for _ in 0..num_bits {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Ok(value)
    }

    /// Read a byte-aligned value
    pub fn read_u8(&mut self) -> Result<u8> {
        self.align();
        if self.pos >= self.data.len() {
            return Err(Error::InvalidData("End of bitstream".to_string()));
        }
        let value = self.data[self.pos];
        self.pos += 1;
        Ok(value)
    }

    /// Read u16 big-endian
    pub fn read_u16(&mut self) -> Result<u16> {
        let b1 = self.read_u8()? as u16;
        let b2 = self.read_u8()? as u16;
        Ok((b1 << 8) | b2)
    }

    /// Align to byte boundary
    pub fn align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.pos += 1;
        }
    }

    /// Get current byte position
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Check if more data available
    pub fn has_more(&self) -> bool {
        self.pos < self.data.len()
    }
}

/// Bitstream writer for ProRes
pub struct ProResBitstreamWriter {
    /// Output buffer
    data: Vec<u8>,
    /// Current byte being written
    current_byte: u8,
    /// Current bit position within byte (0-7)
    bit_pos: u8,
}

impl ProResBitstreamWriter {
    /// Create new bitstream writer
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    /// Write a single bit
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Write multiple bits from u32
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1;
            self.write_bit(bit != 0);
        }
    }

    /// Write a byte-aligned value
    pub fn write_u8(&mut self, value: u8) {
        self.align();
        self.data.push(value);
    }

    /// Write u16 big-endian
    pub fn write_u16(&mut self, value: u16) {
        self.write_u8((value >> 8) as u8);
        self.write_u8(value as u8);
    }

    /// Align to byte boundary
    pub fn align(&mut self) {
        if self.bit_pos != 0 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    /// Get the written data
    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.data
    }

    /// Get current size in bytes
    pub fn len(&self) -> usize {
        self.data.len() + if self.bit_pos > 0 { 1 } else { 0 }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.bit_pos == 0
    }
}

impl Default for ProResBitstreamWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstream_reader_single_bit() {
        let data = vec![0b10101010];
        let mut reader = ProResBitstreamReader::new(&data);

        assert_eq!(reader.read_bit().unwrap(), true);
        assert_eq!(reader.read_bit().unwrap(), false);
        assert_eq!(reader.read_bit().unwrap(), true);
        assert_eq!(reader.read_bit().unwrap(), false);
    }

    #[test]
    fn test_bitstream_reader_multiple_bits() {
        let data = vec![0b11010110, 0b10101100];
        let mut reader = ProResBitstreamReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1101);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01101010);
    }

    #[test]
    fn test_bitstream_reader_u8() {
        let data = vec![0x12, 0x34, 0x56];
        let mut reader = ProResBitstreamReader::new(&data);

        assert_eq!(reader.read_u8().unwrap(), 0x12);
        assert_eq!(reader.read_u8().unwrap(), 0x34);
        assert_eq!(reader.read_u8().unwrap(), 0x56);
    }

    #[test]
    fn test_bitstream_reader_u16() {
        let data = vec![0x12, 0x34];
        let mut reader = ProResBitstreamReader::new(&data);

        assert_eq!(reader.read_u16().unwrap(), 0x1234);
    }

    #[test]
    fn test_bitstream_writer_single_bit() {
        let mut writer = ProResBitstreamWriter::new();

        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);

        let data = writer.finish();
        assert_eq!(data, vec![0b10101010]);
    }

    #[test]
    fn test_bitstream_writer_multiple_bits() {
        let mut writer = ProResBitstreamWriter::new();

        writer.write_bits(0b1101, 4);
        writer.write_bits(0b01101010, 8);

        let data = writer.finish();
        assert_eq!(data, vec![0b11010110, 0b10100000]);
    }

    #[test]
    fn test_bitstream_writer_u8() {
        let mut writer = ProResBitstreamWriter::new();

        writer.write_u8(0x12);
        writer.write_u8(0x34);

        let data = writer.finish();
        assert_eq!(data, vec![0x12, 0x34]);
    }

    #[test]
    fn test_bitstream_writer_u16() {
        let mut writer = ProResBitstreamWriter::new();

        writer.write_u16(0x1234);

        let data = writer.finish();
        assert_eq!(data, vec![0x12, 0x34]);
    }

    #[test]
    fn test_bitstream_align() {
        let mut reader = ProResBitstreamReader::new(&[0b10101010, 0xFF]);

        reader.read_bit().unwrap(); // Read 1 bit
        reader.align(); // Align to next byte

        assert_eq!(reader.pos(), 1);
        assert_eq!(reader.read_u8().unwrap(), 0xFF);
    }
}
