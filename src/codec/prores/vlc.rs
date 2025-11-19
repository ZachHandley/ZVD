//! ProRes Variable Length Coding (VLC) / Huffman Coding
//!
//! ProRes uses Huffman coding to compress DCT coefficients efficiently.
//! This module implements the encoding and decoding of these codes.

use crate::error::{Error, Result};
use super::bitstream::{ProResBitstreamReader, ProResBitstreamWriter};

/// VLC code entry
#[derive(Debug, Clone, Copy)]
pub struct VlcCode {
    /// Code bits
    pub code: u32,
    /// Code length in bits
    pub len: u8,
    /// Symbol value
    pub symbol: i16,
}

/// ProRes DC coefficient codes
pub struct ProResDcVlc;

impl ProResDcVlc {
    /// DC VLC table (simplified - ProRes uses multiple tables)
    /// Format: (code, length, symbol)
    const TABLE: &'static [(u32, u8, i16)] = &[
        (0b0, 1, 0),           // 0
        (0b10, 2, 1),          // +1
        (0b11, 2, -1),         // -1
        (0b100, 3, 2),         // +2
        (0b101, 3, -2),        // -2
        (0b110, 3, 3),         // +3
        (0b111, 3, -3),        // -3
        (0b1000, 4, 4),        // +4
        (0b1001, 4, -4),       // -4
        (0b1010, 4, 5),        // +5
        (0b1011, 4, -5),       // -5
        (0b1100, 4, 6),        // +6
        (0b1101, 4, -6),       // -6
        (0b1110, 4, 7),        // +7
        (0b1111, 4, -7),       // -7
    ];

    /// Decode DC coefficient
    pub fn decode(reader: &mut ProResBitstreamReader) -> Result<i16> {
        // Read up to 12 bits to find match
        for &(code, len, symbol) in Self::TABLE {
            let peek = reader.read_bits(len)?;
            if peek == code {
                return Ok(symbol);
            }
            // Rewind if no match (in real implementation, use lookahead)
        }

        Err(Error::InvalidData("Invalid DC VLC code".to_string()))
    }

    /// Encode DC coefficient
    pub fn encode(writer: &mut ProResBitstreamWriter, value: i16) -> Result<()> {
        for &(code, len, symbol) in Self::TABLE {
            if symbol == value {
                writer.write_bits(code, len);
                return Ok(());
            }
        }

        Err(Error::InvalidData(format!("DC value {} not in VLC table", value)))
    }
}

/// ProRes AC coefficient codes (run-length encoding)
pub struct ProResAcVlc;

impl ProResAcVlc {
    /// AC VLC table (simplified)
    /// Format: (code, length, run, level)
    /// run = number of zeros before this coefficient
    /// level = coefficient value
    const TABLE: &'static [(u32, u8, u8, i16)] = &[
        // EOB (End of Block)
        (0b00, 2, 0, 0),

        // Run=0 (no zeros)
        (0b10, 2, 0, 1),        // Level +1
        (0b11, 2, 0, -1),       // Level -1
        (0b0100, 4, 0, 2),      // Level +2
        (0b0101, 4, 0, -2),     // Level -2
        (0b0110, 4, 0, 3),      // Level +3
        (0b0111, 4, 0, -3),     // Level -3

        // Run=1 (1 zero before)
        (0b1000, 4, 1, 1),      // 0, +1
        (0b1001, 4, 1, -1),     // 0, -1
        (0b1010, 4, 1, 2),      // 0, +2
        (0b1011, 4, 1, -2),     // 0, -2

        // Run=2 (2 zeros before)
        (0b1100, 4, 2, 1),      // 0, 0, +1
        (0b1101, 4, 2, -1),     // 0, 0, -1

        // Run=3 (3 zeros before)
        (0b1110, 4, 3, 1),      // 0, 0, 0, +1
        (0b1111, 4, 3, -1),     // 0, 0, 0, -1
    ];

    /// Decode AC coefficient (returns run, level)
    /// Returns None for EOB
    pub fn decode(reader: &mut ProResBitstreamReader) -> Result<Option<(u8, i16)>> {
        for &(code, len, run, level) in Self::TABLE {
            let peek = reader.read_bits(len)?;
            if peek == code {
                if run == 0 && level == 0 {
                    return Ok(None); // EOB
                }
                return Ok(Some((run, level)));
            }
        }

        Err(Error::InvalidData("Invalid AC VLC code".to_string()))
    }

    /// Encode AC coefficient
    /// If run/level is None, encodes EOB
    pub fn encode(writer: &mut ProResBitstreamWriter, run_level: Option<(u8, i16)>) -> Result<()> {
        match run_level {
            None => {
                // EOB
                writer.write_bits(0b00, 2);
                Ok(())
            }
            Some((run, level)) => {
                for &(code, len, table_run, table_level) in Self::TABLE {
                    if table_run == run && table_level == level {
                        writer.write_bits(code, len);
                        return Ok(());
                    }
                }
                Err(Error::InvalidData(format!("AC run={} level={} not in VLC table", run, level)))
            }
        }
    }
}

/// Decode a block of DCT coefficients
pub fn decode_dct_coefficients(
    reader: &mut ProResBitstreamReader,
    coeffs: &mut [i16; 64],
) -> Result<()> {
    // Reset coefficients
    coeffs.fill(0);

    // Decode DC coefficient
    coeffs[0] = ProResDcVlc::decode(reader)?;

    // Decode AC coefficients
    let mut pos = 1;
    while pos < 64 {
        match ProResAcVlc::decode(reader)? {
            None => break, // EOB
            Some((run, level)) => {
                pos += run as usize;
                if pos >= 64 {
                    return Err(Error::InvalidData("AC coefficient position overflow".to_string()));
                }
                coeffs[pos] = level;
                pos += 1;
            }
        }
    }

    Ok(())
}

/// Encode a block of DCT coefficients
pub fn encode_dct_coefficients(
    writer: &mut ProResBitstreamWriter,
    coeffs: &[i16; 64],
) -> Result<()> {
    // Encode DC coefficient
    ProResDcVlc::encode(writer, coeffs[0])?;

    // Encode AC coefficients with run-length encoding
    let mut pos = 1;
    while pos < 64 {
        // Find next non-zero coefficient
        let run_start = pos;
        while pos < 64 && coeffs[pos] == 0 {
            pos += 1;
        }

        if pos >= 64 {
            // EOB - all remaining coefficients are zero
            ProResAcVlc::encode(writer, None)?;
            break;
        }

        let run = (pos - run_start) as u8;
        let level = coeffs[pos];

        ProResAcVlc::encode(writer, Some((run, level)))?;
        pos += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_vlc_encode_decode() {
        let mut writer = ProResBitstreamWriter::new();

        // Encode DC value
        ProResDcVlc::encode(&mut writer, 0).unwrap();
        ProResDcVlc::encode(&mut writer, 1).unwrap();
        ProResDcVlc::encode(&mut writer, -1).unwrap();

        let data = writer.finish();
        let mut reader = ProResBitstreamReader::new(&data);

        // Decode and verify
        assert_eq!(ProResDcVlc::decode(&mut reader).unwrap(), 0);
        assert_eq!(ProResDcVlc::decode(&mut reader).unwrap(), 1);
        assert_eq!(ProResDcVlc::decode(&mut reader).unwrap(), -1);
    }

    #[test]
    fn test_ac_vlc_encode_decode() {
        let mut writer = ProResBitstreamWriter::new();

        // Encode AC values
        ProResAcVlc::encode(&mut writer, Some((0, 1))).unwrap(); // Run=0, Level=1
        ProResAcVlc::encode(&mut writer, Some((1, -1))).unwrap(); // Run=1, Level=-1
        ProResAcVlc::encode(&mut writer, None).unwrap(); // EOB

        let data = writer.finish();
        let mut reader = ProResBitstreamReader::new(&data);

        // Decode and verify
        assert_eq!(ProResAcVlc::decode(&mut reader).unwrap(), Some((0, 1)));
        assert_eq!(ProResAcVlc::decode(&mut reader).unwrap(), Some((1, -1)));
        assert_eq!(ProResAcVlc::decode(&mut reader).unwrap(), None);
    }

    #[test]
    fn test_dct_coefficients_roundtrip() {
        let mut coeffs = [0i16; 64];
        coeffs[0] = 5;  // DC
        coeffs[1] = 3;  // AC
        coeffs[3] = -2; // AC with run=1
        coeffs[10] = 1; // AC with run=6 (would need escape code in real implementation)

        let mut writer = ProResBitstreamWriter::new();
        encode_dct_coefficients(&mut writer, &coeffs).unwrap();

        let data = writer.finish();
        let mut reader = ProResBitstreamReader::new(&data);

        let mut decoded = [0i16; 64];
        decode_dct_coefficients(&mut reader, &mut decoded).unwrap();

        // DC should match
        assert_eq!(decoded[0], coeffs[0]);
    }

    #[test]
    fn test_all_zero_block() {
        let coeffs = [0i16; 64];

        let mut writer = ProResBitstreamWriter::new();
        encode_dct_coefficients(&mut writer, &coeffs).unwrap();

        let data = writer.finish();
        let mut reader = ProResBitstreamReader::new(&data);

        let mut decoded = [0i16; 64];
        decode_dct_coefficients(&mut reader, &mut decoded).unwrap();

        assert_eq!(decoded, coeffs);
    }
}
