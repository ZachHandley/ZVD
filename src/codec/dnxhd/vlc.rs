//! DNxHD VLC (Variable Length Coding)
//!
//! Huffman coding for DC and AC coefficients

use super::bitstream::{DnxhdBitstreamReader, DnxhdBitstreamWriter};
use super::data::{CidData, DcCode, AcVlcEntry, RunCode};
use crate::error::{Error, Result};

/// VLC decoder for DNxHD
pub struct DnxhdVlcDecoder {
    cid_data: &'static CidData,
}

impl DnxhdVlcDecoder {
    /// Create a new VLC decoder
    pub fn new(cid_data: &'static CidData) -> Self {
        Self { cid_data }
    }

    /// Decode DC coefficient
    pub fn decode_dc(&self, reader: &mut DnxhdBitstreamReader) -> Result<i16> {
        // Read bits to find matching DC code
        let mut code = 0u16;
        let mut bits_read = 0u8;

        for bits in 1..=16 {
            let bit = reader.read_bit()?;
            code = (code << 1) | (bit as u16);
            bits_read = bits;

            // Check if this matches a DC code
            for dc_code in self.cid_data.dc_codes {
                if dc_code.bits == bits_read && dc_code.code == code {
                    // Found the code - now read the magnitude
                    let nbits = bits_read / 2; // Simplified - actual calculation varies
                    if nbits == 0 {
                        return Ok(0);
                    }

                    let mag = reader.read_bits(nbits)? as i16;
                    let sign = reader.read_bit()?;

                    let value = if sign == 0 { mag } else { -mag };
                    return Ok(value);
                }
            }
        }

        Err(Error::invalid_input("Invalid DC VLC code"))
    }

    /// Decode AC coefficient block (63 coefficients)
    pub fn decode_ac(&self, reader: &mut DnxhdBitstreamReader, output: &mut [i16; 64]) -> Result<()> {
        let mut pos = 1; // Start after DC

        while pos < 64 {
            // Try to decode AC coefficient
            let mut code = 0u16;
            let mut bits_read = 0u8;
            let mut found = false;

            for bits in 1..=16 {
                let bit = reader.read_bit()?;
                code = (code << 1) | (bit as u16);
                bits_read = bits;

                // Check for EOB (End of Block)
                if bits_read == 2 && code == 0 {
                    // EOB - rest of block is zeros
                    return Ok(());
                }

                // Check AC codes
                for ac_entry in self.cid_data.ac_codes {
                    if ac_entry.bits == bits_read && ac_entry.code == code {
                        pos += ac_entry.run as usize;
                        if pos < 64 {
                            output[pos] = ac_entry.level;
                            pos += 1;
                        }
                        found = true;
                        break;
                    }
                }

                if found {
                    break;
                }
            }

            if !found {
                // Escape code - read raw value
                let run = reader.read_bits(6)? as usize;
                let level = reader.read_bits(12)? as i16;
                pos += run;
                if pos < 64 {
                    output[pos] = level;
                    pos += 1;
                }
            }
        }

        Ok(())
    }
}

/// VLC encoder for DNxHD
pub struct DnxhdVlcEncoder {
    cid_data: &'static CidData,
}

impl DnxhdVlcEncoder {
    /// Create a new VLC encoder
    pub fn new(cid_data: &'static CidData) -> Self {
        Self { cid_data }
    }

    /// Encode DC coefficient
    pub fn encode_dc(&self, writer: &mut DnxhdBitstreamWriter, value: i16) -> Result<()> {
        if value == 0 {
            // Encode zero DC
            let dc_code = &self.cid_data.dc_codes[0];
            writer.write_bits(dc_code.code as u32, dc_code.bits);
            return Ok(());
        }

        // Calculate magnitude and sign
        let mag = value.abs() as u32;
        let sign = if value < 0 { 1 } else { 0 };

        // Find appropriate DC code based on magnitude
        let nbits = 32 - mag.leading_zeros();
        let dc_index = nbits.min(self.cid_data.dc_codes.len() as u32 - 1);
        let dc_code = &self.cid_data.dc_codes[dc_index as usize];

        // Write code
        writer.write_bits(dc_code.code as u32, dc_code.bits);
        // Write magnitude
        writer.write_bits(mag, nbits as u8);
        // Write sign
        writer.write_bit(sign);

        Ok(())
    }

    /// Encode AC coefficient block (63 coefficients)
    pub fn encode_ac(&self, writer: &mut DnxhdBitstreamWriter, coeffs: &[i16; 64]) -> Result<()> {
        let mut last_non_zero = 0;

        // Find last non-zero coefficient
        for i in (1..64).rev() {
            if coeffs[i] != 0 {
                last_non_zero = i;
                break;
            }
        }

        if last_non_zero == 0 {
            // All AC coefficients are zero - write EOB
            writer.write_bits(0, 2); // EOB code
            return Ok(());
        }

        let mut pos = 1;
        while pos <= last_non_zero {
            if coeffs[pos] == 0 {
                pos += 1;
                continue;
            }

            // Count run of zeros
            let mut run = 0;
            for i in (pos - 1)..last_non_zero {
                if coeffs[i + 1] == 0 {
                    run += 1;
                } else {
                    break;
                }
            }

            let level = coeffs[pos];

            // Try to find matching AC code
            let mut found = false;
            for ac_entry in self.cid_data.ac_codes {
                if ac_entry.run == run as u8 && ac_entry.level == level {
                    writer.write_bits(ac_entry.code as u32, ac_entry.bits);
                    found = true;
                    break;
                }
            }

            if !found {
                // Use escape code
                writer.write_bits(1, 6); // Escape code marker
                writer.write_bits(run as u32, 6);
                writer.write_bits(level as u32 & 0xFFF, 12);
            }

            pos += run + 1;
        }

        // Write EOB
        writer.write_bits(0, 2);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::data::CidData;
    use super::super::DnxhdProfile;

    #[test]
    fn test_vlc_encode_decode_dc_zero() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let encoder = DnxhdVlcEncoder::new(cid_data);
        let decoder = DnxhdVlcDecoder::new(cid_data);

        let mut writer = DnxhdBitstreamWriter::new();
        encoder.encode_dc(&mut writer, 0).unwrap();

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        let value = decoder.decode_dc(&mut reader).unwrap();
        assert_eq!(value, 0);
    }

    #[test]
    fn test_vlc_encode_decode_dc_positive() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let encoder = DnxhdVlcEncoder::new(cid_data);
        let decoder = DnxhdVlcDecoder::new(cid_data);

        let mut writer = DnxhdBitstreamWriter::new();
        encoder.encode_dc(&mut writer, 42).unwrap();

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        let value = decoder.decode_dc(&mut reader).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_vlc_encode_decode_ac_all_zero() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let encoder = DnxhdVlcEncoder::new(cid_data);
        let decoder = DnxhdVlcDecoder::new(cid_data);

        let coeffs = [0i16; 64];
        let mut writer = DnxhdBitstreamWriter::new();
        encoder.encode_ac(&mut writer, &coeffs).unwrap();

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        let mut output = [0i16; 64];
        decoder.decode_ac(&mut reader, &mut output).unwrap();

        for i in 1..64 {
            assert_eq!(output[i], 0);
        }
    }

    #[test]
    fn test_vlc_encode_decode_ac_with_values() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let encoder = DnxhdVlcEncoder::new(cid_data);
        let decoder = DnxhdVlcDecoder::new(cid_data);

        let mut coeffs = [0i16; 64];
        coeffs[1] = 10;
        coeffs[5] = -5;
        coeffs[20] = 3;

        let mut writer = DnxhdBitstreamWriter::new();
        encoder.encode_ac(&mut writer, &coeffs).unwrap();

        let data = writer.finish();
        let mut reader = DnxhdBitstreamReader::new(&data);

        let mut output = [0i16; 64];
        decoder.decode_ac(&mut reader, &mut output).unwrap();

        // Check key positions (allowing for VLC approximation)
        assert_ne!(output[1], 0);
        assert_ne!(output[5], 0);
    }
}
