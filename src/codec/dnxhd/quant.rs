//! DNxHD Quantization
//!
//! Handles quantization and dequantization of DCT coefficients

use super::data::{CidData, ZIGZAG_SCAN};
use crate::error::Result;

/// DNxHD quantizer
pub struct DnxhdQuantizer {
    cid_data: &'static CidData,
    qscale: u16,
}

impl DnxhdQuantizer {
    /// Create a new quantizer
    pub fn new(cid_data: &'static CidData, qscale: u16) -> Self {
        Self { cid_data, qscale }
    }

    /// Quantize DCT coefficients (luma)
    pub fn quantize_luma(&self, input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // DC coefficient is not quantized in DNxHD
        output[0] = input[0];

        // Quantize AC coefficients
        for i in 1..64 {
            let coeff = input[i] as i32;
            let weight = self.cid_data.luma_weight[i] as i32;
            let qscale = self.qscale as i32;

            // Quantization formula: level = (coeff * qscale) / (weight * 8)
            let quantized = (coeff * qscale) / (weight * 8);
            output[i] = quantized.clamp(-2048, 2047) as i16;
        }

        Ok(())
    }

    /// Quantize DCT coefficients (chroma)
    pub fn quantize_chroma(&self, input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // DC coefficient is not quantized in DNxHD
        output[0] = input[0];

        // Quantize AC coefficients
        for i in 1..64 {
            let coeff = input[i] as i32;
            let weight = self.cid_data.chroma_weight[i] as i32;
            let qscale = self.qscale as i32;

            // Quantization formula
            let quantized = (coeff * qscale) / (weight * 8);
            output[i] = quantized.clamp(-2048, 2047) as i16;
        }

        Ok(())
    }

    /// Dequantize DCT coefficients (luma)
    pub fn dequantize_luma(&self, input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // DC coefficient is not quantized
        output[0] = input[0];

        // Dequantize AC coefficients
        for i in 1..64 {
            let quantized = input[i] as i32;
            let weight = self.cid_data.luma_weight[i] as i32;
            let qscale = self.qscale as i32;

            // Dequantization formula: coeff = (level * weight * 8) / qscale
            let dequantized = (quantized * weight * 8) / qscale;
            output[i] = dequantized.clamp(-2048, 2047) as i16;
        }

        Ok(())
    }

    /// Dequantize DCT coefficients (chroma)
    pub fn dequantize_chroma(&self, input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // DC coefficient is not quantized
        output[0] = input[0];

        // Dequantize AC coefficients
        for i in 1..64 {
            let quantized = input[i] as i32;
            let weight = self.cid_data.chroma_weight[i] as i32;
            let qscale = self.qscale as i32;

            // Dequantization formula
            let dequantized = (quantized * weight * 8) / qscale;
            output[i] = dequantized.clamp(-2048, 2047) as i16;
        }

        Ok(())
    }

    /// Reorder coefficients from raster to zigzag order
    pub fn raster_to_zigzag(input: &[i16; 64], output: &mut [i16; 64]) {
        for i in 0..64 {
            output[i] = input[ZIGZAG_SCAN[i]];
        }
    }

    /// Reorder coefficients from zigzag to raster order
    pub fn zigzag_to_raster(input: &[i16; 64], output: &mut [i16; 64]) {
        for i in 0..64 {
            output[ZIGZAG_SCAN[i]] = input[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::data::CidData;
    use super::super::DnxhdProfile;

    #[test]
    fn test_quantize_dequantize_luma() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let quantizer = DnxhdQuantizer::new(cid_data, 1024);

        let mut input = [0i16; 64];
        input[0] = 100; // DC
        input[1] = 50;
        input[2] = 30;
        input[8] = 20;

        let mut quantized = [0i16; 64];
        quantizer.quantize_luma(&input, &mut quantized).unwrap();

        // DC should not be quantized
        assert_eq!(quantized[0], 100);

        let mut dequantized = [0i16; 64];
        quantizer.dequantize_luma(&quantized, &mut dequantized).unwrap();

        // DC should be exact
        assert_eq!(dequantized[0], 100);

        // AC coefficients should be approximately reconstructed
        for i in 1..64 {
            if input[i] != 0 {
                let diff = (input[i] - dequantized[i]).abs();
                assert!(diff < 50, "Position {}: input={}, output={}, diff={}",
                       i, input[i], dequantized[i], diff);
            }
        }
    }

    #[test]
    fn test_quantize_dequantize_chroma() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let quantizer = DnxhdQuantizer::new(cid_data, 1024);

        let mut input = [0i16; 64];
        input[0] = 80; // DC
        input[1] = 40;
        input[5] = 25;

        let mut quantized = [0i16; 64];
        quantizer.quantize_chroma(&input, &mut quantized).unwrap();

        assert_eq!(quantized[0], 80);

        let mut dequantized = [0i16; 64];
        quantizer.dequantize_chroma(&quantized, &mut dequantized).unwrap();

        assert_eq!(dequantized[0], 80);
    }

    #[test]
    fn test_zigzag_roundtrip() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = i as i16;
        }

        let mut zigzag = [0i16; 64];
        DnxhdQuantizer::raster_to_zigzag(&input, &mut zigzag);

        let mut raster = [0i16; 64];
        DnxhdQuantizer::zigzag_to_raster(&zigzag, &mut raster);

        for i in 0..64 {
            assert_eq!(input[i], raster[i]);
        }
    }

    #[test]
    fn test_dc_not_quantized() {
        let cid_data = CidData::for_profile(DnxhdProfile::DnxhrHq);
        let quantizer = DnxhdQuantizer::new(cid_data, 2048);

        let mut input = [0i16; 64];
        input[0] = 512;

        let mut output = [0i16; 64];
        quantizer.quantize_luma(&input, &mut output).unwrap();

        // DC should pass through unchanged
        assert_eq!(output[0], 512);
    }
}
