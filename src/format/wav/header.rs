//! WAV file header structures and parsing

use crate::error::{Error, Result};
use crate::util::SampleFormat;
use std::io::{Read, Seek, SeekFrom};

/// WAV format tag identifying the codec
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum FormatTag {
    /// PCM (uncompressed)
    Pcm = 0x0001,
    /// IEEE Float
    IeeeFloat = 0x0003,
    /// A-Law
    ALaw = 0x0006,
    /// Mu-Law
    MuLaw = 0x0007,
    /// Extensible format
    Extensible = 0xFFFE,
    /// Unknown format
    Unknown(u16),
}

impl From<u16> for FormatTag {
    fn from(val: u16) -> Self {
        match val {
            0x0001 => FormatTag::Pcm,
            0x0003 => FormatTag::IeeeFloat,
            0x0006 => FormatTag::ALaw,
            0x0007 => FormatTag::MuLaw,
            0xFFFE => FormatTag::Extensible,
            other => FormatTag::Unknown(other),
        }
    }
}

impl From<FormatTag> for u16 {
    fn from(tag: FormatTag) -> Self {
        match tag {
            FormatTag::Pcm => 0x0001,
            FormatTag::IeeeFloat => 0x0003,
            FormatTag::ALaw => 0x0006,
            FormatTag::MuLaw => 0x0007,
            FormatTag::Extensible => 0xFFFE,
            FormatTag::Unknown(val) => val,
        }
    }
}

/// WAV format chunk data
#[derive(Debug, Clone)]
pub struct WavFormat {
    /// Format tag (codec ID)
    pub format_tag: FormatTag,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Average bytes per second
    pub byte_rate: u32,
    /// Block alignment
    pub block_align: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Extension size (for extensible format)
    pub extension_size: Option<u16>,
}

impl WavFormat {
    /// Parse WAV format chunk from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(Error::format("WAV format chunk too small"));
        }

        let format_tag = u16::from_le_bytes([data[0], data[1]]).into();
        let channels = u16::from_le_bytes([data[2], data[3]]);
        let sample_rate = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let byte_rate = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let block_align = u16::from_le_bytes([data[12], data[13]]);
        let bits_per_sample = u16::from_le_bytes([data[14], data[15]]);

        let extension_size = if data.len() > 16 {
            Some(u16::from_le_bytes([data[16], data[17]]))
        } else {
            None
        };

        Ok(WavFormat {
            format_tag,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            extension_size,
        })
    }

    /// Convert to bytes for writing
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(18);

        bytes.extend_from_slice(&u16::from(self.format_tag).to_le_bytes());
        bytes.extend_from_slice(&self.channels.to_le_bytes());
        bytes.extend_from_slice(&self.sample_rate.to_le_bytes());
        bytes.extend_from_slice(&self.byte_rate.to_le_bytes());
        bytes.extend_from_slice(&self.block_align.to_le_bytes());
        bytes.extend_from_slice(&self.bits_per_sample.to_le_bytes());

        if let Some(ext_size) = self.extension_size {
            bytes.extend_from_slice(&ext_size.to_le_bytes());
        }

        bytes
    }

    /// Get the internal sample format
    pub fn sample_format(&self) -> SampleFormat {
        match (self.format_tag, self.bits_per_sample) {
            (FormatTag::Pcm, 8) => SampleFormat::U8,
            (FormatTag::Pcm, 16) => SampleFormat::I16,
            (FormatTag::Pcm, 32) => SampleFormat::I32,
            (FormatTag::IeeeFloat, 32) => SampleFormat::F32,
            (FormatTag::IeeeFloat, 64) => SampleFormat::F64,
            _ => SampleFormat::Unknown,
        }
    }

    /// Calculate expected byte rate
    pub fn calculate_byte_rate(&self) -> u32 {
        self.sample_rate * self.block_align as u32
    }

    /// Calculate expected block alignment
    pub fn calculate_block_align(&self) -> u16 {
        self.channels * (self.bits_per_sample / 8)
    }

    /// Validate format parameters
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 {
            return Err(Error::format("Invalid channel count: 0"));
        }

        if self.sample_rate == 0 {
            return Err(Error::format("Invalid sample rate: 0"));
        }

        if self.bits_per_sample == 0 || self.bits_per_sample % 8 != 0 {
            return Err(Error::format(format!(
                "Invalid bits per sample: {}",
                self.bits_per_sample
            )));
        }

        let expected_block_align = self.calculate_block_align();
        if self.block_align != expected_block_align {
            return Err(Error::format(format!(
                "Block align mismatch: expected {}, got {}",
                expected_block_align, self.block_align
            )));
        }

        Ok(())
    }
}

/// Complete WAV file header
#[derive(Debug, Clone)]
pub struct WavHeader {
    /// Total file size (RIFF chunk size + 8)
    pub file_size: u32,
    /// WAV format information
    pub format: WavFormat,
    /// Data chunk size in bytes
    pub data_size: u32,
    /// Data chunk start position in file
    pub data_start: u64,
}

impl WavHeader {
    /// Read and parse WAV header from a reader
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Read RIFF header
        let mut riff_header = [0u8; 12];
        reader
            .read_exact(&mut riff_header)
            .map_err(|e| Error::format(format!("Failed to read RIFF header: {}", e)))?;

        // Verify RIFF magic
        if &riff_header[0..4] != b"RIFF" {
            return Err(Error::format("Not a valid RIFF file"));
        }

        // Verify WAVE magic
        if &riff_header[8..12] != b"WAVE" {
            return Err(Error::format("Not a valid WAVE file"));
        }

        let file_size = u32::from_le_bytes([
            riff_header[4],
            riff_header[5],
            riff_header[6],
            riff_header[7],
        ]) + 8;

        // Find and parse format chunk
        let format = Self::find_and_parse_fmt_chunk(reader)?;

        // Find data chunk
        let (data_size, data_start) = Self::find_data_chunk(reader)?;

        Ok(WavHeader {
            file_size,
            format,
            data_size,
            data_start,
        })
    }

    /// Find and parse the fmt chunk
    fn find_and_parse_fmt_chunk<R: Read + Seek>(reader: &mut R) -> Result<WavFormat> {
        loop {
            let mut chunk_header = [0u8; 8];
            if reader.read_exact(&mut chunk_header).is_err() {
                return Err(Error::format("fmt chunk not found"));
            }

            let chunk_id = &chunk_header[0..4];
            let chunk_size = u32::from_le_bytes([
                chunk_header[4],
                chunk_header[5],
                chunk_header[6],
                chunk_header[7],
            ]);

            if chunk_id == b"fmt " {
                // Read fmt chunk data
                let mut fmt_data = vec![0u8; chunk_size as usize];
                reader
                    .read_exact(&mut fmt_data)
                    .map_err(|e| Error::format(format!("Failed to read fmt chunk: {}", e)))?;

                let format = WavFormat::from_bytes(&fmt_data)?;
                format.validate()?;
                return Ok(format);
            } else {
                // Skip this chunk
                reader
                    .seek(SeekFrom::Current(chunk_size as i64))
                    .map_err(|e| Error::format(format!("Failed to skip chunk: {}", e)))?;

                // WAV chunks are word-aligned
                if chunk_size % 2 != 0 {
                    reader.seek(SeekFrom::Current(1)).ok();
                }
            }
        }
    }

    /// Find the data chunk and return its size and start position
    fn find_data_chunk<R: Read + Seek>(reader: &mut R) -> Result<(u32, u64)> {
        loop {
            let position = reader
                .stream_position()
                .map_err(|e| Error::format(format!("Failed to get position: {}", e)))?;

            let mut chunk_header = [0u8; 8];
            if reader.read_exact(&mut chunk_header).is_err() {
                return Err(Error::format("data chunk not found"));
            }

            let chunk_id = &chunk_header[0..4];
            let chunk_size = u32::from_le_bytes([
                chunk_header[4],
                chunk_header[5],
                chunk_header[6],
                chunk_header[7],
            ]);

            if chunk_id == b"data" {
                let data_start = reader
                    .stream_position()
                    .map_err(|e| Error::format(format!("Failed to get data position: {}", e)))?;
                return Ok((chunk_size, data_start));
            } else {
                // Skip this chunk
                reader
                    .seek(SeekFrom::Current(chunk_size as i64))
                    .map_err(|e| Error::format(format!("Failed to skip chunk: {}", e)))?;

                // WAV chunks are word-aligned
                if chunk_size % 2 != 0 {
                    reader.seek(SeekFrom::Current(1)).ok();
                }
            }
        }
    }

    /// Get duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        let total_samples = self.data_size as f64 / self.format.block_align as f64;
        total_samples / self.format.sample_rate as f64
    }

    /// Get total number of samples (per channel)
    pub fn num_samples(&self) -> u64 {
        self.data_size as u64 / self.format.block_align as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tag_conversion() {
        assert_eq!(u16::from(FormatTag::Pcm), 0x0001);
        assert_eq!(FormatTag::from(0x0001), FormatTag::Pcm);
    }

    #[test]
    fn test_wav_format_calculations() {
        let format = WavFormat {
            format_tag: FormatTag::Pcm,
            channels: 2,
            sample_rate: 44100,
            byte_rate: 176400,
            block_align: 4,
            bits_per_sample: 16,
            extension_size: None,
        };

        assert_eq!(format.calculate_block_align(), 4);
        assert_eq!(format.calculate_byte_rate(), 176400);
        assert_eq!(format.sample_format(), SampleFormat::I16);
    }

    #[test]
    fn test_wav_format_validation() {
        let mut format = WavFormat {
            format_tag: FormatTag::Pcm,
            channels: 2,
            sample_rate: 44100,
            byte_rate: 176400,
            block_align: 4,
            bits_per_sample: 16,
            extension_size: None,
        };

        assert!(format.validate().is_ok());

        // Invalid channel count
        format.channels = 0;
        assert!(format.validate().is_err());
        format.channels = 2;

        // Invalid block align
        format.block_align = 3;
        assert!(format.validate().is_err());
    }
}
