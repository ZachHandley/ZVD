//! WAV file muxer implementation

use super::header::{FormatTag, WavFormat};
use super::{DATA_CHUNK, FMT_CHUNK, RIFF_MAGIC, WAVE_MAGIC};
use crate::error::{Error, Result};
use crate::format::{Muxer, MuxerContext, Packet, Stream};
use crate::util::SampleFormat;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// WAV file muxer
pub struct WavMuxer {
    writer: Option<BufWriter<File>>,
    context: MuxerContext,
    format: Option<WavFormat>,
    data_size_position: Option<u64>,
    riff_size_position: Option<u64>,
    samples_written: u64,
}

impl WavMuxer {
    /// Create a new WAV muxer
    pub fn new() -> Self {
        WavMuxer {
            writer: None,
            context: MuxerContext::new("wav".to_string()),
            format: None,
            data_size_position: None,
            riff_size_position: None,
            samples_written: 0,
        }
    }
}

impl Default for WavMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for WavMuxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::format(format!("Failed to create WAV file: {}", e)))?;

        self.writer = Some(BufWriter::new(file));
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        // WAV only supports one audio stream
        if !self.context.streams().is_empty() {
            return Err(Error::format("WAV only supports one audio stream"));
        }

        // Validate it's an audio stream
        if stream.info.media_type != crate::util::MediaType::Audio {
            return Err(Error::format("WAV only supports audio streams"));
        }

        // Get audio info
        let audio_info = stream
            .info
            .audio_info
            .as_ref()
            .ok_or_else(|| Error::format("Audio stream missing audio info"))?;

        // Determine format tag based on sample format
        let sample_fmt = match audio_info.sample_fmt.as_str() {
            "u8" => SampleFormat::U8,
            "s16" | "i16" => SampleFormat::I16,
            "s32" | "i32" => SampleFormat::I32,
            "f32" => SampleFormat::F32,
            "f64" => SampleFormat::F64,
            _ => {
                return Err(Error::unsupported(format!(
                    "Unsupported sample format: {}",
                    audio_info.sample_fmt
                )))
            }
        };

        let (format_tag, bits_per_sample) = match sample_fmt {
            SampleFormat::U8 => (FormatTag::Pcm, 8),
            SampleFormat::I16 => (FormatTag::Pcm, 16),
            SampleFormat::I32 => (FormatTag::Pcm, 32),
            SampleFormat::F32 => (FormatTag::IeeeFloat, 32),
            SampleFormat::F64 => (FormatTag::IeeeFloat, 64),
            _ => {
                return Err(Error::unsupported(format!(
                    "Sample format not supported for WAV: {:?}",
                    sample_fmt
                )))
            }
        };

        let block_align = audio_info.channels * (bits_per_sample / 8);
        let byte_rate = audio_info.sample_rate * block_align as u32;

        // Create WAV format
        self.format = Some(WavFormat {
            format_tag,
            channels: audio_info.channels,
            sample_rate: audio_info.sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            extension_size: None,
        });

        Ok(self.context.add_stream(stream))
    }

    fn write_header(&mut self) -> Result<()> {
        if self.context.is_header_written() {
            return Err(Error::invalid_state("Header already written"));
        }

        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Muxer not created"))?;

        let format = self
            .format
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No stream added"))?;

        // Write RIFF header
        writer
            .write_all(RIFF_MAGIC)
            .map_err(|e| Error::format(format!("Failed to write RIFF magic: {}", e)))?;

        // Write placeholder file size (will update later)
        self.riff_size_position = Some(
            writer
                .stream_position()
                .map_err(|e| Error::format(format!("Failed to get position: {}", e)))?,
        );
        writer
            .write_all(&0u32.to_le_bytes())
            .map_err(|e| Error::format(format!("Failed to write RIFF size: {}", e)))?;

        // Write WAVE magic
        writer
            .write_all(WAVE_MAGIC)
            .map_err(|e| Error::format(format!("Failed to write WAVE magic: {}", e)))?;

        // Write fmt chunk
        writer
            .write_all(FMT_CHUNK)
            .map_err(|e| Error::format(format!("Failed to write fmt chunk ID: {}", e)))?;

        let fmt_data = format.to_bytes();
        writer
            .write_all(&(fmt_data.len() as u32).to_le_bytes())
            .map_err(|e| Error::format(format!("Failed to write fmt chunk size: {}", e)))?;
        writer
            .write_all(&fmt_data)
            .map_err(|e| Error::format(format!("Failed to write fmt chunk data: {}", e)))?;

        // Write data chunk header
        writer
            .write_all(DATA_CHUNK)
            .map_err(|e| Error::format(format!("Failed to write data chunk ID: {}", e)))?;

        // Write placeholder data size (will update later)
        self.data_size_position = Some(
            writer
                .stream_position()
                .map_err(|e| Error::format(format!("Failed to get position: {}", e)))?,
        );
        writer
            .write_all(&0u32.to_le_bytes())
            .map_err(|e| Error::format(format!("Failed to write data size: {}", e)))?;

        self.context.set_header_written();
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.context.is_header_written() {
            return Err(Error::invalid_state("Header not written"));
        }

        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Muxer not created"))?;

        let format = self
            .format
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No format configured"))?;

        // Write packet data
        writer
            .write_all(packet.data.as_slice())
            .map_err(|e| Error::format(format!("Failed to write packet data: {}", e)))?;

        // Update samples written
        let bytes = packet.data.len() as u64;
        self.samples_written += bytes / format.block_align as u64;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Muxer not created"))?;

        let format = self
            .format
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No format configured"))?;

        // Calculate final data size
        let data_size = self.samples_written * format.block_align as u64;

        // Update data chunk size
        if let Some(pos) = self.data_size_position {
            writer
                .seek(SeekFrom::Start(pos))
                .map_err(|e| Error::format(format!("Failed to seek to data size: {}", e)))?;
            writer
                .write_all(&(data_size as u32).to_le_bytes())
                .map_err(|e| Error::format(format!("Failed to write data size: {}", e)))?;
        }

        // Update RIFF chunk size (file size - 8)
        if let Some(pos) = self.riff_size_position {
            writer
                .seek(SeekFrom::Start(pos))
                .map_err(|e| Error::format(format!("Failed to seek to RIFF size: {}", e)))?;

            // RIFF size = 4 (WAVE) + 8 (fmt header) + fmt_data_size + 8 (data header) + data_size
            let fmt_data_size = format.to_bytes().len() as u32;
            let riff_size = 4 + 8 + fmt_data_size + 8 + data_size as u32;

            writer
                .write_all(&riff_size.to_le_bytes())
                .map_err(|e| Error::format(format!("Failed to write RIFF size: {}", e)))?;
        }

        // Flush writer
        writer
            .flush()
            .map_err(|e| Error::format(format!("Failed to flush writer: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_muxer_creation() {
        let muxer = WavMuxer::new();
        assert!(muxer.writer.is_none());
        assert!(muxer.format.is_none());
    }
}
