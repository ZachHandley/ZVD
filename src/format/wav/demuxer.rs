//! WAV file demuxer implementation

use super::header::{WavHeader, FormatTag};
use crate::error::{Error, Result};
use crate::format::{Demuxer, DemuxerContext, Packet, Stream, StreamInfo};
use crate::util::{Buffer, MediaType, Rational, Timestamp};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// WAV demuxer
pub struct WavDemuxer {
    reader: Option<BufReader<File>>,
    context: DemuxerContext,
    header: Option<WavHeader>,
    samples_read: u64,
    total_samples: u64,
}

impl WavDemuxer {
    /// Create a new WAV demuxer
    pub fn new() -> Self {
        WavDemuxer {
            reader: None,
            context: DemuxerContext::new("wav".to_string()),
            header: None,
            samples_read: 0,
            total_samples: 0,
        }
    }

    /// Get the current header
    pub fn header(&self) -> Option<&WavHeader> {
        self.header.as_ref()
    }
}

impl Default for WavDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for WavDemuxer {
    fn open(&mut self, path: &Path) -> Result<()> {
        // Open the file
        let file = File::open(path).map_err(|e| {
            Error::format(format!("Failed to open WAV file: {}", e))
        })?;

        let mut reader = BufReader::new(file);

        // Parse WAV header
        let header = WavHeader::read(&mut reader)?;

        // Validate format
        if !matches!(
            header.format.format_tag,
            FormatTag::Pcm | FormatTag::IeeeFloat
        ) {
            return Err(Error::unsupported(format!(
                "Unsupported WAV format: {:?}",
                header.format.format_tag
            )));
        }

        // Calculate total samples
        self.total_samples = header.num_samples();

        // Seek to data start
        reader
            .seek(SeekFrom::Start(header.data_start))
            .map_err(|e| Error::format(format!("Failed to seek to data: {}", e)))?;

        // Create stream info
        let mut stream_info = StreamInfo::new(0, MediaType::Audio, "pcm".to_string());

        // Set time base to sample rate
        stream_info.time_base = Rational::new(1, header.format.sample_rate as i64);
        stream_info.duration = self.total_samples as i64;
        stream_info.nb_frames = Some(self.total_samples);

        // Set audio info
        stream_info.audio_info = Some(crate::format::AudioInfo {
            sample_rate: header.format.sample_rate,
            channels: header.format.channels,
            sample_fmt: header.format.sample_format().to_string(),
            bits_per_sample: header.format.bits_per_sample as u8,
            bit_rate: Some(header.format.byte_rate as u64 * 8),
        });

        // Add metadata
        stream_info.metadata.insert(
            "duration".to_string(),
            format!("{:.2}s", header.duration_seconds()),
        );

        // Create stream and add to context
        let stream = Stream::new(stream_info);
        self.context.add_stream(stream);

        // Set duration on context
        self.context.set_duration(self.total_samples as i64);

        self.header = Some(header);
        self.reader = Some(reader);
        self.samples_read = 0;

        Ok(())
    }

    fn streams(&self) -> &[Stream] {
        self.context.streams()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        let header = self
            .header
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No header parsed"))?;

        // Check if we've read all samples
        if self.samples_read >= self.total_samples {
            return Err(Error::EndOfStream);
        }

        // Read one second worth of audio (or remaining samples)
        let samples_per_packet = header.format.sample_rate as u64;
        let samples_to_read = samples_per_packet.min(self.total_samples - self.samples_read);
        let bytes_to_read = (samples_to_read * header.format.block_align as u64) as usize;

        // Read audio data
        let mut buffer = vec![0u8; bytes_to_read];
        reader
            .read_exact(&mut buffer)
            .map_err(|e| Error::format(format!("Failed to read audio data: {}", e)))?;

        // Create packet
        let mut packet = Packet::new(0, Buffer::from_vec(buffer));

        // Set timestamps
        packet.pts = Timestamp::new(self.samples_read as i64);
        packet.dts = packet.pts;
        packet.duration = samples_to_read as i64;

        // PCM is always a keyframe
        packet.set_keyframe(true);

        // Update samples read
        self.samples_read += samples_to_read;

        Ok(packet)
    }

    fn seek(&mut self, _stream_index: usize, timestamp: i64) -> Result<()> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        let header = self
            .header
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No header parsed"))?;

        // Validate timestamp
        if timestamp < 0 || timestamp > self.total_samples as i64 {
            return Err(Error::invalid_input(format!(
                "Seek timestamp {} out of range (0-{})",
                timestamp, self.total_samples
            )));
        }

        // Calculate byte offset
        let byte_offset = timestamp as u64 * header.format.block_align as u64;
        let file_offset = header.data_start + byte_offset;

        // Seek to position
        reader
            .seek(SeekFrom::Start(file_offset))
            .map_err(|e| Error::format(format!("Failed to seek: {}", e)))?;

        self.samples_read = timestamp as u64;

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.reader = None;
        self.header = None;
        self.samples_read = 0;
        self.total_samples = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_demuxer_creation() {
        let demuxer = WavDemuxer::new();
        assert!(demuxer.header().is_none());
        assert_eq!(demuxer.streams().len(), 0);
    }

    // Additional tests would require sample WAV files
}
