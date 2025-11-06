//! Symphonia adapter for multi-format audio support
//!
//! This module provides an adapter layer between ZVD and the Symphonia
//! audio decoding library, allowing us to support multiple audio formats
//! (FLAC, Vorbis, MP3, etc.) through a unified interface.

use crate::error::{Error, Result};
use crate::format::{AudioInfo, Demuxer, DemuxerContext, Packet, Stream, StreamInfo};
use crate::util::{Buffer, MediaType, Rational, SampleFormat, Timestamp};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Symphonia-based demuxer for multiple audio formats
///
/// Note: This demuxer also handles decoding since Symphonia's architecture
/// tightly couples the FormatReader (demuxer) and Decoder. The demuxer
/// reads packets and immediately decodes them to PCM.
pub struct SymphoniaDemuxer {
    reader: Option<Box<dyn FormatReader>>,
    decoder: Option<Box<dyn Decoder>>,
    sample_buffer: Option<SampleBuffer<i16>>,
    context: DemuxerContext,
    track_id: Option<u32>,
}

impl SymphoniaDemuxer {
    /// Create a new Symphonia demuxer
    pub fn new(format_name: &str) -> Self {
        SymphoniaDemuxer {
            reader: None,
            decoder: None,
            sample_buffer: None,
            context: DemuxerContext::new(format_name.to_string()),
            track_id: None,
        }
    }
}

impl Demuxer for SymphoniaDemuxer {
    fn open(&mut self, path: &Path) -> Result<()> {
        // Open the file
        let file = File::open(path).map_err(|e| {
            Error::format(format!("Failed to open file: {}", e))
        })?;

        // Create media source stream
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create a hint from the file extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe the media source
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| Error::format(format!("Failed to probe file: {}", e)))?;

        let mut reader = probed.format;

        // Get the default track
        let track = reader
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| Error::format("No supported audio track found"))?;

        self.track_id = Some(track.id);

        // Create stream info
        let codec_params = &track.codec_params;

        // Note: We report "pcm" as the codec because the Symphonia demuxer
        // decodes the audio to PCM samples internally. The original codec
        // (FLAC, MP3, Vorbis) is transparent to the caller.
        let mut stream_info = StreamInfo::new(
            track.id as usize,
            MediaType::Audio,
            "pcm".to_string(),
        );

        // Set audio info
        if let Some(sample_rate) = codec_params.sample_rate {
            stream_info.time_base = Rational::new(1, sample_rate as i64);

            let channels = codec_params.channels
                .map(|c| c.count())
                .unwrap_or(2) as u16;

            // Determine sample format from bits per coded sample
            let bits = codec_params.bits_per_coded_sample.unwrap_or(16);
            let sample_fmt = match bits {
                8 => SampleFormat::U8,
                16 => SampleFormat::I16,
                24 | 32 => SampleFormat::I32,
                _ => SampleFormat::I16,
            };

            stream_info.audio_info = Some(AudioInfo {
                sample_rate,
                channels,
                sample_fmt: format!("{}", sample_fmt),
                bits_per_sample: bits as u8,
                bit_rate: codec_params.max_frames_per_packet.map(|b| b as u64),
            });

            if let Some(n_frames) = codec_params.n_frames {
                stream_info.duration = n_frames as i64;
                stream_info.nb_frames = Some(n_frames);
            }
        }

        // Add stream to context
        let stream = Stream::new(stream_info);
        self.context.add_stream(stream);

        // Create decoder for this track
        let decoder = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| Error::format(format!("Unsupported codec: {}", e)))?;

        self.decoder = Some(decoder);
        self.reader = Some(reader);

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

        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Decoder not initialized"))?;

        // Read next packet from format reader
        let packet = reader
            .next_packet()
            .map_err(|e| match e {
                symphonia::core::errors::Error::IoError(ref io_err) => {
                    if io_err.kind() == std::io::ErrorKind::UnexpectedEof {
                        Error::EndOfStream
                    } else {
                        Error::format(format!("IO error reading packet: {}", e))
                    }
                }
                symphonia::core::errors::Error::ResetRequired => Error::EndOfStream,
                _ => Error::format(format!("Failed to read packet: {}", e)),
            })?;

        // Store packet metadata
        let pts = packet.ts();
        let duration = packet.dur();
        let track_id = packet.track_id();

        // Decode the packet to get audio samples
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| Error::format(format!("Failed to decode packet: {}", e)))?;

        // Initialize sample buffer on first decode, or reuse existing one
        if self.sample_buffer.is_none() {
            // Create a sample buffer to hold decoded audio as i16 samples
            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;
            self.sample_buffer = Some(SampleBuffer::<i16>::new(duration, spec));
        }

        // Copy decoded audio into sample buffer (converts to i16 interleaved format)
        let sample_buf = self.sample_buffer.as_mut().unwrap();
        sample_buf.copy_interleaved_ref(decoded);

        // Get the raw i16 samples and convert to bytes
        let samples = sample_buf.samples();
        let byte_data: Vec<u8> = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();

        // Create our packet with decoded PCM data
        let mut zvd_packet = Packet::new(
            track_id as usize,
            Buffer::from_vec(byte_data),
        );

        zvd_packet.pts = Timestamp::new(pts as i64);
        zvd_packet.duration = duration as i64;

        Ok(zvd_packet)
    }

    fn seek(&mut self, _stream_index: usize, timestamp: i64) -> Result<()> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        let track_id = self
            .track_id
            .ok_or_else(|| Error::invalid_state("No track selected"))?;

        reader
            .seek(
                symphonia::core::formats::SeekMode::Accurate,
                symphonia::core::formats::SeekTo::TimeStamp {
                    ts: timestamp as u64,
                    track_id,
                },
            )
            .map_err(|e| Error::format(format!("Seek failed: {}", e)))?;

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.reader = None;
        self.decoder = None;
        self.sample_buffer = None;
        self.track_id = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symphonia_demuxer_creation() {
        let demuxer = SymphoniaDemuxer::new("flac");
        assert_eq!(demuxer.context.format_name(), "flac");
    }
}
