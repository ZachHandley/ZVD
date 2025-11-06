//! Audio filters

use super::Filter;
use crate::codec::{AudioFrame, Frame};
use crate::error::{Error, Result};
use crate::util::{Buffer, Timestamp};

/// Volume adjustment filter
pub struct VolumeFilter {
    volume: f32,
}

impl VolumeFilter {
    /// Create a new volume filter
    /// volume: 1.0 = 100%, 0.5 = 50%, 2.0 = 200%
    pub fn new(volume: f32) -> Self {
        VolumeFilter { volume }
    }

    /// Apply volume adjustment to audio samples
    fn adjust_volume(&self, audio_frame: &AudioFrame) -> Result<AudioFrame> {
        let mut output_frame = audio_frame.clone();

        // Process each plane (channel for planar, or single buffer for interleaved)
        for plane in &mut output_frame.data {
            let samples = plane.as_slice();
            let mut adjusted_samples = Vec::with_capacity(samples.len());

            // Determine sample format based on frame info
            // For now, assume i16 PCM format (most common)
            if samples.len() % 2 == 0 {
                // Process as i16 samples
                for chunk in samples.chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    let adjusted = (sample as f32 * self.volume).clamp(-32768.0, 32767.0) as i16;
                    adjusted_samples.extend_from_slice(&adjusted.to_le_bytes());
                }
            } else {
                // If not even bytes, just pass through
                return Err(Error::filter(
                    "Unsupported audio sample format for volume filter",
                ));
            }

            *plane = Buffer::from_vec(adjusted_samples);
        }

        Ok(output_frame)
    }
}

impl Filter for VolumeFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Audio(audio_frame) => {
                let adjusted = self.adjust_volume(&audio_frame)?;
                Ok(vec![Frame::Audio(adjusted)])
            }
            Frame::Video(_) => Err(Error::filter("Volume filter only accepts audio frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Audio resampling filter
pub struct ResampleFilter {
    target_sample_rate: u32,
}

impl ResampleFilter {
    /// Create a new resample filter
    pub fn new(sample_rate: u32) -> Self {
        ResampleFilter {
            target_sample_rate: sample_rate,
        }
    }
}

impl ResampleFilter {
    /// Simple linear resampling for i16 PCM audio
    fn resample_linear(&self, audio_frame: &AudioFrame, src_rate: u32) -> Result<AudioFrame> {
        if src_rate == self.target_sample_rate {
            return Ok(audio_frame.clone());
        }

        let mut output_frame = audio_frame.clone();
        let ratio = self.target_sample_rate as f64 / src_rate as f64;

        // Process each plane (channel)
        for plane in &mut output_frame.data {
            let samples = plane.as_slice();
            if samples.len() % 2 != 0 {
                return Err(Error::filter("Audio data must be i16 PCM (even byte count)"));
            }

            let input_samples: Vec<i16> = samples
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            let output_count = (input_samples.len() as f64 * ratio) as usize;
            let mut output_samples = Vec::with_capacity(output_count);

            for i in 0..output_count {
                let src_pos = i as f64 / ratio;
                let src_idx = src_pos as usize;

                if src_idx + 1 < input_samples.len() {
                    // Linear interpolation
                    let frac = src_pos - src_idx as f64;
                    let s0 = input_samples[src_idx] as f64;
                    let s1 = input_samples[src_idx + 1] as f64;
                    let interpolated = s0 + (s1 - s0) * frac;
                    output_samples.push(interpolated.clamp(-32768.0, 32767.0) as i16);
                } else if src_idx < input_samples.len() {
                    // Use last sample
                    output_samples.push(input_samples[src_idx]);
                }
            }

            // Convert back to bytes
            let mut resampled_bytes = Vec::with_capacity(output_samples.len() * 2);
            for sample in output_samples {
                resampled_bytes.extend_from_slice(&sample.to_le_bytes());
            }

            *plane = Buffer::from_vec(resampled_bytes);
        }

        output_frame.sample_rate = self.target_sample_rate;
        output_frame.nb_samples = output_frame.data[0].len() / 2;
        // Update duration proportionally to sample count change
        if audio_frame.nb_samples > 0 {
            output_frame.duration = (audio_frame.duration as f64 * ratio) as i64;
        }

        Ok(output_frame)
    }
}

impl Filter for ResampleFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Audio(audio_frame) => {
                let src_rate = audio_frame.sample_rate;
                let resampled = self.resample_linear(&audio_frame, src_rate)?;
                Ok(vec![Frame::Audio(resampled)])
            }
            Frame::Video(_) => Err(Error::filter("Resample filter only accepts audio frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Audio normalization filter
pub struct NormalizeFilter {
    target_level: f32, // Target RMS level (0.0 to 1.0)
}

impl NormalizeFilter {
    /// Create a new normalize filter
    /// target_level: Target RMS level between 0.0 and 1.0 (default 0.25 = -12dB)
    pub fn new(target_level: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&target_level) {
            return Err(Error::filter(format!(
                "Target level must be between 0.0 and 1.0, got {}",
                target_level
            )));
        }
        Ok(NormalizeFilter { target_level })
    }

    /// Calculate RMS level of audio samples
    fn calculate_rms(&self, samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = samples
            .iter()
            .map(|&s| (s as f64 / 32768.0).powi(2))
            .sum();
        (sum_squares / samples.len() as f64).sqrt() as f32
    }

    /// Normalize audio frame
    fn normalize_audio(&self, audio_frame: &AudioFrame) -> Result<AudioFrame> {
        let mut output_frame = audio_frame.clone();

        for plane in &mut output_frame.data {
            let samples_bytes = plane.as_slice();
            if samples_bytes.len() % 2 != 0 {
                return Err(Error::filter("Audio data must be i16 PCM (even byte count)"));
            }

            // Convert to i16 samples
            let samples: Vec<i16> = samples_bytes
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            // Calculate current RMS
            let current_rms = self.calculate_rms(&samples);
            if current_rms < 0.001 {
                // Very quiet, don't normalize
                continue;
            }

            // Calculate gain
            let gain = self.target_level / current_rms;

            // Apply gain with soft limiting
            let normalized: Vec<u8> = samples
                .iter()
                .flat_map(|&sample| {
                    let adjusted = (sample as f32 * gain).clamp(-32768.0, 32767.0) as i16;
                    adjusted.to_le_bytes()
                })
                .collect();

            *plane = Buffer::from_vec(normalized);
        }

        Ok(output_frame)
    }
}

impl Filter for NormalizeFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Audio(audio_frame) => {
                let normalized = self.normalize_audio(&audio_frame)?;
                Ok(vec![Frame::Audio(normalized)])
            }
            Frame::Video(_) => Err(Error::filter("Normalize filter only accepts audio frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}
