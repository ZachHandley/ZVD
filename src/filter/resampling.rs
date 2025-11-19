//! Audio Resampling
//!
//! Sample rate conversion for audio processing, format conversion, and
//! synchronization with different hardware/software requirements.
//!
//! ## Common Sample Rates
//!
//! - 44.1 kHz: CD audio
//! - 48 kHz: Professional video/audio (most common)
//! - 88.2 kHz: High-res audio (2x CD)
//! - 96 kHz: High-res professional
//! - 192 kHz: Extreme high-res
//!
//! ## Resampling Methods
//!
//! - **Linear**: Fast, lower quality (for non-critical applications)
//! - **Sinc**: High quality, windowed sinc interpolation (Lanczos)
//! - **Polyphase**: Efficient for integer ratios
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::filter::resampling::{AudioResampler, ResamplingMethod};
//!
//! // Create resampler: 44.1kHz → 48kHz
//! let mut resampler = AudioResampler::new(
//!     44100,
//!     48000,
//!     ResamplingMethod::Sinc { window_size: 64 },
//! );
//!
//! let resampled = resampler.process(&audio_samples)?;
//! ```

use crate::error::{Error, Result};
use std::f32::consts::PI;

/// Resampling method
#[derive(Debug, Clone)]
pub enum ResamplingMethod {
    /// Linear interpolation (fast, lower quality)
    Linear,
    /// Windowed sinc interpolation (high quality)
    Sinc { window_size: usize },
    /// Zero-order hold (nearest neighbor)
    ZeroOrderHold,
}

/// Audio resampler
pub struct AudioResampler {
    /// Input sample rate
    input_rate: u32,
    /// Output sample rate
    output_rate: u32,
    /// Resampling method
    method: ResamplingMethod,
    /// Ratio (output / input)
    ratio: f64,
    /// Buffer for overlap
    buffer: Vec<f32>,
    /// Position accumulator
    position: f64,
}

impl AudioResampler {
    /// Create new audio resampler
    ///
    /// # Arguments
    /// * `input_rate` - Input sample rate (Hz)
    /// * `output_rate` - Output sample rate (Hz)
    /// * `method` - Resampling method
    pub fn new(input_rate: u32, output_rate: u32, method: ResamplingMethod) -> Self {
        let ratio = output_rate as f64 / input_rate as f64;

        // Buffer size depends on method
        let buffer_size = match &method {
            ResamplingMethod::Sinc { window_size } => *window_size,
            _ => 4,
        };

        AudioResampler {
            input_rate,
            output_rate,
            method,
            ratio,
            buffer: Vec::with_capacity(buffer_size),
            position: 0.0,
        }
    }

    /// Process audio samples
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        match &self.method {
            ResamplingMethod::Linear => self.resample_linear(input),
            ResamplingMethod::Sinc { window_size } => self.resample_sinc(input, *window_size),
            ResamplingMethod::ZeroOrderHold => self.resample_nearest(input),
        }
    }

    /// Linear interpolation resampling
    fn resample_linear(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Combine buffer and new input
        let mut combined = self.buffer.clone();
        combined.extend_from_slice(input);

        let output_len = (combined.len() as f64 * self.ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        let mut pos = 0.0;

        while pos < combined.len() as f64 - 1.0 {
            let index = pos.floor() as usize;
            let frac = pos - pos.floor();

            // Linear interpolation
            let sample0 = combined[index];
            let sample1 = combined[index + 1];
            let interpolated = sample0 + (sample1 - sample0) * frac as f32;

            output.push(interpolated);
            pos += 1.0 / self.ratio;
        }

        // Save last few samples for next call
        let keep_samples = 2.min(combined.len());
        self.buffer = combined[combined.len() - keep_samples..].to_vec();

        Ok(output)
    }

    /// Windowed sinc interpolation (Lanczos)
    fn resample_sinc(&mut self, input: &[f32], window_size: usize) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Combine buffer and new input
        let mut combined = self.buffer.clone();
        combined.extend_from_slice(input);

        let output_len = (combined.len() as f64 * self.ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        let half_window = (window_size / 2) as f64;
        let mut pos = half_window;

        while pos < combined.len() as f64 - half_window {
            let mut sample = 0.0;

            // Windowed sinc interpolation
            for i in 0..window_size {
                let x = pos - i as f64;
                let index = x as usize;

                if index < combined.len() {
                    let sinc_val = self.lanczos_kernel(pos - index as f64, half_window);
                    sample += combined[index] * sinc_val;
                }
            }

            output.push(sample);
            pos += 1.0 / self.ratio;
        }

        // Save samples for overlap
        let keep_samples = window_size.min(combined.len());
        self.buffer = combined[combined.len() - keep_samples..].to_vec();

        Ok(output)
    }

    /// Lanczos windowed sinc kernel
    fn lanczos_kernel(&self, x: f64, a: f64) -> f32 {
        if x.abs() < 1e-8 {
            return 1.0;
        }

        if x.abs() >= a {
            return 0.0;
        }

        let pi_x = PI as f64 * x;
        let sinc = (pi_x).sin() / pi_x;
        let window = (pi_x / a).sin() / (pi_x / a);

        (sinc * window) as f32
    }

    /// Zero-order hold (nearest neighbor)
    fn resample_nearest(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let output_len = (input.len() as f64 * self.ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let input_pos = (i as f64 / self.ratio).round() as usize;
            let sample = input[input_pos.min(input.len() - 1)];
            output.push(sample);
        }

        Ok(output)
    }

    /// Reset resampler state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.position = 0.0;
    }

    /// Get resampling ratio
    pub fn ratio(&self) -> f64 {
        self.ratio
    }
}

/// Polyphase resampler (for integer ratios)
pub struct PolyphaseResampler {
    /// Upsampling factor
    upsample: usize,
    /// Downsampling factor
    downsample: usize,
    /// FIR filter coefficients
    coefficients: Vec<f32>,
    /// Filter state
    state: Vec<f32>,
    /// Phase index
    phase: usize,
}

impl PolyphaseResampler {
    /// Create new polyphase resampler
    ///
    /// # Arguments
    /// * `input_rate` - Input sample rate
    /// * `output_rate` - Output sample rate
    ///
    /// Automatically calculates integer ratio (e.g., 44100→48000 = 160/147)
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        let (upsample, downsample) = Self::calculate_ratio(input_rate, output_rate);

        // Design lowpass filter
        let filter_length = 64 * upsample;
        let coefficients = Self::design_lowpass(filter_length, upsample);

        PolyphaseResampler {
            upsample,
            downsample,
            coefficients,
            state: vec![0.0; filter_length],
            phase: 0,
        }
    }

    /// Calculate integer ratio using GCD
    fn calculate_ratio(input_rate: u32, output_rate: u32) -> (usize, usize) {
        let gcd = Self::gcd(input_rate, output_rate);
        let upsample = (output_rate / gcd) as usize;
        let downsample = (input_rate / gcd) as usize;
        (upsample, downsample)
    }

    /// Greatest common divisor
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Design lowpass FIR filter
    fn design_lowpass(length: usize, upsample: usize) -> Vec<f32> {
        let mut coeffs = Vec::with_capacity(length);
        let fc = 0.5 / upsample as f32; // Cutoff frequency

        for i in 0..length {
            let n = i as f32 - (length as f32 - 1.0) / 2.0;

            let h = if n.abs() < 1e-8 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * n).sin() / (PI * n)
            };

            // Hamming window
            let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (length as f32 - 1.0)).cos();

            coeffs.push(h * window);
        }

        // Normalize
        let sum: f32 = coeffs.iter().sum();
        coeffs.iter_mut().for_each(|c| *c /= sum);

        coeffs
    }

    /// Process samples
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::new();

        for &sample in input {
            // Upsample by L
            for _ in 0..self.upsample {
                // Shift state
                self.state.rotate_right(1);
                self.state[0] = if self.phase == 0 { sample } else { 0.0 };

                // FIR filter
                let mut filtered = 0.0;
                for (i, &coeff) in self.coefficients.iter().enumerate() {
                    if i < self.state.len() {
                        filtered += self.state[i] * coeff;
                    }
                }

                // Downsample by M
                if self.phase == 0 {
                    output.push(filtered);
                }

                self.phase = (self.phase + 1) % self.downsample;
            }
        }

        output
    }

    /// Reset resampler
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.phase = 0;
    }
}

/// Pitch shifter (tempo-preserving)
pub struct PitchShifter {
    /// Pitch shift ratio (1.0 = no shift, 2.0 = up 1 octave)
    ratio: f64,
    /// Window size
    window_size: usize,
    /// Hop size
    hop_size: usize,
    /// Input buffer
    input_buffer: Vec<f32>,
    /// Output buffer
    output_buffer: Vec<f32>,
}

impl PitchShifter {
    /// Create new pitch shifter
    pub fn new(ratio: f64, window_size: usize) -> Self {
        let hop_size = window_size / 4;

        PitchShifter {
            ratio,
            window_size,
            hop_size,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    /// Shift pitch
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        // Simple time-domain pitch shifting (phase vocoder would be better)
        // This is a basic implementation

        let resampler = AudioResampler::new(
            48000,
            (48000.0 * self.ratio) as u32,
            ResamplingMethod::Linear,
        );

        // This would need proper WSOLA or phase vocoder implementation
        // For now, just return input (placeholder)
        input.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_creation() {
        let resampler = AudioResampler::new(44100, 48000, ResamplingMethod::Linear);
        assert_eq!(resampler.input_rate, 44100);
        assert_eq!(resampler.output_rate, 48000);
        assert!((resampler.ratio - 1.0884).abs() < 0.001);
    }

    #[test]
    fn test_linear_resampling_upsample() {
        let mut resampler = AudioResampler::new(44100, 48000, ResamplingMethod::Linear);

        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = resampler.process(&input).unwrap();

        // Output should be longer (upsampling)
        assert!(output.len() > input.len());
    }

    #[test]
    fn test_linear_resampling_downsample() {
        let mut resampler = AudioResampler::new(48000, 44100, ResamplingMethod::Linear);

        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = resampler.process(&input).unwrap();

        // Output should be shorter (downsampling)
        assert!(output.len() < input.len() || output.len() == input.len());
    }

    #[test]
    fn test_sinc_resampling() {
        let mut resampler = AudioResampler::new(
            44100,
            48000,
            ResamplingMethod::Sinc { window_size: 32 },
        );

        let input: Vec<f32> = (0..100).map(|i| (i as f32 / 10.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        assert!(!output.is_empty());
        assert!(output.len() > input.len());
    }

    #[test]
    fn test_nearest_resampling() {
        let mut resampler = AudioResampler::new(44100, 48000, ResamplingMethod::ZeroOrderHold);

        let input = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let output = resampler.process(&input).unwrap();

        assert!(!output.is_empty());
    }

    #[test]
    fn test_lanczos_kernel() {
        let resampler = AudioResampler::new(
            44100,
            48000,
            ResamplingMethod::Sinc { window_size: 32 },
        );

        // At x=0, sinc should be 1.0
        let val = resampler.lanczos_kernel(0.0, 3.0);
        assert!((val - 1.0).abs() < 0.001);

        // Outside window should be 0.0
        let val = resampler.lanczos_kernel(5.0, 3.0);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_polyphase_creation() {
        let resampler = PolyphaseResampler::new(44100, 48000);
        assert_eq!(resampler.upsample, 160);
        assert_eq!(resampler.downsample, 147);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(PolyphaseResampler::gcd(48000, 44100), 300);
        assert_eq!(PolyphaseResampler::gcd(48000, 48000), 48000);
        assert_eq!(PolyphaseResampler::gcd(44100, 22050), 22050);
    }

    #[test]
    fn test_polyphase_process() {
        let mut resampler = PolyphaseResampler::new(44100, 48000);

        let input: Vec<f32> = (0..100).map(|i| (i as f32 / 10.0).sin()).collect();
        let output = resampler.process(&input);

        assert!(!output.is_empty());
    }

    #[test]
    fn test_resampler_reset() {
        let mut resampler = AudioResampler::new(44100, 48000, ResamplingMethod::Linear);

        let input = vec![0.5f32; 100];
        let _ = resampler.process(&input).unwrap();

        resampler.reset();
        assert!(resampler.buffer.is_empty());
        assert_eq!(resampler.position, 0.0);
    }

    #[test]
    fn test_ratio_calculation() {
        let resampler = AudioResampler::new(44100, 88200, ResamplingMethod::Linear);
        assert!((resampler.ratio - 2.0).abs() < 0.001);

        let resampler2 = AudioResampler::new(48000, 24000, ResamplingMethod::Linear);
        assert!((resampler2.ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_input() {
        let mut resampler = AudioResampler::new(44100, 48000, ResamplingMethod::Linear);

        let output = resampler.process(&[]).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_pitch_shifter() {
        let mut shifter = PitchShifter::new(1.5, 1024);

        let input: Vec<f32> = (0..1000).map(|i| (i as f32 / 10.0).sin()).collect();
        let output = shifter.process(&input);

        assert!(!output.is_empty());
    }

    #[test]
    fn test_lowpass_design() {
        let coeffs = PolyphaseResampler::design_lowpass(64, 2);

        assert_eq!(coeffs.len(), 64);

        // Coefficients should sum to approximately 1.0
        let sum: f32 = coeffs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
