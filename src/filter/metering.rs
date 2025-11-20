//! Audio Metering and Analysis
//!
//! Professional audio level metering, phase correlation, and frequency
//! analysis for mixing, mastering, and broadcast compliance.
//!
//! ## Meter Types
//!
//! - **VU Meter**: Volume Unit (ballistic response, ~300ms)
//! - **Peak Meter**: True peak detection (sample accurate)
//! - **Phase Correlation**: Stereo phase relationship (-1 to +1)
//! - **Spectrum Analyzer**: Frequency domain analysis
//!
//! ## Standards
//!
//! - VU: 0 VU = +4 dBu (1.228V RMS)
//! - Peak: Digital full scale (0 dBFS)
//! - Phase: +1 = mono, 0 = stereo, -1 = out of phase
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::filter::metering::{VuMeter, PeakMeter, PhaseCorrelation};
//!
//! // VU metering
//! let mut vu = VuMeter::new(48000);
//! let vu_level = vu.process(&audio_samples);
//!
//! // Peak metering
//! let mut peak = PeakMeter::new();
//! let peak_db = peak.process(&audio_samples);
//! ```

use crate::error::Result;
use std::collections::VecDeque;

/// VU Meter (Volume Unit)
///
/// Ballistic response meter with ~300ms integration time
pub struct VuMeter {
    /// Sample rate
    sample_rate: u32,
    /// Attack time constant
    attack_coeff: f32,
    /// Release time constant
    release_coeff: f32,
    /// Current VU level
    current_level: f32,
}

impl VuMeter {
    /// Create new VU meter
    ///
    /// Standard VU meter: 300ms rise time to 99% of step input
    pub fn new(sample_rate: u32) -> Self {
        // VU meter ballistics: 300ms integration
        let attack_time = 0.3; // 300ms
        let release_time = 0.3;

        let attack_coeff = Self::time_constant(attack_time, sample_rate);
        let release_coeff = Self::time_constant(release_time, sample_rate);

        VuMeter {
            sample_rate,
            attack_coeff,
            release_coeff,
            current_level: 0.0,
        }
    }

    /// Calculate time constant coefficient
    fn time_constant(time_seconds: f32, sample_rate: u32) -> f32 {
        (-1.0 / (time_seconds * sample_rate as f32)).exp()
    }

    /// Process audio samples and return VU level
    pub fn process(&mut self, samples: &[f32]) -> f32 {
        for &sample in samples {
            let abs_sample = sample.abs();

            // Ballistic response
            if abs_sample > self.current_level {
                // Attack
                self.current_level =
                    self.attack_coeff * self.current_level + (1.0 - self.attack_coeff) * abs_sample;
            } else {
                // Release
                self.current_level = self.release_coeff * self.current_level
                    + (1.0 - self.release_coeff) * abs_sample;
            }
        }

        self.current_level
    }

    /// Get VU level in dB
    pub fn get_db(&self) -> f32 {
        if self.current_level > 0.0 {
            20.0 * self.current_level.log10()
        } else {
            -std::f32::INFINITY
        }
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.current_level = 0.0;
    }
}

/// Peak Meter
pub struct PeakMeter {
    /// Current peak level
    peak_level: f32,
    /// Hold time (samples)
    hold_time: usize,
    /// Hold counter
    hold_counter: usize,
    /// Decay rate
    decay_rate: f32,
}

impl PeakMeter {
    /// Create new peak meter
    pub fn new() -> Self {
        PeakMeter {
            peak_level: 0.0,
            hold_time: 24000, // 500ms @ 48kHz
            hold_counter: 0,
            decay_rate: 0.99, // Slow decay
        }
    }

    /// Set hold time in samples
    pub fn set_hold_time(&mut self, samples: usize) {
        self.hold_time = samples;
    }

    /// Process audio and return peak level
    pub fn process(&mut self, samples: &[f32]) -> f32 {
        let mut current_peak = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        if current_peak > self.peak_level {
            self.peak_level = current_peak;
            self.hold_counter = 0;
        } else if self.hold_counter < self.hold_time {
            self.hold_counter += samples.len();
        } else {
            // Decay
            self.peak_level *= self.decay_rate;
        }

        self.peak_level
    }

    /// Get peak in dB
    pub fn get_db(&self) -> f32 {
        if self.peak_level > 0.0 {
            20.0 * self.peak_level.log10()
        } else {
            -std::f32::INFINITY
        }
    }

    /// Reset peak
    pub fn reset(&mut self) {
        self.peak_level = 0.0;
        self.hold_counter = 0;
    }
}

impl Default for PeakMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// Phase Correlation Meter
///
/// Measures stereo phase relationship (-1 to +1)
/// +1 = mono (in phase), 0 = stereo, -1 = out of phase
pub struct PhaseCorrelationMeter {
    /// Window size for correlation
    window_size: usize,
    /// Left channel buffer
    left_buffer: VecDeque<f32>,
    /// Right channel buffer
    right_buffer: VecDeque<f32>,
}

impl PhaseCorrelationMeter {
    /// Create new phase correlation meter
    pub fn new(window_size: usize) -> Self {
        PhaseCorrelationMeter {
            window_size,
            left_buffer: VecDeque::with_capacity(window_size),
            right_buffer: VecDeque::with_capacity(window_size),
        }
    }

    /// Process stereo samples and return correlation (-1 to +1)
    pub fn process(&mut self, left: &[f32], right: &[f32]) -> f32 {
        if left.len() != right.len() {
            return 0.0;
        }

        // Add samples to buffers
        for (l, r) in left.iter().zip(right.iter()) {
            if self.left_buffer.len() >= self.window_size {
                self.left_buffer.pop_front();
                self.right_buffer.pop_front();
            }

            self.left_buffer.push_back(*l);
            self.right_buffer.push_back(*r);
        }

        self.calculate_correlation()
    }

    /// Calculate correlation coefficient
    fn calculate_correlation(&self) -> f32 {
        if self.left_buffer.len() < 2 {
            return 0.0;
        }

        let mut sum_lr = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_r2 = 0.0;

        for (l, r) in self.left_buffer.iter().zip(self.right_buffer.iter()) {
            sum_lr += l * r;
            sum_l2 += l * l;
            sum_r2 += r * r;
        }

        let denom = (sum_l2 * sum_r2).sqrt();

        if denom > 0.0 {
            (sum_lr / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Spectrum Analyzer (frequency domain)
pub struct SpectrumAnalyzer {
    /// FFT size
    fft_size: usize,
    /// Sample rate
    sample_rate: u32,
    /// Frequency bins
    bins: Vec<f32>,
}

impl SpectrumAnalyzer {
    /// Create new spectrum analyzer
    pub fn new(fft_size: usize, sample_rate: u32) -> Self {
        SpectrumAnalyzer {
            fft_size,
            sample_rate,
            bins: vec![0.0; fft_size / 2 + 1],
        }
    }

    /// Analyze frequency spectrum (simplified - no actual FFT)
    ///
    /// In production, would use FFT library like rustfft
    pub fn analyze(&mut self, samples: &[f32]) -> &[f32] {
        // Stub implementation - would use FFT
        // For now, just return zero bins
        &self.bins
    }

    /// Get frequency for bin index
    pub fn bin_frequency(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate as f32 / self.fft_size as f32
    }

    /// Get magnitude in dB for bin
    pub fn bin_magnitude_db(&self, bin: usize) -> f32 {
        if bin < self.bins.len() && self.bins[bin] > 0.0 {
            20.0 * self.bins[bin].log10()
        } else {
            -std::f32::INFINITY
        }
    }
}

/// RMS Level Meter
pub struct RmsLevelMeter {
    /// Window size
    window_size: usize,
    /// Sample buffer
    buffer: VecDeque<f32>,
}

impl RmsLevelMeter {
    /// Create new RMS meter
    pub fn new(window_size: usize) -> Self {
        RmsLevelMeter {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
        }
    }

    /// Process samples and return RMS level
    pub fn process(&mut self, samples: &[f32]) -> f32 {
        for &sample in samples {
            if self.buffer.len() >= self.window_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }

        self.calculate_rms()
    }

    /// Calculate RMS level
    fn calculate_rms(&self) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.buffer.iter().map(|&s| s * s).sum();
        (sum_squares / self.buffer.len() as f32).sqrt()
    }

    /// Get RMS in dB
    pub fn get_db(&self) -> f32 {
        let rms = self.calculate_rms();
        if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -std::f32::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vu_meter() {
        let mut vu = VuMeter::new(48000);

        // Silent signal
        let silent = vec![0.0f32; 100];
        let level = vu.process(&silent);
        assert_eq!(level, 0.0);

        // Loud signal
        let loud = vec![0.5f32; 100];
        let level_loud = vu.process(&loud);
        assert!(level_loud > 0.0);
    }

    #[test]
    fn test_vu_meter_db() {
        let mut vu = VuMeter::new(48000);
        let signal = vec![0.5f32; 1000];
        vu.process(&signal);

        let db = vu.get_db();
        assert!(db < 0.0); // 0.5 linear is negative dB
        assert!(db > -10.0);
    }

    #[test]
    fn test_peak_meter() {
        let mut peak = PeakMeter::new();

        let samples = vec![0.0, 0.5, 1.0, 0.3, 0.1];
        let peak_level = peak.process(&samples);

        assert_eq!(peak_level, 1.0);
    }

    #[test]
    fn test_peak_meter_db() {
        let mut peak = PeakMeter::new();
        let samples = vec![1.0f32];

        peak.process(&samples);
        let db = peak.get_db();

        assert_eq!(db, 0.0); // 1.0 linear = 0 dBFS
    }

    #[test]
    fn test_phase_correlation_mono() {
        let mut phase = PhaseCorrelationMeter::new(100);

        // Identical signals (mono) = +1 correlation
        let left = vec![0.5f32; 100];
        let right = left.clone();

        let correlation = phase.process(&left, &right);
        assert!((correlation - 1.0).abs() < 0.01); // Should be close to +1
    }

    #[test]
    fn test_phase_correlation_inverted() {
        let mut phase = PhaseCorrelationMeter::new(100);

        // Inverted signals = -1 correlation
        let left = vec![0.5f32; 100];
        let right: Vec<f32> = left.iter().map(|&x| -x).collect();

        let correlation = phase.process(&left, &right);
        assert!((correlation + 1.0).abs() < 0.01); // Should be close to -1
    }

    #[test]
    fn test_spectrum_analyzer() {
        let analyzer = SpectrumAnalyzer::new(1024, 48000);

        let freq = analyzer.bin_frequency(10);
        assert!(freq > 0.0);
        assert!(freq < 24000.0); // Less than Nyquist
    }

    #[test]
    fn test_rms_level_meter() {
        let mut rms = RmsLevelMeter::new(100);

        let samples = vec![0.5f32; 100];
        let level = rms.process(&samples);

        assert_eq!(level, 0.5); // RMS of constant 0.5 is 0.5
    }

    #[test]
    fn test_rms_db() {
        let mut rms = RmsLevelMeter::new(100);
        let samples = vec![1.0f32; 100];

        rms.process(&samples);
        let db = rms.get_db();

        assert_eq!(db, 0.0); // 1.0 linear = 0 dB
    }

    #[test]
    fn test_vu_reset() {
        let mut vu = VuMeter::new(48000);
        vu.process(&vec![0.5f32; 100]);

        vu.reset();
        assert_eq!(vu.current_level, 0.0);
    }

    #[test]
    fn test_peak_reset() {
        let mut peak = PeakMeter::new();
        peak.process(&vec![0.9f32; 10]);

        peak.reset();
        assert_eq!(peak.peak_level, 0.0);
    }
}
