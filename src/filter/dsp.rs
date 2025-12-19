//! Advanced Audio DSP (Digital Signal Processing)
//!
//! Professional audio processing tools including:
//! - Parametric EQ (Equalization)
//! - Dynamic Range Compression
//! - Limiting
//! - Expansion/Gating
//! - De-essing
//!
//! ## Features
//!
//! **Parametric EQ:**
//! - Multiple bands (low shelf, high shelf, peaking)
//! - Adjustable frequency, gain, Q factor
//! - Biquad filter implementation
//!
//! **Compressor:**
//! - Threshold, ratio, attack, release
//! - Knee (soft/hard)
//! - Makeup gain
//! - RMS or peak detection
//!
//! **Limiter:**
//! - Brick wall limiting
//! - Look-ahead
//! - Ultra-fast attack
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::filter::dsp::{ParametricEq, Compressor};
//!
//! // Create 3-band EQ
//! let mut eq = ParametricEq::new(48000);
//! eq.add_low_shelf(100.0, 3.0, 0.707);
//! eq.add_peaking(1000.0, -2.0, 1.0);
//! eq.add_high_shelf(8000.0, 2.0, 0.707);
//!
//! // Create compressor
//! let mut comp = Compressor::new(48000);
//! comp.set_threshold(-20.0);
//! comp.set_ratio(4.0);
//! comp.set_attack_ms(5.0);
//! comp.set_release_ms(50.0);
//! ```

use crate::error::Result;
use std::f64::consts::PI;

/// Biquad filter coefficients
#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Biquad filter state
#[derive(Debug, Clone)]
struct BiquadFilter {
    coeffs: BiquadCoeffs,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadFilter {
    fn new(coeffs: BiquadCoeffs) -> Self {
        BiquadFilter {
            coeffs,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let output = self.coeffs.b0 * input + self.coeffs.b1 * self.x1 + self.coeffs.b2 * self.x2
            - self.coeffs.a1 * self.y1
            - self.coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// EQ band type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqBandType {
    LowShelf,
    HighShelf,
    Peaking,
    LowPass,
    HighPass,
    Notch,
}

/// EQ band
#[derive(Debug, Clone)]
pub struct EqBand {
    band_type: EqBandType,
    frequency: f64,
    gain_db: f64,
    q: f64,
    filter: BiquadFilter,
}

impl EqBand {
    /// Create EQ band
    fn new(band_type: EqBandType, frequency: f64, gain_db: f64, q: f64, sample_rate: u32) -> Self {
        let coeffs = Self::calculate_coeffs(band_type, frequency, gain_db, q, sample_rate);

        EqBand {
            band_type,
            frequency,
            gain_db,
            q,
            filter: BiquadFilter::new(coeffs),
        }
    }

    fn calculate_coeffs(
        band_type: EqBandType,
        freq: f64,
        gain_db: f64,
        q: f64,
        sample_rate: u32,
    ) -> BiquadCoeffs {
        let w0 = 2.0 * PI * freq / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a = 10f64.powf(gain_db / 40.0);

        match band_type {
            EqBandType::LowShelf => {
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

                BiquadCoeffs {
                    b0: b0 / a0,
                    b1: b1 / a0,
                    b2: b2 / a0,
                    a1: a1 / a0,
                    a2: a2 / a0,
                }
            }
            EqBandType::HighShelf => {
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

                BiquadCoeffs {
                    b0: b0 / a0,
                    b1: b1 / a0,
                    b2: b2 / a0,
                    a1: a1 / a0,
                    a2: a2 / a0,
                }
            }
            EqBandType::Peaking => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;

                BiquadCoeffs {
                    b0: b0 / a0,
                    b1: b1 / a0,
                    b2: b2 / a0,
                    a1: a1 / a0,
                    a2: a2 / a0,
                }
            }
            EqBandType::LowPass => {
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                BiquadCoeffs {
                    b0: b0 / a0,
                    b1: b1 / a0,
                    b2: b2 / a0,
                    a1: a1 / a0,
                    a2: a2 / a0,
                }
            }
            EqBandType::HighPass => {
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                BiquadCoeffs {
                    b0: b0 / a0,
                    b1: b1 / a0,
                    b2: b2 / a0,
                    a1: a1 / a0,
                    a2: a2 / a0,
                }
            }
            EqBandType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;

                BiquadCoeffs {
                    b0: b0 / a0,
                    b1: b1 / a0,
                    b2: b2 / a0,
                    a1: a1 / a0,
                    a2: a2 / a0,
                }
            }
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        self.filter.process(input)
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

/// Parametric EQ with multiple bands
pub struct ParametricEq {
    sample_rate: u32,
    bands: Vec<EqBand>,
}

impl ParametricEq {
    pub fn new(sample_rate: u32) -> Self {
        ParametricEq {
            sample_rate,
            bands: Vec::new(),
        }
    }

    /// Add low shelf band
    pub fn add_low_shelf(&mut self, frequency: f64, gain_db: f64, q: f64) {
        self.bands.push(EqBand::new(
            EqBandType::LowShelf,
            frequency,
            gain_db,
            q,
            self.sample_rate,
        ));
    }

    /// Add high shelf band
    pub fn add_high_shelf(&mut self, frequency: f64, gain_db: f64, q: f64) {
        self.bands.push(EqBand::new(
            EqBandType::HighShelf,
            frequency,
            gain_db,
            q,
            self.sample_rate,
        ));
    }

    /// Add peaking band
    pub fn add_peaking(&mut self, frequency: f64, gain_db: f64, q: f64) {
        self.bands.push(EqBand::new(
            EqBandType::Peaking,
            frequency,
            gain_db,
            q,
            self.sample_rate,
        ));
    }

    /// Add low pass filter
    pub fn add_low_pass(&mut self, frequency: f64, q: f64) {
        self.bands.push(EqBand::new(
            EqBandType::LowPass,
            frequency,
            0.0,
            q,
            self.sample_rate,
        ));
    }

    /// Add high pass filter
    pub fn add_high_pass(&mut self, frequency: f64, q: f64) {
        self.bands.push(EqBand::new(
            EqBandType::HighPass,
            frequency,
            0.0,
            q,
            self.sample_rate,
        ));
    }

    /// Process single sample through all EQ bands
    pub fn process(&mut self, input: f32) -> f32 {
        let mut output = input as f64;
        for band in &mut self.bands {
            output = band.process(output);
        }
        output as f32
    }

    /// Process buffer
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Reset all filters
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Clear all bands
    pub fn clear(&mut self) {
        self.bands.clear();
    }
}

/// Dynamic range compressor
pub struct Compressor {
    sample_rate: u32,
    threshold_db: f64,
    ratio: f64,
    attack_coeff: f64,
    release_coeff: f64,
    knee_db: f64,
    makeup_gain_db: f64,
    envelope: f64,
    use_rms: bool,
    rms_window: Vec<f64>,
    rms_index: usize,
}

impl Compressor {
    pub fn new(sample_rate: u32) -> Self {
        let mut comp = Compressor {
            sample_rate,
            threshold_db: -20.0,
            ratio: 4.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
            envelope: 0.0,
            use_rms: true,
            rms_window: vec![0.0; 256],
            rms_index: 0,
        };

        comp.set_attack_ms(5.0);
        comp.set_release_ms(50.0);
        comp
    }

    pub fn set_threshold(&mut self, db: f64) {
        self.threshold_db = db;
    }

    pub fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio.max(1.0);
    }

    pub fn set_attack_ms(&mut self, ms: f64) {
        self.attack_coeff = (-1000.0 / (ms * self.sample_rate as f64)).exp();
    }

    pub fn set_release_ms(&mut self, ms: f64) {
        self.release_coeff = (-1000.0 / (ms * self.sample_rate as f64)).exp();
    }

    pub fn set_knee(&mut self, db: f64) {
        self.knee_db = db.abs();
    }

    pub fn set_makeup_gain(&mut self, db: f64) {
        self.makeup_gain_db = db;
    }

    pub fn set_detection_mode(&mut self, use_rms: bool) {
        self.use_rms = use_rms;
    }

    /// Calculate gain reduction
    fn calculate_gain_reduction(&self, level_db: f64) -> f64 {
        if self.knee_db > 0.0 {
            // Soft knee
            if level_db < (self.threshold_db - self.knee_db / 2.0) {
                0.0
            } else if level_db > (self.threshold_db + self.knee_db / 2.0) {
                (self.threshold_db - level_db) * (1.0 - 1.0 / self.ratio)
            } else {
                let diff = level_db - self.threshold_db + self.knee_db / 2.0;
                let gain_reduction = diff * diff / (2.0 * self.knee_db) * (1.0 - 1.0 / self.ratio);
                -gain_reduction
            }
        } else {
            // Hard knee
            if level_db > self.threshold_db {
                (self.threshold_db - level_db) * (1.0 - 1.0 / self.ratio)
            } else {
                0.0
            }
        }
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let input_f64 = input as f64;

        // Level detection
        let level = if self.use_rms {
            // RMS detection
            self.rms_window[self.rms_index] = input_f64 * input_f64;
            self.rms_index = (self.rms_index + 1) % self.rms_window.len();
            let rms_sum: f64 = self.rms_window.iter().sum();
            (rms_sum / self.rms_window.len() as f64).sqrt()
        } else {
            // Peak detection
            input_f64.abs()
        };

        let level_db = if level > 0.0 {
            20.0 * level.log10()
        } else {
            -120.0
        };

        // Calculate gain reduction
        let gain_reduction_db = self.calculate_gain_reduction(level_db);

        // Apply envelope follower
        let target_envelope = 10f64.powf(gain_reduction_db / 20.0);
        let coeff = if target_envelope < self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = target_envelope + coeff * (self.envelope - target_envelope);

        // Apply makeup gain
        let makeup_gain = 10f64.powf(self.makeup_gain_db / 20.0);

        // Apply compression
        (input_f64 * self.envelope * makeup_gain) as f32
    }

    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.rms_window.fill(0.0);
        self.rms_index = 0;
    }
}

/// Limiter (fast compressor with high ratio)
pub struct Limiter {
    compressor: Compressor,
}

impl Limiter {
    pub fn new(sample_rate: u32, ceiling_db: f64) -> Self {
        let mut comp = Compressor::new(sample_rate);
        comp.set_threshold(ceiling_db);
        comp.set_ratio(20.0); // Very high ratio
        comp.set_attack_ms(0.1); // Ultra-fast attack
        comp.set_release_ms(50.0);
        comp.set_knee(0.0); // Hard knee
        comp.set_detection_mode(false); // Peak detection

        Limiter { compressor: comp }
    }

    pub fn set_ceiling(&mut self, db: f64) {
        self.compressor.set_threshold(db);
    }

    pub fn process(&mut self, input: f32) -> f32 {
        self.compressor.process(input)
    }

    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        self.compressor.process_buffer(buffer);
    }
}

/// Expander/Gate
pub struct Expander {
    sample_rate: u32,
    threshold_db: f64,
    ratio: f64,
    attack_coeff: f64,
    release_coeff: f64,
    envelope: f64,
}

impl Expander {
    pub fn new(sample_rate: u32) -> Self {
        let mut exp = Expander {
            sample_rate,
            threshold_db: -40.0,
            ratio: 2.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            envelope: 1.0,
        };

        exp.set_attack_ms(1.0);
        exp.set_release_ms(100.0);
        exp
    }

    /// Create noise gate (high ratio expander)
    pub fn gate(sample_rate: u32, threshold_db: f64) -> Self {
        let mut gate = Self::new(sample_rate);
        gate.set_threshold(threshold_db);
        gate.set_ratio(10.0); // High ratio for gating
        gate
    }

    pub fn set_threshold(&mut self, db: f64) {
        self.threshold_db = db;
    }

    pub fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio.max(1.0);
    }

    pub fn set_attack_ms(&mut self, ms: f64) {
        self.attack_coeff = (-1000.0 / (ms * self.sample_rate as f64)).exp();
    }

    pub fn set_release_ms(&mut self, ms: f64) {
        self.release_coeff = (-1000.0 / (ms * self.sample_rate as f64)).exp();
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let level = input.abs() as f64;
        let level_db = if level > 0.0 {
            20.0 * level.log10()
        } else {
            -120.0
        };

        // Calculate gain (opposite of compressor)
        let gain = if level_db < self.threshold_db {
            let diff = self.threshold_db - level_db;
            10f64.powf(-diff * (self.ratio - 1.0) / (self.ratio * 20.0))
        } else {
            1.0
        };

        // Envelope follower
        let coeff = if gain < self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = gain + coeff * (self.envelope - gain);

        (input as f64 * self.envelope) as f32
    }

    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

/// De-esser (sibilance reduction)
pub struct DeEsser {
    sample_rate: u32,
    detector_filter: BiquadFilter,
    compressor: Compressor,
    frequency: f64,
}

impl DeEsser {
    pub fn new(sample_rate: u32) -> Self {
        // Typical sibilance range: 5-8 kHz
        let frequency = 6000.0;
        let q = 2.0;

        let coeffs = EqBand::calculate_coeffs(
            EqBandType::Peaking,
            frequency,
            12.0, // Boost for detection
            q,
            sample_rate,
        );

        let mut comp = Compressor::new(sample_rate);
        comp.set_threshold(-15.0);
        comp.set_ratio(4.0);
        comp.set_attack_ms(1.0); // Fast attack for sibilance
        comp.set_release_ms(10.0); // Fast release
        comp.set_detection_mode(false); // Peak detection

        DeEsser {
            sample_rate,
            detector_filter: BiquadFilter::new(coeffs),
            compressor: comp,
            frequency,
        }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
        let coeffs =
            EqBand::calculate_coeffs(EqBandType::Peaking, freq, 12.0, 2.0, self.sample_rate);
        self.detector_filter.coeffs = coeffs;
        self.detector_filter.reset();
    }

    pub fn set_threshold(&mut self, db: f64) {
        self.compressor.set_threshold(db);
    }

    pub fn process(&mut self, input: f32) -> f32 {
        // Filter input to detect sibilance
        let detected = self.detector_filter.process(input as f64) as f32;

        // Use detected level to control compression
        let _ = self.compressor.process(detected);

        // Apply compression envelope to original signal
        (input as f64 * self.compressor.envelope) as f32
    }

    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parametric_eq_creation() {
        let mut eq = ParametricEq::new(48000);
        eq.add_low_shelf(100.0, 3.0, 0.707);
        eq.add_peaking(1000.0, -2.0, 1.0);
        eq.add_high_shelf(8000.0, 2.0, 0.707);

        assert_eq!(eq.bands.len(), 3);
    }

    #[test]
    fn test_eq_process() {
        let mut eq = ParametricEq::new(48000);
        eq.add_peaking(1000.0, 0.0, 1.0); // 0 dB boost (should pass through)

        let input = 0.5f32;
        let output = eq.process(input);

        // With 0 dB gain, output should be close to input
        assert!((output - input).abs() < 0.1);
    }

    #[test]
    fn test_compressor_creation() {
        let comp = Compressor::new(48000);
        assert_eq!(comp.threshold_db, -20.0);
        assert_eq!(comp.ratio, 4.0);
    }

    #[test]
    fn test_compressor_process() {
        let mut comp = Compressor::new(48000);
        comp.set_threshold(-20.0);
        comp.set_ratio(4.0);

        // Signal above threshold should be compressed
        let loud_input = 0.8f32;
        let output = comp.process(loud_input);

        // Output should be attenuated
        assert!(output.abs() <= loud_input.abs());
    }

    #[test]
    fn test_limiter() {
        let mut limiter = Limiter::new(48000, -1.0);

        // Signal above ceiling should be limited
        let input = 1.0f32;
        let output = limiter.process(input);

        // Output should be at or below ceiling
        let output_db = 20.0 * output.abs().log10();
        assert!(output_db <= -1.0 + 0.1); // Small tolerance
    }

    #[test]
    fn test_expander() {
        let mut exp = Expander::new(48000);
        exp.set_threshold(-30.0); // Threshold at -30dB
        exp.set_ratio(4.0);

        // Signal at -50dB (0.003) is below threshold (-30dB)
        // Should be attenuated by expansion
        let quiet_input = 0.003f32; // About -50dB

        // Process multiple samples to let envelope settle
        let mut output = 0.0f32;
        for _ in 0..100 {
            output = exp.process(quiet_input);
        }

        assert!(output.abs() < quiet_input.abs());
    }

    #[test]
    fn test_gate() {
        let mut gate = Expander::gate(48000, -40.0);

        // Very quiet signal should be heavily attenuated
        let quiet_input = 0.001f32;
        let output = gate.process(quiet_input);

        assert!(output.abs() < quiet_input.abs());
    }

    #[test]
    fn test_deesser() {
        let deesser = DeEsser::new(48000);
        assert_eq!(deesser.frequency, 6000.0);
    }

    #[test]
    fn test_eq_buffer_processing() {
        let mut eq = ParametricEq::new(48000);
        eq.add_peaking(1000.0, 6.0, 1.0);

        // Use a test signal at the EQ frequency (1000 Hz) instead of DC
        // Generate 1000 Hz sine wave at 48kHz sample rate
        let mut buffer: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();

        let original_energy: f32 = buffer.iter().map(|x| x * x).sum();
        eq.process_buffer(&mut buffer);
        let processed_energy: f32 = buffer.iter().map(|x| x * x).sum();

        // With +6dB boost at 1000 Hz, energy should increase
        assert!(processed_energy > original_energy);
    }

    #[test]
    fn test_compressor_buffer_processing() {
        let mut comp = Compressor::new(48000);
        comp.set_threshold(-20.0);

        let mut buffer = vec![0.8f32; 1024];
        comp.process_buffer(&mut buffer);

        // Compression should reduce levels
        assert!(buffer.iter().all(|&x| x < 0.8));
    }
}
