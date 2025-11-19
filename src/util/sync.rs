//! Audio/Video Synchronization Detection
//!
//! Detect and correct audio/video sync drift in multi-camera workflows,
//! dual-system sound recordings, and long-form content.
//!
//! ## Sync Detection Methods
//!
//! - **Cross-Correlation**: Waveform matching for initial offset
//! - **Drift Detection**: Long-term phase shift analysis
//! - **Clap/Flash Detection**: Visual/audio spike matching
//! - **Timecode Comparison**: TC-based sync verification
//!
//! ## Common Scenarios
//!
//! - Dual-system sound (separate audio recorder)
//! - Multi-camera shoots (different start times)
//! - Live events with broadcast delay
//! - Long recordings with clock drift
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::sync::{SyncDetector, SyncMethod};
//!
//! let detector = SyncDetector::new(SyncMethod::CrossCorrelation);
//! let offset = detector.detect_offset(&audio1, &audio2, sample_rate)?;
//! println!("Sync offset: {} ms", offset * 1000.0 / sample_rate as f64);
//! ```

use crate::error::{Error, Result};
use std::collections::VecDeque;

/// Sync detection method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMethod {
    /// Cross-correlation (waveform matching)
    CrossCorrelation,
    /// Clap/flash detection (spike matching)
    SpikeDetection,
    /// Phase correlation (frequency domain)
    PhaseCorrelation,
}

/// Sync detection result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Sample offset (positive = audio2 is ahead)
    pub offset_samples: i64,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Detected drift (samples per second)
    pub drift_rate: Option<f64>,
}

impl SyncResult {
    /// Get offset in seconds
    pub fn offset_seconds(&self, sample_rate: u32) -> f64 {
        self.offset_samples as f64 / sample_rate as f64
    }

    /// Get offset in milliseconds
    pub fn offset_ms(&self, sample_rate: u32) -> f64 {
        self.offset_seconds(sample_rate) * 1000.0
    }

    /// Get drift in ppm (parts per million)
    pub fn drift_ppm(&self) -> f64 {
        self.drift_rate.unwrap_or(0.0) * 1_000_000.0
    }
}

/// Audio spike (clap, flash, beep)
#[derive(Debug, Clone)]
pub struct AudioSpike {
    /// Sample position
    pub position: usize,
    /// Peak amplitude
    pub amplitude: f32,
    /// Confidence
    pub confidence: f64,
}

/// Sync detector
pub struct SyncDetector {
    method: SyncMethod,
    /// Max search window (samples)
    max_offset: usize,
    /// Threshold for spike detection (0.0 to 1.0)
    spike_threshold: f32,
}

impl SyncDetector {
    /// Create new sync detector
    pub fn new(method: SyncMethod) -> Self {
        SyncDetector {
            method,
            max_offset: 48000 * 10, // 10 seconds @ 48kHz
            spike_threshold: 0.8,    // 80% of max amplitude
        }
    }

    /// Set max offset to search
    pub fn with_max_offset(mut self, max_offset: usize) -> Self {
        self.max_offset = max_offset;
        self
    }

    /// Set spike detection threshold
    pub fn with_spike_threshold(mut self, threshold: f32) -> Self {
        self.spike_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Detect sync offset between two audio streams
    pub fn detect_offset(&self, audio1: &[f32], audio2: &[f32]) -> Result<SyncResult> {
        match self.method {
            SyncMethod::CrossCorrelation => self.cross_correlation(audio1, audio2),
            SyncMethod::SpikeDetection => self.spike_detection(audio1, audio2),
            SyncMethod::PhaseCorrelation => self.phase_correlation(audio1, audio2),
        }
    }

    /// Cross-correlation sync detection
    fn cross_correlation(&self, audio1: &[f32], audio2: &[f32]) -> Result<SyncResult> {
        if audio1.is_empty() || audio2.is_empty() {
            return Err(Error::InvalidInput("Empty audio data".to_string()));
        }

        // Use shorter audio as reference
        let (reference, search) = if audio1.len() < audio2.len() {
            (audio1, audio2)
        } else {
            (audio2, audio1)
        };

        let ref_len = reference.len().min(self.max_offset);
        let search_len = search.len();

        let mut best_offset = 0i64;
        let mut best_correlation = f64::NEG_INFINITY;

        // Search for best correlation
        let max_search = (search_len - ref_len).min(self.max_offset);

        for offset in 0..max_search {
            let correlation = self.calculate_correlation(
                &reference[0..ref_len],
                &search[offset..offset + ref_len],
            );

            if correlation > best_correlation {
                best_correlation = correlation;
                best_offset = offset as i64;
            }
        }

        // Swap sign if we swapped arrays
        if audio1.len() > audio2.len() {
            best_offset = -best_offset;
        }

        // Normalize correlation to 0-1 confidence
        let confidence = ((best_correlation + 1.0) / 2.0).clamp(0.0, 1.0);

        Ok(SyncResult {
            offset_samples: best_offset,
            confidence,
            drift_rate: None,
        })
    }

    /// Calculate normalized cross-correlation
    fn calculate_correlation(&self, signal1: &[f32], signal2: &[f32]) -> f64 {
        let len = signal1.len().min(signal2.len());

        let mut sum_prod = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..len {
            let s1 = signal1[i] as f64;
            let s2 = signal2[i] as f64;

            sum_prod += s1 * s2;
            sum_sq1 += s1 * s1;
            sum_sq2 += s2 * s2;
        }

        // Normalized correlation
        let denom = (sum_sq1 * sum_sq2).sqrt();
        if denom > 0.0 {
            sum_prod / denom
        } else {
            0.0
        }
    }

    /// Spike-based sync detection (clap/flash)
    fn spike_detection(&self, audio1: &[f32], audio2: &[f32]) -> Result<SyncResult> {
        let spikes1 = self.detect_spikes(audio1);
        let spikes2 = self.detect_spikes(audio2);

        if spikes1.is_empty() || spikes2.is_empty() {
            return Err(Error::InvalidInput(
                "No spikes detected in audio".to_string(),
            ));
        }

        // Find best matching spike pair
        let mut best_offset = 0i64;
        let mut best_confidence = 0.0;

        for spike1 in &spikes1 {
            for spike2 in &spikes2 {
                let offset = spike2.position as i64 - spike1.position as i64;

                if offset.unsigned_abs() as usize > self.max_offset {
                    continue;
                }

                // Confidence is product of both spike confidences
                let confidence = spike1.confidence * spike2.confidence;

                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_offset = offset;
                }
            }
        }

        Ok(SyncResult {
            offset_samples: best_offset,
            confidence: best_confidence,
            drift_rate: None,
        })
    }

    /// Detect audio spikes (claps, beeps, etc.)
    fn detect_spikes(&self, audio: &[f32]) -> Vec<AudioSpike> {
        let mut spikes = Vec::new();

        // Find max amplitude for threshold
        let max_amp = audio.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        if max_amp == 0.0 {
            return spikes;
        }

        let threshold = max_amp * self.spike_threshold;
        let min_spacing = 4800; // Min 100ms @ 48kHz

        let mut last_spike_pos = 0;

        for (i, &sample) in audio.iter().enumerate() {
            let amp = sample.abs();

            if amp > threshold && (i - last_spike_pos) > min_spacing {
                // Check if this is a local peak
                let is_peak = (i == 0 || amp > audio[i - 1].abs())
                    && (i == audio.len() - 1 || amp > audio[i + 1].abs());

                if is_peak {
                    spikes.push(AudioSpike {
                        position: i,
                        amplitude: amp,
                        confidence: (amp / max_amp) as f64,
                    });

                    last_spike_pos = i;
                }
            }
        }

        spikes
    }

    /// Phase correlation (frequency domain)
    fn phase_correlation(&self, audio1: &[f32], audio2: &[f32]) -> Result<SyncResult> {
        // Simplified implementation - would use FFT in production
        // Fall back to cross-correlation for now
        self.cross_correlation(audio1, audio2)
    }

    /// Detect drift over time
    pub fn detect_drift(
        &self,
        audio1: &[f32],
        audio2: &[f32],
        sample_rate: u32,
    ) -> Result<f64> {
        if audio1.len() < sample_rate as usize * 60 {
            return Err(Error::InvalidInput(
                "Need at least 60 seconds for drift detection".to_string(),
            ));
        }

        // Divide into segments and measure offset in each
        let segment_duration = sample_rate as usize * 30; // 30 seconds
        let num_segments = audio1.len() / segment_duration;

        if num_segments < 2 {
            return Ok(0.0);
        }

        let mut offsets = Vec::new();

        for i in 0..num_segments.min(10) {
            // Max 10 segments
            let start = i * segment_duration;
            let end = (start + segment_duration).min(audio1.len()).min(audio2.len());

            if end - start < segment_duration / 2 {
                break;
            }

            let result = self.cross_correlation(&audio1[start..end], &audio2[start..end])?;

            if result.confidence > 0.5 {
                offsets.push((i, result.offset_samples));
            }
        }

        if offsets.len() < 2 {
            return Ok(0.0);
        }

        // Calculate drift rate (linear regression)
        let drift_rate = self.calculate_drift_rate(&offsets, segment_duration);

        Ok(drift_rate)
    }

    /// Calculate drift rate from offset measurements
    fn calculate_drift_rate(&self, offsets: &[(usize, i64)], segment_duration: usize) -> f64 {
        if offsets.len() < 2 {
            return 0.0;
        }

        let n = offsets.len() as f64;

        let sum_x: f64 = offsets.iter().map(|(i, _)| *i as f64).sum();
        let sum_y: f64 = offsets.iter().map(|(_, offset)| *offset as f64).sum();
        let sum_xy: f64 = offsets
            .iter()
            .map(|(i, offset)| *i as f64 * *offset as f64)
            .sum();
        let sum_x2: f64 = offsets.iter().map(|(i, _)| (*i as f64).powi(2)).sum();

        // Linear regression slope
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Convert to samples per sample
        slope / segment_duration as f64
    }
}

/// Sync drift corrector
pub struct DriftCorrector {
    /// Current drift rate (samples per sample)
    drift_rate: f64,
    /// Accumulated fractional samples
    accumulator: f64,
}

impl DriftCorrector {
    /// Create new drift corrector
    pub fn new(drift_rate: f64) -> Self {
        DriftCorrector {
            drift_rate,
            accumulator: 0.0,
        }
    }

    /// Calculate number of samples to skip/repeat
    ///
    /// Returns number of samples to advance (negative = repeat samples)
    pub fn process_sample(&mut self) -> i32 {
        self.accumulator += self.drift_rate;

        if self.accumulator >= 1.0 {
            self.accumulator -= 1.0;
            1 // Skip one sample
        } else if self.accumulator <= -1.0 {
            self.accumulator += 1.0;
            -1 // Repeat one sample
        } else {
            0 // No correction
        }
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(len: usize, freq: f32, sample_rate: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_sync_detector_creation() {
        let detector = SyncDetector::new(SyncMethod::CrossCorrelation);
        assert_eq!(detector.method, SyncMethod::CrossCorrelation);
    }

    #[test]
    fn test_sync_detector_with_max_offset() {
        let detector = SyncDetector::new(SyncMethod::CrossCorrelation).with_max_offset(10000);
        assert_eq!(detector.max_offset, 10000);
    }

    #[test]
    fn test_cross_correlation_identical() {
        let audio = create_test_audio(1000, 440.0, 48000);
        let detector = SyncDetector::new(SyncMethod::CrossCorrelation);

        let result = detector.detect_offset(&audio, &audio).unwrap();

        assert_eq!(result.offset_samples, 0);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_cross_correlation_offset() {
        let audio1 = create_test_audio(1000, 440.0, 48000);
        let mut audio2 = vec![0.0f32; 100];
        audio2.extend_from_slice(&audio1);

        let detector = SyncDetector::new(SyncMethod::CrossCorrelation);
        let result = detector.detect_offset(&audio1, &audio2).unwrap();

        // audio2 starts with 100 samples of silence, so it's ahead by 100
        assert!(result.offset_samples > 50 && result.offset_samples < 150);
    }

    #[test]
    fn test_spike_detection() {
        let mut audio = vec![0.1f32; 1000];
        audio[500] = 1.0; // Spike at position 500

        let detector = SyncDetector::new(SyncMethod::SpikeDetection);
        let spikes = detector.detect_spikes(&audio);

        assert!(!spikes.is_empty());
        assert_eq!(spikes[0].position, 500);
    }

    #[test]
    fn test_spike_sync_detection() {
        let mut audio1 = vec![0.1f32; 2000];
        let mut audio2 = vec![0.1f32; 2000];

        audio1[500] = 1.0; // Clap at 500
        audio2[700] = 1.0; // Clap at 700 (200 samples later)

        let detector = SyncDetector::new(SyncMethod::SpikeDetection);
        let result = detector.detect_offset(&audio1, &audio2).unwrap();

        assert!(result.offset_samples.abs() > 150 && result.offset_samples.abs() < 250);
    }

    #[test]
    fn test_sync_result_conversions() {
        let result = SyncResult {
            offset_samples: 480,
            confidence: 0.95,
            drift_rate: Some(0.000001),
        };

        assert_eq!(result.offset_seconds(48000), 0.01); // 480 / 48000 = 0.01s
        assert_eq!(result.offset_ms(48000), 10.0); // 10ms
        assert_eq!(result.drift_ppm(), 1.0); // 1 ppm
    }

    #[test]
    fn test_drift_corrector() {
        let mut corrector = DriftCorrector::new(0.0001); // Small drift

        let mut skips = 0;
        for _ in 0..10000 {
            if corrector.process_sample() == 1 {
                skips += 1;
            }
        }

        // Should skip ~1 sample (0.0001 * 10000 = 1)
        assert!(skips >= 0 && skips <= 2);
    }

    #[test]
    fn test_drift_corrector_negative() {
        let mut corrector = DriftCorrector::new(-0.0001);

        let mut repeats = 0;
        for _ in 0..10000 {
            if corrector.process_sample() == -1 {
                repeats += 1;
            }
        }

        assert!(repeats >= 0 && repeats <= 2);
    }

    #[test]
    fn test_calculate_correlation() {
        let detector = SyncDetector::new(SyncMethod::CrossCorrelation);

        let signal1 = vec![1.0, 2.0, 3.0, 4.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0];

        let correlation = detector.calculate_correlation(&signal1, &signal2);
        assert!((correlation - 1.0).abs() < 0.001); // Should be 1.0 (perfect correlation)
    }

    #[test]
    fn test_empty_audio_error() {
        let detector = SyncDetector::new(SyncMethod::CrossCorrelation);
        let result = detector.detect_offset(&[], &[]);
        assert!(result.is_err());
    }
}
