//! EBU R128 Loudness Normalization
//!
//! Implementation of ITU-R BS.1770-4 loudness measurement and EBU R128 loudness normalization.
//!
//! ## Standards
//!
//! - **ITU-R BS.1770-4**: Algorithms to measure audio programme loudness and true-peak
//! - **EBU R128**: Loudness normalisation and permitted maximum level of audio signals
//! - **ATSC A/85**: Advanced Television Systems Committee digital audio compression standard
//!
//! ## Key Concepts
//!
//! **LUFS (Loudness Units relative to Full Scale):**
//! - Perceptual loudness measurement (not peak level)
//! - Takes into account human hearing sensitivity
//! - -23 LUFS is broadcast standard (EBU R128)
//! - -14 LUFS is streaming standard (Spotify, YouTube, etc.)
//!
//! **Loudness Range (LRA):**
//! - Measures dynamic range of program
//! - LRA = 0-5 LU: very compressed/limited
//! - LRA = 5-10 LU: moderately dynamic
//! - LRA = 10-20 LU: highly dynamic
//!
//! **True Peak:**
//! - Maximum level reached by audio signal between samples
//! - Prevents inter-sample peaks that cause clipping in DACs
//! - Measured using 4x oversampling
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::filter::loudness::{LoudnessMeter, LoudnessNormalizer};
//!
//! // Measure loudness
//! let mut meter = LoudnessMeter::new(48000);
//! meter.process_samples(&audio_data, 2);
//! let integrated = meter.integrated_loudness();
//! let lra = meter.loudness_range();
//!
//! // Normalize to -23 LUFS
//! let mut normalizer = LoudnessNormalizer::new(48000, -23.0);
//! let normalized = normalizer.normalize(&audio_data, 2);
//! ```

use crate::error::{Error, Result};
use std::collections::VecDeque;

/// EBU R128 loudness targets (LUFS)
pub mod targets {
    /// Broadcast standard (EBU R128)
    pub const BROADCAST: f64 = -23.0;
    /// Streaming platforms (Spotify, YouTube, Apple Music)
    pub const STREAMING: f64 = -14.0;
    /// CD mastering
    pub const CD: f64 = -9.0;
    /// Film/Cinema
    pub const FILM: f64 = -27.0;
}

/// K-weighting filter coefficients for ITU-R BS.1770-4
///
/// The K-weighting consists of:
/// 1. Pre-filter (high-pass ~80Hz) - models head-related transfer function
/// 2. RLB filter (high-shelf ~1500Hz) - models acoustic radiation of loudspeaker
struct KWeightFilter {
    // Pre-filter state (2nd order high-pass Butterworth @ 80Hz)
    pre_b0: f64,
    pre_b1: f64,
    pre_b2: f64,
    pre_a1: f64,
    pre_a2: f64,
    pre_z1: f64,
    pre_z2: f64,

    // RLB filter state (2nd order high-shelf @ 1500Hz)
    rlb_b0: f64,
    rlb_b1: f64,
    rlb_b2: f64,
    rlb_a1: f64,
    rlb_a2: f64,
    rlb_z1: f64,
    rlb_z2: f64,
}

impl KWeightFilter {
    /// Create K-weighting filter for given sample rate
    fn new(sample_rate: u32) -> Self {
        let fs = sample_rate as f64;

        // Pre-filter coefficients (high-pass @ 80Hz)
        let f_pre = 80.0;
        let q_pre = 0.707; // Butterworth
        let k_pre = (std::f64::consts::PI * f_pre / fs).tan();
        let norm_pre = 1.0 / (1.0 + k_pre / q_pre + k_pre * k_pre);

        let pre_b0 = norm_pre;
        let pre_b1 = -2.0 * norm_pre;
        let pre_b2 = norm_pre;
        let pre_a1 = 2.0 * (k_pre * k_pre - 1.0) * norm_pre;
        let pre_a2 = (1.0 - k_pre / q_pre + k_pre * k_pre) * norm_pre;

        // RLB filter coefficients (high-shelf @ 1500Hz, +4dB)
        let f_rlb = 1500.0;
        let q_rlb = 0.707;
        let db_gain = 4.0;
        let a_rlb = 10f64.powf(db_gain / 40.0);
        let k_rlb = (std::f64::consts::PI * f_rlb / fs).tan();
        let norm_rlb = 1.0 / (1.0 + k_rlb / q_rlb + k_rlb * k_rlb);

        let rlb_b0 = (a_rlb + a_rlb.sqrt() * k_rlb / q_rlb + k_rlb * k_rlb) * norm_rlb;
        let rlb_b1 = 2.0 * (k_rlb * k_rlb - a_rlb) * norm_rlb;
        let rlb_b2 = (a_rlb - a_rlb.sqrt() * k_rlb / q_rlb + k_rlb * k_rlb) * norm_rlb;
        let rlb_a1 = 2.0 * (k_rlb * k_rlb - 1.0) * norm_rlb;
        let rlb_a2 = (1.0 - k_rlb / q_rlb + k_rlb * k_rlb) * norm_rlb;

        KWeightFilter {
            pre_b0,
            pre_b1,
            pre_b2,
            pre_a1,
            pre_a2,
            pre_z1: 0.0,
            pre_z2: 0.0,
            rlb_b0,
            rlb_b1,
            rlb_b2,
            rlb_a1,
            rlb_a2,
            rlb_z1: 0.0,
            rlb_z2: 0.0,
        }
    }

    /// Process a single sample through K-weighting filter
    fn process(&mut self, input: f64) -> f64 {
        // Pre-filter (high-pass)
        let pre_out = self.pre_b0 * input + self.pre_z1;
        self.pre_z1 = self.pre_b1 * input - self.pre_a1 * pre_out + self.pre_z2;
        self.pre_z2 = self.pre_b2 * input - self.pre_a2 * pre_out;

        // RLB filter (high-shelf)
        let rlb_out = self.rlb_b0 * pre_out + self.rlb_z1;
        self.rlb_z1 = self.rlb_b1 * pre_out - self.rlb_a1 * rlb_out + self.rlb_z2;
        self.rlb_z2 = self.rlb_b2 * pre_out - self.rlb_a2 * rlb_out;

        rlb_out
    }

    /// Reset filter state
    fn reset(&mut self) {
        self.pre_z1 = 0.0;
        self.pre_z2 = 0.0;
        self.rlb_z1 = 0.0;
        self.rlb_z2 = 0.0;
    }
}

/// EBU R128 loudness meter
pub struct LoudnessMeter {
    sample_rate: u32,
    /// K-weighting filters per channel
    filters: Vec<KWeightFilter>,
    /// Gating blocks (400ms blocks with 75% overlap)
    blocks: VecDeque<LoudnessBlock>,
    /// Block size in samples (400ms)
    block_size: usize,
    /// Hop size (100ms for 75% overlap)
    hop_size: usize,
    /// Current block accumulator
    current_block: Vec<f64>,
    /// Samples processed in current block
    current_block_samples: usize,
}

/// Loudness measurement block (400ms)
#[derive(Debug, Clone)]
struct LoudnessBlock {
    /// Mean square power per channel (after K-weighting)
    channel_power: Vec<f64>,
    /// Block loudness in LUFS
    loudness: f64,
}

impl LoudnessMeter {
    /// Create a new loudness meter
    pub fn new(sample_rate: u32) -> Self {
        let block_size = ((sample_rate as f64 * 0.4) as usize).max(1); // 400ms
        let hop_size = ((sample_rate as f64 * 0.1) as usize).max(1); // 100ms

        LoudnessMeter {
            sample_rate,
            filters: Vec::new(),
            blocks: VecDeque::new(),
            block_size,
            hop_size,
            current_block: Vec::new(),
            current_block_samples: 0,
        }
    }

    /// Process audio samples (interleaved)
    ///
    /// # Arguments
    /// * `samples` - Interleaved audio samples (normalized to -1.0 to 1.0)
    /// * `channels` - Number of channels
    pub fn process_samples(&mut self, samples: &[f32], channels: usize) -> Result<()> {
        if channels == 0 {
            return Err(Error::InvalidInput("Channel count must be > 0".to_string()));
        }

        // Initialize filters if needed
        if self.filters.is_empty() {
            self.filters = (0..channels)
                .map(|_| KWeightFilter::new(self.sample_rate))
                .collect();
        }

        // Process samples in frames
        for frame in samples.chunks_exact(channels) {
            if frame.len() != channels {
                break;
            }

            // Apply K-weighting to each channel
            let mut channel_powers = vec![0.0; channels];
            for (ch, &sample) in frame.iter().enumerate() {
                let filtered = self.filters[ch].process(sample as f64);
                channel_powers[ch] = filtered * filtered;
            }

            // Calculate channel-weighted power
            // ITU-R BS.1770-4 channel weighting:
            // L, R = 1.0, C = 1.0, Ls, Rs = 1.41 (~+1.5dB)
            let weighted_power = if channels >= 5 {
                // 5.1 surround: L, R, C, LFE (ignored), Ls, Rs
                channel_powers[0]
                    + channel_powers[1]
                    + channel_powers[2]
                    + 1.41 * channel_powers[4]
                    + 1.41 * channel_powers[5]
            } else {
                // Stereo or mono: equal weighting
                channel_powers.iter().sum::<f64>()
            };

            self.current_block.push(weighted_power);
            self.current_block_samples += 1;

            // Process block when full
            if self.current_block_samples >= self.block_size {
                self.process_block(channels);

                // Keep overlap samples
                self.current_block.drain(0..self.hop_size);
                self.current_block_samples -= self.hop_size;
            }
        }

        Ok(())
    }

    /// Process a complete loudness block
    fn process_block(&mut self, channels: usize) {
        if self.current_block.is_empty() {
            return;
        }

        // Calculate mean square for the block
        let mean_square: f64 =
            self.current_block.iter().sum::<f64>() / self.current_block.len() as f64;

        // Convert to loudness (LUFS)
        let loudness = if mean_square > 0.0 {
            -0.691 + 10.0 * mean_square.log10()
        } else {
            -f64::INFINITY
        };

        // Store block
        self.blocks.push_back(LoudnessBlock {
            channel_power: vec![mean_square; channels],
            loudness,
        });

        // Limit buffer size (keep last ~60 seconds worth of blocks)
        let max_blocks = (60.0 * self.sample_rate as f64 / self.hop_size as f64) as usize;
        while self.blocks.len() > max_blocks {
            self.blocks.pop_front();
        }
    }

    /// Calculate integrated loudness (LUFS) using gating
    pub fn integrated_loudness(&self) -> f64 {
        if self.blocks.is_empty() {
            return -f64::INFINITY;
        }

        // Absolute gate: -70 LUFS
        let absolute_gate = -70.0;
        let gated_blocks: Vec<&LoudnessBlock> = self
            .blocks
            .iter()
            .filter(|b| b.loudness > absolute_gate)
            .collect();

        if gated_blocks.is_empty() {
            return -f64::INFINITY;
        }

        // Calculate relative gate: mean - 10 LU
        let mean_loudness: f64 =
            gated_blocks.iter().map(|b| b.loudness).sum::<f64>() / gated_blocks.len() as f64;
        let relative_gate = mean_loudness - 10.0;

        // Apply relative gate and calculate final integrated loudness
        let final_gated: Vec<&LoudnessBlock> = gated_blocks
            .iter()
            .filter(|b| b.loudness > relative_gate)
            .copied()
            .collect();

        if final_gated.is_empty() {
            return -f64::INFINITY;
        }

        final_gated.iter().map(|b| b.loudness).sum::<f64>() / final_gated.len() as f64
    }

    /// Calculate loudness range (LRA) in LU
    pub fn loudness_range(&self) -> f64 {
        if self.blocks.len() < 2 {
            return 0.0;
        }

        let mut loudnesses: Vec<f64> = self
            .blocks
            .iter()
            .map(|b| b.loudness)
            .filter(|&l| l > -70.0) // Absolute gate
            .collect();

        if loudnesses.len() < 2 {
            return 0.0;
        }

        loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate 10th and 95th percentiles
        let p10_idx = (loudnesses.len() as f64 * 0.10) as usize;
        let p95_idx = (loudnesses.len() as f64 * 0.95) as usize;

        let p10 = loudnesses[p10_idx.min(loudnesses.len() - 1)];
        let p95 = loudnesses[p95_idx.min(loudnesses.len() - 1)];

        (p95 - p10).max(0.0)
    }

    /// Calculate true peak (dBTP) using 4x oversampling
    pub fn true_peak(&self, samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return -f64::INFINITY;
        }

        // Simple true peak approximation (4x oversampling with linear interpolation)
        let mut max_peak = 0.0f64;

        for i in 0..(samples.len() - 1) {
            let s1 = samples[i] as f64;
            let s2 = samples[i + 1] as f64;

            // Check original sample
            max_peak = max_peak.max(s1.abs());

            // Check interpolated samples
            for j in 1..4 {
                let t = j as f64 / 4.0;
                let interpolated = s1 + (s2 - s1) * t;
                max_peak = max_peak.max(interpolated.abs());
            }
        }

        // Last sample
        max_peak = max_peak.max(samples.last().unwrap().abs() as f64);

        // Convert to dBTP
        if max_peak > 0.0 {
            20.0 * max_peak.log10()
        } else {
            -f64::INFINITY
        }
    }

    /// Reset meter state
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.current_block.clear();
        self.current_block_samples = 0;
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    /// Get measurement summary
    pub fn summary(&self, samples: &[f32]) -> String {
        format!(
            "EBU R128 Loudness:\n\
             - Integrated: {:.1} LUFS\n\
             - Loudness Range: {:.1} LU\n\
             - True Peak: {:.1} dBTP",
            self.integrated_loudness(),
            self.loudness_range(),
            self.true_peak(samples)
        )
    }
}

/// Loudness normalizer
pub struct LoudnessNormalizer {
    meter: LoudnessMeter,
    target_lufs: f64,
    max_true_peak: f64,
}

impl LoudnessNormalizer {
    /// Create a new loudness normalizer
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `target_lufs` - Target loudness (e.g., -23.0 for broadcast, -14.0 for streaming)
    pub fn new(sample_rate: u32, target_lufs: f64) -> Self {
        LoudnessNormalizer {
            meter: LoudnessMeter::new(sample_rate),
            target_lufs,
            max_true_peak: -1.0, // -1.0 dBTP default (EBU R128 recommendation)
        }
    }

    /// Set maximum true peak level (dBTP)
    pub fn set_max_true_peak(&mut self, max_dbtp: f64) {
        self.max_true_peak = max_dbtp;
    }

    /// Analyze audio and calculate normalization gain
    pub fn analyze(&mut self, samples: &[f32], channels: usize) -> Result<f64> {
        self.meter.reset();
        self.meter.process_samples(samples, channels)?;

        let integrated = self.meter.integrated_loudness();
        if integrated.is_infinite() && integrated.is_sign_negative() {
            return Ok(1.0); // Silent audio
        }

        // Calculate gain to reach target
        let gain_db = self.target_lufs - integrated;
        let gain_linear = 10f64.powf(gain_db / 20.0);

        // Check true peak limiter
        let true_peak = self.meter.true_peak(samples);
        let new_peak = true_peak + gain_db;

        if new_peak > self.max_true_peak {
            // Reduce gain to meet true peak limit
            let limited_gain_db = self.max_true_peak - true_peak;
            Ok(10f64.powf(limited_gain_db / 20.0))
        } else {
            Ok(gain_linear)
        }
    }

    /// Normalize audio to target loudness
    pub fn normalize(&mut self, samples: &[f32], channels: usize) -> Result<Vec<f32>> {
        let gain = self.analyze(samples, channels)?;

        let normalized: Vec<f32> = samples
            .iter()
            .map(|&s| (s as f64 * gain).clamp(-1.0, 1.0) as f32)
            .collect();

        Ok(normalized)
    }

    /// Get current meter
    pub fn meter(&self) -> &LoudnessMeter {
        &self.meter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_tone(freq: f64, duration: f64, sample_rate: u32, amplitude: f32) -> Vec<f32> {
        let samples = (duration * sample_rate as f64) as usize;
        (0..samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (amplitude as f64 * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn test_k_weight_filter() {
        let mut filter = KWeightFilter::new(48000);

        // Process some samples
        let input = vec![0.5, 0.3, -0.2, -0.5];
        let outputs: Vec<f64> = input.iter().map(|&x| filter.process(x as f64)).collect();

        // Filter should produce output
        assert!(outputs.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_loudness_meter_creation() {
        let meter = LoudnessMeter::new(48000);
        assert_eq!(meter.sample_rate, 48000);
        assert_eq!(meter.block_size, 19200); // 400ms at 48kHz
        assert_eq!(meter.hop_size, 4800); // 100ms at 48kHz
    }

    #[test]
    fn test_loudness_meter_process() {
        let mut meter = LoudnessMeter::new(48000);

        // Generate 1 second of stereo 1kHz tone at -20 dBFS
        let amplitude = 10f32.powf(-20.0 / 20.0);
        let samples = generate_tone(1000.0, 1.0, 48000, amplitude);
        let stereo: Vec<f32> = samples.iter().flat_map(|&s| vec![s, s]).collect();

        meter.process_samples(&stereo, 2).unwrap();

        let integrated = meter.integrated_loudness();
        // Should be around -20 LUFS (may vary due to K-weighting)
        assert!(integrated > -30.0 && integrated < -10.0);
    }

    #[test]
    fn test_integrated_loudness() {
        let mut meter = LoudnessMeter::new(48000);

        // Generate quiet tone
        let amplitude = 10f32.powf(-30.0 / 20.0);
        let samples = generate_tone(1000.0, 2.0, 48000, amplitude);
        let stereo: Vec<f32> = samples.iter().flat_map(|&s| vec![s, s]).collect();

        meter.process_samples(&stereo, 2).unwrap();

        let integrated = meter.integrated_loudness();
        assert!(integrated.is_finite());
        assert!(integrated < -20.0); // Quiet signal
    }

    #[test]
    fn test_loudness_range() {
        let mut meter = LoudnessMeter::new(48000);

        // Generate signal with varying amplitude
        let mut samples = Vec::new();
        let amp1 = 10f32.powf(-20.0 / 20.0);
        let amp2 = 10f32.powf(-30.0 / 20.0);

        samples.extend(generate_tone(1000.0, 1.0, 48000, amp1));
        samples.extend(generate_tone(1000.0, 1.0, 48000, amp2));

        let stereo: Vec<f32> = samples.iter().flat_map(|&s| vec![s, s]).collect();

        meter.process_samples(&stereo, 2).unwrap();

        let lra = meter.loudness_range();
        assert!(lra > 0.0); // Should have some dynamic range
    }

    #[test]
    fn test_true_peak() {
        let meter = LoudnessMeter::new(48000);

        // Generate tone at -6 dBFS
        let amplitude = 10f32.powf(-6.0 / 20.0);
        let samples = generate_tone(1000.0, 0.1, 48000, amplitude);

        let peak = meter.true_peak(&samples);
        // Should be close to -6 dBTP
        assert!(peak > -7.0 && peak < -5.0);
    }

    #[test]
    fn test_normalizer_analyze() {
        let mut normalizer = LoudnessNormalizer::new(48000, -23.0);

        // Generate tone at -30 LUFS
        let amplitude = 10f32.powf(-30.0 / 20.0);
        let samples = generate_tone(1000.0, 2.0, 48000, amplitude);
        let stereo: Vec<f32> = samples.iter().flat_map(|&s| vec![s, s]).collect();

        let gain = normalizer.analyze(&stereo, 2).unwrap();

        // Gain should be positive (boost needed)
        assert!(gain > 1.0);
        // Should boost by roughly 7 dB (from -30 to -23)
        let gain_db = 20.0 * gain.log10();
        assert!(gain_db > 5.0 && gain_db < 9.0);
    }

    #[test]
    fn test_normalizer_normalize() {
        let mut normalizer = LoudnessNormalizer::new(48000, -14.0);

        // Generate quiet tone
        let amplitude = 10f32.powf(-25.0 / 20.0);
        let samples = generate_tone(1000.0, 1.0, 48000, amplitude);
        let stereo: Vec<f32> = samples.iter().flat_map(|&s| vec![s, s]).collect();

        let normalized = normalizer.normalize(&stereo, 2).unwrap();

        assert_eq!(normalized.len(), stereo.len());
        // Normalized samples should be louder
        let max_orig = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let max_norm = normalized.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_norm > max_orig);
    }

    #[test]
    fn test_true_peak_limiter() {
        let mut normalizer = LoudnessNormalizer::new(48000, 0.0); // Target 0 LUFS (very loud)
        normalizer.set_max_true_peak(-6.0); // Max peak at -6 dBTP

        // Generate tone at -20 dBFS (moderate level)
        let amplitude = 10f32.powf(-20.0 / 20.0); // 0.1 amplitude
        let samples = generate_tone(1000.0, 2.0, 48000, amplitude);
        let stereo: Vec<f32> = samples.iter().flat_map(|&s| vec![s, s]).collect();

        let gain = normalizer.analyze(&stereo, 2).unwrap();

        // Calculate what true peak would be after gain
        // Original peak is ~-20 dBTP, target is 0 LUFS which would require ~+20dB boost
        // But max_true_peak is -6 dBTP, so gain should be limited
        // Actual gain should bring peak from -20 dBTP to at most -6 dBTP (i.e., 14dB gain max)
        let gain_db = 20.0 * (gain as f64).log10();
        assert!(gain_db <= 14.5); // Limited gain
    }

    #[test]
    fn test_loudness_targets() {
        assert_eq!(targets::BROADCAST, -23.0);
        assert_eq!(targets::STREAMING, -14.0);
        assert_eq!(targets::CD, -9.0);
        assert_eq!(targets::FILM, -27.0);
    }
}
