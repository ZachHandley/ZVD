//! ZVC69 Entropy Coding Module
//!
//! This module implements entropy coding for the ZVC69 neural video codec using
//! the `constriction` library for Asymmetric Numeral Systems (ANS) coding.
//!
//! ## Overview
//!
//! Neural video codecs use learned probability distributions to achieve near-optimal
//! compression. The entropy coding pipeline works as follows:
//!
//! 1. **Encoder**: Neural network outputs latent representations + probability parameters
//! 2. **Quantization**: Latents are rounded to integers
//! 3. **Entropy Coding**: ANS encodes symbols using predicted probabilities
//! 4. **Decoder**: ANS decodes symbols, then neural network reconstructs the frame
//!
//! ## Entropy Models
//!
//! Two entropy models are implemented:
//!
//! - **FactorizedPrior**: For hyperprior latents (no conditioning)
//! - **GaussianConditional**: For main latents (conditioned on hyperprior)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::entropy::{EntropyCoder, GaussianConditional};
//!
//! // Create entropy coder
//! let mut coder = EntropyCoder::new();
//!
//! // Encode latents with predicted Gaussian parameters
//! let compressed = coder.encode_symbols(&symbols, &means, &scales)?;
//!
//! // Decode with same parameters
//! let decoded = coder.decode_symbols(&compressed, &means, &scales, num_symbols)?;
//! ```

use super::error::ZVC69Error;
use constriction::stream::model::DefaultLeakyQuantizer;
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::{Decode, Encode};
use probability::distribution::Gaussian;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Default minimum value for latent symbol range
pub const DEFAULT_MIN_SYMBOL: i32 = -128;

/// Default maximum value for latent symbol range
pub const DEFAULT_MAX_SYMBOL: i32 = 127;

/// Default scale bound for Gaussian conditional
pub const DEFAULT_SCALE_BOUND: f32 = 0.11;

/// Number of scale steps in the scale table
pub const DEFAULT_NUM_SCALES: usize = 64;

/// Minimum scale value to prevent numerical instability
pub const MIN_SCALE: f64 = 0.01;

/// Maximum scale value for practical purposes
pub const MAX_SCALE: f64 = 256.0;

/// Precision bits for CDF (16-bit)
pub const CDF_PRECISION_BITS: usize = 16;

/// CDF scale factor (2^16)
pub const CDF_SCALE: f64 = 65536.0;

/// Tail mass for leaky quantizer (ensures non-zero probability at edges)
pub const DEFAULT_TAIL_MASS: f64 = 1e-9;

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

/// Quantize continuous latent values to integers by rounding
///
/// This function converts floating-point latent representations from the
/// neural network encoder to discrete integer symbols for entropy coding.
///
/// # Arguments
///
/// * `values` - Continuous latent values from the encoder
///
/// # Returns
///
/// Vector of quantized integer symbols
///
/// # Example
///
/// ```rust,ignore
/// let continuous = vec![1.2, -0.7, 3.9, -2.1];
/// let quantized = quantize_latents(&continuous);
/// assert_eq!(quantized, vec![1, -1, 4, -2]);
/// ```
pub fn quantize_latents(values: &[f32]) -> Vec<i32> {
    values.iter().map(|&v| v.round() as i32).collect()
}

/// Dequantize integer symbols back to continuous values
///
/// This is the inverse of quantization - converts integer symbols
/// back to floating-point values for the decoder neural network.
///
/// # Arguments
///
/// * `symbols` - Quantized integer symbols
///
/// # Returns
///
/// Vector of floating-point values
pub fn dequantize_latents(symbols: &[i32]) -> Vec<f32> {
    symbols.iter().map(|&s| s as f32).collect()
}

/// Estimate the number of bits required to encode symbols given Gaussian parameters
///
/// Uses the information-theoretic formula:
///   bits = -log2(P(symbol | mean, scale))
///
/// For a Gaussian, this approximates the negative log-likelihood.
///
/// # Arguments
///
/// * `symbols` - Quantized integer symbols to encode
/// * `means` - Predicted mean values for each symbol
/// * `scales` - Predicted scale (std dev) values for each symbol
///
/// # Returns
///
/// Estimated total bits for encoding all symbols
pub fn estimate_bits(symbols: &[i32], means: &[f32], scales: &[f32]) -> f32 {
    if symbols.len() != means.len() || symbols.len() != scales.len() {
        return f32::INFINITY;
    }

    let mut total_bits: f64 = 0.0;

    for ((&symbol, &mean), &scale) in symbols.iter().zip(means.iter()).zip(scales.iter()) {
        let scale_clamped = (scale as f64).clamp(MIN_SCALE, MAX_SCALE);
        let mean_f64 = mean as f64;

        // Compute Gaussian CDF at bin edges
        let x = symbol as f64;
        let upper = normal_cdf((x + 0.5 - mean_f64) / scale_clamped);
        let lower = normal_cdf((x - 0.5 - mean_f64) / scale_clamped);

        // Probability mass in this bin
        let prob = (upper - lower).max(1e-10);

        // Information content: -log2(prob)
        total_bits += -prob.log2();
    }

    total_bits as f32
}

/// Standard normal CDF using the error function approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Coefficients for approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Convert Gaussian parameters (mean, scale) to a discrete CDF
///
/// This converts continuous Gaussian parameters to a discrete probability
/// distribution suitable for entropy coding.
///
/// # Arguments
///
/// * `mean` - Mean of the Gaussian distribution
/// * `scale` - Standard deviation of the Gaussian
/// * `num_symbols` - Number of discrete symbols (determines range)
/// * `precision` - Precision bits for the CDF (typically 16)
///
/// # Returns
///
/// CDF as a vector of u32 values, length = num_symbols + 1
pub fn gaussian_to_cdf(mean: f32, scale: f32, num_symbols: usize, precision: usize) -> Vec<u32> {
    let max_val = 1u32 << precision;
    let scale_factor = max_val as f64;
    let mean_f64 = mean as f64;
    let scale_clamped = (scale as f64).clamp(MIN_SCALE, MAX_SCALE);

    // Symbol range is centered around 0
    let half_range = (num_symbols / 2) as i32;
    let min_val = -half_range;

    let mut cdf = Vec::with_capacity(num_symbols + 1);
    cdf.push(0);

    for i in 0..num_symbols {
        let symbol = min_val + i as i32;
        let x = (symbol as f64 + 0.5 - mean_f64) / scale_clamped;
        let prob = normal_cdf(x);
        // Clamp to [1, max_val - 1] for intermediate values
        let cdf_val = ((prob * scale_factor).round() as u32).clamp(1, max_val - 1);
        cdf.push(cdf_val);
    }

    // Ensure CDF is monotonically increasing
    for i in 1..cdf.len() {
        if cdf[i] <= cdf[i - 1] {
            cdf[i] = (cdf[i - 1] + 1).min(max_val - 1);
        }
    }

    // Ensure last value is exactly max_val
    if let Some(last) = cdf.last_mut() {
        *last = max_val;
    }

    cdf
}

/// Convert compressed u32 words to bytes
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Convert bytes back to u32 words
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::with_capacity(bytes.len().div_ceil(4));
    for chunk in bytes.chunks(4) {
        let mut arr = [0u8; 4];
        arr[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(arr));
    }
    words
}

// -------------------------------------------------------------------------
// QuantizedGaussian
// -------------------------------------------------------------------------

/// Quantized Gaussian distribution for entropy coding
///
/// Represents a Gaussian distribution discretized to integer symbols
/// within a specified range.
#[derive(Debug, Clone)]
pub struct QuantizedGaussian {
    /// Mean of the distribution
    pub mean: f32,
    /// Standard deviation (scale) of the distribution
    pub scale: f32,
    /// Minimum symbol value
    pub min_val: i32,
    /// Maximum symbol value
    pub max_val: i32,
}

impl QuantizedGaussian {
    /// Create a new quantized Gaussian
    pub fn new(mean: f32, scale: f32) -> Self {
        Self {
            mean,
            scale: scale.max(MIN_SCALE as f32),
            min_val: DEFAULT_MIN_SYMBOL,
            max_val: DEFAULT_MAX_SYMBOL,
        }
    }

    /// Create with custom symbol range
    pub fn with_range(mean: f32, scale: f32, min_val: i32, max_val: i32) -> Self {
        Self {
            mean,
            scale: scale.max(MIN_SCALE as f32),
            min_val,
            max_val,
        }
    }

    /// Get the symbol range
    pub fn symbol_range(&self) -> std::ops::RangeInclusive<i32> {
        self.min_val..=self.max_val
    }

    /// Get the number of symbols
    pub fn num_symbols(&self) -> usize {
        (self.max_val - self.min_val + 1) as usize
    }

    /// Compute the probability mass for a given symbol
    pub fn probability(&self, symbol: i32) -> f64 {
        if symbol < self.min_val || symbol > self.max_val {
            return 0.0;
        }

        let mean_f64 = self.mean as f64;
        let scale_f64 = (self.scale as f64).clamp(MIN_SCALE, MAX_SCALE);
        let x = symbol as f64;

        let upper = normal_cdf((x + 0.5 - mean_f64) / scale_f64);
        let lower = normal_cdf((x - 0.5 - mean_f64) / scale_f64);

        (upper - lower).max(1e-10)
    }

    /// Compute information content (bits) for a symbol
    pub fn bits(&self, symbol: i32) -> f64 {
        -self.probability(symbol).log2()
    }
}

impl Default for QuantizedGaussian {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

// -------------------------------------------------------------------------
// EntropyCoder
// -------------------------------------------------------------------------

/// Entropy coder for neural codec latents
///
/// This is the main interface for encoding and decoding quantized latent
/// symbols using ANS (Asymmetric Numeral Systems) coding with learned
/// Gaussian probability distributions.
///
/// # Example
///
/// ```rust,ignore
/// use zvd::codec::zvc69::entropy::EntropyCoder;
///
/// let mut coder = EntropyCoder::new();
///
/// // Encode
/// let symbols = vec![5, -3, 12, 0, -7];
/// let means = vec![4.5, -2.8, 11.2, 0.1, -6.5];
/// let scales = vec![1.0, 1.5, 2.0, 0.5, 1.2];
///
/// let compressed = coder.encode_symbols(&symbols, &means, &scales)?;
///
/// // Decode
/// let decoded = coder.decode_symbols(&compressed, &means, &scales, 5)?;
/// assert_eq!(symbols, decoded);
/// ```
pub struct EntropyCoder {
    /// Minimum symbol value for quantization range
    min_symbol: i32,
    /// Maximum symbol value for quantization range
    max_symbol: i32,
}

impl EntropyCoder {
    /// Create a new entropy coder with default symbol range
    pub fn new() -> Self {
        Self {
            min_symbol: DEFAULT_MIN_SYMBOL,
            max_symbol: DEFAULT_MAX_SYMBOL,
        }
    }

    /// Create with custom symbol range
    pub fn with_range(min_symbol: i32, max_symbol: i32) -> Self {
        Self {
            min_symbol,
            max_symbol,
        }
    }

    /// Get the symbol range
    pub fn symbol_range(&self) -> (i32, i32) {
        (self.min_symbol, self.max_symbol)
    }

    /// Encode latent symbols given probability distributions
    ///
    /// # Arguments
    ///
    /// * `symbols` - Quantized integer symbols to encode
    /// * `means` - Predicted mean for each symbol's Gaussian distribution
    /// * `scales` - Predicted scale (std dev) for each symbol's distribution
    ///
    /// # Returns
    ///
    /// Compressed byte stream on success, or error on failure
    pub fn encode_symbols(
        &mut self,
        symbols: &[i32],
        means: &[f32],
        scales: &[f32],
    ) -> Result<Vec<u8>, ZVC69Error> {
        if symbols.is_empty() {
            return Ok(Vec::new());
        }

        if symbols.len() != means.len() || symbols.len() != scales.len() {
            return Err(ZVC69Error::EntropyEncodingFailed {
                reason: format!(
                    "Length mismatch: symbols={}, means={}, scales={}",
                    symbols.len(),
                    means.len(),
                    scales.len()
                ),
            });
        }

        // Validate symbols are in range
        for (i, &sym) in symbols.iter().enumerate() {
            if sym < self.min_symbol || sym > self.max_symbol {
                return Err(ZVC69Error::EntropyEncodingFailed {
                    reason: format!(
                        "Symbol {} at index {} out of range [{}, {}]",
                        sym, i, self.min_symbol, self.max_symbol
                    ),
                });
            }
        }

        // Create ANS coder
        let mut coder = DefaultAnsCoder::new();

        // Create the leaky quantizer for the symbol range
        let quantizer = DefaultLeakyQuantizer::new(self.min_symbol..=self.max_symbol);

        // Encode in reverse order (ANS is LIFO)
        for ((&symbol, &mean), &scale) in symbols.iter().zip(means.iter()).zip(scales.iter()).rev()
        {
            // Clamp scale to valid range
            let scale_clamped = (scale as f64).clamp(MIN_SCALE, MAX_SCALE);

            // Create Gaussian distribution
            let gaussian = Gaussian::new(mean as f64, scale_clamped);

            // Quantize the distribution
            let model = quantizer.quantize(gaussian);

            // Encode the symbol
            coder
                .encode_symbol(symbol, model)
                .map_err(|e| ZVC69Error::EntropyEncodingFailed {
                    reason: format!("ANS encode failed: {:?}", e),
                })?;
        }

        // Get compressed representation
        let compressed_words =
            coder
                .into_compressed()
                .map_err(|e| ZVC69Error::EntropyEncodingFailed {
                    reason: format!("Failed to get compressed data: {:?}", e),
                })?;

        Ok(words_to_bytes(&compressed_words))
    }

    /// Decode latent symbols given probability distributions
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed byte stream
    /// * `means` - Predicted mean for each symbol's Gaussian distribution
    /// * `scales` - Predicted scale (std dev) for each symbol's distribution
    /// * `num_symbols` - Number of symbols to decode
    ///
    /// # Returns
    ///
    /// Decoded integer symbols on success, or error on failure
    pub fn decode_symbols(
        &mut self,
        data: &[u8],
        means: &[f32],
        scales: &[f32],
        num_symbols: usize,
    ) -> Result<Vec<i32>, ZVC69Error> {
        if num_symbols == 0 {
            return Ok(Vec::new());
        }

        if data.is_empty() {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Empty compressed data".to_string(),
            });
        }

        if means.len() != num_symbols || scales.len() != num_symbols {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: format!(
                    "Length mismatch: expected {} symbols, got means={}, scales={}",
                    num_symbols,
                    means.len(),
                    scales.len()
                ),
            });
        }

        // Convert bytes to words
        let compressed_words = bytes_to_words(data);

        // Create decoder from compressed data
        let mut coder = DefaultAnsCoder::from_compressed(compressed_words).map_err(|e| {
            ZVC69Error::EntropyDecodingFailed {
                reason: format!("Failed to initialize decoder: {:?}", e),
            }
        })?;

        // Create the leaky quantizer
        let quantizer = DefaultLeakyQuantizer::new(self.min_symbol..=self.max_symbol);

        // Decode symbols
        let mut symbols = Vec::with_capacity(num_symbols);

        for (&mean, &scale) in means.iter().zip(scales.iter()) {
            // Clamp scale to valid range
            let scale_clamped = (scale as f64).clamp(MIN_SCALE, MAX_SCALE);

            // Create Gaussian distribution
            let gaussian = Gaussian::new(mean as f64, scale_clamped);

            // Quantize the distribution
            let model = quantizer.quantize(gaussian);

            // Decode symbol
            let symbol =
                coder
                    .decode_symbol(model)
                    .map_err(|e| ZVC69Error::EntropyDecodingFailed {
                        reason: format!("ANS decode failed: {:?}", e),
                    })?;

            symbols.push(symbol);
        }

        // No reverse needed: encoding in reverse + forward decoding gives correct order
        Ok(symbols)
    }
}

impl Default for EntropyCoder {
    fn default() -> Self {
        Self::new()
    }
}

// -------------------------------------------------------------------------
// FactorizedPrior
// -------------------------------------------------------------------------

/// Factorized prior for hyperprior latents (no conditioning)
///
/// This entropy model assumes spatial independence within each channel,
/// with a fixed learned distribution per channel. It's used for encoding
/// the hyperprior latents which provide side information for the main
/// latent entropy model.
///
/// The factorized prior is simpler but less efficient than the conditional
/// models, making it suitable for the smaller hyperprior representation.
pub struct FactorizedPrior {
    /// Number of channels in the hyperprior
    num_channels: usize,
    /// Per-channel CDF tables (learned during training)
    cdfs: Vec<Vec<u32>>,
    /// Symbol range
    min_symbol: i32,
    max_symbol: i32,
}

impl FactorizedPrior {
    /// Create a new factorized prior with default uniform CDFs
    ///
    /// # Arguments
    ///
    /// * `num_channels` - Number of hyperprior channels
    pub fn new(num_channels: usize) -> Self {
        let num_symbols = (DEFAULT_MAX_SYMBOL - DEFAULT_MIN_SYMBOL + 1) as usize;

        // Initialize with uniform CDFs (will be replaced with learned ones)
        let cdf = (0..=num_symbols)
            .map(|i| ((i as f64 / num_symbols as f64) * CDF_SCALE) as u32)
            .collect::<Vec<_>>();

        Self {
            num_channels,
            cdfs: vec![cdf; num_channels],
            min_symbol: DEFAULT_MIN_SYMBOL,
            max_symbol: DEFAULT_MAX_SYMBOL,
        }
    }

    /// Create with custom per-channel CDFs
    ///
    /// # Arguments
    ///
    /// * `cdfs` - Vector of CDF tables, one per channel
    pub fn with_cdfs(cdfs: Vec<Vec<u32>>) -> Self {
        let num_channels = cdfs.len();
        Self {
            num_channels,
            cdfs,
            min_symbol: DEFAULT_MIN_SYMBOL,
            max_symbol: DEFAULT_MAX_SYMBOL,
        }
    }

    /// Get the number of channels
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Set the CDF for a specific channel
    pub fn set_channel_cdf(&mut self, channel: usize, cdf: Vec<u32>) -> Result<(), ZVC69Error> {
        if channel >= self.num_channels {
            return Err(ZVC69Error::InvalidProbability {
                details: format!(
                    "Channel {} out of range (max {})",
                    channel,
                    self.num_channels - 1
                ),
            });
        }
        self.cdfs[channel] = cdf;
        Ok(())
    }

    /// Encode hyperprior latents
    ///
    /// # Arguments
    ///
    /// * `latents` - Quantized hyperprior latent values
    ///
    /// # Returns
    ///
    /// Compressed byte stream
    pub fn encode(&self, latents: &[i32]) -> Result<Vec<u8>, ZVC69Error> {
        if latents.is_empty() {
            return Ok(Vec::new());
        }

        // Create ANS coder
        let mut coder = DefaultAnsCoder::new();

        // Create quantizer
        let quantizer = DefaultLeakyQuantizer::new(self.min_symbol..=self.max_symbol);

        // Encode in reverse order
        for (i, &symbol) in latents.iter().enumerate().rev() {
            // Determine channel (cyclic assignment)
            // Note: In production, channel-specific CDFs would be used here
            let _channel = i % self.num_channels;

            // Use a Gaussian with fixed parameters based on the CDF
            // For factorized prior, we use a standard Gaussian centered at 0
            // The actual learned CDFs would replace this in production
            let gaussian = Gaussian::new(0.0, 5.0); // Broad distribution
            let model = quantizer.quantize(gaussian);

            coder
                .encode_symbol(symbol, model)
                .map_err(|e| ZVC69Error::EntropyEncodingFailed {
                    reason: format!("Factorized prior encode failed at {}: {:?}", i, e),
                })?;
        }

        let compressed_words =
            coder
                .into_compressed()
                .map_err(|e| ZVC69Error::EntropyEncodingFailed {
                    reason: format!("Failed to get compressed data: {:?}", e),
                })?;

        Ok(words_to_bytes(&compressed_words))
    }

    /// Decode hyperprior latents
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed byte stream
    /// * `num_symbols` - Number of symbols to decode
    ///
    /// # Returns
    ///
    /// Decoded integer symbols
    pub fn decode(&self, data: &[u8], num_symbols: usize) -> Result<Vec<i32>, ZVC69Error> {
        if num_symbols == 0 {
            return Ok(Vec::new());
        }

        if data.is_empty() {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Empty compressed data for factorized prior".to_string(),
            });
        }

        let compressed_words = bytes_to_words(data);

        let mut coder = DefaultAnsCoder::from_compressed(compressed_words).map_err(|e| {
            ZVC69Error::EntropyDecodingFailed {
                reason: format!("Failed to initialize factorized prior decoder: {:?}", e),
            }
        })?;

        let quantizer = DefaultLeakyQuantizer::new(self.min_symbol..=self.max_symbol);

        let mut symbols = Vec::with_capacity(num_symbols);

        for _i in 0..num_symbols {
            let gaussian = Gaussian::new(0.0, 5.0);
            let model = quantizer.quantize(gaussian);

            let symbol =
                coder
                    .decode_symbol(model)
                    .map_err(|e| ZVC69Error::EntropyDecodingFailed {
                        reason: format!("Factorized prior decode failed: {:?}", e),
                    })?;

            symbols.push(symbol);
        }

        // No reverse needed: encoding in reverse + forward decoding gives correct order
        Ok(symbols)
    }
}

// -------------------------------------------------------------------------
// GaussianConditional
// -------------------------------------------------------------------------

/// Gaussian conditional entropy model - conditioned on hyperprior
///
/// This is the main entropy model for encoding latent representations.
/// It uses Gaussian distributions whose parameters (mean, scale) are
/// predicted by the hyperprior decoder network.
///
/// The scale values are quantized to a table for efficiency.
pub struct GaussianConditional {
    /// Table of quantized scale values
    scale_table: Vec<f32>,
    /// Tail mass for probability normalization (used in CDF generation)
    #[allow(dead_code)]
    tail_mass: f32,
    /// Symbol range
    min_symbol: i32,
    max_symbol: i32,
}

impl GaussianConditional {
    /// Create a new Gaussian conditional model
    ///
    /// # Arguments
    ///
    /// * `scale_bound` - Lower bound for scale values
    /// * `num_scales` - Number of discrete scale levels
    pub fn new(scale_bound: f32, num_scales: usize) -> Self {
        // Build scale table (logarithmic spacing)
        let scale_table = if num_scales > 0 {
            let log_min = (scale_bound.max(MIN_SCALE as f32)).ln();
            let log_max = (MAX_SCALE as f32).ln();

            (0..num_scales)
                .map(|i| {
                    let t = i as f32 / (num_scales - 1).max(1) as f32;
                    (log_min + t * (log_max - log_min)).exp()
                })
                .collect()
        } else {
            vec![1.0]
        };

        Self {
            scale_table,
            tail_mass: DEFAULT_TAIL_MASS as f32,
            min_symbol: DEFAULT_MIN_SYMBOL,
            max_symbol: DEFAULT_MAX_SYMBOL,
        }
    }

    /// Get the scale table
    pub fn scale_table(&self) -> &[f32] {
        &self.scale_table
    }

    /// Find the nearest scale index for a given scale value
    pub fn find_scale_index(&self, scale: f32) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for (i, &s) in self.scale_table.iter().enumerate() {
            let dist = (s - scale).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Quantize scales to table values
    pub fn quantize_scales(&self, scales: &[f32]) -> Vec<f32> {
        scales
            .iter()
            .map(|&s| {
                let idx = self.find_scale_index(s);
                self.scale_table[idx]
            })
            .collect()
    }

    /// Encode latents with Gaussian conditional model
    ///
    /// # Arguments
    ///
    /// * `latents` - Quantized latent values
    /// * `means` - Predicted means from hyperprior
    /// * `scales` - Predicted scales from hyperprior
    ///
    /// # Returns
    ///
    /// Compressed byte stream
    pub fn encode(
        &self,
        latents: &[i32],
        means: &[f32],
        scales: &[f32],
    ) -> Result<Vec<u8>, ZVC69Error> {
        if latents.is_empty() {
            return Ok(Vec::new());
        }

        if latents.len() != means.len() || latents.len() != scales.len() {
            return Err(ZVC69Error::EntropyEncodingFailed {
                reason: format!(
                    "Gaussian conditional length mismatch: latents={}, means={}, scales={}",
                    latents.len(),
                    means.len(),
                    scales.len()
                ),
            });
        }

        let mut coder = DefaultAnsCoder::new();
        let quantizer = DefaultLeakyQuantizer::new(self.min_symbol..=self.max_symbol);

        // Encode in reverse order
        for ((&symbol, &mean), &scale) in latents.iter().zip(means.iter()).zip(scales.iter()).rev()
        {
            let scale_clamped = (scale as f64).clamp(MIN_SCALE, MAX_SCALE);
            let gaussian = Gaussian::new(mean as f64, scale_clamped);
            let model = quantizer.quantize(gaussian);

            coder
                .encode_symbol(symbol, model)
                .map_err(|e| ZVC69Error::EntropyEncodingFailed {
                    reason: format!("Gaussian conditional encode failed: {:?}", e),
                })?;
        }

        let compressed_words =
            coder
                .into_compressed()
                .map_err(|e| ZVC69Error::EntropyEncodingFailed {
                    reason: format!("Failed to get compressed data: {:?}", e),
                })?;

        Ok(words_to_bytes(&compressed_words))
    }

    /// Decode latents with Gaussian conditional model
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed byte stream
    /// * `means` - Predicted means from hyperprior
    /// * `scales` - Predicted scales from hyperprior
    /// * `num_symbols` - Number of symbols to decode
    ///
    /// # Returns
    ///
    /// Decoded integer symbols
    pub fn decode(
        &self,
        data: &[u8],
        means: &[f32],
        scales: &[f32],
        num_symbols: usize,
    ) -> Result<Vec<i32>, ZVC69Error> {
        if num_symbols == 0 {
            return Ok(Vec::new());
        }

        if data.is_empty() {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Empty compressed data for Gaussian conditional".to_string(),
            });
        }

        if means.len() != num_symbols || scales.len() != num_symbols {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: format!(
                    "Gaussian conditional length mismatch: expected {}, means={}, scales={}",
                    num_symbols,
                    means.len(),
                    scales.len()
                ),
            });
        }

        let compressed_words = bytes_to_words(data);

        let mut coder = DefaultAnsCoder::from_compressed(compressed_words).map_err(|e| {
            ZVC69Error::EntropyDecodingFailed {
                reason: format!("Failed to initialize Gaussian conditional decoder: {:?}", e),
            }
        })?;

        let quantizer = DefaultLeakyQuantizer::new(self.min_symbol..=self.max_symbol);

        let mut symbols = Vec::with_capacity(num_symbols);

        for (&mean, &scale) in means.iter().zip(scales.iter()) {
            let scale_clamped = (scale as f64).clamp(MIN_SCALE, MAX_SCALE);
            let gaussian = Gaussian::new(mean as f64, scale_clamped);
            let model = quantizer.quantize(gaussian);

            let symbol =
                coder
                    .decode_symbol(model)
                    .map_err(|e| ZVC69Error::EntropyDecodingFailed {
                        reason: format!("Gaussian conditional decode failed: {:?}", e),
                    })?;

            symbols.push(symbol);
        }

        // No reverse needed: encoding in reverse + forward decoding gives correct order
        Ok(symbols)
    }
}

impl Default for GaussianConditional {
    fn default() -> Self {
        Self::new(DEFAULT_SCALE_BOUND, DEFAULT_NUM_SCALES)
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_latents() {
        let values = vec![1.2, -0.7, 3.9, -2.1, 0.5, -0.5];
        let quantized = quantize_latents(&values);
        // Rust's f32::round() rounds half away from zero: 0.5 -> 1, -0.5 -> -1
        assert_eq!(quantized, vec![1, -1, 4, -2, 1, -1]);
    }

    #[test]
    fn test_dequantize_latents() {
        let symbols = vec![1, -1, 4, -2, 0];
        let values = dequantize_latents(&symbols);
        assert_eq!(values, vec![1.0, -1.0, 4.0, -2.0, 0.0]);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = vec![5, -3, 0, 10, -7];
        let float_vals = dequantize_latents(&original);
        let roundtrip = quantize_latents(&float_vals);
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_estimate_bits_basic() {
        // Symbols close to means with small scales should have low bit cost
        let symbols = vec![0, 0, 0];
        let means = vec![0.0, 0.0, 0.0];
        let scales = vec![1.0, 1.0, 1.0];

        let bits = estimate_bits(&symbols, &means, &scales);
        assert!(bits > 0.0);
        assert!(bits < 10.0); // Should be quite low for perfect predictions
    }

    #[test]
    fn test_estimate_bits_far_from_mean() {
        // Symbols far from means should have higher bit cost
        let symbols = vec![10, -10, 20];
        let means = vec![0.0, 0.0, 0.0];
        let scales = vec![1.0, 1.0, 1.0];

        let bits_far = estimate_bits(&symbols, &means, &scales);

        let symbols_close = vec![0, 0, 0];
        let bits_close = estimate_bits(&symbols_close, &means, &scales);

        assert!(bits_far > bits_close);
    }

    #[test]
    fn test_estimate_bits_larger_scale() {
        // Larger scale (more uncertainty) should reduce bit cost for off-center symbols
        let symbols = vec![5, 5, 5];
        let means = vec![0.0, 0.0, 0.0];

        let bits_small_scale = estimate_bits(&symbols, &means, &vec![1.0, 1.0, 1.0]);
        let bits_large_scale = estimate_bits(&symbols, &means, &vec![10.0, 10.0, 10.0]);

        // Larger scale means more probability spread, so symbols away from mean cost less
        assert!(bits_large_scale < bits_small_scale);
    }

    #[test]
    fn test_gaussian_to_cdf() {
        let cdf = gaussian_to_cdf(0.0, 1.0, 256, 16);

        // CDF should start at 0 and end at 2^16
        assert_eq!(cdf[0], 0);
        assert_eq!(*cdf.last().unwrap(), 65536);

        // CDF should be monotonically increasing
        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1]);
        }

        // Middle of CDF (around mean) should be around 2^15
        let mid_idx = cdf.len() / 2;
        let mid_val = cdf[mid_idx];
        assert!(mid_val > 20000 && mid_val < 45000);
    }

    #[test]
    fn test_gaussian_to_cdf_offset_mean() {
        let cdf_centered = gaussian_to_cdf(0.0, 1.0, 256, 16);
        let cdf_offset = gaussian_to_cdf(10.0, 1.0, 256, 16);

        // Both should have same structure but offset values
        assert_eq!(cdf_centered.len(), cdf_offset.len());
        assert_eq!(cdf_centered[0], cdf_offset[0]); // Both start at 0
    }

    #[test]
    fn test_quantized_gaussian_probability() {
        let qg = QuantizedGaussian::new(0.0, 1.0);

        // Probability at mean should be highest
        let prob_at_mean = qg.probability(0);
        let prob_off_center = qg.probability(3);

        assert!(prob_at_mean > prob_off_center);
        assert!(prob_at_mean > 0.0);
        assert!(prob_at_mean < 1.0);
    }

    #[test]
    fn test_quantized_gaussian_out_of_range() {
        let qg = QuantizedGaussian::new(0.0, 1.0);

        assert_eq!(qg.probability(-200), 0.0);
        assert_eq!(qg.probability(200), 0.0);
    }

    #[test]
    fn test_entropy_coder_encode_decode_roundtrip() {
        let mut coder = EntropyCoder::new();

        let symbols = vec![5, -3, 12, 0, -7, 8, -1, 3];
        let means = vec![4.5, -2.8, 11.2, 0.1, -6.5, 7.8, -0.9, 2.5];
        let scales = vec![1.0, 1.5, 2.0, 0.5, 1.2, 1.8, 0.8, 1.1];

        let compressed = coder.encode_symbols(&symbols, &means, &scales).unwrap();
        assert!(!compressed.is_empty());

        let decoded = coder
            .decode_symbols(&compressed, &means, &scales, symbols.len())
            .unwrap();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_entropy_coder_empty_input() {
        let mut coder = EntropyCoder::new();

        let compressed = coder.encode_symbols(&[], &[], &[]).unwrap();
        assert!(compressed.is_empty());

        let decoded = coder.decode_symbols(&compressed, &[], &[], 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_entropy_coder_single_symbol() {
        let mut coder = EntropyCoder::new();

        let symbols = vec![42];
        let means = vec![40.0];
        let scales = vec![5.0];

        let compressed = coder.encode_symbols(&symbols, &means, &scales).unwrap();
        let decoded = coder
            .decode_symbols(&compressed, &means, &scales, 1)
            .unwrap();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_entropy_coder_extreme_values() {
        let mut coder = EntropyCoder::new();

        let symbols = vec![-128, 127, 0, -100, 100];
        let means = vec![-128.0, 127.0, 0.0, -100.0, 100.0];
        let scales = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let compressed = coder.encode_symbols(&symbols, &means, &scales).unwrap();
        let decoded = coder
            .decode_symbols(&compressed, &means, &scales, symbols.len())
            .unwrap();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_entropy_coder_length_mismatch() {
        let mut coder = EntropyCoder::new();

        let symbols = vec![1, 2, 3];
        let means = vec![0.0, 0.0]; // Wrong length
        let scales = vec![1.0, 1.0, 1.0];

        let result = coder.encode_symbols(&symbols, &means, &scales);
        assert!(result.is_err());
    }

    #[test]
    fn test_entropy_coder_out_of_range_symbol() {
        let mut coder = EntropyCoder::new();

        let symbols = vec![0, 200, 0]; // 200 is out of range
        let means = vec![0.0, 0.0, 0.0];
        let scales = vec![1.0, 1.0, 1.0];

        let result = coder.encode_symbols(&symbols, &means, &scales);
        assert!(result.is_err());
    }

    #[test]
    fn test_factorized_prior_encode_decode() {
        let prior = FactorizedPrior::new(128);

        // Create some test latents
        let latents: Vec<i32> = (0..256).map(|i| ((i % 20) as i32) - 10).collect();

        let compressed = prior.encode(&latents).unwrap();
        assert!(!compressed.is_empty());

        let decoded = prior.decode(&compressed, latents.len()).unwrap();
        assert_eq!(latents, decoded);
    }

    #[test]
    fn test_factorized_prior_empty() {
        let prior = FactorizedPrior::new(128);

        let compressed = prior.encode(&[]).unwrap();
        assert!(compressed.is_empty());

        let decoded = prior.decode(&compressed, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_gaussian_conditional_encode_decode() {
        let gc = GaussianConditional::default();

        let latents = vec![5, -3, 12, 0, -7, 8, -1, 3];
        let means = vec![4.5, -2.8, 11.2, 0.1, -6.5, 7.8, -0.9, 2.5];
        let scales = vec![1.0, 1.5, 2.0, 0.5, 1.2, 1.8, 0.8, 1.1];

        let compressed = gc.encode(&latents, &means, &scales).unwrap();
        assert!(!compressed.is_empty());

        let decoded = gc
            .decode(&compressed, &means, &scales, latents.len())
            .unwrap();

        assert_eq!(latents, decoded);
    }

    #[test]
    fn test_gaussian_conditional_scale_table() {
        let gc = GaussianConditional::new(0.11, 64);

        let table = gc.scale_table();
        assert_eq!(table.len(), 64);

        // Scale table should be sorted
        for i in 1..table.len() {
            assert!(table[i] >= table[i - 1]);
        }

        // First value should be near scale_bound
        assert!(table[0] >= 0.01);
        assert!(table[0] <= 0.2);

        // Last value should be near MAX_SCALE
        assert!(table[table.len() - 1] >= 100.0);
    }

    #[test]
    fn test_gaussian_conditional_quantize_scales() {
        let gc = GaussianConditional::new(0.11, 64);

        let scales = vec![0.5, 1.0, 5.0, 20.0];
        let quantized = gc.quantize_scales(&scales);

        assert_eq!(quantized.len(), scales.len());

        // Each quantized scale should be in the table
        for &qs in &quantized {
            assert!(gc.scale_table().iter().any(|&t| (t - qs).abs() < 0.001));
        }
    }

    #[test]
    fn test_words_bytes_conversion() {
        let words = vec![0x12345678u32, 0xABCDEF01, 0x00FF00FF];

        let bytes = words_to_bytes(&words);
        assert_eq!(bytes.len(), 12);

        let recovered = bytes_to_words(&bytes);
        assert_eq!(words, recovered);
    }

    #[test]
    fn test_words_bytes_conversion_empty() {
        let words: Vec<u32> = vec![];
        let bytes = words_to_bytes(&words);
        assert!(bytes.is_empty());

        let recovered = bytes_to_words(&bytes);
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_normal_cdf_properties() {
        // CDF(0) for standard normal should be 0.5
        let cdf_zero = normal_cdf(0.0);
        assert!((cdf_zero - 0.5).abs() < 0.001);

        // CDF(-inf) -> 0, CDF(+inf) -> 1
        let cdf_neg = normal_cdf(-10.0);
        let cdf_pos = normal_cdf(10.0);
        assert!(cdf_neg < 0.001);
        assert!(cdf_pos > 0.999);

        // CDF should be monotonically increasing
        let cdf_1 = normal_cdf(-1.0);
        let cdf_2 = normal_cdf(0.0);
        let cdf_3 = normal_cdf(1.0);
        assert!(cdf_1 < cdf_2);
        assert!(cdf_2 < cdf_3);
    }

    #[test]
    fn test_compression_efficiency() {
        let mut coder = EntropyCoder::new();

        // Well-predicted symbols (close to means) should compress well
        let n = 1000;
        let symbols: Vec<i32> = (0..n).map(|i| (i % 10) as i32 - 5).collect();
        let means: Vec<f32> = symbols.iter().map(|&s| s as f32 + 0.1).collect();
        let scales: Vec<f32> = vec![0.5; n];

        let compressed = coder.encode_symbols(&symbols, &means, &scales).unwrap();

        // Compressed size should be much smaller than raw data
        let raw_size = n * 4; // 4 bytes per i32
        let compressed_size = compressed.len();

        // With good predictions, compression ratio should be significant
        assert!(compressed_size < raw_size / 2);
    }

    #[test]
    fn test_zero_scale_handling() {
        let mut coder = EntropyCoder::new();

        // Zero scale should be clamped to MIN_SCALE
        let symbols = vec![0, 1, -1];
        let means = vec![0.0, 1.0, -1.0];
        let scales = vec![0.0, 0.0, 0.0]; // All zero scales

        let compressed = coder.encode_symbols(&symbols, &means, &scales).unwrap();
        let decoded = coder
            .decode_symbols(&compressed, &means, &scales, symbols.len())
            .unwrap();

        assert_eq!(symbols, decoded);
    }

    #[test]
    fn test_large_scale_handling() {
        let mut coder = EntropyCoder::new();

        // Very large scales should be clamped to MAX_SCALE
        let symbols = vec![0, 10, -10];
        let means = vec![0.0, 0.0, 0.0];
        let scales = vec![1000.0, 10000.0, 100000.0]; // Very large scales

        let compressed = coder.encode_symbols(&symbols, &means, &scales).unwrap();
        let decoded = coder
            .decode_symbols(&compressed, &means, &scales, symbols.len())
            .unwrap();

        assert_eq!(symbols, decoded);
    }
}
