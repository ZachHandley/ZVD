//! ZVC69 Neural Video Codec configuration
//!
//! This module provides comprehensive configuration options for the ZVC69 neural codec,
//! including quality levels, encoding presets, and fine-grained control over the
//! neural network inference pipeline.

use super::error::ZVC69Error;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Quality Level
// ─────────────────────────────────────────────────────────────────────────────

/// Quality level for ZVC69 encoding (1-8)
///
/// Higher values produce better visual quality at the cost of higher bitrates.
/// Quality levels roughly correspond to:
/// - 1-2: Low quality (aggressive compression, suitable for previews)
/// - 3-4: Medium quality (balanced compression/quality)
/// - 5-6: High quality (good visual fidelity)
/// - 7-8: Very high quality (near-lossless perceptual quality)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Quality {
    /// Level 1 - Lowest quality, smallest file size
    Q1 = 1,
    /// Level 2
    Q2 = 2,
    /// Level 3
    Q3 = 3,
    /// Level 4 - Medium quality (default)
    Q4 = 4,
    /// Level 5
    Q5 = 5,
    /// Level 6 - High quality
    Q6 = 6,
    /// Level 7
    Q7 = 7,
    /// Level 8 - Highest quality, largest file size
    Q8 = 8,
}

impl Quality {
    /// Get the quality level as a numeric value (1-8)
    pub fn level(&self) -> u8 {
        *self as u8
    }

    /// Get the internal quantization scale factor
    ///
    /// Returns a value in [0.0, 1.0] where lower means more compression
    pub fn quant_scale(&self) -> f32 {
        match self {
            Quality::Q1 => 0.05,
            Quality::Q2 => 0.10,
            Quality::Q3 => 0.18,
            Quality::Q4 => 0.28,
            Quality::Q5 => 0.42,
            Quality::Q6 => 0.58,
            Quality::Q7 => 0.75,
            Quality::Q8 => 0.95,
        }
    }

    /// Create from a numeric level (clamped to valid range)
    pub fn from_level(level: u8) -> Self {
        match level {
            0 | 1 => Quality::Q1,
            2 => Quality::Q2,
            3 => Quality::Q3,
            4 => Quality::Q4,
            5 => Quality::Q5,
            6 => Quality::Q6,
            7 => Quality::Q7,
            _ => Quality::Q8,
        }
    }

    /// Get the approximate bitrate multiplier relative to Q4
    pub fn bitrate_multiplier(&self) -> f32 {
        match self {
            Quality::Q1 => 0.25,
            Quality::Q2 => 0.40,
            Quality::Q3 => 0.60,
            Quality::Q4 => 1.00,
            Quality::Q5 => 1.50,
            Quality::Q6 => 2.20,
            Quality::Q7 => 3.50,
            Quality::Q8 => 5.50,
        }
    }
}

impl Default for Quality {
    fn default() -> Self {
        Quality::Q4
    }
}

impl fmt::Display for Quality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q{}", self.level())
    }
}

impl TryFrom<u8> for Quality {
    type Error = ZVC69Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Quality::Q1),
            2 => Ok(Quality::Q2),
            3 => Ok(Quality::Q3),
            4 => Ok(Quality::Q4),
            5 => Ok(Quality::Q5),
            6 => Ok(Quality::Q6),
            7 => Ok(Quality::Q7),
            8 => Ok(Quality::Q8),
            _ => Err(ZVC69Error::InvalidQuantParam {
                param: "quality".to_string(),
                value: value as i32,
                min: 1,
                max: 8,
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoding Preset
// ─────────────────────────────────────────────────────────────────────────────

/// Encoding preset controlling speed vs compression efficiency trade-off
///
/// Faster presets encode quickly but may produce larger files at the same quality.
/// Slower presets use more advanced neural network features and optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Ultra fast encoding - minimal neural processing
    ///
    /// - Single-pass inference
    /// - No motion refinement
    /// - Basic entropy coding
    /// - Target: Real-time encoding on mid-range GPU
    Ultrafast,

    /// Fast encoding - quick neural processing
    ///
    /// - Single-pass inference
    /// - Basic motion estimation
    /// - Standard entropy coding
    /// - Target: Near real-time encoding
    Fast,

    /// Medium encoding (default) - balanced speed/compression
    ///
    /// - Standard neural inference
    /// - Motion estimation with refinement
    /// - Adaptive entropy coding
    /// - Target: ~2x real-time on high-end GPU
    Medium,

    /// Slow encoding - better compression
    ///
    /// - Multi-scale neural inference
    /// - Advanced motion estimation
    /// - Full entropy model
    /// - Target: ~0.5x real-time
    Slow,

    /// Very slow encoding - maximum compression
    ///
    /// - Multi-pass neural inference
    /// - Exhaustive motion search
    /// - Optimal entropy coding with look-ahead
    /// - Target: Offline encoding only
    Veryslow,
}

impl Preset {
    /// Get the name of this preset
    pub fn name(&self) -> &'static str {
        match self {
            Preset::Ultrafast => "ultrafast",
            Preset::Fast => "fast",
            Preset::Medium => "medium",
            Preset::Slow => "slow",
            Preset::Veryslow => "veryslow",
        }
    }

    /// Get the number of inference passes
    pub fn inference_passes(&self) -> u32 {
        match self {
            Preset::Ultrafast => 1,
            Preset::Fast => 1,
            Preset::Medium => 1,
            Preset::Slow => 2,
            Preset::Veryslow => 3,
        }
    }

    /// Get the motion estimation search range multiplier
    pub fn motion_search_range(&self) -> f32 {
        match self {
            Preset::Ultrafast => 0.25,
            Preset::Fast => 0.5,
            Preset::Medium => 1.0,
            Preset::Slow => 2.0,
            Preset::Veryslow => 4.0,
        }
    }

    /// Get the lookahead buffer size in frames
    pub fn lookahead_frames(&self) -> u32 {
        match self {
            Preset::Ultrafast => 0,
            Preset::Fast => 4,
            Preset::Medium => 16,
            Preset::Slow => 32,
            Preset::Veryslow => 64,
        }
    }

    /// Check if this preset enables motion refinement
    pub fn motion_refinement(&self) -> bool {
        !matches!(self, Preset::Ultrafast)
    }

    /// Check if this preset uses multi-scale inference
    pub fn multiscale(&self) -> bool {
        matches!(self, Preset::Slow | Preset::Veryslow)
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ultrafast" => Some(Preset::Ultrafast),
            "fast" => Some(Preset::Fast),
            "medium" => Some(Preset::Medium),
            "slow" => Some(Preset::Slow),
            "veryslow" => Some(Preset::Veryslow),
            _ => None,
        }
    }
}

impl Default for Preset {
    fn default() -> Self {
        Preset::Medium
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rate Control Mode
// ─────────────────────────────────────────────────────────────────────────────

/// Rate control mode for bitrate management
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateControlMode {
    /// Constant Rate Factor - targets perceptual quality
    ///
    /// The CRF value controls quality directly (lower = better).
    /// File size varies based on content complexity.
    Crf {
        /// CRF value (0-51, default 23)
        crf: f32,
    },

    /// Variable Bitrate - targets average bitrate
    ///
    /// Quality varies to maintain target bitrate.
    Vbr {
        /// Target bitrate in bits per second
        target_bitrate: u64,
        /// Maximum bitrate (optional, for constrained VBR)
        max_bitrate: Option<u64>,
    },

    /// Constant Bitrate - maintains fixed bitrate
    ///
    /// Uses buffer model to ensure consistent bitrate.
    Cbr {
        /// Target bitrate in bits per second
        bitrate: u64,
        /// VBV buffer size in bits
        vbv_buffer_size: u64,
    },

    /// Constant Quantization Parameter
    ///
    /// Fixed quantization, bitrate varies freely.
    Cqp {
        /// Quantization parameter for I-frames
        qp_i: u8,
        /// Quantization parameter for P-frames
        qp_p: u8,
        /// Quantization parameter for B-frames
        qp_b: u8,
    },
}

impl Default for RateControlMode {
    fn default() -> Self {
        RateControlMode::Crf { crf: 23.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GOP Structure
// ─────────────────────────────────────────────────────────────────────────────

/// Group of Pictures (GOP) structure configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GopConfig {
    /// Keyframe interval (I-frame distance)
    ///
    /// Set to 0 for intra-only encoding.
    pub keyframe_interval: u32,

    /// B-frame count between P-frames
    ///
    /// Set to 0 to disable B-frames (low latency mode).
    pub bframes: u32,

    /// Enable adaptive B-frame placement
    pub adaptive_bframes: bool,

    /// Enable scene change detection for keyframe insertion
    pub scene_detection: bool,

    /// Minimum keyframe interval (prevents too frequent keyframes)
    pub min_keyframe_interval: u32,
}

impl Default for GopConfig {
    fn default() -> Self {
        GopConfig {
            keyframe_interval: 250,
            bframes: 3,
            adaptive_bframes: true,
            scene_detection: true,
            min_keyframe_interval: 25,
        }
    }
}

impl GopConfig {
    /// Create intra-only configuration (no inter-frame prediction)
    pub fn intra_only() -> Self {
        GopConfig {
            keyframe_interval: 1,
            bframes: 0,
            adaptive_bframes: false,
            scene_detection: false,
            min_keyframe_interval: 1,
        }
    }

    /// Create low-latency configuration (no B-frames)
    pub fn low_latency() -> Self {
        GopConfig {
            keyframe_interval: 60,
            bframes: 0,
            adaptive_bframes: false,
            scene_detection: true,
            min_keyframe_interval: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural Model Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Neural network model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the encoder model (ONNX format)
    pub encoder_model_path: Option<String>,

    /// Path to the decoder model (ONNX format)
    pub decoder_model_path: Option<String>,

    /// Path to the entropy model
    pub entropy_model_path: Option<String>,

    /// Enable FP16 inference for speed
    pub use_fp16: bool,

    /// Enable INT8 quantized inference (fastest, slight quality loss)
    pub use_int8: bool,

    /// Batch size for inference
    pub batch_size: u32,

    /// Use TensorRT optimization (NVIDIA GPUs only)
    pub use_tensorrt: bool,

    /// GPU device index (-1 for CPU)
    pub gpu_device: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            encoder_model_path: None,
            decoder_model_path: None,
            entropy_model_path: None,
            use_fp16: true,
            use_int8: false,
            batch_size: 1,
            use_tensorrt: false,
            gpu_device: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Configuration Struct
// ─────────────────────────────────────────────────────────────────────────────

/// Complete ZVC69 encoder/decoder configuration
///
/// Use the builder pattern via `ZVC69Config::builder()` for ergonomic construction.
#[derive(Debug, Clone)]
pub struct ZVC69Config {
    // ── Video Dimensions ──
    /// Frame width in pixels (must be divisible by 16)
    pub width: u32,

    /// Frame height in pixels (must be divisible by 16)
    pub height: u32,

    // ── Encoding Parameters ──
    /// Quality level (1-8)
    pub quality: Quality,

    /// Encoding preset
    pub preset: Preset,

    /// Rate control mode
    pub rate_control: RateControlMode,

    /// GOP structure
    pub gop: GopConfig,

    // ── Frame Rate ──
    /// Frame rate numerator
    pub framerate_num: u32,

    /// Frame rate denominator
    pub framerate_den: u32,

    // ── Color Space ──
    /// Color primaries (BT.709, BT.2020, etc.)
    pub color_primaries: ColorPrimaries,

    /// Transfer characteristics
    pub transfer_characteristics: TransferCharacteristics,

    /// Matrix coefficients
    pub matrix_coefficients: MatrixCoefficients,

    /// Full range vs limited range
    pub full_range: bool,

    // ── Neural Model ──
    /// Neural model configuration
    pub model: ModelConfig,

    // ── Threading ──
    /// Number of encoding threads (0 = auto-detect)
    pub threads: u32,

    /// Enable tile-based parallel encoding
    pub tile_columns: u32,

    /// Enable tile-based parallel encoding
    pub tile_rows: u32,
}

impl Default for ZVC69Config {
    fn default() -> Self {
        ZVC69Config {
            width: 1920,
            height: 1080,
            quality: Quality::default(),
            preset: Preset::default(),
            rate_control: RateControlMode::default(),
            gop: GopConfig::default(),
            framerate_num: 30,
            framerate_den: 1,
            color_primaries: ColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Bt709,
            matrix_coefficients: MatrixCoefficients::Bt709,
            full_range: false,
            model: ModelConfig::default(),
            threads: 0,
            tile_columns: 1,
            tile_rows: 1,
        }
    }
}

impl ZVC69Config {
    /// Create a configuration builder
    pub fn builder() -> ZVC69ConfigBuilder {
        ZVC69ConfigBuilder::default()
    }

    /// Create a default config for given dimensions
    pub fn new(width: u32, height: u32) -> Self {
        ZVC69Config {
            width,
            height,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ZVC69Error> {
        // Check dimensions
        if self.width == 0 || self.height == 0 {
            return Err(ZVC69Error::invalid_config("Dimensions cannot be zero"));
        }

        if self.width > 8192 || self.height > 8192 {
            return Err(ZVC69Error::UnsupportedResolution {
                width: self.width,
                height: self.height,
                min_dim: 64,
                max_dim: 8192,
                alignment: 16,
            });
        }

        if self.width % 16 != 0 || self.height % 16 != 0 {
            return Err(ZVC69Error::invalid_config(format!(
                "Dimensions {}x{} must be divisible by 16",
                self.width, self.height
            )));
        }

        // Check framerate
        if self.framerate_num == 0 || self.framerate_den == 0 {
            return Err(ZVC69Error::invalid_config("Invalid framerate"));
        }

        // Check GOP
        if self.gop.bframes > 16 {
            return Err(ZVC69Error::invalid_config("B-frame count cannot exceed 16"));
        }

        // Check tiles
        if self.tile_columns == 0 || self.tile_rows == 0 {
            return Err(ZVC69Error::invalid_config(
                "Tile columns and rows must be at least 1",
            ));
        }

        Ok(())
    }

    /// Get the framerate as a float
    pub fn framerate(&self) -> f32 {
        self.framerate_num as f32 / self.framerate_den as f32
    }

    /// Get recommended bitrate for this configuration
    pub fn recommended_bitrate(&self) -> u64 {
        let pixels_per_frame = (self.width * self.height) as u64;
        let fps = self.framerate();
        let base_bpp = 0.1; // bits per pixel at Q4
        let quality_mult = self.quality.bitrate_multiplier();

        (pixels_per_frame as f32 * fps * base_bpp * quality_mult) as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder Pattern
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for ZVC69Config
#[derive(Debug, Clone, Default)]
pub struct ZVC69ConfigBuilder {
    config: ZVC69Config,
}

impl ZVC69ConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set frame dimensions
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Set frame width
    pub fn width(mut self, width: u32) -> Self {
        self.config.width = width;
        self
    }

    /// Set frame height
    pub fn height(mut self, height: u32) -> Self {
        self.config.height = height;
        self
    }

    /// Set quality level
    pub fn quality(mut self, quality: Quality) -> Self {
        self.config.quality = quality;
        self
    }

    /// Set quality by numeric level (1-8)
    pub fn quality_level(mut self, level: u8) -> Self {
        self.config.quality = Quality::from_level(level);
        self
    }

    /// Set encoding preset
    pub fn preset(mut self, preset: Preset) -> Self {
        self.config.preset = preset;
        self
    }

    /// Set preset by name
    pub fn preset_name(mut self, name: &str) -> Self {
        if let Some(preset) = Preset::from_str(name) {
            self.config.preset = preset;
        }
        self
    }

    /// Set rate control mode
    pub fn rate_control(mut self, mode: RateControlMode) -> Self {
        self.config.rate_control = mode;
        self
    }

    /// Set CRF rate control
    pub fn crf(mut self, crf: f32) -> Self {
        self.config.rate_control = RateControlMode::Crf { crf };
        self
    }

    /// Set target bitrate (VBR mode)
    pub fn bitrate(mut self, bitrate: u64) -> Self {
        self.config.rate_control = RateControlMode::Vbr {
            target_bitrate: bitrate,
            max_bitrate: None,
        };
        self
    }

    /// Set GOP configuration
    pub fn gop(mut self, gop: GopConfig) -> Self {
        self.config.gop = gop;
        self
    }

    /// Set keyframe interval
    pub fn keyframe_interval(mut self, interval: u32) -> Self {
        self.config.gop.keyframe_interval = interval;
        self
    }

    /// Set B-frame count
    pub fn bframes(mut self, count: u32) -> Self {
        self.config.gop.bframes = count;
        self
    }

    /// Set framerate
    pub fn framerate(mut self, num: u32, den: u32) -> Self {
        self.config.framerate_num = num;
        self.config.framerate_den = den;
        self
    }

    /// Set framerate as integer fps
    pub fn fps(mut self, fps: u32) -> Self {
        self.config.framerate_num = fps;
        self.config.framerate_den = 1;
        self
    }

    /// Set thread count
    pub fn threads(mut self, threads: u32) -> Self {
        self.config.threads = threads;
        self
    }

    /// Set tile configuration
    pub fn tiles(mut self, columns: u32, rows: u32) -> Self {
        self.config.tile_columns = columns;
        self.config.tile_rows = rows;
        self
    }

    /// Enable intra-only mode
    pub fn intra_only(mut self) -> Self {
        self.config.gop = GopConfig::intra_only();
        self
    }

    /// Enable low-latency mode
    pub fn low_latency(mut self) -> Self {
        self.config.gop = GopConfig::low_latency();
        self.config.preset = Preset::Fast;
        self
    }

    /// Set model configuration
    pub fn model(mut self, model: ModelConfig) -> Self {
        self.config.model = model;
        self
    }

    /// Set GPU device
    pub fn gpu(mut self, device: i32) -> Self {
        self.config.model.gpu_device = device;
        self
    }

    /// Enable TensorRT
    pub fn tensorrt(mut self, enable: bool) -> Self {
        self.config.model.use_tensorrt = enable;
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> Result<ZVC69Config, ZVC69Error> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation (use carefully)
    pub fn build_unchecked(self) -> ZVC69Config {
        self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Color Space Types
// ─────────────────────────────────────────────────────────────────────────────

/// Color primaries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPrimaries {
    /// BT.709 (sRGB, HD)
    #[default]
    Bt709,
    /// BT.2020 (UHD/HDR)
    Bt2020,
    /// SMPTE 170M (NTSC)
    Smpte170m,
    /// SMPTE 240M
    Smpte240m,
    /// Generic film
    Film,
    /// DCI-P3
    DciP3,
    /// Unspecified
    Unspecified,
}

/// Transfer characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferCharacteristics {
    /// BT.709
    #[default]
    Bt709,
    /// BT.2020 10-bit
    Bt202010,
    /// BT.2020 12-bit
    Bt202012,
    /// SMPTE ST 2084 (PQ / HDR10)
    SmpteSt2084,
    /// ARIB STD-B67 (HLG)
    AribStdB67,
    /// sRGB
    Srgb,
    /// Linear
    Linear,
    /// Unspecified
    Unspecified,
}

/// Matrix coefficients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixCoefficients {
    /// BT.709
    #[default]
    Bt709,
    /// BT.2020 non-constant luminance
    Bt2020Ncl,
    /// BT.2020 constant luminance
    Bt2020Cl,
    /// SMPTE 170M (NTSC)
    Smpte170m,
    /// SMPTE 240M
    Smpte240m,
    /// Identity (RGB)
    Identity,
    /// Unspecified
    Unspecified,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_levels() {
        assert_eq!(Quality::Q1.level(), 1);
        assert_eq!(Quality::Q8.level(), 8);
        assert!(Quality::Q1.quant_scale() < Quality::Q8.quant_scale());
    }

    #[test]
    fn test_quality_from_level() {
        assert_eq!(Quality::from_level(0), Quality::Q1);
        assert_eq!(Quality::from_level(4), Quality::Q4);
        assert_eq!(Quality::from_level(100), Quality::Q8);
    }

    #[test]
    fn test_preset_names() {
        assert_eq!(Preset::Ultrafast.name(), "ultrafast");
        assert_eq!(Preset::from_str("MEDIUM"), Some(Preset::Medium));
        assert_eq!(Preset::from_str("invalid"), None);
    }

    #[test]
    fn test_config_builder() {
        // Use dimensions divisible by 16 (1920x1088 instead of 1920x1080)
        let config = ZVC69Config::builder()
            .dimensions(1920, 1088)
            .quality(Quality::Q6)
            .preset(Preset::Fast)
            .fps(60)
            .build()
            .unwrap();

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1088);
        assert_eq!(config.quality, Quality::Q6);
        assert_eq!(config.preset, Preset::Fast);
        assert_eq!(config.framerate_num, 60);
    }

    #[test]
    fn test_config_validation() {
        // Valid config (dimensions divisible by 16)
        let config = ZVC69Config::new(1920, 1088);
        assert!(config.validate().is_ok());

        // Also valid: 1280x720
        let config = ZVC69Config::new(1280, 720);
        assert!(config.validate().is_ok());

        // Invalid: not divisible by 16
        let mut config = ZVC69Config::new(1920, 1088);
        config.width = 100;
        assert!(config.validate().is_err());

        // Invalid: zero dimension
        let mut config = ZVC69Config::new(1920, 1088);
        config.width = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gop_configs() {
        let intra = GopConfig::intra_only();
        assert_eq!(intra.keyframe_interval, 1);
        assert_eq!(intra.bframes, 0);

        let low_latency = GopConfig::low_latency();
        assert_eq!(low_latency.bframes, 0);
    }

    #[test]
    fn test_recommended_bitrate() {
        let config = ZVC69Config::builder()
            .dimensions(1920, 1088)
            .quality(Quality::Q4)
            .fps(30)
            .build_unchecked();

        let bitrate = config.recommended_bitrate();
        // Should be in reasonable range for 1080p30
        assert!(bitrate > 1_000_000); // > 1 Mbps
        assert!(bitrate < 100_000_000); // < 100 Mbps
    }
}
