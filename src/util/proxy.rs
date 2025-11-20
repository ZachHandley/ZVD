//! Proxy Video Generation
//!
//! Generate low-resolution proxy files for faster editing and playback.
//! Proxy workflows are essential for editing high-resolution footage (4K/8K)
//! on systems with limited resources.
//!
//! ## Proxy Workflow
//!
//! 1. **Offline Editing**: Edit using low-res proxies
//! 2. **Online Finishing**: Relink to full-res originals for final export
//!
//! ## Common Proxy Formats
//!
//! - **H.264/H.265**: Good compression, wide compatibility
//! - **MJPEG**: Frame-accurate, no GOP, easier scrubbing
//! - **ProRes Proxy**: Professional standard, Apple ecosystem
//! - **DNxHR LB**: Avid ecosystem, low bitrate
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::proxy::{ProxyGenerator, ProxyPreset};
//!
//! let generator = ProxyGenerator::new(ProxyPreset::H264Quarter)?;
//! generator.generate("source.mp4", "proxy.mp4")?;
//! ```

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// Proxy resolution preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyResolution {
    /// Full resolution (1:1) - for reference
    Full,
    /// Half resolution (1:2) - 1920x1080 -> 960x540
    Half,
    /// Quarter resolution (1:4) - 1920x1080 -> 480x270
    Quarter,
    /// Eighth resolution (1:8) - 3840x2160 -> 480x270
    Eighth,
    /// Custom resolution
    Custom { width: u32, height: u32 },
}

impl ProxyResolution {
    /// Calculate proxy dimensions from source dimensions
    pub fn calculate_dimensions(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        match self {
            ProxyResolution::Full => (src_width, src_height),
            ProxyResolution::Half => (src_width / 2, src_height / 2),
            ProxyResolution::Quarter => (src_width / 4, src_height / 4),
            ProxyResolution::Eighth => (src_width / 8, src_height / 8),
            ProxyResolution::Custom { width, height } => (*width, *height),
        }
    }

    /// Get scale factor
    pub fn scale_factor(&self) -> f32 {
        match self {
            ProxyResolution::Full => 1.0,
            ProxyResolution::Half => 0.5,
            ProxyResolution::Quarter => 0.25,
            ProxyResolution::Eighth => 0.125,
            ProxyResolution::Custom { .. } => 1.0, // Calculated separately
        }
    }
}

/// Proxy codec format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyCodec {
    /// H.264 (AVC) - widely compatible
    H264,
    /// H.265 (HEVC) - better compression
    H265,
    /// MJPEG - frame-accurate, no GOP
    MJPEG,
    /// ProRes Proxy - professional standard
    ProResProxy,
    /// DNxHR LB (Low Bandwidth) - Avid standard
    DNxHRLB,
}

impl ProxyCodec {
    /// Get recommended bitrate for resolution
    pub fn recommended_bitrate(&self, width: u32, height: u32) -> u32 {
        let pixels = width * height;
        match self {
            ProxyCodec::H264 => {
                // ~0.1 bits per pixel
                (pixels as f32 * 0.1 * 30.0) as u32 // Assuming 30fps
            }
            ProxyCodec::H265 => {
                // ~0.05 bits per pixel (50% of H.264)
                (pixels as f32 * 0.05 * 30.0) as u32
            }
            ProxyCodec::MJPEG => {
                // ~0.3 bits per pixel (intra-only)
                (pixels as f32 * 0.3 * 30.0) as u32
            }
            ProxyCodec::ProResProxy => {
                // Fixed bitrate based on resolution
                match (width, height) {
                    (_, h) if h <= 540 => 10_000_000,  // 10 Mbps for SD/540p
                    (_, h) if h <= 720 => 18_000_000,  // 18 Mbps for 720p
                    (_, h) if h <= 1080 => 36_000_000, // 36 Mbps for 1080p
                    _ => 45_000_000,                   // 45 Mbps for higher
                }
            }
            ProxyCodec::DNxHRLB => {
                // DNxHR LB: ~8 Mbps @ 1080p
                match (width, height) {
                    (_, h) if h <= 540 => 4_000_000,
                    (_, h) if h <= 720 => 6_000_000,
                    (_, h) if h <= 1080 => 8_000_000,
                    _ => 12_000_000,
                }
            }
        }
    }

    /// Get file extension for codec
    pub fn file_extension(&self) -> &'static str {
        match self {
            ProxyCodec::H264 => "mp4",
            ProxyCodec::H265 => "mp4",
            ProxyCodec::MJPEG => "mov",
            ProxyCodec::ProResProxy => "mov",
            ProxyCodec::DNxHRLB => "mxf",
        }
    }
}

/// Proxy quality preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyQuality {
    /// Draft quality (lowest bitrate, fastest encoding)
    Draft,
    /// Standard proxy quality
    Standard,
    /// High quality proxy (higher bitrate)
    High,
}

impl ProxyQuality {
    /// Get bitrate multiplier
    pub fn bitrate_multiplier(&self) -> f32 {
        match self {
            ProxyQuality::Draft => 0.5,
            ProxyQuality::Standard => 1.0,
            ProxyQuality::High => 1.5,
        }
    }
}

/// Pre-configured proxy presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyPreset {
    /// H.264 Half Resolution (fast, compatible)
    H264Half,
    /// H.264 Quarter Resolution (very fast)
    H264Quarter,
    /// MJPEG Quarter (frame-accurate scrubbing)
    MJPEGQuarter,
    /// ProRes Proxy Quarter (professional workflow)
    ProResProxyQuarter,
    /// DNxHR LB Quarter (Avid workflow)
    DNxHRQuarter,
}

impl ProxyPreset {
    /// Get codec for preset
    pub fn codec(&self) -> ProxyCodec {
        match self {
            ProxyPreset::H264Half | ProxyPreset::H264Quarter => ProxyCodec::H264,
            ProxyPreset::MJPEGQuarter => ProxyCodec::MJPEG,
            ProxyPreset::ProResProxyQuarter => ProxyCodec::ProResProxy,
            ProxyPreset::DNxHRQuarter => ProxyCodec::DNxHRLB,
        }
    }

    /// Get resolution for preset
    pub fn resolution(&self) -> ProxyResolution {
        match self {
            ProxyPreset::H264Half => ProxyResolution::Half,
            ProxyPreset::H264Quarter
            | ProxyPreset::MJPEGQuarter
            | ProxyPreset::ProResProxyQuarter
            | ProxyPreset::DNxHRQuarter => ProxyResolution::Quarter,
        }
    }

    /// Get quality for preset
    pub fn quality(&self) -> ProxyQuality {
        ProxyQuality::Standard
    }
}

/// Proxy configuration
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Resolution preset
    pub resolution: ProxyResolution,
    /// Codec to use
    pub codec: ProxyCodec,
    /// Quality preset
    pub quality: ProxyQuality,
    /// Target bitrate (overrides automatic calculation)
    pub target_bitrate: Option<u32>,
    /// Audio bitrate (kbps)
    pub audio_bitrate: u32,
    /// Preserve audio channels
    pub audio_channels: Option<u32>,
    /// Frame rate (None = preserve original)
    pub frame_rate: Option<f64>,
}

impl ProxyConfig {
    /// Create from preset
    pub fn from_preset(preset: ProxyPreset) -> Self {
        ProxyConfig {
            resolution: preset.resolution(),
            codec: preset.codec(),
            quality: preset.quality(),
            target_bitrate: None,
            audio_bitrate: 128, // 128 kbps for proxy audio
            audio_channels: Some(2), // Stereo
            frame_rate: None,
        }
    }

    /// Calculate effective bitrate
    pub fn effective_bitrate(&self, width: u32, height: u32) -> u32 {
        if let Some(bitrate) = self.target_bitrate {
            bitrate
        } else {
            let base_bitrate = self.codec.recommended_bitrate(width, height);
            (base_bitrate as f32 * self.quality.bitrate_multiplier()) as u32
        }
    }
}

impl Default for ProxyConfig {
    fn default() -> Self {
        ProxyConfig::from_preset(ProxyPreset::H264Quarter)
    }
}

/// Proxy metadata (for linking back to original)
#[derive(Debug, Clone)]
pub struct ProxyMetadata {
    /// Original file path
    pub original_path: PathBuf,
    /// Original resolution
    pub original_width: u32,
    pub original_height: u32,
    /// Proxy resolution
    pub proxy_width: u32,
    pub proxy_height: u32,
    /// Scale factor
    pub scale_factor: f32,
    /// Codec used
    pub codec: ProxyCodec,
    /// Timecode offset (if any)
    pub timecode_offset: Option<String>,
}

impl ProxyMetadata {
    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        let json = format!(
            r#"{{
  "original_path": "{}",
  "original_resolution": [{}, {}],
  "proxy_resolution": [{}, {}],
  "scale_factor": {},
  "codec": "{:?}",
  "timecode_offset": {}
}}"#,
            self.original_path.display(),
            self.original_width,
            self.original_height,
            self.proxy_width,
            self.proxy_height,
            self.scale_factor,
            self.codec,
            self.timecode_offset
                .as_ref()
                .map(|s| format!("\"{}\"", s))
                .unwrap_or_else(|| "null".to_string())
        );
        Ok(json)
    }

    /// Save to sidecar file (.proxy.json)
    pub fn save_sidecar<P: AsRef<Path>>(&self, proxy_path: P) -> Result<()> {
        let json = self.to_json()?;
        let sidecar_path = proxy_path.as_ref().with_extension("proxy.json");
        std::fs::write(&sidecar_path, json).map_err(Error::Io)?;
        Ok(())
    }
}

/// Proxy generator
pub struct ProxyGenerator {
    config: ProxyConfig,
}

impl ProxyGenerator {
    /// Create new proxy generator with config
    pub fn new(config: ProxyConfig) -> Result<Self> {
        Ok(ProxyGenerator { config })
    }

    /// Create from preset
    pub fn from_preset(preset: ProxyPreset) -> Result<Self> {
        Self::new(ProxyConfig::from_preset(preset))
    }

    /// Generate proxy file
    ///
    /// This is a stub implementation. In a real implementation, you would:
    /// 1. Open source file with demuxer
    /// 2. Get source video/audio stream info
    /// 3. Calculate proxy dimensions
    /// 4. Set up scale filter
    /// 5. Set up encoder with proxy settings
    /// 6. Process frames: decode -> scale -> encode
    /// 7. Mux to output file
    /// 8. Generate and save metadata sidecar
    pub fn generate<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        source_path: P,
        output_path: Q,
    ) -> Result<ProxyMetadata> {
        let source_path = source_path.as_ref();
        let output_path = output_path.as_ref();

        // Stub: In real implementation, get from source
        let src_width = 1920;
        let src_height = 1080;

        let (proxy_width, proxy_height) = self
            .config
            .resolution
            .calculate_dimensions(src_width, src_height);

        let metadata = ProxyMetadata {
            original_path: source_path.to_path_buf(),
            original_width: src_width,
            original_height: src_height,
            proxy_width,
            proxy_height,
            scale_factor: self.config.resolution.scale_factor(),
            codec: self.config.codec,
            timecode_offset: None,
        };

        // Save metadata sidecar
        metadata.save_sidecar(output_path)?;

        // Stub: Real implementation would encode here
        // 1. Open source with demuxer
        // 2. Set up scale filter
        // 3. Set up encoder
        // 4. Process: decode -> scale -> encode
        // 5. Write to muxer

        Ok(metadata)
    }

    /// Generate proxy with automatic naming
    pub fn generate_auto<P: AsRef<Path>>(&self, source_path: P) -> Result<ProxyMetadata> {
        let source_path = source_path.as_ref();
        let ext = self.config.codec.file_extension();

        let proxy_path = source_path.with_extension(format!("proxy.{}", ext));

        self.generate(source_path, proxy_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_calculation() {
        let half = ProxyResolution::Half;
        assert_eq!(half.calculate_dimensions(1920, 1080), (960, 540));

        let quarter = ProxyResolution::Quarter;
        assert_eq!(quarter.calculate_dimensions(3840, 2160), (960, 540));

        let eighth = ProxyResolution::Eighth;
        assert_eq!(eighth.calculate_dimensions(7680, 4320), (960, 540));
    }

    #[test]
    fn test_h264_bitrate() {
        let codec = ProxyCodec::H264;
        let bitrate = codec.recommended_bitrate(1920, 1080);
        assert!(bitrate > 0);
        assert!(bitrate < 10_000_000); // Should be reasonable for proxy
    }

    #[test]
    fn test_prores_proxy_bitrate() {
        let codec = ProxyCodec::ProResProxy;
        let bitrate_1080 = codec.recommended_bitrate(1920, 1080);
        let bitrate_720 = codec.recommended_bitrate(1280, 720);

        assert_eq!(bitrate_1080, 36_000_000); // 36 Mbps for 1080p
        assert_eq!(bitrate_720, 18_000_000); // 18 Mbps for 720p
        assert!(bitrate_1080 > bitrate_720);
    }

    #[test]
    fn test_preset_h264_quarter() {
        let preset = ProxyPreset::H264Quarter;
        assert_eq!(preset.codec(), ProxyCodec::H264);
        assert_eq!(preset.resolution(), ProxyResolution::Quarter);
    }

    #[test]
    fn test_config_from_preset() {
        let config = ProxyConfig::from_preset(ProxyPreset::H264Quarter);
        assert_eq!(config.codec, ProxyCodec::H264);
        assert_eq!(config.resolution, ProxyResolution::Quarter);
        assert_eq!(config.audio_bitrate, 128);
    }

    #[test]
    fn test_effective_bitrate() {
        let config = ProxyConfig::from_preset(ProxyPreset::H264Quarter);
        let bitrate = config.effective_bitrate(1920, 1080);
        assert!(bitrate > 0);
    }

    #[test]
    fn test_quality_multiplier() {
        assert_eq!(ProxyQuality::Draft.bitrate_multiplier(), 0.5);
        assert_eq!(ProxyQuality::Standard.bitrate_multiplier(), 1.0);
        assert_eq!(ProxyQuality::High.bitrate_multiplier(), 1.5);
    }

    #[test]
    fn test_metadata_creation() {
        let metadata = ProxyMetadata {
            original_path: PathBuf::from("/source/video.mp4"),
            original_width: 1920,
            original_height: 1080,
            proxy_width: 480,
            proxy_height: 270,
            scale_factor: 0.25,
            codec: ProxyCodec::H264,
            timecode_offset: Some("00:00:00:00".to_string()),
        };

        let json = metadata.to_json().unwrap();
        assert!(json.contains("1920"));
        assert!(json.contains("480"));
        assert!(json.contains("0.25"));
    }

    #[test]
    fn test_codec_extensions() {
        assert_eq!(ProxyCodec::H264.file_extension(), "mp4");
        assert_eq!(ProxyCodec::H265.file_extension(), "mp4");
        assert_eq!(ProxyCodec::MJPEG.file_extension(), "mov");
        assert_eq!(ProxyCodec::ProResProxy.file_extension(), "mov");
        assert_eq!(ProxyCodec::DNxHRLB.file_extension(), "mxf");
    }
}
