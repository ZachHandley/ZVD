//! HDR (High Dynamic Range) Metadata Support
//!
//! This module provides comprehensive HDR metadata handling for modern video workflows.
//!
//! ## Supported Standards
//!
//! - **HDR10** - Static metadata (SMPTE ST 2086, SMPTE ST 2094-10)
//! - **HDR10+** - Dynamic metadata (SMPTE ST 2094-40)
//! - **Dolby Vision** - Dynamic metadata (proprietary)
//! - **HLG (Hybrid Log-Gamma)** - BBC/NHK standard
//!
//! ## Key Concepts
//!
//! **EOTF (Electro-Optical Transfer Function):**
//! - PQ (Perceptual Quantizer) - SMPTE ST 2084
//! - HLG (Hybrid Log-Gamma) - ITU-R BT.2100
//! - SDR (Standard Dynamic Range) - ITU-R BT.709
//!
//! **Color Primaries:**
//! - BT.2020 (Wide Color Gamut) - HDR standard
//! - DCI-P3 - Digital cinema
//! - BT.709 - SDR standard
//!
//! **Mastering Display:**
//! - Display primaries (R, G, B)
//! - White point
//! - Min/Max luminance
//!
//! **Content Light Level:**
//! - MaxCLL - Maximum Content Light Level
//! - MaxFALL - Maximum Frame-Average Light Level
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::format::hdr::{HdrMetadata, MasteringDisplayMetadata, ContentLightLevel};
//!
//! // Create HDR10 metadata
//! let mut hdr = HdrMetadata::hdr10();
//! hdr.mastering_display = Some(MasteringDisplayMetadata::rec2020_d65());
//! hdr.content_light = Some(ContentLightLevel::new(1000, 400));
//!
//! // Apply to video stream
//! stream.hdr_metadata = Some(hdr);
//! ```

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// HDR transfer function (EOTF)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferFunction {
    /// SDR - BT.709
    BT709,
    /// PQ (Perceptual Quantizer) - SMPTE ST 2084
    PQ,
    /// HLG (Hybrid Log-Gamma) - ITU-R BT.2100
    HLG,
    /// Linear
    Linear,
    /// SMPTE 240M
    SMPTE240M,
}

impl TransferFunction {
    /// Get transfer function name
    pub fn name(&self) -> &'static str {
        match self {
            TransferFunction::BT709 => "BT.709",
            TransferFunction::PQ => "PQ (SMPTE ST 2084)",
            TransferFunction::HLG => "HLG (BT.2100)",
            TransferFunction::Linear => "Linear",
            TransferFunction::SMPTE240M => "SMPTE 240M",
        }
    }

    /// Is HDR transfer function
    pub fn is_hdr(&self) -> bool {
        matches!(self, TransferFunction::PQ | TransferFunction::HLG)
    }
}

/// Color primaries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorPrimaries {
    /// BT.709 (SDR)
    BT709,
    /// BT.2020 (HDR)
    BT2020,
    /// DCI-P3 (Digital Cinema)
    DCIP3,
    /// Display P3
    DisplayP3,
    /// BT.601 NTSC
    BT601NTSC,
    /// BT.601 PAL
    BT601PAL,
}

impl ColorPrimaries {
    /// Get primaries name
    pub fn name(&self) -> &'static str {
        match self {
            ColorPrimaries::BT709 => "BT.709",
            ColorPrimaries::BT2020 => "BT.2020",
            ColorPrimaries::DCIP3 => "DCI-P3",
            ColorPrimaries::DisplayP3 => "Display P3",
            ColorPrimaries::BT601NTSC => "BT.601 NTSC",
            ColorPrimaries::BT601PAL => "BT.601 PAL",
        }
    }

    /// Get CIE 1931 xy chromaticity coordinates
    /// Returns (R, G, B, White Point) as (x, y) pairs
    pub fn chromaticity(&self) -> ((f64, f64), (f64, f64), (f64, f64), (f64, f64)) {
        match self {
            ColorPrimaries::BT709 => (
                (0.64, 0.33),   // R
                (0.30, 0.60),   // G
                (0.15, 0.06),   // B
                (0.3127, 0.3290), // D65 white point
            ),
            ColorPrimaries::BT2020 => (
                (0.708, 0.292), // R
                (0.170, 0.797), // G
                (0.131, 0.046), // B
                (0.3127, 0.3290), // D65 white point
            ),
            ColorPrimaries::DCIP3 => (
                (0.680, 0.320), // R
                (0.265, 0.690), // G
                (0.150, 0.060), // B
                (0.314, 0.351), // DCI white point
            ),
            ColorPrimaries::DisplayP3 => (
                (0.680, 0.320), // R
                (0.265, 0.690), // G
                (0.150, 0.060), // B
                (0.3127, 0.3290), // D65 white point
            ),
            ColorPrimaries::BT601NTSC => (
                (0.630, 0.340), // R
                (0.310, 0.595), // G
                (0.155, 0.070), // B
                (0.3127, 0.3290), // D65 white point
            ),
            ColorPrimaries::BT601PAL => (
                (0.640, 0.330), // R
                (0.290, 0.600), // G
                (0.150, 0.060), // B
                (0.3127, 0.3290), // D65 white point
            ),
        }
    }
}

/// Chromaticity coordinates (CIE 1931 xy)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Chromaticity {
    pub x: f64,
    pub y: f64,
}

impl Chromaticity {
    pub fn new(x: f64, y: f64) -> Self {
        Chromaticity { x, y }
    }
}

/// Mastering Display Color Volume (MDCV) - SMPTE ST 2086
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MasteringDisplayMetadata {
    /// Red primary (x, y)
    pub red_primary: Chromaticity,
    /// Green primary (x, y)
    pub green_primary: Chromaticity,
    /// Blue primary (x, y)
    pub blue_primary: Chromaticity,
    /// White point (x, y)
    pub white_point: Chromaticity,
    /// Maximum luminance in cd/m² (nits)
    pub max_luminance: f64,
    /// Minimum luminance in cd/m² (nits)
    pub min_luminance: f64,
}

impl MasteringDisplayMetadata {
    /// Create metadata for BT.2020 with D65 white point
    pub fn rec2020_d65() -> Self {
        MasteringDisplayMetadata {
            red_primary: Chromaticity::new(0.708, 0.292),
            green_primary: Chromaticity::new(0.170, 0.797),
            blue_primary: Chromaticity::new(0.131, 0.046),
            white_point: Chromaticity::new(0.3127, 0.3290),
            max_luminance: 1000.0, // 1000 nits (typical HDR mastering)
            min_luminance: 0.0001, // 0.0001 nits
        }
    }

    /// Create metadata for DCI-P3 D65
    pub fn dci_p3_d65() -> Self {
        MasteringDisplayMetadata {
            red_primary: Chromaticity::new(0.680, 0.320),
            green_primary: Chromaticity::new(0.265, 0.690),
            blue_primary: Chromaticity::new(0.150, 0.060),
            white_point: Chromaticity::new(0.3127, 0.3290),
            max_luminance: 1000.0,
            min_luminance: 0.0001,
        }
    }

    /// Validate metadata
    pub fn validate(&self) -> Result<()> {
        // Check chromaticity coordinates are in valid range [0, 1]
        let coords = [
            ("red_primary.x", self.red_primary.x),
            ("red_primary.y", self.red_primary.y),
            ("green_primary.x", self.green_primary.x),
            ("green_primary.y", self.green_primary.y),
            ("blue_primary.x", self.blue_primary.x),
            ("blue_primary.y", self.blue_primary.y),
            ("white_point.x", self.white_point.x),
            ("white_point.y", self.white_point.y),
        ];

        for (name, value) in &coords {
            if *value < 0.0 || *value > 1.0 {
                return Err(Error::InvalidInput(format!(
                    "Invalid chromaticity {}: {} (must be 0-1)",
                    name, value
                )));
            }
        }

        // Check luminance values
        if self.max_luminance <= 0.0 || self.max_luminance > 10000.0 {
            return Err(Error::InvalidInput(format!(
                "Invalid max_luminance: {} (must be 0-10000 nits)",
                self.max_luminance
            )));
        }

        if self.min_luminance < 0.0 || self.min_luminance >= self.max_luminance {
            return Err(Error::InvalidInput(format!(
                "Invalid min_luminance: {} (must be 0 to max_luminance)",
                self.min_luminance
            )));
        }

        Ok(())
    }

    /// Generate SMPTE ST 2086 string representation
    pub fn to_smpte_string(&self) -> String {
        format!(
            "G({:.4},{:.4})B({:.4},{:.4})R({:.4},{:.4})WP({:.4},{:.4})L({:.0},{:.4})",
            self.green_primary.x,
            self.green_primary.y,
            self.blue_primary.x,
            self.blue_primary.y,
            self.red_primary.x,
            self.red_primary.y,
            self.white_point.x,
            self.white_point.y,
            self.max_luminance,
            self.min_luminance
        )
    }
}

/// Content Light Level Information - SMPTE ST 2086
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (MaxCLL) in cd/m² (nits)
    pub max_cll: u32,
    /// Maximum Frame-Average Light Level (MaxFALL) in cd/m² (nits)
    pub max_fall: u32,
}

impl ContentLightLevel {
    pub fn new(max_cll: u32, max_fall: u32) -> Self {
        ContentLightLevel { max_cll, max_fall }
    }

    /// Validate light level values
    pub fn validate(&self) -> Result<()> {
        if self.max_cll == 0 || self.max_cll > 10000 {
            return Err(Error::InvalidInput(format!(
                "Invalid MaxCLL: {} (must be 1-10000 nits)",
                self.max_cll
            )));
        }

        if self.max_fall == 0 || self.max_fall > self.max_cll {
            return Err(Error::InvalidInput(format!(
                "Invalid MaxFALL: {} (must be 1 to MaxCLL)",
                self.max_fall
            )));
        }

        Ok(())
    }
}

/// HDR10+ Dynamic Metadata (SMPTE ST 2094-40)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdr10PlusMetadata {
    /// Application version
    pub application_version: u8,
    /// Per-frame metadata
    pub frames: Vec<Hdr10PlusFrame>,
}

/// HDR10+ per-frame metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdr10PlusFrame {
    /// Target system display maximum luminance
    pub targeted_system_display_max_luminance: f64,
    /// Average RGB value for the frame
    pub average_rgb: f64,
    /// Distribution values (percentiles)
    pub distribution_values: Vec<f64>,
    /// Knee point (tone mapping)
    pub knee_point_x: f64,
    pub knee_point_y: f64,
    /// Bezier curve anchors
    pub bezier_curve_anchors: Vec<(f64, f64)>,
}

/// Dolby Vision metadata (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DolbyVisionMetadata {
    /// Profile (5, 7, 8, etc.)
    pub profile: u8,
    /// Level
    pub level: u8,
    /// RPU (Reference Processing Unit) data
    pub rpu_data: Vec<u8>,
}

/// Complete HDR metadata container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdrMetadata {
    /// Transfer function (EOTF)
    pub transfer_function: TransferFunction,
    /// Color primaries
    pub color_primaries: ColorPrimaries,
    /// Mastering display metadata (HDR10 static)
    pub mastering_display: Option<MasteringDisplayMetadata>,
    /// Content light level (HDR10 static)
    pub content_light: Option<ContentLightLevel>,
    /// HDR10+ dynamic metadata
    pub hdr10_plus: Option<Hdr10PlusMetadata>,
    /// Dolby Vision metadata
    pub dolby_vision: Option<DolbyVisionMetadata>,
}

impl HdrMetadata {
    /// Create SDR metadata (BT.709)
    pub fn sdr() -> Self {
        HdrMetadata {
            transfer_function: TransferFunction::BT709,
            color_primaries: ColorPrimaries::BT709,
            mastering_display: None,
            content_light: None,
            hdr10_plus: None,
            dolby_vision: None,
        }
    }

    /// Create HDR10 metadata (BT.2020 + PQ)
    pub fn hdr10() -> Self {
        HdrMetadata {
            transfer_function: TransferFunction::PQ,
            color_primaries: ColorPrimaries::BT2020,
            mastering_display: Some(MasteringDisplayMetadata::rec2020_d65()),
            content_light: Some(ContentLightLevel::new(1000, 400)),
            hdr10_plus: None,
            dolby_vision: None,
        }
    }

    /// Create HLG metadata (BT.2020 + HLG)
    pub fn hlg() -> Self {
        HdrMetadata {
            transfer_function: TransferFunction::HLG,
            color_primaries: ColorPrimaries::BT2020,
            mastering_display: None, // HLG doesn't require mastering display
            content_light: None,
            hdr10_plus: None,
            dolby_vision: None,
        }
    }

    /// Is this HDR content?
    pub fn is_hdr(&self) -> bool {
        self.transfer_function.is_hdr()
    }

    /// Get HDR format name
    pub fn format_name(&self) -> &'static str {
        if self.dolby_vision.is_some() {
            "Dolby Vision"
        } else if self.hdr10_plus.is_some() {
            "HDR10+"
        } else if self.transfer_function == TransferFunction::PQ {
            "HDR10"
        } else if self.transfer_function == TransferFunction::HLG {
            "HLG"
        } else {
            "SDR"
        }
    }

    /// Validate all metadata
    pub fn validate(&self) -> Result<()> {
        if let Some(ref mdcv) = self.mastering_display {
            mdcv.validate()?;
        }

        if let Some(ref cll) = self.content_light {
            cll.validate()?;
        }

        Ok(())
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        let mut s = format!("HDR Format: {}\n", self.format_name());
        s.push_str(&format!("Transfer Function: {}\n", self.transfer_function.name()));
        s.push_str(&format!("Color Primaries: {}\n", self.color_primaries.name()));

        if let Some(ref mdcv) = self.mastering_display {
            s.push_str(&format!(
                "Mastering Display: {:.0}-{:.4} nits\n",
                mdcv.min_luminance, mdcv.max_luminance
            ));
        }

        if let Some(ref cll) = self.content_light {
            s.push_str(&format!(
                "Content Light Level: MaxCLL={} nits, MaxFALL={} nits\n",
                cll.max_cll, cll.max_fall
            ));
        }

        s
    }
}

/// Simple tone mapping from HDR to SDR
pub struct ToneMapper {
    /// Target peak luminance (nits)
    target_peak: f64,
    /// Knee point (start of compression)
    knee_point: f64,
}

impl ToneMapper {
    /// Create tone mapper for SDR output (100 nits)
    pub fn sdr() -> Self {
        ToneMapper {
            target_peak: 100.0,
            knee_point: 75.0,
        }
    }

    /// Create tone mapper with custom target
    pub fn new(target_peak: f64) -> Self {
        ToneMapper {
            target_peak,
            knee_point: target_peak * 0.75,
        }
    }

    /// Apply tone mapping to linear light value
    pub fn map(&self, linear_hdr: f64, source_peak: f64) -> f64 {
        // Normalize to target peak
        let normalized = linear_hdr * (self.target_peak / source_peak);

        // Simple Reinhard-style tone mapping with knee point
        if normalized <= self.knee_point {
            normalized
        } else {
            let excess = normalized - self.knee_point;
            let range = self.target_peak - self.knee_point;
            self.knee_point + (excess * range) / (excess + range)
        }
    }

    /// Apply PQ EOTF (Perceptual Quantizer) - SMPTE ST 2084
    pub fn pq_eotf(pq_value: f64) -> f64 {
        const M1: f64 = 0.1593017578125; // 2610 / 16384
        const M2: f64 = 78.84375; // 2523 / 32
        const C1: f64 = 0.8359375; // 3424 / 4096
        const C2: f64 = 18.8515625; // 2413 / 128
        const C3: f64 = 18.6875; // 2392 / 128

        let v = pq_value.clamp(0.0, 1.0);
        let v_pow = v.powf(1.0 / M2);
        let numerator = (v_pow - C1).max(0.0);
        let denominator = C2 - C3 * v_pow;

        let linear = (numerator / denominator).powf(1.0 / M1);

        // Scale to nits (10000 nits peak)
        linear * 10000.0
    }

    /// Apply inverse PQ EOTF (linear to PQ)
    pub fn pq_inverse_eotf(linear_nits: f64) -> f64 {
        const M1: f64 = 0.1593017578125;
        const M2: f64 = 78.84375;
        const C1: f64 = 0.8359375;
        const C2: f64 = 18.8515625;
        const C3: f64 = 18.6875;

        let y = (linear_nits / 10000.0).clamp(0.0, 1.0);
        let y_pow = y.powf(M1);
        let numerator = C1 + C2 * y_pow;
        let denominator = 1.0 + C3 * y_pow;

        (numerator / denominator).powf(M2)
    }

    /// Apply HLG OETF (Hybrid Log-Gamma) - ITU-R BT.2100
    pub fn hlg_oetf(linear: f64) -> f64 {
        const A: f64 = 0.17883277;
        const B: f64 = 0.28466892; // 1 - 4*a
        const C: f64 = 0.55991073; // 0.5 - a * ln(4*a)

        if linear <= 1.0 / 12.0 {
            (3.0 * linear).sqrt()
        } else {
            A * (12.0 * linear - B).ln() + C
        }
    }

    /// Apply inverse HLG OETF (HLG to linear)
    pub fn hlg_inverse_oetf(hlg: f64) -> f64 {
        const A: f64 = 0.17883277;
        const B: f64 = 0.28466892;
        const C: f64 = 0.55991073;

        if hlg <= 0.5 {
            (hlg * hlg) / 3.0
        } else {
            ((hlg - C) / A).exp() / 12.0 + B / 12.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_function() {
        assert!(TransferFunction::PQ.is_hdr());
        assert!(TransferFunction::HLG.is_hdr());
        assert!(!TransferFunction::BT709.is_hdr());

        assert_eq!(TransferFunction::PQ.name(), "PQ (SMPTE ST 2084)");
    }

    #[test]
    fn test_color_primaries_chromaticity() {
        let (r, g, b, w) = ColorPrimaries::BT2020.chromaticity();

        // Check BT.2020 red primary
        assert!((r.0 - 0.708).abs() < 0.001);
        assert!((r.1 - 0.292).abs() < 0.001);

        // Check D65 white point
        assert!((w.0 - 0.3127).abs() < 0.001);
        assert!((w.1 - 0.329).abs() < 0.001);
    }

    #[test]
    fn test_mastering_display_metadata() {
        let mdcv = MasteringDisplayMetadata::rec2020_d65();

        assert_eq!(mdcv.max_luminance, 1000.0);
        assert_eq!(mdcv.min_luminance, 0.0001);
        assert!(mdcv.validate().is_ok());

        // Test SMPTE string
        let smpte = mdcv.to_smpte_string();
        assert!(smpte.contains("G(0.1700,0.7970)"));
        assert!(smpte.contains("L(1000,0.0001)"));
    }

    #[test]
    fn test_content_light_level() {
        let cll = ContentLightLevel::new(1000, 400);
        assert!(cll.validate().is_ok());

        // Invalid: MaxFALL > MaxCLL
        let invalid_cll = ContentLightLevel::new(400, 1000);
        assert!(invalid_cll.validate().is_err());
    }

    #[test]
    fn test_hdr_metadata_creation() {
        let sdr = HdrMetadata::sdr();
        assert!(!sdr.is_hdr());
        assert_eq!(sdr.format_name(), "SDR");

        let hdr10 = HdrMetadata::hdr10();
        assert!(hdr10.is_hdr());
        assert_eq!(hdr10.format_name(), "HDR10");
        assert!(hdr10.mastering_display.is_some());
        assert!(hdr10.content_light.is_some());

        let hlg = HdrMetadata::hlg();
        assert!(hlg.is_hdr());
        assert_eq!(hlg.format_name(), "HLG");
    }

    #[test]
    fn test_hdr_metadata_validation() {
        let mut hdr = HdrMetadata::hdr10();
        assert!(hdr.validate().is_ok());

        // Invalid mastering display
        if let Some(ref mut mdcv) = hdr.mastering_display {
            mdcv.max_luminance = -100.0; // Invalid
        }
        assert!(hdr.validate().is_err());
    }

    #[test]
    fn test_tone_mapper() {
        let mapper = ToneMapper::sdr();
        assert_eq!(mapper.target_peak, 100.0);

        // Map 1000 nit HDR to 100 nit SDR
        let mapped = mapper.map(1000.0, 1000.0);
        assert!(mapped <= 100.0);

        // Values below knee point pass through
        let low = mapper.map(50.0, 1000.0);
        assert!((low - 5.0).abs() < 0.1); // 50 * (100/1000) = 5
    }

    #[test]
    fn test_pq_eotf() {
        // Test PQ round-trip
        let linear_nits = 100.0;
        let pq = ToneMapper::pq_inverse_eotf(linear_nits);
        let recovered = ToneMapper::pq_eotf(pq);

        assert!((recovered - linear_nits).abs() < 0.01);

        // Test peak white (10000 nits)
        let peak_pq = ToneMapper::pq_inverse_eotf(10000.0);
        assert!((peak_pq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hlg_oetf() {
        // Test HLG round-trip
        let linear = 0.5;
        let hlg = ToneMapper::hlg_oetf(linear);
        let recovered = ToneMapper::hlg_inverse_oetf(hlg);

        assert!((recovered - linear).abs() < 0.01);

        // Test transition point (1/12)
        let transition = ToneMapper::hlg_oetf(1.0 / 12.0);
        assert!(transition > 0.0 && transition < 1.0);
    }
}
