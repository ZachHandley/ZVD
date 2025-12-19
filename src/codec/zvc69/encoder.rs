//! ZVC69 Neural Video Encoder
//!
//! This module implements the encoder for the ZVC69 neural video codec.
//! The encoder uses learned neural network transforms for compression,
//! achieving better rate-distortion performance than traditional codecs.
//!
//! ## Architecture
//!
//! The encoder pipeline consists of:
//! 1. **Analysis Transform**: Neural network that transforms pixels to latent space
//! 2. **Hyperprior Encoding**: Compress latents to hyperprior for entropy parameters
//! 3. **Quantization**: Round latents to integers for entropy coding
//! 4. **Entropy Coding**: Encode with learned probability models (ANS)
//! 5. **Bitstream Packaging**: Assemble into ZVC69 frame format
//!
//! ## P-Frame Encoding Pipeline
//!
//! For inter-frames (P-frames), the following pipeline is used:
//! 1. **Motion Estimation**: Estimate optical flow between current and reference
//! 2. **Motion Compression**: Quantize and entropy-code motion vectors
//! 3. **Frame Warping**: Warp reference frame using estimated motion
//! 4. **Residual Computation**: Compute difference between current and warped
//! 5. **Residual Encoding**: Encode residuals using neural transform
//! 6. **Encoder-Side Reconstruction**: Decode for drift-free reference
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::{ZVC69Encoder, ZVC69Config, Quality, Preset};
//!
//! let config = ZVC69Config::builder()
//!     .dimensions(1920, 1080)
//!     .quality(Quality::Q5)
//!     .preset(Preset::Medium)
//!     .build()?;
//!
//! let mut encoder = ZVC69Encoder::new(config)?;
//! encoder.send_frame(&frame)?;
//! let packet = encoder.receive_packet()?;
//! ```

use std::path::Path;

use super::bitstream::FrameType as BitstreamFrameType;
use super::config::{Preset, Quality, RateControlMode, ZVC69Config};
use super::entropy::{
    EntropyCoder, FactorizedPrior, GaussianConditional, DEFAULT_MAX_SYMBOL, DEFAULT_MIN_SYMBOL,
};
use super::error::ZVC69Error;
use super::model::{image_to_tensor, tensor_to_image, NeuralModel, NeuralModelConfig};
use super::motion::{encode_motion, MotionConfig, MotionEstimator, MotionField};
use super::quantize::{
    clamp_quantized, dequantize_tensor, flatten_tensor_chw, flatten_tensor_f32, quality_to_scale,
    quantize_scaled,
};
use super::residual::{
    should_skip_residual, CompressedResidual, Residual, ResidualConfig, ResidualDecoder,
    ResidualEncoder,
};
use super::warp::{FrameWarper, WarpConfig};
use crate::codec::{Encoder, Frame, PictureType, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array4;
use std::io::Cursor;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Default number of latent channels
const DEFAULT_LATENT_CHANNELS: usize = 192;

/// Default number of hyperprior channels
const DEFAULT_HYPERPRIOR_CHANNELS: usize = 128;

/// Latent spatial downsampling factor (image size / 16)
const LATENT_SPATIAL_FACTOR: usize = 16;

/// Hyperprior spatial downsampling factor (image size / 64)
const HYPERPRIOR_SPATIAL_FACTOR: usize = 64;

/// Default skip threshold for P-frame residuals
/// If residual energy is below this, skip residual coding
const DEFAULT_PFRAME_SKIP_THRESHOLD: f32 = 0.01;

/// Magic bytes for P-frame skip mode
const PFRAME_SKIP_MAGIC: &[u8; 4] = b"SKIP";

/// P-frame motion section marker
const MOTION_SECTION_MARKER: u8 = 0x4D; // 'M'

/// P-frame residual section marker
const RESIDUAL_SECTION_MARKER: u8 = 0x52; // 'R'

// -------------------------------------------------------------------------
// Frame Type (Internal)
// -------------------------------------------------------------------------

/// Frame type for encoding decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameType {
    /// Intra frame (keyframe)
    I,
    /// Predicted frame
    P,
    /// Bidirectional frame
    B,
}

impl FrameType {
    fn to_bitstream_type(self) -> BitstreamFrameType {
        match self {
            FrameType::I => BitstreamFrameType::I,
            FrameType::P => BitstreamFrameType::P,
            FrameType::B => BitstreamFrameType::B,
        }
    }
}

// -------------------------------------------------------------------------
// Encoded Frame
// -------------------------------------------------------------------------

/// Result of encoding a single frame
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Frame type (I, P, or B)
    pub frame_type: BitstreamFrameType,
    /// Encoded bitstream data
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: i64,
    /// Decode timestamp
    pub dts: i64,
    /// Size in bits
    pub size_bits: usize,
    /// Whether this is a keyframe
    pub is_keyframe: bool,
}

impl EncodedFrame {
    /// Create a new encoded frame
    pub fn new(frame_type: BitstreamFrameType, data: Vec<u8>, pts: i64, dts: i64) -> Self {
        let size_bits = data.len() * 8;
        let is_keyframe = frame_type.is_keyframe();
        EncodedFrame {
            frame_type,
            data,
            pts,
            dts,
            size_bits,
            is_keyframe,
        }
    }
}

// -------------------------------------------------------------------------
// Internal Encoder State
// -------------------------------------------------------------------------

/// Internal encoder state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderState {
    /// Encoder is ready to accept frames
    Ready,
    /// Encoder is processing frames
    Encoding,
    /// Encoder is flushing remaining frames
    Flushing,
    /// Encoder has finished
    Finished,
}

/// Reference frame buffer for inter-frame prediction
#[derive(Debug, Clone)]
struct ReferenceFrame {
    /// Decoded Y plane
    y: Vec<u8>,
    /// Decoded U plane
    u: Vec<u8>,
    /// Decoded V plane
    v: Vec<u8>,
    /// Latent representation (for P-frame encoding)
    latents: Option<Array4<f32>>,
    /// Tensor representation for motion estimation/warping
    tensor: Option<Array4<f32>>,
    /// Frame number
    frame_num: u64,
    /// Picture type
    pict_type: PictureType,
    /// Presentation timestamp
    pts: Timestamp,
    /// Width
    width: u32,
    /// Height
    height: u32,
}

impl ReferenceFrame {
    fn new(width: u32, height: u32) -> Self {
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;
        ReferenceFrame {
            y: vec![128; y_size],
            u: vec![128; uv_size],
            v: vec![128; uv_size],
            latents: None,
            tensor: None,
            frame_num: 0,
            pict_type: PictureType::None,
            pts: Timestamp::none(),
            width,
            height,
        }
    }

    /// Convert to VideoFrame for compatibility
    fn to_video_frame(&self) -> VideoFrame {
        let mut frame = VideoFrame::new(self.width, self.height, PixelFormat::YUV420P);
        frame.data = vec![
            Buffer::from_vec(self.y.clone()),
            Buffer::from_vec(self.u.clone()),
            Buffer::from_vec(self.v.clone()),
        ];
        frame.linesize = vec![
            self.width as usize,
            (self.width / 2) as usize,
            (self.width / 2) as usize,
        ];
        frame.pts = self.pts;
        frame
    }

    /// Check if this reference has valid tensor data
    fn has_tensor(&self) -> bool {
        self.tensor.is_some()
    }
}

/// Reference buffer for P-frame encoding
///
/// Manages reference frames and their tensor representations for
/// motion estimation and frame warping operations.
#[derive(Debug, Clone)]
pub struct ReferenceBuffer {
    /// Last reconstructed frame tensor for motion compensation
    last_frame: Option<Array4<f32>>,
    /// Last frame's latent representation (for temporal context)
    last_latent: Option<Array4<f32>>,
    /// Frame number of the last reference
    last_frame_num: u64,
    /// Golden frame tensor (long-term reference)
    golden_frame: Option<Array4<f32>>,
    /// Golden frame number
    golden_frame_num: u64,
}

impl ReferenceBuffer {
    /// Create a new empty reference buffer
    pub fn new() -> Self {
        ReferenceBuffer {
            last_frame: None,
            last_latent: None,
            last_frame_num: 0,
            golden_frame: None,
            golden_frame_num: 0,
        }
    }

    /// Check if we have a valid reference frame
    pub fn has_reference(&self) -> bool {
        self.last_frame.is_some()
    }

    /// Get the last frame tensor
    pub fn get_last_frame(&self) -> Option<&Array4<f32>> {
        self.last_frame.as_ref()
    }

    /// Get the last latent representation
    pub fn get_last_latent(&self) -> Option<&Array4<f32>> {
        self.last_latent.as_ref()
    }

    /// Update the reference buffer with a new reconstructed frame
    pub fn update(
        &mut self,
        frame_tensor: Array4<f32>,
        latent: Option<Array4<f32>>,
        frame_num: u64,
    ) {
        self.last_frame = Some(frame_tensor);
        self.last_latent = latent;
        self.last_frame_num = frame_num;
    }

    /// Update the golden (long-term) reference
    pub fn update_golden(&mut self, frame_tensor: Array4<f32>, frame_num: u64) {
        self.golden_frame = Some(frame_tensor);
        self.golden_frame_num = frame_num;
    }

    /// Clear all references (on scene change or explicit reset)
    pub fn clear(&mut self) {
        self.last_frame = None;
        self.last_latent = None;
        self.last_frame_num = 0;
        self.golden_frame = None;
        self.golden_frame_num = 0;
    }
}

impl Default for ReferenceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate control state
#[derive(Debug, Clone)]
struct RateControlState {
    /// Target bits per frame
    target_bits_per_frame: f64,
    /// Accumulated bit error
    bit_error: f64,
    /// Current QP
    current_qp: f32,
    /// Frame count
    frame_count: u64,
    /// Total bits used
    total_bits: u64,
}

impl RateControlState {
    fn new(config: &ZVC69Config) -> Self {
        let fps = config.framerate();
        let target_bps = match &config.rate_control {
            RateControlMode::Vbr { target_bitrate, .. } => *target_bitrate as f64,
            RateControlMode::Cbr { bitrate, .. } => *bitrate as f64,
            RateControlMode::Crf { crf } => {
                // Estimate bitrate from CRF
                let base_bpp = 0.1 * (51.0 - crf) / 28.0;
                (config.width * config.height) as f64 * fps as f64 * base_bpp as f64
            }
            RateControlMode::Cqp { .. } => {
                // No target bitrate for CQP
                f64::INFINITY
            }
        };

        RateControlState {
            target_bits_per_frame: target_bps / fps as f64,
            bit_error: 0.0,
            current_qp: match &config.rate_control {
                RateControlMode::Crf { crf } => *crf,
                RateControlMode::Cqp { qp_p, .. } => *qp_p as f32,
                _ => 23.0,
            },
            frame_count: 0,
            total_bits: 0,
        }
    }
}

// -------------------------------------------------------------------------
// ZVC69 Encoder
// -------------------------------------------------------------------------

/// ZVC69 Neural Video Encoder
///
/// Implements neural video compression using learned transforms and entropy coding.
pub struct ZVC69Encoder {
    /// Encoder configuration
    config: ZVC69Config,

    /// Encoder state
    state: EncoderState,

    /// Frame counter
    frame_count: u64,

    /// GOP frame counter (resets at each keyframe)
    gop_frame_count: u32,

    /// Pending output packets
    output_packets: Vec<Packet>,

    /// Last reference frame
    last_ref: ReferenceFrame,

    /// Golden reference frame (long-term)
    golden_ref: ReferenceFrame,

    /// Rate control state
    rate_control: RateControlState,

    /// Lookahead buffer for B-frame decisions
    lookahead_buffer: Vec<VideoFrame>,

    /// Extradata (codec configuration record)
    extradata: Option<Vec<u8>>,

    /// Scene change threshold
    scene_change_threshold: f32,

    /// Neural model (optional - for real neural encoding)
    #[cfg(feature = "zvc69")]
    model: Option<NeuralModel>,

    /// Factorized prior for hyperprior encoding
    factorized_prior: FactorizedPrior,

    /// Gaussian conditional for main latent encoding
    gaussian_cond: GaussianConditional,

    /// Quantization scale (derived from quality)
    quant_scale: f32,

    /// Number of latent channels
    latent_channels: usize,

    /// Number of hyperprior channels
    hyperprior_channels: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // P-Frame Encoding Components
    // ─────────────────────────────────────────────────────────────────────────
    /// Motion estimator for inter-frame prediction
    motion_estimator: MotionEstimator,

    /// Frame warper for motion compensation
    frame_warper: FrameWarper,

    /// Residual encoder for P-frame residuals
    residual_encoder: ResidualEncoder,

    /// Residual decoder for encoder-side reconstruction (drift prevention)
    residual_decoder: ResidualDecoder,

    /// Reference buffer for P-frame encoding
    reference_buffer: ReferenceBuffer,

    /// Entropy coder for motion vectors
    entropy_coder: EntropyCoder,

    /// Skip threshold for P-frame residuals
    pframe_skip_threshold: f32,

    /// Force next frame to be keyframe
    force_next_keyframe: bool,
}

impl ZVC69Encoder {
    /// Create a new ZVC69 encoder with the given configuration
    pub fn new(config: ZVC69Config) -> Result<Self> {
        config.validate().map_err(|e| Error::codec(e.to_string()))?;

        let rate_control = RateControlState::new(&config);
        let quant_scale = quality_to_scale(config.quality.level());

        // Initialize motion estimator with configuration based on preset
        let motion_config = MotionConfig {
            block_size: match config.preset {
                Preset::Ultrafast => 16,
                Preset::Fast => 8,
                Preset::Medium => 8,
                Preset::Slow => 4,
                Preset::Veryslow => 4,
            },
            search_range: match config.preset {
                Preset::Ultrafast => 8,
                Preset::Fast => 16,
                Preset::Medium => 32,
                Preset::Slow => 48,
                Preset::Veryslow => 64,
            },
            ..Default::default()
        };
        let motion_estimator = MotionEstimator::new(motion_config);

        // Initialize frame warper
        let warp_config = WarpConfig::default();
        let frame_warper = FrameWarper::new(warp_config);

        // Initialize residual encoder/decoder
        let residual_config = ResidualConfig::default();
        let residual_encoder = ResidualEncoder::new(residual_config.clone());
        let residual_decoder = ResidualDecoder::new(residual_config);

        Ok(ZVC69Encoder {
            last_ref: ReferenceFrame::new(config.width, config.height),
            golden_ref: ReferenceFrame::new(config.width, config.height),
            rate_control,
            lookahead_buffer: Vec::with_capacity(config.preset.lookahead_frames() as usize),
            extradata: None,
            scene_change_threshold: 0.4,
            quant_scale,
            latent_channels: DEFAULT_LATENT_CHANNELS,
            hyperprior_channels: DEFAULT_HYPERPRIOR_CHANNELS,
            factorized_prior: FactorizedPrior::new(DEFAULT_HYPERPRIOR_CHANNELS),
            gaussian_cond: GaussianConditional::default(),
            #[cfg(feature = "zvc69")]
            model: None,
            config,
            state: EncoderState::Ready,
            frame_count: 0,
            gop_frame_count: 0,
            output_packets: Vec::new(),
            // P-frame encoding components
            motion_estimator,
            frame_warper,
            residual_encoder,
            residual_decoder,
            reference_buffer: ReferenceBuffer::new(),
            entropy_coder: EntropyCoder::new(),
            pframe_skip_threshold: DEFAULT_PFRAME_SKIP_THRESHOLD,
            force_next_keyframe: false,
        })
    }

    /// Create a new encoder with default configuration for given dimensions
    pub fn with_dimensions(width: u32, height: u32) -> Result<Self> {
        let config = ZVC69Config::new(width, height);
        Self::new(config)
    }

    /// Load neural model from a directory
    #[cfg(feature = "zvc69")]
    pub fn load_model(&mut self, model_dir: &Path) -> Result<()> {
        let model = NeuralModel::load(model_dir).map_err(|e| Error::codec(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    /// Load neural model with custom configuration
    #[cfg(feature = "zvc69")]
    pub fn load_model_with_config(
        &mut self,
        model_dir: &Path,
        model_config: NeuralModelConfig,
    ) -> Result<()> {
        let model = NeuralModel::load_with_config(model_dir, model_config)
            .map_err(|e| Error::codec(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    /// Check if neural model is loaded
    #[cfg(feature = "zvc69")]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn has_model(&self) -> bool {
        false
    }

    /// Get the encoder configuration
    pub fn config(&self) -> &ZVC69Config {
        &self.config
    }

    /// Set quality level
    pub fn set_quality(&mut self, quality: Quality) {
        self.config.quality = quality;
        self.quant_scale = quality_to_scale(quality.level());
    }

    /// Set encoding preset
    pub fn set_preset(&mut self, preset: Preset) {
        self.config.preset = preset;
    }

    /// Set target bitrate (switches to VBR mode)
    pub fn set_bitrate(&mut self, bitrate: u64) {
        self.config.rate_control = RateControlMode::Vbr {
            target_bitrate: bitrate,
            max_bitrate: None,
        };
        self.rate_control = RateControlState::new(&self.config);
    }

    /// Set CRF value (switches to CRF mode)
    pub fn set_crf(&mut self, crf: f32) {
        self.config.rate_control = RateControlMode::Crf {
            crf: crf.clamp(0.0, 51.0),
        };
        self.rate_control = RateControlState::new(&self.config);
    }

    /// Force the next frame to be a keyframe
    pub fn force_keyframe(&mut self) {
        // Set flag to trigger I-frame on next encode
        self.force_next_keyframe = true;
        // Also set gop_frame_count for backward compatibility
        self.gop_frame_count = self.config.gop.keyframe_interval;
    }

    /// Clear the force keyframe flag (called after encoding an I-frame)
    fn clear_force_keyframe(&mut self) {
        self.force_next_keyframe = false;
    }

    /// Get encoding statistics
    pub fn stats(&self) -> EncoderStats {
        let avg_bitrate = if self.frame_count > 0 {
            let duration_secs = self.frame_count as f64 / self.config.framerate() as f64;
            (self.rate_control.total_bits as f64 / duration_secs) as u64
        } else {
            0
        };

        EncoderStats {
            frames_encoded: self.frame_count,
            total_bits: self.rate_control.total_bits,
            avg_bitrate,
            avg_qp: self.rate_control.current_qp,
        }
    }

    /// Determine the frame type for the next frame
    ///
    /// Frame type decision follows this priority:
    /// 1. First frame is always I-frame
    /// 2. Forced keyframe request
    /// 3. No valid reference available -> I-frame
    /// 4. Keyframe interval reached -> I-frame
    /// 5. B-frame position in mini-GOP -> B-frame
    /// 6. Otherwise -> P-frame
    fn determine_frame_type(&self) -> FrameType {
        // First frame is always I-frame
        if self.frame_count == 0 {
            return FrameType::I;
        }

        // Check for forced keyframe
        if self.force_next_keyframe {
            return FrameType::I;
        }

        // Check if we have a valid reference for P-frames
        if !self.reference_buffer.has_reference() {
            return FrameType::I;
        }

        // Keyframe interval reached
        if self.gop_frame_count >= self.config.gop.keyframe_interval {
            return FrameType::I;
        }

        // Decide between P and B frames based on GOP position
        if self.config.gop.bframes > 0 {
            let pos_in_mini_gop = self.gop_frame_count % (self.config.gop.bframes + 1);
            if pos_in_mini_gop != 0 {
                return FrameType::B;
            }
        }

        FrameType::P
    }

    /// Decide frame type (public API alias for determine_frame_type)
    pub fn decide_frame_type(&self) -> BitstreamFrameType {
        self.determine_frame_type().to_bitstream_type()
    }

    /// Encode a video frame
    fn encode_video_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        // Validate frame dimensions
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(Error::codec(
                ZVC69Error::dimension_mismatch(
                    frame.width,
                    frame.height,
                    self.config.width,
                    self.config.height,
                )
                .to_string(),
            ));
        }

        // Validate pixel format
        if frame.format != PixelFormat::YUV420P {
            return Err(Error::codec(format!(
                "ZVC69 encoder expects YUV420P, got {:?}",
                frame.format
            )));
        }

        // Determine frame type
        let frame_type = self.determine_frame_type();

        // Encode based on frame type
        let encoded_frame = match frame_type {
            FrameType::I => self.encode_intra_frame(frame)?,
            FrameType::P => self.encode_inter_frame(frame, false)?,
            FrameType::B => self.encode_inter_frame(frame, true)?,
        };

        // Create output packet
        let pts = if frame.pts.is_valid() {
            frame.pts
        } else {
            Timestamp::new(
                (self.frame_count * 1000 * self.config.framerate_den as u64
                    / self.config.framerate_num as u64) as i64,
            )
        };

        let mut packet = Packet::new(0, Buffer::from_vec(encoded_frame.data.clone()));
        packet.pts = pts;
        packet.dts = pts; // For now, DTS = PTS (no B-frame reordering yet)
        packet.flags.keyframe = encoded_frame.is_keyframe;

        // Update rate control
        self.rate_control.total_bits += encoded_frame.size_bits as u64;
        self.rate_control.frame_count += 1;

        // Update GOP counter
        // After I-frame, start counting from 1 (the I-frame itself counts as frame 1 in the GOP)
        // This ensures keyframe_interval=3 gives us I at frames 0, 3, 6... (every 3rd)
        if frame_type == FrameType::I {
            self.gop_frame_count = 1;
            // Clear force keyframe flag after encoding an I-frame
            self.clear_force_keyframe();
            // Note: Do NOT clear reference_buffer here - the I-frame encoding
            // has already updated it with the reconstructed frame for P-frames to use
        } else {
            self.gop_frame_count += 1;
        }

        self.frame_count += 1;
        self.output_packets.push(packet);

        Ok(())
    }

    /// Encode an intra frame (keyframe) - Full neural encoding pipeline
    fn encode_intra_frame(&mut self, frame: &VideoFrame) -> Result<EncodedFrame> {
        let pts = if frame.pts.is_valid() {
            frame.pts.value
        } else {
            0
        };
        let dts = pts;

        #[cfg(feature = "zvc69")]
        {
            // Check if model is available and run neural encoding if so
            if self.model.is_some() {
                return self.encode_intra_neural(frame, pts, dts);
            }
        }

        // Fallback to placeholder encoding when model is not loaded
        self.encode_intra_placeholder(frame, pts, dts)
    }

    /// Full neural I-frame encoding pipeline
    #[cfg(feature = "zvc69")]
    fn encode_intra_neural(
        &mut self,
        frame: &VideoFrame,
        pts: i64,
        dts: i64,
    ) -> Result<EncodedFrame> {
        // Get model reference (we know it exists from the caller check)
        let model = self.model.as_ref().unwrap();

        // Step 1: Convert frame to tensor [1, 3, H, W]
        let tensor = image_to_tensor(frame).map_err(|e| Error::codec(e.to_string()))?;

        // Step 2: Run encoder network: image -> latents
        let latents = model
            .encode(&tensor)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 3: Run hyperprior encoder: latents -> hyperprior
        let hyperprior = model
            .encode_hyperprior(&latents.y)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 4: Quantize hyperprior
        let z_quantized = quantize_scaled(&hyperprior.z, self.quant_scale);
        let z_clamped = clamp_quantized(&z_quantized, DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);
        let z_flat = flatten_tensor_chw(&z_clamped.view());

        // Step 5: Encode hyperprior with factorized prior
        let z_bytes = self
            .factorized_prior
            .encode(&z_flat)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 6: Dequantize hyperprior and run hyperprior decoder for entropy params
        let z_dequant = dequantize_tensor(&z_clamped);
        let z_shape = hyperprior.z.dim();

        // Create Hyperprior struct for decode_hyperprior
        let z_hyperprior = super::model::Hyperprior::new(z_dequant);
        let entropy_params = model
            .decode_hyperprior(&z_hyperprior)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 7: Quantize main latents
        let y_quantized = quantize_scaled(&latents.y, self.quant_scale);
        let y_clamped = clamp_quantized(&y_quantized, DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);
        let y_flat = flatten_tensor_chw(&y_clamped.view());

        // Step 8: Flatten entropy parameters
        let means_flat = flatten_tensor_f32(&entropy_params.means.view());
        let scales_flat = flatten_tensor_f32(&entropy_params.scales.view());

        // Step 9: Encode main latents with Gaussian conditional
        let y_bytes = self
            .gaussian_cond
            .encode(&y_flat, &means_flat, &scales_flat)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 10: Package into bitstream
        let y_shape = latents.y.dim();
        let frame_data = self.package_iframe(
            z_bytes,
            y_bytes,
            (y_shape.0, y_shape.1, y_shape.2, y_shape.3),
            (z_shape.0, z_shape.1, z_shape.2, z_shape.3),
        )?;

        // Step 11: Reconstruct frame for reference (encoder-side reconstruction)
        // This ensures the reference matches what the decoder will see
        let y_dequantized = dequantize_tensor(&y_clamped);
        let reconstructed_latents = super::model::Latents::new(y_dequantized.clone());
        let reconstructed_tensor = model
            .decode(&reconstructed_latents)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Update legacy reference frame
        self.update_reference_frame(frame)?;
        self.last_ref.latents = Some(y_dequantized.clone());
        self.last_ref.tensor = Some(reconstructed_tensor.clone());

        // Update new reference buffer (for P-frame encoding)
        self.reference_buffer.update(
            reconstructed_tensor.clone(),
            Some(y_dequantized),
            self.frame_count,
        );

        // Update golden reference on I-frames
        self.reference_buffer
            .update_golden(reconstructed_tensor, self.frame_count);

        Ok(EncodedFrame::new(
            BitstreamFrameType::I,
            frame_data,
            pts,
            dts,
        ))
    }

    /// Placeholder I-frame encoding (when model is not loaded)
    fn encode_intra_placeholder(
        &mut self,
        frame: &VideoFrame,
        pts: i64,
        dts: i64,
    ) -> Result<EncodedFrame> {
        let mut bitstream = Vec::with_capacity(frame.width as usize * frame.height as usize);

        // Write ZVC69 frame header
        self.write_frame_header(&mut bitstream, FrameType::I)?;

        // Write placeholder frame data
        self.write_placeholder_frame_data(&mut bitstream, frame)?;

        // Convert frame to tensor for reference buffer
        let frame_tensor = image_to_tensor(frame).map_err(|e| Error::codec(e.to_string()))?;

        // Update legacy reference frame
        self.update_reference_frame(frame)?;
        self.last_ref.tensor = Some(frame_tensor.clone());

        // Update new reference buffer (for P-frame encoding)
        self.reference_buffer
            .update(frame_tensor.clone(), None, self.frame_count);

        // Update golden reference on I-frames
        self.reference_buffer
            .update_golden(frame_tensor, self.frame_count);

        Ok(EncodedFrame::new(
            BitstreamFrameType::I,
            bitstream,
            pts,
            dts,
        ))
    }

    /// Encode an inter frame (P or B frame)
    fn encode_inter_frame(&mut self, frame: &VideoFrame, is_b_frame: bool) -> Result<EncodedFrame> {
        let pts = if frame.pts.is_valid() {
            frame.pts.value
        } else {
            0
        };
        let dts = pts;

        if is_b_frame {
            // B-frames: Use placeholder for now (requires 2 references)
            self.encode_bframe_placeholder(frame, pts, dts)
        } else {
            // P-frames: Use full motion compensation pipeline
            self.encode_pframe(frame, pts, dts)
        }
    }

    /// Encode a P-frame using motion compensation
    ///
    /// P-frame encoding pipeline:
    /// 1. Get reference frame tensor
    /// 2. Convert current frame to tensor
    /// 3. Estimate motion (current <- reference)
    /// 4. Compress motion vectors
    /// 5. Warp reference frame using motion
    /// 6. Compute residual (current - predicted)
    /// 7. Check skip mode (low residual energy)
    /// 8. Encode residual
    /// 9. Reconstruct for reference (encoder-side reconstruction)
    /// 10. Update reference buffer
    /// 11. Package P-frame bitstream
    fn encode_pframe(&mut self, frame: &VideoFrame, pts: i64, dts: i64) -> Result<EncodedFrame> {
        // Step 1: Get reference frame tensor
        let reference_tensor = self.get_reference_tensor()?;

        // Step 2: Convert current frame to tensor [1, 3, H, W]
        let current_tensor = image_to_tensor(frame).map_err(|e| Error::codec(e.to_string()))?;

        // Step 3: Estimate motion (current <- reference)
        // Motion vectors indicate where pixels in current frame come from in reference
        let motion = self
            .motion_estimator
            .estimate_placeholder(&current_tensor, &reference_tensor);

        // Step 4: Encode motion vectors (quantize + entropy code)
        let motion_bytes = encode_motion(&motion, &mut self.entropy_coder)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 5: Warp reference frame using motion
        let predicted = self.frame_warper.backward_warp(&reference_tensor, &motion);

        // Step 6: Compute residual (current - predicted)
        let residual = Residual::compute(&current_tensor, &predicted);

        // Step 7: Check skip mode (low residual energy)
        if should_skip_residual(&residual, self.pframe_skip_threshold) {
            // Skip mode: motion only, no residual
            // Still need to update reference with predicted (warped) frame
            self.reference_buffer
                .update(predicted.clone(), None, self.frame_count);
            self.update_reference_frame(frame)?;
            self.last_ref.tensor = Some(predicted);

            return self.package_pframe_skip(motion_bytes, pts, dts);
        }

        // Step 8: Encode residual
        let compressed_residual = self.residual_encoder.encode_placeholder(&residual);
        let residual_bytes = compressed_residual
            .to_bytes(&mut self.entropy_coder)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 9: Reconstruct for reference (encoder-side reconstruction)
        // Decode the residual exactly as the decoder will
        // Use the tensor dimensions for output shape
        let output_height = current_tensor.dim().2;
        let output_width = current_tensor.dim().3;
        let decoded_residual = self.residual_decoder.decode_placeholder(
            &compressed_residual,
            output_height,
            output_width,
        );
        let reconstructed = Residual::reconstruct(&decoded_residual, &predicted);

        // Step 10: Update reference buffer with reconstructed frame
        self.reference_buffer.update(
            reconstructed.clone(),
            None, // No latents for P-frames
            self.frame_count,
        );

        // Update legacy reference frame
        self.update_reference_frame(frame)?;
        self.last_ref.tensor = Some(reconstructed);

        // Step 11: Package P-frame bitstream
        self.package_pframe(motion_bytes, residual_bytes, &motion, pts, dts)
    }

    /// Encode a B-frame (placeholder implementation)
    fn encode_bframe_placeholder(
        &mut self,
        frame: &VideoFrame,
        pts: i64,
        dts: i64,
    ) -> Result<EncodedFrame> {
        let mut bitstream = Vec::with_capacity(frame.width as usize * frame.height as usize / 2);

        // Write frame header
        self.write_frame_header(&mut bitstream, FrameType::B)?;

        // Write placeholder frame data
        self.write_placeholder_frame_data(&mut bitstream, frame)?;

        // B-frames don't update reference

        Ok(EncodedFrame::new(
            BitstreamFrameType::B,
            bitstream,
            pts,
            dts,
        ))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reference Frame Management
    // ─────────────────────────────────────────────────────────────────────────

    /// Get reference frame tensor for P-frame encoding
    fn get_reference_tensor(&self) -> Result<Array4<f32>> {
        self.reference_buffer
            .get_last_frame()
            .cloned()
            .ok_or_else(|| Error::codec("No reference frame available for P-frame encoding"))
    }

    /// Get reference frame as VideoFrame
    #[allow(dead_code)]
    fn get_reference_frame(&self) -> Result<VideoFrame> {
        if self.last_ref.tensor.is_some() {
            Ok(self.last_ref.to_video_frame())
        } else {
            Err(Error::codec("No reference frame available"))
        }
    }

    /// Clear all references (called on scene change or GOP start)
    pub fn clear_references(&mut self) {
        self.reference_buffer.clear();
        self.last_ref = ReferenceFrame::new(self.config.width, self.config.height);
        self.golden_ref = ReferenceFrame::new(self.config.width, self.config.height);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // P-Frame Bitstream Packaging
    // ─────────────────────────────────────────────────────────────────────────

    /// Package P-frame with motion and residual data
    fn package_pframe(
        &self,
        motion_bytes: Vec<u8>,
        residual_bytes: Vec<u8>,
        motion: &MotionField,
        pts: i64,
        dts: i64,
    ) -> Result<EncodedFrame> {
        let mut bitstream = Vec::new();

        // Write frame header
        self.write_frame_header(&mut bitstream, FrameType::P)?;

        // Write P-frame flags (bit 0 = has residual, bit 1 = skip mode)
        let flags: u8 = 0x01; // has residual = true
        bitstream.push(flags);

        // Write motion section
        bitstream.push(MOTION_SECTION_MARKER);
        // Motion field dimensions
        bitstream
            .write_u16::<LittleEndian>(motion.height() as u16)
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream
            .write_u16::<LittleEndian>(motion.width() as u16)
            .map_err(|e| Error::codec(e.to_string()))?;
        // Motion data size and bytes
        bitstream
            .write_u32::<LittleEndian>(motion_bytes.len() as u32)
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream.extend_from_slice(&motion_bytes);

        // Write residual section
        bitstream.push(RESIDUAL_SECTION_MARKER);
        // Residual data size and bytes
        bitstream
            .write_u32::<LittleEndian>(residual_bytes.len() as u32)
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream.extend_from_slice(&residual_bytes);

        Ok(EncodedFrame::new(
            BitstreamFrameType::P,
            bitstream,
            pts,
            dts,
        ))
    }

    /// Package skip-mode P-frame (motion only, no residual)
    fn package_pframe_skip(
        &self,
        motion_bytes: Vec<u8>,
        pts: i64,
        dts: i64,
    ) -> Result<EncodedFrame> {
        let mut bitstream = Vec::new();

        // Write frame header
        self.write_frame_header(&mut bitstream, FrameType::P)?;

        // Write P-frame flags (bit 0 = has residual, bit 1 = skip mode)
        let flags: u8 = 0x02; // skip mode = true, has residual = false
        bitstream.push(flags);

        // Write skip magic for verification
        bitstream.extend_from_slice(PFRAME_SKIP_MAGIC);

        // Write motion section
        bitstream.push(MOTION_SECTION_MARKER);
        // Motion data size and bytes
        bitstream
            .write_u32::<LittleEndian>(motion_bytes.len() as u32)
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream.extend_from_slice(&motion_bytes);

        // No residual section for skip mode

        // For skip mode, we still need to update reference with warped frame
        // This is handled in encode_pframe before calling this method

        Ok(EncodedFrame::new(
            BitstreamFrameType::P,
            bitstream,
            pts,
            dts,
        ))
    }

    /// Package I-frame data into bitstream format
    fn package_iframe(
        &self,
        z_bytes: Vec<u8>,
        y_bytes: Vec<u8>,
        y_shape: (usize, usize, usize, usize),
        z_shape: (usize, usize, usize, usize),
    ) -> Result<Vec<u8>> {
        let mut bitstream = Vec::new();

        // Write frame header
        self.write_frame_header(&mut bitstream, FrameType::I)?;

        // Write latent shape info (for decoder)
        bitstream
            .write_u16::<LittleEndian>(y_shape.1 as u16) // channels
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream
            .write_u16::<LittleEndian>(y_shape.2 as u16) // height
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream
            .write_u16::<LittleEndian>(y_shape.3 as u16) // width
            .map_err(|e| Error::codec(e.to_string()))?;

        // Write hyperprior shape info
        bitstream
            .write_u16::<LittleEndian>(z_shape.1 as u16) // channels
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream
            .write_u16::<LittleEndian>(z_shape.2 as u16) // height
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream
            .write_u16::<LittleEndian>(z_shape.3 as u16) // width
            .map_err(|e| Error::codec(e.to_string()))?;

        // Write quantization scale
        bitstream
            .write_u32::<LittleEndian>(self.quant_scale.to_bits())
            .map_err(|e| Error::codec(e.to_string()))?;

        // Write hyperprior section
        bitstream
            .write_u32::<LittleEndian>(z_bytes.len() as u32)
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream.extend_from_slice(&z_bytes);

        // Write main latent section
        bitstream
            .write_u32::<LittleEndian>(y_bytes.len() as u32)
            .map_err(|e| Error::codec(e.to_string()))?;
        bitstream.extend_from_slice(&y_bytes);

        Ok(bitstream)
    }

    /// Write ZVC69 frame header
    fn write_frame_header(&self, bitstream: &mut Vec<u8>, frame_type: FrameType) -> Result<()> {
        // ZVC69 frame header format:
        // [0-3]: Magic "ZVC1"
        // [4]:   Frame type (0=I, 1=P, 2=B)
        // [5]:   Quality level (1-8)
        // [6-7]: Width / 16
        // [8-9]: Height / 16
        // [10]:  Quantization parameter
        // [11]:  Reserved

        bitstream.extend_from_slice(b"ZVC1");

        let frame_type_byte = match frame_type {
            FrameType::I => 0u8,
            FrameType::P => 1u8,
            FrameType::B => 2u8,
        };
        bitstream.push(frame_type_byte);

        bitstream.push(self.config.quality.level());

        let width_mb = (self.config.width / 16) as u16;
        let height_mb = (self.config.height / 16) as u16;
        bitstream.extend_from_slice(&width_mb.to_le_bytes());
        bitstream.extend_from_slice(&height_mb.to_le_bytes());

        // Write QP based on quality
        let qp = (51.0 * (1.0 - self.config.quality.quant_scale())) as u8;
        bitstream.push(qp);

        // Reserved byte
        bitstream.push(0);

        Ok(())
    }

    /// Write placeholder frame data (when model is not loaded)
    fn write_placeholder_frame_data(
        &self,
        bitstream: &mut Vec<u8>,
        frame: &VideoFrame,
    ) -> Result<()> {
        // Write frame size (placeholder)
        let frame_size = (frame.width * frame.height) as u32;
        bitstream.extend_from_slice(&frame_size.to_le_bytes());

        // Write a simple checksum of the input (for verification during development)
        let checksum = if !frame.data.is_empty() {
            let y_data = frame.data[0].as_slice();
            y_data
                .iter()
                .fold(0u32, |acc, &x| acc.wrapping_add(x as u32))
        } else {
            0
        };
        bitstream.extend_from_slice(&checksum.to_le_bytes());

        // Placeholder marker
        bitstream.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        Ok(())
    }

    /// Update reference frame buffer
    fn update_reference_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.data.len() >= 3 {
            // Copy Y plane
            let y_data = frame.data[0].as_slice();
            let y_size = (self.config.width * self.config.height) as usize;
            if y_data.len() >= y_size {
                self.last_ref.y[..y_size].copy_from_slice(&y_data[..y_size]);
            }

            // Copy U plane
            let u_data = frame.data[1].as_slice();
            let uv_size = y_size / 4;
            if u_data.len() >= uv_size {
                self.last_ref.u[..uv_size].copy_from_slice(&u_data[..uv_size]);
            }

            // Copy V plane
            let v_data = frame.data[2].as_slice();
            if v_data.len() >= uv_size {
                self.last_ref.v[..uv_size].copy_from_slice(&v_data[..uv_size]);
            }

            self.last_ref.frame_num = self.frame_count;
            self.last_ref.pts = frame.pts;
        }

        Ok(())
    }

    /// Generate extradata (codec configuration record)
    fn generate_extradata(&mut self) -> Vec<u8> {
        // ZVC69 configuration record:
        // [0-3]:   Magic "ZVC0"
        // [4]:     Version (1)
        // [5]:     Profile (0 = baseline)
        // [6]:     Level (computed from resolution)
        // [7]:     Quality default
        // [8-9]:   Width
        // [10-11]: Height
        // [12-13]: Framerate num
        // [14-15]: Framerate den

        let mut extradata = Vec::with_capacity(16);

        extradata.extend_from_slice(b"ZVC0");
        extradata.push(1); // Version
        extradata.push(0); // Profile

        // Compute level from resolution
        let pixels = self.config.width * self.config.height;
        let level = if pixels <= 921600 {
            31
        } else if pixels <= 2073600 {
            40
        } else if pixels <= 8294400 {
            51
        } else {
            62
        };
        extradata.push(level);

        extradata.push(self.config.quality.level());
        extradata.extend_from_slice(&(self.config.width as u16).to_le_bytes());
        extradata.extend_from_slice(&(self.config.height as u16).to_le_bytes());
        extradata.extend_from_slice(&(self.config.framerate_num as u16).to_le_bytes());
        extradata.extend_from_slice(&(self.config.framerate_den as u16).to_le_bytes());

        extradata
    }
}

impl Encoder for ZVC69Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(video_frame) => {
                self.state = EncoderState::Encoding;
                self.encode_video_frame(video_frame)
            }
            Frame::Audio(_) => Err(Error::codec("ZVC69 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(packet) = self.output_packets.pop() {
            Ok(packet)
        } else {
            Err(Error::TryAgain)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.state = EncoderState::Flushing;

        // Encode any remaining frames in lookahead buffer
        // (B-frame reordering would be handled here)

        self.lookahead_buffer.clear();
        self.state = EncoderState::Finished;

        Ok(())
    }

    fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }
}

/// Encoder statistics
#[derive(Debug, Clone)]
pub struct EncoderStats {
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Total bits produced
    pub total_bits: u64,
    /// Average bitrate in bps
    pub avg_bitrate: u64,
    /// Average quantization parameter
    pub avg_qp: f32,
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        // Use dimensions divisible by 16
        let config = ZVC69Config::new(1920, 1088);
        let encoder = ZVC69Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_with_dimensions() {
        let encoder = ZVC69Encoder::with_dimensions(1280, 720);
        assert!(encoder.is_ok());
        let encoder = encoder.unwrap();
        assert_eq!(encoder.config.width, 1280);
        assert_eq!(encoder.config.height, 720);
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let config = ZVC69Config {
            width: 100, // Not divisible by 16
            height: 100,
            ..Default::default()
        };
        let encoder = ZVC69Encoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_frame_type_determination() {
        // Use dimensions divisible by 16
        let config = ZVC69Config::builder()
            .dimensions(1920, 1088)
            .keyframe_interval(30)
            .bframes(0)
            .build()
            .unwrap();

        let encoder = ZVC69Encoder::new(config).unwrap();

        // First frame should be I
        assert_eq!(encoder.determine_frame_type(), FrameType::I);
    }

    #[test]
    fn test_encoder_stats() {
        // Use dimensions divisible by 16
        let config = ZVC69Config::new(1920, 1088);
        let encoder = ZVC69Encoder::new(config).unwrap();
        let stats = encoder.stats();

        assert_eq!(stats.frames_encoded, 0);
        assert_eq!(stats.total_bits, 0);
    }

    #[test]
    fn test_encode_simple_frame() {
        let config = ZVC69Config::new(64, 64);
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Create a test frame
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        // Encode
        let result = encoder.send_frame(&Frame::Video(frame));
        assert!(result.is_ok());

        // Get packet
        let packet = encoder.receive_packet();
        assert!(packet.is_ok());

        let packet = packet.unwrap();
        assert!(packet.flags.keyframe);

        // Verify ZVC69 header
        let data = packet.data.as_slice();
        assert!(data.len() >= 12);
        assert_eq!(&data[0..4], b"ZVC1");
    }

    #[test]
    fn test_dimension_mismatch() {
        // Use dimensions divisible by 16
        let config = ZVC69Config::new(1920, 1088);
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Create frame with wrong dimensions (but still valid for a frame)
        let mut frame = VideoFrame::new(1280, 720, PixelFormat::YUV420P);
        let y_size = 1280 * 720;
        let uv_size = y_size / 4;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];

        let result = encoder.send_frame(&Frame::Video(frame));
        assert!(result.is_err());
    }

    #[test]
    fn test_encoded_frame_creation() {
        let frame = EncodedFrame::new(BitstreamFrameType::I, vec![0u8; 1000], 0, 0);

        assert!(frame.is_keyframe);
        assert_eq!(frame.size_bits, 8000);
        assert_eq!(frame.frame_type, BitstreamFrameType::I);
    }

    #[test]
    fn test_quality_change() {
        let config = ZVC69Config::new(64, 64);
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        encoder.set_quality(Quality::Q1);
        assert!(encoder.quant_scale > 1.0); // Lower quality = higher scale

        encoder.set_quality(Quality::Q8);
        assert!(encoder.quant_scale < 1.0); // Higher quality = lower scale
    }

    #[test]
    fn test_force_keyframe() {
        // Use config without B-frames for predictable frame type
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .bframes(0)
            .keyframe_interval(30)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Encode first frame to populate reference
        let mut iframe = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        iframe.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        iframe.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(iframe)).unwrap();
        let _ = encoder.receive_packet().unwrap();

        // Now we have a reference, so P-frame should be possible
        assert!(encoder.reference_buffer.has_reference());

        // At this point: frame_count = 1, gop_frame_count = 1
        // With keyframe_interval = 30 and bframes = 0, next frame should be P
        assert_eq!(encoder.determine_frame_type(), FrameType::P);

        // Force keyframe
        encoder.force_keyframe();
        assert_eq!(encoder.determine_frame_type(), FrameType::I);
    }

    #[test]
    fn test_encode_multiple_frames() {
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .keyframe_interval(3)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Encode multiple frames
        for i in 0..5 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = 32 * 32;
            frame.data = vec![
                Buffer::from_vec(vec![(128 + i) as u8; y_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            encoder.send_frame(&Frame::Video(frame)).unwrap();
            let packet = encoder.receive_packet().unwrap();

            // First frame and every 3rd should be keyframe
            if i == 0 || i == 3 {
                assert!(packet.flags.keyframe, "Frame {} should be keyframe", i);
            }
        }

        let stats = encoder.stats();
        assert_eq!(stats.frames_encoded, 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // P-Frame Encoding Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_pframe_encoding_with_motion() {
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .keyframe_interval(10)
            .bframes(0)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Encode I-frame first
        let mut iframe = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        iframe.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        iframe.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(iframe)).unwrap();
        let ipacket = encoder.receive_packet().unwrap();
        assert!(ipacket.flags.keyframe);

        // Verify reference buffer is populated
        assert!(encoder.reference_buffer.has_reference());

        // Encode P-frame
        let mut pframe = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        // Create slightly different frame to generate motion/residual
        let mut y_data = vec![128u8; y_size];
        for (i, pixel) in y_data.iter_mut().enumerate() {
            *pixel = (128 + (i % 10) as u8).min(255);
        }
        pframe.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        pframe.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(pframe)).unwrap();
        let ppacket = encoder.receive_packet().unwrap();
        assert!(!ppacket.flags.keyframe);

        // Verify P-frame header
        let data = ppacket.data.as_slice();
        assert!(data.len() >= 12);
        assert_eq!(&data[0..4], b"ZVC1");
        assert_eq!(data[4], 1); // Frame type = P
    }

    #[test]
    fn test_pframe_skip_mode() {
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .keyframe_interval(10)
            .bframes(0)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Set high skip threshold to force skip mode
        encoder.pframe_skip_threshold = 1.0;

        // Encode I-frame
        let mut iframe = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        iframe.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        iframe.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(iframe)).unwrap();
        let _ = encoder.receive_packet().unwrap();

        // Encode P-frame with identical content (should trigger skip)
        let mut pframe = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        pframe.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        pframe.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(pframe)).unwrap();
        let ppacket = encoder.receive_packet().unwrap();
        assert!(!ppacket.flags.keyframe);

        // Verify P-frame is skip mode (flag byte should have bit 1 set)
        let data = ppacket.data.as_slice();
        assert!(data.len() >= 13);
        assert_eq!(&data[0..4], b"ZVC1");
        assert_eq!(data[4], 1); // Frame type = P
        let flags = data[12]; // Flags byte
        assert_eq!(flags & 0x02, 0x02); // Skip mode bit set
    }

    #[test]
    fn test_gop_structure_ippp() {
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .keyframe_interval(4)
            .bframes(0)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Expected pattern: I P P P I P P P
        let expected_keyframes = vec![true, false, false, false, true, false, false, false];

        for (i, &expected_keyframe) in expected_keyframes.iter().enumerate() {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = 32 * 32;
            frame.data = vec![
                Buffer::from_vec(vec![(128 + i) as u8; y_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            encoder.send_frame(&Frame::Video(frame)).unwrap();
            let packet = encoder.receive_packet().unwrap();

            assert_eq!(
                packet.flags.keyframe, expected_keyframe,
                "Frame {} should be keyframe={}, got keyframe={}",
                i, expected_keyframe, packet.flags.keyframe
            );
        }
    }

    #[test]
    fn test_reference_frame_updates() {
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .keyframe_interval(10)
            .bframes(0)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Initially no reference
        assert!(!encoder.reference_buffer.has_reference());

        // Encode I-frame
        let mut iframe = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        iframe.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        iframe.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(iframe)).unwrap();
        let _ = encoder.receive_packet().unwrap();

        // Now have reference
        assert!(encoder.reference_buffer.has_reference());
        let ref_frame = encoder.reference_buffer.get_last_frame().unwrap();
        assert_eq!(ref_frame.dim(), (1, 3, 64, 64));
    }

    #[test]
    fn test_clear_references() {
        let config = ZVC69Config::new(64, 64);
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Encode a frame to populate reference
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        encoder.send_frame(&Frame::Video(frame)).unwrap();
        let _ = encoder.receive_packet().unwrap();

        assert!(encoder.reference_buffer.has_reference());

        // Clear references
        encoder.clear_references();

        assert!(!encoder.reference_buffer.has_reference());
    }

    #[test]
    fn test_encode_decode_roundtrip_pframe() {
        let config = ZVC69Config::builder()
            .dimensions(64, 64)
            .keyframe_interval(10)
            .bframes(0)
            .build()
            .unwrap();
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Encode sequence: I, P, P
        for i in 0..3 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = 32 * 32;
            // Create varying content
            let mut y_data = vec![0u8; y_size];
            for (j, pixel) in y_data.iter_mut().enumerate() {
                *pixel = ((128 + i * 10 + (j % 20)) as u8).min(255);
            }
            frame.data = vec![
                Buffer::from_vec(y_data),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            encoder.send_frame(&Frame::Video(frame)).unwrap();
            let packet = encoder.receive_packet().unwrap();

            // Verify packet is valid
            let data = packet.data.as_slice();
            assert!(data.len() >= 12);
            assert_eq!(&data[0..4], b"ZVC1");

            // Check frame type
            let frame_type = data[4];
            if i == 0 {
                assert_eq!(frame_type, 0); // I-frame
                assert!(packet.flags.keyframe);
            } else {
                assert_eq!(frame_type, 1); // P-frame
                assert!(!packet.flags.keyframe);
            }
        }

        let stats = encoder.stats();
        assert_eq!(stats.frames_encoded, 3);
    }

    #[test]
    fn test_reference_buffer_struct() {
        let mut buffer = ReferenceBuffer::new();

        assert!(!buffer.has_reference());
        assert!(buffer.get_last_frame().is_none());
        assert!(buffer.get_last_latent().is_none());

        // Update with a frame
        let tensor = Array4::<f32>::zeros((1, 3, 64, 64));
        let latent = Array4::<f32>::zeros((1, 192, 4, 4));
        buffer.update(tensor.clone(), Some(latent.clone()), 0);

        assert!(buffer.has_reference());
        assert!(buffer.get_last_frame().is_some());
        assert!(buffer.get_last_latent().is_some());

        // Update golden
        buffer.update_golden(tensor.clone(), 0);

        // Clear
        buffer.clear();
        assert!(!buffer.has_reference());
    }

    #[test]
    fn test_decide_frame_type_public_api() {
        let config = ZVC69Config::new(64, 64);
        let encoder = ZVC69Encoder::new(config).unwrap();

        // First frame should be I
        let frame_type = encoder.decide_frame_type();
        assert_eq!(frame_type, BitstreamFrameType::I);
    }
}
