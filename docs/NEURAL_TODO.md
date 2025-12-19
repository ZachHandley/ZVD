# ZVC69 Neural Video Codec Implementation TODO

> **Document Version**: 1.0
> **Target Completion**: 13 weeks (3 months)
> **Reference**: [NEURAL_CODEC_ZVC69.md](./NEURAL_CODEC_ZVC69.md)
> **Last Updated**: December 9, 2025

---

## Overview

This document provides a step-by-step implementation guide for building the ZVC69 neural video codec. The goal is to achieve **20%+ bitrate savings over AV1** with **real-time 1080p encoding/decoding**.

### Success Milestones

| Milestone | Target | Acceptance Criteria |
|-----------|--------|---------------------|
| **M1** | I-frame encode/decode working | Compress/decompress single images, PSNR > 30dB at 0.5 bpp |
| **M2** | P-frame encode/decode working | Encode video sequences with motion compensation |
| **M3** | Real-time 720p achieved | 30+ fps encode, 60+ fps decode at 720p on RTX 3060 |
| **M4** | Real-time 1080p achieved | 30+ fps encode, 60+ fps decode at 1080p on RTX 3080 |
| **M5** | Production-ready release | Full API, documentation, benchmarks vs AV1/H.265 |

---

## Phase 1: Foundation (Weeks 1-3)

### Week 1: Module Structure & Dependencies

#### Day 1-2: Project Setup

- [ ] **Create ZVC69 module structure**
  - **Files**: `src/codec/zvc69/mod.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Module compiles and exports public API stubs

  ```rust
  // src/codec/zvc69/mod.rs
  pub mod encoder;
  pub mod decoder;
  pub mod model;
  pub mod entropy;
  pub mod bitstream;
  pub mod config;
  pub mod quantize;

  pub use encoder::ZVC69Encoder;
  pub use decoder::ZVC69Decoder;
  pub use config::{ZVC69Config, Quality, Preset};
  ```

- [ ] **Add Cargo.toml dependencies**
  - **Files**: `Cargo.toml`
  - **Effort**: 1 hour
  - **Dependencies**:

  ```toml
  # Neural inference
  ort = { version = "2.0.0-rc.10", features = ["cuda", "tensorrt"] }

  # Entropy coding
  constriction = "0.4"

  # Tensor operations
  ndarray = { version = "0.16", features = ["rayon"] }

  # Serialization
  byteorder = "1.5"
  serde = { version = "1.0", features = ["derive"] }

  # Error handling
  thiserror = "1.0"
  anyhow = "1.0"

  # Optional: image loading for tests
  image = "0.25"
  ```

- [ ] **Create config.rs with quality presets**
  - **Files**: `src/codec/zvc69/config.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Config struct with builder pattern

  ```rust
  // src/codec/zvc69/config.rs
  #[derive(Debug, Clone)]
  pub struct ZVC69Config {
      pub width: u32,
      pub height: u32,
      pub quality: Quality,
      pub preset: Preset,
      pub gop_size: u8,
      pub latent_channels: u32,
      pub hyperprior_channels: u32,
  }

  #[derive(Debug, Clone, Copy)]
  pub enum Quality {
      Low = 1,      // ~0.1 bpp
      Medium = 3,   // ~0.25 bpp
      High = 5,     // ~0.5 bpp
      VeryHigh = 7, // ~1.0 bpp
  }

  #[derive(Debug, Clone, Copy)]
  pub enum Preset {
      Ultrafast,  // Minimal processing
      Fast,       // Balanced speed/quality
      Medium,     // Default
      Slow,       // Better compression
      Veryslow,   // Maximum compression
  }
  ```

#### Day 3-4: Bitstream Format Implementation

- [ ] **Define bitstream header structures**
  - **Files**: `src/codec/zvc69/bitstream.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Can serialize/deserialize file and frame headers

  ```rust
  // src/codec/zvc69/bitstream.rs
  pub const MAGIC: &[u8; 6] = b"ZVC69\0";
  pub const VERSION_MAJOR: u8 = 1;
  pub const VERSION_MINOR: u8 = 0;

  #[repr(C)]
  pub struct FileHeader {
      pub magic: [u8; 6],           // "ZVC69\0"
      pub version_major: u8,
      pub version_minor: u8,
      pub flags: u32,               // Feature flags
      pub width: u16,
      pub height: u16,
      pub framerate_num: u16,
      pub framerate_den: u16,
      pub total_frames: u32,
      pub gop_size: u8,
      pub quality_level: u8,
      pub color_space: u8,          // 0=YUV420, 1=YUV444, 2=RGB
      pub bit_depth: u8,            // 8, 10, or 12
      pub model_hash: [u8; 4],      // First 4 bytes of model SHA-256
      pub index_offset: u64,        // Byte offset to index table
      pub latent_channels: u32,
      pub hyperprior_channels: u32,
      pub reserved: [u8; 16],
  }

  #[repr(C)]
  pub struct FrameHeader {
      pub frame_type: u8,           // 0=I, 1=P, 2=B
      pub temporal_layer: u8,
      pub reference_flags: u16,
      pub frame_size: u32,
      pub pts: u64,
      pub dts: u64,
      pub qp_offset: i16,
      pub checksum: u16,
      pub reserved: u32,
  }

  pub enum FrameType {
      I = 0,
      P = 1,
      B = 2,
  }
  ```

- [ ] **Implement BitstreamWriter**
  - **Files**: `src/codec/zvc69/bitstream.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Write valid ZVC69 bitstream to file/buffer

- [ ] **Implement BitstreamReader**
  - **Files**: `src/codec/zvc69/bitstream.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Parse ZVC69 bitstream, validate magic/version

#### Day 5: Index Table for Seeking

- [ ] **Implement index table structure**
  - **Files**: `src/codec/zvc69/bitstream.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Random access seeking to any keyframe

  ```rust
  pub struct IndexEntry {
      pub frame_number: u32,
      pub byte_offset: u64,
      pub flags: u32,  // IS_KEYFRAME, IS_REFERENCE
  }

  pub struct IndexTable {
      pub entries: Vec<IndexEntry>,
  }

  impl IndexTable {
      pub fn seek_to_frame(&self, target: u32) -> Option<&IndexEntry> {
          // Find nearest keyframe at or before target
          self.entries
              .iter()
              .filter(|e| e.frame_number <= target && e.is_keyframe())
              .max_by_key(|e| e.frame_number)
      }
  }
  ```

### Week 2: Entropy Coding & ONNX Model Loading

#### Day 1-2: Entropy Coding with constriction

- [ ] **Create entropy.rs module**
  - **Files**: `src/codec/zvc69/entropy.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Encode/decode integer symbols with Gaussian model

  ```rust
  // src/codec/zvc69/entropy.rs
  use constriction::stream::stack::DefaultAnsCoder;
  use constriction::stream::model::DefaultLeakyQuantizer;

  pub struct EntropyEncoder {
      coder: DefaultAnsCoder,
  }

  impl EntropyEncoder {
      pub fn new() -> Self {
          Self {
              coder: DefaultAnsCoder::new(),
          }
      }

      /// Encode latent tensor with predicted mean/scale
      pub fn encode_gaussian(
          &mut self,
          symbols: &[i32],
          means: &[f64],
          scales: &[f64],
      ) -> Result<(), EntropyError> {
          // Encode in reverse order (ANS is LIFO)
          for ((&y, &mu), &sigma) in symbols.iter()
              .zip(means.iter())
              .zip(scales.iter())
              .rev()
          {
              let quantizer = DefaultLeakyQuantizer::new(-128..=127);
              let gaussian = probability::distribution::Gaussian::new(mu, sigma.max(0.11));
              let model = quantizer.quantize(gaussian);
              self.coder.encode_symbol(y, model)?;
          }
          Ok(())
      }

      pub fn finish(self) -> Vec<u32> {
          self.coder.into_compressed().unwrap()
      }
  }
  ```

- [ ] **Implement factorized prior for hyperprior**
  - **Files**: `src/codec/zvc69/entropy.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Encode hyperprior latents with learned CDF

- [ ] **Add entropy decoder**
  - **Files**: `src/codec/zvc69/entropy.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Symmetric decode with encode

#### Day 3-4: ONNX Model Loading with ort

- [ ] **Create model.rs module**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Load ONNX model and run inference

  ```rust
  // src/codec/zvc69/model.rs
  use ort::{Environment, Session, SessionBuilder, Value};
  use std::sync::Arc;

  pub struct ModelLoader {
      environment: Arc<Environment>,
  }

  impl ModelLoader {
      pub fn new() -> Result<Self, ModelError> {
          let environment = Environment::builder()
              .with_name("ZVC69")
              .with_execution_providers([
                  ort::CUDAExecutionProvider::default().build(),
                  ort::TensorrtExecutionProvider::default()
                      .with_fp16(true)
                      .build(),
              ])
              .build()?;

          Ok(Self {
              environment: Arc::new(environment),
          })
      }

      pub fn load_session(&self, model_path: &str) -> Result<Session, ModelError> {
          SessionBuilder::new(&self.environment)?
              .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
              .with_model_from_file(model_path)
              .map_err(ModelError::from)
      }
  }
  ```

- [ ] **Implement AnalysisTransform wrapper**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Run encoder network, get latent tensor

- [ ] **Implement SynthesisTransform wrapper**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Run decoder network, reconstruct image

#### Day 5: Hyperprior Networks

- [ ] **Implement HyperpriorEncoder wrapper**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Encode side information z from latent y

- [ ] **Implement HyperpriorDecoder wrapper**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Decode mean/scale parameters from z

### Week 3: I-Frame Encoder/Decoder

#### Day 1-2: I-Frame Encoder

- [ ] **Create encoder.rs skeleton**
  - **Files**: `src/codec/zvc69/encoder.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Encoder struct with encode_iframe method

  ```rust
  // src/codec/zvc69/encoder.rs
  pub struct ZVC69Encoder {
      config: ZVC69Config,
      model_loader: ModelLoader,
      analysis: Session,
      hyperprior_enc: Session,
      entropy_encoder: EntropyEncoder,
      frame_count: u64,
  }

  impl ZVC69Encoder {
      pub fn new(config: ZVC69Config, model_dir: &Path) -> Result<Self, EncoderError>;

      pub fn encode_iframe(&mut self, frame: &VideoFrame) -> Result<EncodedFrame, EncoderError> {
          // 1. Run analysis transform: frame -> latent y
          let y = self.analysis.run(frame.to_tensor())?;

          // 2. Quantize latent
          let y_hat = quantize(&y);

          // 3. Run hyperprior encoder: y -> z
          let z = self.hyperprior_enc.run(&y)?;
          let z_hat = quantize(&z);

          // 4. Entropy encode z with factorized prior
          let z_stream = self.encode_hyperprior(&z_hat)?;

          // 5. Run hyperprior decoder: z_hat -> (mean, scale)
          let (mean, scale) = self.hyperprior_dec.run(&z_hat)?;

          // 6. Entropy encode y_hat with Gaussian conditional
          let y_stream = self.encode_latent(&y_hat, &mean, &scale)?;

          // 7. Pack into bitstream
          Ok(EncodedFrame::new_iframe(z_stream, y_stream))
      }
  }
  ```

- [ ] **Implement quantize.rs**
  - **Files**: `src/codec/zvc69/quantize.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Round-to-nearest quantization with STE gradient

  ```rust
  // src/codec/zvc69/quantize.rs
  pub fn quantize(tensor: &Tensor) -> Tensor {
      tensor.mapv(|x| x.round())
  }

  pub fn dequantize(tensor: &Tensor, scale: f32) -> Tensor {
      tensor.mapv(|x| x * scale)
  }
  ```

#### Day 3-4: I-Frame Decoder

- [ ] **Create decoder.rs skeleton**
  - **Files**: `src/codec/zvc69/decoder.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Decoder struct with decode_iframe method

  ```rust
  // src/codec/zvc69/decoder.rs
  pub struct ZVC69Decoder {
      model_loader: ModelLoader,
      synthesis: Session,
      hyperprior_dec: Session,
      entropy_decoder: EntropyDecoder,
      reference_frame: Option<VideoFrame>,
  }

  impl ZVC69Decoder {
      pub fn new(model_dir: &Path) -> Result<Self, DecoderError>;

      pub fn decode_iframe(&mut self, data: &EncodedFrame) -> Result<VideoFrame, DecoderError> {
          // 1. Entropy decode z
          let z_hat = self.decode_hyperprior(&data.hyperprior_stream)?;

          // 2. Run hyperprior decoder: z_hat -> (mean, scale)
          let (mean, scale) = self.hyperprior_dec.run(&z_hat)?;

          // 3. Entropy decode y
          let y_hat = self.decode_latent(&data.latent_stream, &mean, &scale)?;

          // 4. Run synthesis transform: y_hat -> reconstructed frame
          let frame = self.synthesis.run(&y_hat)?;

          // 5. Store as reference for P-frames
          self.reference_frame = Some(frame.clone());

          Ok(frame)
      }
  }
  ```

#### Day 5: Integration Tests & Milestone M1

- [ ] **Create I-frame round-trip test**
  - **Files**: `tests/zvc69_iframe_test.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Encode then decode image, PSNR > 30dB

  ```rust
  #[test]
  fn test_iframe_roundtrip() {
      let config = ZVC69Config::default();
      let mut encoder = ZVC69Encoder::new(config.clone(), MODEL_DIR).unwrap();
      let mut decoder = ZVC69Decoder::new(MODEL_DIR).unwrap();

      // Load test image
      let original = load_test_image("test_1080p.png");

      // Encode
      let encoded = encoder.encode_iframe(&original).unwrap();
      assert!(encoded.size_bytes() < original.size_bytes() / 4);

      // Decode
      let decoded = decoder.decode_iframe(&encoded).unwrap();

      // Quality check
      let psnr = calculate_psnr(&original, &decoded);
      assert!(psnr > 30.0, "PSNR too low: {}", psnr);
  }
  ```

- [ ] **Verify Milestone M1: I-frame codec working**
  - **Acceptance**: Single images compress/decompress with PSNR > 30dB at 0.5 bpp

---

## Phase 2: Video Codec (Weeks 4-7)

### Week 4: Motion Estimation Network

#### Day 1-2: Motion Estimator Implementation

- [ ] **Create motion.rs module**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Estimate optical flow between frames

  ```rust
  // src/codec/zvc69/motion.rs
  pub struct MotionEstimator {
      session: Session,
  }

  impl MotionEstimator {
      pub fn new(model_loader: &ModelLoader, model_path: &Path) -> Result<Self, MotionError>;

      /// Estimate optical flow from reference to current frame
      pub fn estimate_flow(
          &self,
          current: &VideoFrame,
          reference: &VideoFrame,
      ) -> Result<FlowField, MotionError> {
          // Concatenate frames: [B, 6, H, W]
          let input = concat_frames(current, reference);

          // Run motion network
          let flow = self.session.run(&input)?;

          // Output: [B, 2, H, W] - (dx, dy) per pixel
          Ok(FlowField::from_tensor(flow))
      }
  }

  pub struct FlowField {
      pub dx: Tensor2D,
      pub dy: Tensor2D,
  }
  ```

- [ ] **Implement flow encoding/decoding**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Quantize and entropy code flow field

#### Day 3-4: Frame Warping

- [ ] **Implement bilinear warping**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Warp reference frame using flow field

  ```rust
  // src/codec/zvc69/motion.rs
  impl FlowField {
      /// Warp reference frame to predict current frame
      pub fn warp(&self, reference: &VideoFrame) -> VideoFrame {
          let (h, w) = (reference.height(), reference.width());
          let mut warped = VideoFrame::zeros(h, w);

          for y in 0..h {
              for x in 0..w {
                  // Get flow at this position
                  let dx = self.dx[[y, x]];
                  let dy = self.dy[[y, x]];

                  // Source position (with bilinear interpolation)
                  let src_x = x as f32 + dx;
                  let src_y = y as f32 + dy;

                  // Bilinear sample from reference
                  warped.set_pixel(y, x, reference.bilinear_sample(src_y, src_x));
              }
          }

          warped
      }
  }
  ```

- [ ] **Optimize warping with SIMD/GPU**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Warping < 5ms for 1080p frame

#### Day 5: Motion Network ONNX Export

- [ ] **Create Python script to export motion network**
  - **Files**: `train/export_motion.py`
  - **Effort**: 3 hours
  - **Acceptance**: ONNX model loads in Rust

### Week 5: Frame Warping & Prediction

#### Day 1-2: Temporal Context Model

- [ ] **Implement temporal context extraction**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Extract features from reference frames

- [ ] **Add multi-reference support**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Support L0/L1 reference lists

#### Day 3-4: Implicit Motion Model (DCVC-RT Style)

- [ ] **Implement implicit temporal modeling**
  - **Files**: `src/codec/zvc69/motion.rs`
  - **Effort**: 6 hours
  - **Acceptance**: Single network replaces explicit motion estimation

  ```rust
  // DCVC-RT uses implicit temporal modeling instead of explicit flow
  pub struct TemporalContextModel {
      session: Session,
  }

  impl TemporalContextModel {
      /// Process current frame with temporal context from references
      pub fn extract_context(
          &self,
          current: &VideoFrame,
          references: &[VideoFrame],
      ) -> Result<TemporalContext, MotionError> {
          // Concatenate current + references
          let input = self.prepare_input(current, references);

          // Run single network that implicitly handles motion
          let context = self.session.run(&input)?;

          Ok(TemporalContext::from_tensor(context))
      }
  }
  ```

#### Day 5: Reference Frame Buffer

- [ ] **Implement decoded picture buffer (DPB)**
  - **Files**: `src/codec/zvc69/buffer.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Manage reference frames for P/B-frame decoding

  ```rust
  // src/codec/zvc69/buffer.rs
  pub struct DecodedPictureBuffer {
      max_size: usize,
      frames: VecDeque<ReferenceFrame>,
  }

  pub struct ReferenceFrame {
      pub frame: VideoFrame,
      pub poc: u64,  // Picture order count
      pub is_long_term: bool,
  }

  impl DecodedPictureBuffer {
      pub fn add_frame(&mut self, frame: VideoFrame, poc: u64);
      pub fn get_reference(&self, poc: u64) -> Option<&VideoFrame>;
      pub fn get_recent(&self, count: usize) -> Vec<&VideoFrame>;
      pub fn mark_for_removal(&mut self, poc: u64);
  }
  ```

### Week 6: Residual Coding

#### Day 1-2: Residual Encoder/Decoder

- [ ] **Create residual.rs module**
  - **Files**: `src/codec/zvc69/residual.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Encode/decode prediction residuals

  ```rust
  // src/codec/zvc69/residual.rs
  pub struct ResidualCodec {
      analysis: Session,
      synthesis: Session,
  }

  impl ResidualCodec {
      /// Encode residual (actual - predicted)
      pub fn encode(&self, residual: &VideoFrame) -> Result<ResidualLatent, ResidualError> {
          // Residual network is smaller than main codec
          // because residuals have lower entropy
          let latent = self.analysis.run(residual.to_tensor())?;
          let quantized = quantize(&latent);
          Ok(ResidualLatent::new(quantized))
      }

      /// Decode residual latent back to pixel domain
      pub fn decode(&self, latent: &ResidualLatent) -> Result<VideoFrame, ResidualError> {
          let residual = self.synthesis.run(&latent.tensor)?;
          Ok(VideoFrame::from_tensor(residual))
      }
  }
  ```

- [ ] **Implement residual entropy model**
  - **Files**: `src/codec/zvc69/residual.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Entropy code residual latents

#### Day 3-4: P-Frame Full Pipeline

- [ ] **Implement P-frame encoder**
  - **Files**: `src/codec/zvc69/encoder.rs`
  - **Effort**: 6 hours
  - **Acceptance**: Full P-frame encoding pipeline

  ```rust
  impl ZVC69Encoder {
      pub fn encode_pframe(
          &mut self,
          frame: &VideoFrame,
          reference: &VideoFrame,
      ) -> Result<EncodedFrame, EncoderError> {
          // 1. Extract temporal context
          let context = self.temporal_model.extract_context(frame, &[reference])?;

          // 2. Entropy encode context
          let context_stream = self.encode_context(&context)?;

          // 3. Predict frame from context
          let predicted = self.predictor.predict(&context, reference)?;

          // 4. Compute residual
          let residual = frame.subtract(&predicted);

          // 5. Encode residual
          let residual_latent = self.residual_codec.encode(&residual)?;

          // 6. Encode residual with hyperprior
          let (z_stream, y_stream) = self.encode_residual_with_hyperprior(&residual_latent)?;

          Ok(EncodedFrame::new_pframe(context_stream, z_stream, y_stream))
      }
  }
  ```

#### Day 5: P-Frame Decoder

- [ ] **Implement P-frame decoder**
  - **Files**: `src/codec/zvc69/decoder.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Decode P-frames with motion compensation

  ```rust
  impl ZVC69Decoder {
      pub fn decode_pframe(&mut self, data: &EncodedFrame) -> Result<VideoFrame, DecoderError> {
          let reference = self.reference_frame.as_ref()
              .ok_or(DecoderError::NoReference)?;

          // 1. Decode temporal context
          let context = self.decode_context(&data.context_stream)?;

          // 2. Predict frame
          let predicted = self.predictor.predict(&context, reference)?;

          // 3. Decode residual hyperprior
          let z_hat = self.decode_hyperprior(&data.hyperprior_stream)?;
          let (mean, scale) = self.hyperprior_dec.run(&z_hat)?;

          // 4. Decode residual latent
          let residual_latent = self.decode_latent(&data.latent_stream, &mean, &scale)?;
          let residual = self.residual_codec.decode(&residual_latent)?;

          // 5. Reconstruct frame
          let frame = predicted.add(&residual);

          // 6. Update reference
          self.reference_frame = Some(frame.clone());

          Ok(frame)
      }
  }
  ```

### Week 7: P-Frame Integration & Testing

#### Day 1-2: GOP Structure Implementation

- [ ] **Implement GOP manager**
  - **Files**: `src/codec/zvc69/gop.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Automatic I/P frame decision based on GOP size

  ```rust
  // src/codec/zvc69/gop.rs
  pub struct GopManager {
      gop_size: u8,
      frame_count: u64,
  }

  impl GopManager {
      pub fn frame_type(&self, frame_num: u64) -> FrameType {
          if frame_num % self.gop_size as u64 == 0 {
              FrameType::I
          } else {
              FrameType::P
          }
      }

      pub fn is_keyframe(&self, frame_num: u64) -> bool {
          self.frame_type(frame_num) == FrameType::I
      }
  }
  ```

- [ ] **Add scene change detection**
  - **Files**: `src/codec/zvc69/gop.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Force I-frame on scene changes

#### Day 3-4: Video Sequence Encoding

- [ ] **Implement full video encoding loop**
  - **Files**: `src/codec/zvc69/encoder.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Encode video file frame by frame

  ```rust
  impl ZVC69Encoder {
      pub fn encode_sequence<I>(&mut self, frames: I) -> Result<Vec<EncodedFrame>, EncoderError>
      where
          I: Iterator<Item = VideoFrame>,
      {
          let mut encoded = Vec::new();
          let mut reference: Option<VideoFrame> = None;

          for (i, frame) in frames.enumerate() {
              let frame_type = self.gop.frame_type(i as u64);

              let encoded_frame = match frame_type {
                  FrameType::I => {
                      let ef = self.encode_iframe(&frame)?;
                      reference = Some(self.decode_for_reference(&ef)?);
                      ef
                  }
                  FrameType::P => {
                      let ref_frame = reference.as_ref().unwrap();
                      let ef = self.encode_pframe(&frame, ref_frame)?;
                      reference = Some(self.decode_for_reference(&ef)?);
                      ef
                  }
                  _ => unimplemented!("B-frames"),
              };

              encoded.push(encoded_frame);
          }

          Ok(encoded)
      }
  }
  ```

#### Day 5: P-Frame Tests & Milestone M2

- [ ] **Create P-frame round-trip test**
  - **Files**: `tests/zvc69_pframe_test.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Encode then decode video sequence

- [ ] **Verify Milestone M2: P-frame codec working**
  - **Acceptance**: Video sequences encode/decode with motion compensation

---

## Phase 3: Optimization (Weeks 8-10)

### Week 8: TensorRT Integration

#### Day 1-2: TensorRT Engine Building

- [ ] **Add TensorRT execution provider configuration**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Models run with TensorRT acceleration

  ```rust
  impl ModelLoader {
      pub fn with_tensorrt(&mut self, fp16: bool, int8: bool) -> &mut Self {
          self.execution_providers.push(
              ort::TensorrtExecutionProvider::default()
                  .with_fp16(fp16)
                  .with_int8(int8)
                  .with_engine_cache_enable(true)
                  .with_engine_cache_path("./trt_cache")
                  .build()
          );
          self
      }
  }
  ```

- [ ] **Implement TensorRT engine caching**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 2 hours
  - **Acceptance**: First run builds engine, subsequent runs load from cache

#### Day 3-4: FP16 Quantization

- [ ] **Configure FP16 precision for all models**
  - **Files**: `train/quantize_models.py`
  - **Effort**: 4 hours
  - **Acceptance**: 60-80% throughput improvement with <0.1 dB PSNR loss

- [ ] **Test mixed precision (FP16 compute, FP32 accumulation)**
  - **Files**: `src/codec/zvc69/model.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Numerical stability verified

#### Day 5: INT8 Calibration (Optional)

- [ ] **Create calibration dataset**
  - **Files**: `train/create_calibration.py`
  - **Effort**: 2 hours
  - **Acceptance**: Representative sample of training data

- [ ] **Run INT8 calibration**
  - **Files**: `train/calibrate_int8.py`
  - **Effort**: 3 hours
  - **Acceptance**: INT8 model with <0.5 dB PSNR loss

### Week 9: Performance Profiling & Tuning

#### Day 1-2: Profiling Infrastructure

- [ ] **Add timing instrumentation**
  - **Files**: `src/codec/zvc69/profiler.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Per-stage timing breakdown

  ```rust
  // src/codec/zvc69/profiler.rs
  pub struct Profiler {
      timings: HashMap<String, Vec<Duration>>,
  }

  impl Profiler {
      pub fn time<T, F: FnOnce() -> T>(&mut self, name: &str, f: F) -> T {
          let start = Instant::now();
          let result = f();
          self.timings.entry(name.to_string())
              .or_default()
              .push(start.elapsed());
          result
      }

      pub fn report(&self) {
          for (name, times) in &self.timings {
              let avg = times.iter().sum::<Duration>() / times.len() as u32;
              println!("{}: {:?} avg ({} samples)", name, avg, times.len());
          }
      }
  }
  ```

- [ ] **Profile encoder pipeline**
  - **Files**: `benches/encoder_bench.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Identify top 3 bottlenecks

#### Day 3-4: Memory Optimization

- [ ] **Implement frame buffer pooling**
  - **Files**: `src/codec/zvc69/buffer.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Zero allocation during steady-state encoding

  ```rust
  pub struct FramePool {
      available: Vec<VideoFrame>,
      in_use: HashSet<usize>,
  }

  impl FramePool {
      pub fn acquire(&mut self) -> PooledFrame {
          if let Some(frame) = self.available.pop() {
              PooledFrame::new(frame, self)
          } else {
              PooledFrame::new(VideoFrame::new_uninit(), self)
          }
      }

      fn release(&mut self, frame: VideoFrame) {
          self.available.push(frame);
      }
  }
  ```

- [ ] **Pre-allocate GPU memory**
  - **Files**: `src/codec/zvc69/gpu.rs`
  - **Effort**: 3 hours
  - **Acceptance**: GPU memory stable during encoding

#### Day 5: CUDA Stream Parallelism

- [ ] **Implement multi-stream inference**
  - **Files**: `src/codec/zvc69/gpu.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Overlap compute with memory transfers

  ```rust
  pub struct CudaStreamManager {
      streams: Vec<cudaStream_t>,
      current: usize,
  }

  impl CudaStreamManager {
      pub fn next_stream(&mut self) -> cudaStream_t {
          let stream = self.streams[self.current];
          self.current = (self.current + 1) % self.streams.len();
          stream
      }
  }
  ```

### Week 10: Memory Optimization & Testing

#### Day 1-2: Activation Checkpointing

- [ ] **Implement gradient checkpointing for training**
  - **Files**: `train/checkpoint_training.py`
  - **Effort**: 3 hours
  - **Acceptance**: 30% memory reduction for training

- [ ] **Optimize inference memory layout**
  - **Files**: `src/codec/zvc69/memory.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Minimum memory footprint for inference

#### Day 3-4: Benchmark Suite

- [ ] **Create comprehensive benchmark suite**
  - **Files**: `benches/full_benchmark.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Automated speed/quality benchmarks

  ```rust
  #[bench]
  fn bench_encode_1080p(b: &mut Bencher) {
      let encoder = setup_encoder();
      let frame = load_test_frame_1080p();

      b.iter(|| {
          encoder.encode_iframe(&frame).unwrap()
      });
  }

  #[bench]
  fn bench_decode_1080p(b: &mut Bencher) {
      let decoder = setup_decoder();
      let encoded = load_encoded_iframe_1080p();

      b.iter(|| {
          decoder.decode_iframe(&encoded).unwrap()
      });
  }
  ```

#### Day 5: Milestone M3 Verification

- [ ] **Verify Milestone M3: Real-time 720p**
  - **Target GPU**: RTX 3060
  - **Acceptance**: 30+ fps encode, 60+ fps decode at 720p

- [ ] **Performance regression tests**
  - **Files**: `tests/performance_test.rs`
  - **Effort**: 2 hours
  - **Acceptance**: CI fails if performance regresses > 10%

---

## Phase 4: Production (Weeks 11-13)

### Week 11: Rate Control

#### Day 1-2: CBR Rate Control

- [ ] **Implement constant bitrate controller**
  - **Files**: `src/codec/zvc69/ratecontrol.rs`
  - **Effort**: 6 hours
  - **Acceptance**: Output bitrate within 5% of target

  ```rust
  // src/codec/zvc69/ratecontrol.rs
  pub struct RateController {
      target_bitrate: u64,
      buffer_size: u64,
      buffer_fullness: u64,
      qp_min: i32,
      qp_max: i32,
  }

  impl RateController {
      pub fn calculate_qp(&mut self, frame_type: FrameType, frame_complexity: f32) -> i32 {
          // PID-style rate control
          let target_bits = self.target_bitrate / self.framerate as u64;
          let buffer_deviation = self.buffer_fullness as f32 - (self.buffer_size as f32 / 2.0);

          // Adjust QP based on buffer state
          let qp_adjustment = (buffer_deviation / self.buffer_size as f32 * 10.0) as i32;

          let base_qp = match frame_type {
              FrameType::I => 22,
              FrameType::P => 24,
              FrameType::B => 26,
          };

          (base_qp + qp_adjustment).clamp(self.qp_min, self.qp_max)
      }

      pub fn update_buffer(&mut self, actual_bits: u64) {
          let target = self.target_bitrate / self.framerate as u64;
          self.buffer_fullness = self.buffer_fullness.saturating_add(target).saturating_sub(actual_bits);
          self.buffer_fullness = self.buffer_fullness.min(self.buffer_size);
      }
  }
  ```

#### Day 3-4: VBR and CRF Modes

- [ ] **Implement variable bitrate mode**
  - **Files**: `src/codec/zvc69/ratecontrol.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Quality consistent, bitrate varies

- [ ] **Implement constant rate factor (CRF)**
  - **Files**: `src/codec/zvc69/ratecontrol.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Perceptually consistent quality

#### Day 5: Two-Pass Encoding

- [ ] **Implement first pass (analysis)**
  - **Files**: `src/codec/zvc69/twopass.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Collect frame complexity statistics

- [ ] **Implement second pass (optimal allocation)**
  - **Files**: `src/codec/zvc69/twopass.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Optimal bitrate distribution

### Week 12: Quality Presets & API Polish

#### Day 1-2: Quality Presets

- [ ] **Implement ultrafast preset**
  - **Files**: `src/codec/zvc69/presets.rs`
  - **Effort**: 2 hours
  - **Acceptance**: Maximum speed, reduced quality

  ```rust
  // src/codec/zvc69/presets.rs
  pub fn apply_preset(config: &mut ZVC69Config, preset: Preset) {
      match preset {
          Preset::Ultrafast => {
              config.gop_size = 30;
              config.reference_frames = 1;
              config.entropy_context = false;
              config.lookahead = 0;
          }
          Preset::Fast => {
              config.gop_size = 20;
              config.reference_frames = 2;
              config.entropy_context = true;
              config.lookahead = 5;
          }
          Preset::Medium => {
              config.gop_size = 10;
              config.reference_frames = 4;
              config.entropy_context = true;
              config.lookahead = 20;
          }
          Preset::Slow => {
              config.gop_size = 10;
              config.reference_frames = 8;
              config.entropy_context = true;
              config.lookahead = 40;
          }
          Preset::Veryslow => {
              config.gop_size = 10;
              config.reference_frames = 16;
              config.entropy_context = true;
              config.lookahead = 60;
          }
      }
  }
  ```

- [ ] **Implement all preset levels**
  - **Files**: `src/codec/zvc69/presets.rs`
  - **Effort**: 4 hours
  - **Acceptance**: 5 presets from ultrafast to veryslow

#### Day 3-4: Public API Design

- [ ] **Design ergonomic public API**
  - **Files**: `src/codec/zvc69/mod.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Clean, documented API

  ```rust
  // Public API example usage
  let encoder = ZVC69Encoder::builder()
      .width(1920)
      .height(1080)
      .quality(Quality::High)
      .preset(Preset::Fast)
      .build()?;

  let mut output = Vec::new();
  for frame in video_frames {
      let packet = encoder.encode_frame(&frame)?;
      output.extend(packet.data());
  }
  encoder.flush_into(&mut output)?;
  ```

- [ ] **Add streaming encoder API**
  - **Files**: `src/codec/zvc69/streaming.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Low-latency frame-by-frame encoding

#### Day 5: Error Handling & Recovery

- [ ] **Implement comprehensive error types**
  - **Files**: `src/codec/zvc69/error.rs`
  - **Effort**: 2 hours
  - **Acceptance**: All error cases have specific error types

- [ ] **Add error recovery mechanisms**
  - **Files**: `src/codec/zvc69/recovery.rs`
  - **Effort**: 3 hours
  - **Acceptance**: Graceful handling of corrupted frames

### Week 13: Testing, Benchmarking & Documentation

#### Day 1-2: Comprehensive Testing

- [ ] **Create unit test suite**
  - **Files**: `src/codec/zvc69/tests/*.rs`
  - **Effort**: 4 hours
  - **Acceptance**: >80% code coverage

- [ ] **Create integration test suite**
  - **Files**: `tests/zvc69_integration.rs`
  - **Effort**: 4 hours
  - **Acceptance**: End-to-end encoding/decoding tests

#### Day 3: Benchmarking vs AV1/H.265

- [ ] **Create BD-rate benchmark suite**
  - **Files**: `benches/bdrate_comparison.rs`
  - **Effort**: 4 hours
  - **Acceptance**: Automated comparison vs SVT-AV1 and x265

  ```rust
  // BD-rate calculation
  pub fn calculate_bdrate(
      anchor_points: &[(f64, f64)],  // (bitrate, PSNR)
      test_points: &[(f64, f64)],
  ) -> f64 {
      // Bjontegaard delta rate calculation
      let anchor_curve = fit_polynomial(anchor_points, 3);
      let test_curve = fit_polynomial(test_points, 3);

      // Integrate difference
      let (psnr_min, psnr_max) = common_psnr_range(anchor_points, test_points);
      integrate_bdrate(&anchor_curve, &test_curve, psnr_min, psnr_max)
  }
  ```

- [ ] **Run benchmark on standard test sequences**
  - **Sequences**: UVG, MCL-JCV, HEVC Class B/C/D
  - **Effort**: 3 hours
  - **Acceptance**: Results documented in benchmark report

#### Day 4: Documentation

- [ ] **Write API documentation**
  - **Files**: `src/codec/zvc69/*.rs` (doc comments)
  - **Effort**: 4 hours
  - **Acceptance**: All public items documented

- [ ] **Create user guide**
  - **Files**: `docs/ZVC69_USER_GUIDE.md`
  - **Effort**: 3 hours
  - **Acceptance**: Getting started, API reference, examples

- [ ] **Create architecture document**
  - **Files**: `docs/ZVC69_ARCHITECTURE.md`
  - **Effort**: 2 hours
  - **Acceptance**: Block diagram, data flow, design decisions

#### Day 5: Final Integration & Release

- [ ] **Verify Milestone M4: Real-time 1080p**
  - **Target GPU**: RTX 3080
  - **Acceptance**: 30+ fps encode, 60+ fps decode at 1080p

- [ ] **Verify Milestone M5: Production-ready**
  - **Acceptance**: Full API, documentation, benchmarks complete

- [ ] **Final release checklist**
  - [ ] All tests passing
  - [ ] Benchmarks meet targets
  - [ ] Documentation complete
  - [ ] Example code working
  - [ ] Changelog updated
  - [ ] Version bumped

---

## Appendix A: File Structure Summary

```
src/codec/zvc69/
|-- mod.rs              # Module root, public API exports
|-- encoder.rs          # ZVC69Encoder implementation
|-- decoder.rs          # ZVC69Decoder implementation
|-- model.rs            # ONNX model loading with ort
|-- entropy.rs          # Entropy coding with constriction
|-- bitstream.rs        # Bitstream format read/write
|-- motion.rs           # Motion estimation/compensation
|-- residual.rs         # Residual coding network
|-- quantize.rs         # Quantization utilities
|-- config.rs           # Configuration and presets
|-- presets.rs          # Quality preset definitions
|-- ratecontrol.rs      # Rate control (CBR/VBR/CRF)
|-- gop.rs              # GOP structure management
|-- buffer.rs           # Frame buffer management
|-- gpu.rs              # GPU/CUDA utilities
|-- profiler.rs         # Performance profiling
|-- error.rs            # Error types
|-- recovery.rs         # Error recovery
|-- streaming.rs        # Low-latency streaming API
|-- twopass.rs          # Two-pass encoding
|-- tests/
|   |-- mod.rs
|   |-- encoder_tests.rs
|   |-- decoder_tests.rs
|   |-- bitstream_tests.rs
|   |-- entropy_tests.rs
|   |-- motion_tests.rs

tests/
|-- zvc69_iframe_test.rs
|-- zvc69_pframe_test.rs
|-- zvc69_integration.rs
|-- performance_test.rs

benches/
|-- encoder_bench.rs
|-- decoder_bench.rs
|-- full_benchmark.rs
|-- bdrate_comparison.rs

train/
|-- export_baseline.py
|-- export_motion.py
|-- quantize_models.py
|-- create_calibration.py
|-- calibrate_int8.py
|-- checkpoint_training.py

docs/
|-- NEURAL_CODEC_ZVC69.md   # Research document
|-- NEURAL_TODO.md          # This file
|-- ZVC69_USER_GUIDE.md     # User documentation
|-- ZVC69_ARCHITECTURE.md   # Architecture documentation
```

---

## Appendix B: Key Dependencies

```toml
[dependencies]
# Neural inference (choose one primary)
ort = { version = "2.0.0-rc.10", features = ["cuda", "tensorrt"] }

# Entropy coding
constriction = "0.4"

# Tensor operations
ndarray = { version = "0.16", features = ["rayon"] }

# Serialization
byteorder = "1.5"
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Image handling
image = "0.25"

# Parallelism
rayon = "1.10"

# Logging
tracing = "0.1"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
```

---

## Appendix C: Training Pipeline (Python)

```bash
# Environment setup
python -m venv zvc69_train
source zvc69_train/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install compressai==1.2.8 lpips pytorch-msssim tensorboard onnx

# Download training data
mkdir -p data/vimeo90k && cd data/vimeo90k
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
unzip vimeo_septuplet.zip

# Train image codec (Stage 1)
python train/train_image_codec.py \
    --dataset data/vimeo90k \
    --epochs 100 \
    --batch-size 8 \
    --lambda 0.013 \
    --output models/image_codec.pth

# Train video codec (Stage 2)
python train/train_video_codec.py \
    --dataset data/vimeo90k \
    --pretrained models/image_codec.pth \
    --epochs 50 \
    --sequence-length 7 \
    --output models/video_codec.pth

# Export to ONNX
python train/export_onnx.py \
    --model models/video_codec.pth \
    --output models/zvc69_encoder.onnx models/zvc69_decoder.onnx

# Optimize with TensorRT (optional)
trtexec --onnx=models/zvc69_encoder.onnx \
    --saveEngine=models/zvc69_encoder.trt \
    --fp16
```

---

## Appendix D: Performance Targets Reference

| Resolution | Encode FPS (Target) | Decode FPS (Target) | Target GPU |
|------------|---------------------|---------------------|------------|
| 720p30     | 60+                 | 120+                | GTX 1060   |
| 1080p30    | 30+                 | 60+                 | RTX 3060   |
| 1080p60    | 60+                 | 120+                | RTX 3080   |
| 4K30       | 30+                 | 60+                 | RTX 4090   |

| Quality Target | vs AV1 BD-rate | vs H.265 BD-rate |
|----------------|----------------|------------------|
| Phase 1        | -5%            | -25%             |
| Phase 2        | -15%           | -35%             |
| Phase 3        | -20%           | -40%             |

---

*Document Version: 1.0*
*Last Updated: December 9, 2025*
