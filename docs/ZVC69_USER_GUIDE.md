# ZVC69 Neural Video Codec - User Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Performance Tuning](#performance-tuning)
5. [API Reference](#api-reference)
6. [Examples](#examples)

---

## Overview

### What is ZVC69?

ZVC69 is a next-generation **neural video codec** that uses learned neural network transforms for video compression. Unlike traditional codecs (H.264, H.265, AV1) that rely on hand-crafted algorithms and transforms, ZVC69 uses deep learning to achieve:

- **Superior rate-distortion performance**: 20-40% bitrate savings compared to H.265 at equivalent quality
- **Perceptually optimized compression**: Neural networks learn to preserve details that matter to human perception
- **Flexible quality-bitrate tradeoffs**: 8 quality levels (Q1-Q8) for fine-grained control
- **GPU-accelerated encoding/decoding**: TensorRT optimization enables real-time 1080p processing

### Key Features

| Feature | Description |
|---------|-------------|
| **Neural Compression** | Learned analysis/synthesis transforms replace DCT |
| **Adaptive Quality** | 8 quality levels (Q1-Q8) with continuous quality scaling |
| **Multiple Presets** | From `ultrafast` to `veryslow` for speed/quality tradeoffs |
| **Modern Rate Control** | CRF, VBR, CBR, and CQP modes |
| **GPU Acceleration** | ONNX Runtime with optional TensorRT support |
| **Real-time Performance** | 30+ fps encoding at 1080p with TensorRT FP16 |

### When to Use ZVC69

**Ideal Use Cases:**
- Video streaming where bandwidth is premium
- Archival storage requiring best-in-class compression
- Professional video production workflows
- Research and development in neural compression

**Consider Alternatives When:**
- Maximum compatibility is required (use H.264/H.265)
- CPU-only encoding on low-power devices
- Ultra-low latency requirements (<5ms)

---

## Quick Start

### Feature Flag

ZVC69 requires the `zvc69` feature to be enabled:

```toml
[dependencies]
zvd = { version = "0.1", features = ["zvc69"] }
```

### Basic Encoding Example

```rust
use zvd::codec::zvc69::{ZVC69Encoder, ZVC69Config, Quality, Preset};
use zvd::codec::{Encoder, Frame};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build configuration for 1920x1088 (dimensions must be divisible by 16)
    let config = ZVC69Config::builder()
        .dimensions(1920, 1088)
        .quality(Quality::Q5)
        .preset(Preset::Medium)
        .fps(30)
        .build()?;

    // Create encoder
    let mut encoder = ZVC69Encoder::new(config)?;

    // Encode frames
    for frame in video_frames {
        encoder.send_frame(&Frame::Video(frame))?;

        // Retrieve encoded packets
        while let Ok(packet) = encoder.receive_packet() {
            // Write packet.data to output file/stream
            output.write_all(&packet.data)?;
        }
    }

    // Flush remaining frames
    encoder.flush()?;
    while let Ok(packet) = encoder.receive_packet() {
        output.write_all(&packet.data)?;
    }

    Ok(())
}
```

### Basic Decoding Example

```rust
use zvd::codec::zvc69::ZVC69Decoder;
use zvd::codec::{Decoder, Frame};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create decoder
    let mut decoder = ZVC69Decoder::new()?;

    // Decode packets from bitstream
    for packet in bitstream_packets {
        decoder.send_packet(&packet)?;

        // Retrieve decoded frames
        while let Ok(frame) = decoder.receive_frame() {
            if let Frame::Video(video_frame) = frame {
                // Process decoded video frame
                display_frame(&video_frame);
            }
        }
    }

    // Flush decoder
    decoder.flush()?;
    while let Ok(frame) = decoder.receive_frame() {
        // Handle remaining frames
    }

    Ok(())
}
```

---

## Configuration

### Quality Levels

ZVC69 provides 8 quality levels that control the compression-quality tradeoff:

| Level | Use Case | Bitrate Multiplier | Visual Quality |
|-------|----------|-------------------|----------------|
| Q1 | Aggressive compression, previews | 0.25x | Low |
| Q2 | Low bandwidth streaming | 0.40x | Low-Medium |
| Q3 | Standard streaming | 0.60x | Medium |
| Q4 | **Default**, balanced | 1.00x | Medium-High |
| Q5 | High quality streaming | 1.50x | High |
| Q6 | Professional distribution | 2.20x | Very High |
| Q7 | Archival quality | 3.50x | Near-Lossless |
| Q8 | Maximum quality | 5.50x | Perceptually Lossless |

```rust
// Set quality level
let config = ZVC69Config::builder()
    .quality(Quality::Q6)  // High quality
    // or by numeric level:
    .quality_level(6)
    .build()?;
```

### Encoding Presets

Presets control the speed-compression efficiency tradeoff:

| Preset | Speed | Compression | Use Case |
|--------|-------|-------------|----------|
| `Ultrafast` | Fastest | Lower | Real-time on mid-range GPU |
| `Fast` | Fast | Moderate | Near real-time encoding |
| `Medium` | Balanced | Good | Default, ~2x realtime |
| `Slow` | Slow | Better | Offline encoding |
| `Veryslow` | Slowest | Best | Maximum compression |

```rust
use zvd::codec::zvc69::Preset;

let config = ZVC69Config::builder()
    .preset(Preset::Fast)
    // or by name:
    .preset_name("fast")
    .build()?;
```

### Rate Control Modes

#### CRF (Constant Rate Factor) - Recommended

Targets consistent visual quality. Bitrate varies based on content complexity.

```rust
use zvd::codec::zvc69::RateControlMode;

let config = ZVC69Config::builder()
    .crf(23.0)  // Lower = better quality, higher bitrate
    .build()?;
```

#### VBR (Variable Bitrate)

Targets average bitrate while varying quality.

```rust
let config = ZVC69Config::builder()
    .rate_control(RateControlMode::Vbr {
        target_bitrate: 5_000_000,  // 5 Mbps
        max_bitrate: Some(10_000_000),  // Optional cap
    })
    .build()?;
```

#### CBR (Constant Bitrate)

Maintains fixed bitrate using buffer model.

```rust
let config = ZVC69Config::builder()
    .rate_control(RateControlMode::Cbr {
        bitrate: 5_000_000,
        vbv_buffer_size: 10_000_000,
    })
    .build()?;
```

### GOP Configuration

Control keyframe placement and B-frame usage:

```rust
use zvd::codec::zvc69::GopConfig;

// Standard configuration
let config = ZVC69Config::builder()
    .keyframe_interval(250)  // I-frame every 250 frames
    .bframes(3)              // 3 B-frames between P-frames
    .build()?;

// Low-latency (no B-frames)
let config = ZVC69Config::builder()
    .gop(GopConfig::low_latency())
    .build()?;

// Intra-only (all keyframes)
let config = ZVC69Config::builder()
    .gop(GopConfig::intra_only())
    .build()?;
```

### TensorRT Options

Enable TensorRT for maximum GPU performance:

```rust
use zvd::codec::zvc69::{TensorRTConfig, ModelConfig};

// Enable TensorRT with FP16
let model_config = ModelConfig {
    use_tensorrt: true,
    use_fp16: true,
    use_int8: false,
    gpu_device: 0,
    batch_size: 1,
    ..Default::default()
};

let config = ZVC69Config::builder()
    .model(model_config)
    .build()?;
```

#### TensorRT Presets

```rust
use zvd::codec::zvc69::TensorRTConfig;

// Real-time 1080p (FP16)
let trt_config = TensorRTConfig::realtime_1080p();

// Maximum quality (FP32)
let trt_config = TensorRTConfig::quality();

// Maximum speed (INT8, requires calibration)
let trt_config = TensorRTConfig::int8();

// Fast build, good inference
let trt_config = TensorRTConfig::fast();
```

---

## Performance Tuning

### Achieving Real-time Performance

#### Hardware Requirements

| Resolution | Minimum GPU | Recommended GPU |
|------------|-------------|-----------------|
| 720p @ 30fps | GTX 1080 | RTX 3060 |
| 1080p @ 30fps | RTX 2080 | RTX 3080 |
| 1080p @ 60fps | RTX 3080 | RTX 4080 |
| 4K @ 30fps | RTX 4090 | A100 |

#### TensorRT Optimization

TensorRT provides 2-4x speedup over standard ONNX Runtime:

```rust
use zvd::codec::zvc69::{TensorRTConfig, OptimizedZVC69};
use std::path::Path;

// Load optimized codec
let trt_config = TensorRTConfig::realtime_1080p()
    .with_cache_dir("/path/to/cache")  // Cache compiled engines
    .with_fp16(true);                  // Use FP16 for speed

let codec = OptimizedZVC69::load(
    Path::new("models/"),
    trt_config
)?;

// Warm up for JIT compilation
codec.warmup()?;
```

#### Pipelined Encoding

For sustained throughput, use the pipelined encoder:

```rust
use zvd::codec::zvc69::pipeline::{PipelinedEncoder, PipelineConfig};

// Real-time 1080p configuration
let pipeline_config = PipelineConfig::realtime_1080p();

let mut encoder = PipelinedEncoder::new(zvc69_config, pipeline_config)?;

// Non-blocking frame submission
for frame in video_frames {
    encoder.submit(frame)?;

    // Non-blocking output retrieval
    if let Some(encoded) = encoder.try_recv() {
        output.write_all(&encoded.data)?;
    }
}

// Flush remaining
for encoded in encoder.flush() {
    output.write_all(&encoded.data)?;
}

// Check performance
let stats = encoder.stats();
println!("Throughput: {:.1} fps", stats.throughput_fps);
println!("Avg latency: {:.1} ms", stats.avg_latency_ms);
```

#### Parallel Entropy Coding

Enable parallel entropy coding for faster bitstream generation:

```rust
use zvd::codec::zvc69::entropy::ParallelEntropyConfig;

// Use 4 parallel slices
let entropy_config = ParallelEntropyConfig::new(4)
    .with_min_slice_height(8);
```

### Memory Optimization

Use memory pooling to reduce allocation overhead:

```rust
use zvd::codec::zvc69::memory::{EncoderMemoryContext, PoolConfig};

// Pre-allocate pools for 1080p
let pool_config = PoolConfig::preset_1080p();
let memory_ctx = EncoderMemoryContext::new(pool_config)?;

// Reuse buffers from pool
let frame_buffer = memory_ctx.get_frame()?;
// Buffer automatically returns to pool when dropped
```

### Precision vs Speed Tradeoffs

| Precision | Speed | Quality Loss | Use Case |
|-----------|-------|--------------|----------|
| FP32 | 1x (baseline) | None | Reference quality |
| FP16 | ~2x | <0.1 dB PSNR | **Recommended** |
| INT8 | ~4x | ~0.3-0.5 dB PSNR | Edge deployment |

```rust
// Validate precision impact
use zvd::codec::zvc69::benchmark::{
    compare_fp16_fp32_quality,
    assert_fp16_quality
};

let result = compare_fp16_fp32_quality(&test_frames)?;
assert_fp16_quality(&result)?;  // Verify <0.1 dB loss
```

---

## API Reference

### Core Types

#### `ZVC69Config`

Main configuration struct for encoder/decoder.

```rust
pub struct ZVC69Config {
    pub width: u32,           // Frame width (must be divisible by 16)
    pub height: u32,          // Frame height (must be divisible by 16)
    pub quality: Quality,     // Quality level (Q1-Q8)
    pub preset: Preset,       // Encoding speed preset
    pub rate_control: RateControlMode,
    pub gop: GopConfig,
    pub framerate_num: u32,
    pub framerate_den: u32,
    pub model: ModelConfig,
    pub threads: u32,         // 0 = auto-detect
    // ... color space options
}
```

#### `ZVC69Encoder`

Main encoder interface.

```rust
impl ZVC69Encoder {
    pub fn new(config: ZVC69Config) -> Result<Self, ZVC69Error>;
    pub fn send_frame(&mut self, frame: &Frame) -> Result<(), Error>;
    pub fn receive_packet(&mut self) -> Result<Packet, Error>;
    pub fn flush(&mut self) -> Result<(), Error>;
    pub fn stats(&self) -> EncoderStats;
}
```

#### `ZVC69Decoder`

Main decoder interface.

```rust
impl ZVC69Decoder {
    pub fn new() -> Result<Self, ZVC69Error>;
    pub fn send_packet(&mut self, packet: &Packet) -> Result<(), Error>;
    pub fn receive_frame(&mut self) -> Result<Frame, Error>;
    pub fn flush(&mut self) -> Result<(), Error>;
    pub fn stats(&self) -> DecoderStats;
}
```

#### `Quality`

Quality level enumeration.

```rust
pub enum Quality {
    Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8
}

impl Quality {
    pub fn level(&self) -> u8;
    pub fn quant_scale(&self) -> f32;
    pub fn bitrate_multiplier(&self) -> f32;
    pub fn from_level(level: u8) -> Self;
}
```

#### `Preset`

Encoding speed preset.

```rust
pub enum Preset {
    Ultrafast, Fast, Medium, Slow, Veryslow
}

impl Preset {
    pub fn name(&self) -> &'static str;
    pub fn inference_passes(&self) -> u32;
    pub fn motion_search_range(&self) -> f32;
    pub fn lookahead_frames(&self) -> u32;
}
```

### TensorRT Types

#### `TensorRTConfig`

TensorRT optimization configuration.

```rust
pub struct TensorRTConfig {
    pub fp16_enabled: bool,
    pub int8_enabled: bool,
    pub max_workspace_size: usize,
    pub engine_cache_dir: Option<PathBuf>,
    pub max_batch_size: usize,
    pub enable_cuda_graphs: bool,
    pub num_streams: usize,
    pub device_id: usize,
}

impl TensorRTConfig {
    pub fn new() -> Self;
    pub fn fast() -> Self;
    pub fn quality() -> Self;
    pub fn int8() -> Self;
    pub fn realtime_1080p() -> Self;
    pub fn realtime_1080p_int8() -> Self;
}
```

#### `Precision`

Inference precision mode.

```rust
pub enum Precision {
    FP32,  // Full precision
    FP16,  // Half precision (recommended)
    INT8,  // 8-bit quantized
}
```

### Entropy Coding Types

#### `EntropyCoder`

Main entropy coding interface.

```rust
impl EntropyCoder {
    pub fn new() -> Self;
    pub fn encode_symbols(&mut self, symbols: &[i32], means: &[f32], scales: &[f32])
        -> Result<Vec<u8>, ZVC69Error>;
    pub fn decode_symbols(&mut self, data: &[u8], means: &[f32], scales: &[f32], n: usize)
        -> Result<Vec<i32>, ZVC69Error>;
    pub fn encode_parallel(&self, latent: &Array4<f32>, means: &Array4<f32>,
        scales: &Array4<f32>, config: &ParallelEntropyConfig)
        -> Result<Vec<u8>, ZVC69Error>;
}
```

### Helper Functions

```rust
// Resolution validation
pub fn is_valid_resolution(width: u32, height: u32) -> bool;
pub fn align_dimension(dim: u32) -> u32;
pub fn align_dimensions(width: u32, height: u32) -> (u32, u32);

// Bitrate estimation
pub fn estimate_bitrate(width: u32, height: u32, fps: f32, quality: Quality) -> u64;

// Quantization
pub fn quantize_latents(values: &[f32]) -> Vec<i32>;
pub fn dequantize_latents(symbols: &[i32]) -> Vec<f32>;
```

---

## Examples

### Example 1: Transcode Video File

```rust
use zvd::codec::zvc69::{ZVC69Encoder, ZVC69Decoder, ZVC69Config, Quality, Preset};
use zvd::codec::{Encoder, Decoder, Frame};
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn transcode_to_zvc69(input_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Open input video (using external demuxer)
    let (width, height, fps) = get_video_info(input_path)?;

    // Configure encoder for high quality
    let config = ZVC69Config::builder()
        .dimensions(align_dimension(width), align_dimension(height))
        .quality(Quality::Q6)
        .preset(Preset::Medium)
        .fps(fps as u32)
        .crf(20.0)
        .build()?;

    let mut encoder = ZVC69Encoder::new(config)?;

    // Open output file
    let mut output = BufWriter::new(File::create(output_path)?);

    // Process each frame
    for frame in decode_video_frames(input_path)? {
        encoder.send_frame(&Frame::Video(frame))?;

        while let Ok(packet) = encoder.receive_packet() {
            output.write_all(&packet.data)?;
        }
    }

    // Flush encoder
    encoder.flush()?;
    while let Ok(packet) = encoder.receive_packet() {
        output.write_all(&packet.data)?;
    }

    let stats = encoder.stats();
    println!("Encoded {} frames", stats.frames_encoded);
    println!("Average bitrate: {:.2} Mbps", stats.avg_bitrate / 1_000_000.0);

    Ok(())
}
```

### Example 2: Live Streaming with Low Latency

```rust
use zvd::codec::zvc69::{ZVC69Encoder, ZVC69Config, Quality, Preset, GopConfig};
use zvd::codec::zvc69::pipeline::{PipelinedEncoder, PipelineConfig};

fn setup_live_encoder(width: u32, height: u32) -> Result<PipelinedEncoder, Box<dyn std::error::Error>> {
    // Low-latency configuration
    let config = ZVC69Config::builder()
        .dimensions(width, height)
        .quality(Quality::Q4)
        .preset(Preset::Fast)
        .gop(GopConfig::low_latency())  // No B-frames
        .fps(30)
        .build()?;

    // Pipeline for real-time processing
    let pipeline_config = PipelineConfig::low_latency();

    let encoder = PipelinedEncoder::new(config, pipeline_config)?;

    Ok(encoder)
}

async fn stream_frames(
    mut encoder: PipelinedEncoder,
    mut frame_source: impl Iterator<Item = VideoFrame>,
    mut output: impl AsyncWrite + Unpin,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // Submit frame (non-blocking)
        if let Some(frame) = frame_source.next() {
            encoder.submit(frame)?;
        } else {
            break;
        }

        // Retrieve and send encoded packets
        while let Some(encoded) = encoder.try_recv() {
            output.write_all(&encoded.data).await?;

            // Log latency
            if encoded.latency_ms > 33.0 {
                eprintln!("Warning: frame latency {:.1}ms exceeds target", encoded.latency_ms);
            }
        }
    }

    // Flush remaining
    for encoded in encoder.flush() {
        output.write_all(&encoded.data).await?;
    }

    Ok(())
}
```

### Example 3: Batch Processing with TensorRT

```rust
use zvd::codec::zvc69::{OptimizedZVC69, TensorRTConfig};
use std::path::Path;

fn batch_encode_with_tensorrt(
    video_files: &[&str],
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Configure TensorRT for maximum throughput
    let trt_config = TensorRTConfig::fast()
        .with_cache_dir(format!("{}/engine_cache", output_dir))
        .with_fp16(true);

    // Load optimized codec (compile engines once)
    let codec = OptimizedZVC69::load(
        Path::new("models/zvc69/"),
        trt_config
    )?;

    // Warm up GPU
    println!("Warming up TensorRT engines...");
    codec.warmup()?;

    // Process each video
    for input_path in video_files {
        let output_path = format!("{}/{}.zvc69", output_dir,
            Path::new(input_path).file_stem().unwrap().to_str().unwrap());

        println!("Processing: {}", input_path);

        let start = std::time::Instant::now();
        codec.encode_video(input_path, &output_path)?;

        let elapsed = start.elapsed();
        println!("  Completed in {:.1}s", elapsed.as_secs_f64());
    }

    Ok(())
}
```

### Example 4: Quality Comparison

```rust
use zvd::codec::zvc69::{ZVC69Config, Quality, benchmark::*};

fn compare_quality_levels() -> Result<(), Box<dyn std::error::Error>> {
    let test_frames = generate_test_frames(1920, 1088, 100)?;

    let results = benchmark_quality_levels(&test_frames, &[
        Quality::Q2,
        Quality::Q4,
        Quality::Q6,
        Quality::Q8,
    ])?;

    println!("Quality Level Comparison:");
    println!("{:-<60}", "");
    println!("{:>8} {:>12} {:>12} {:>12}", "Quality", "PSNR (dB)", "Bitrate", "Time (ms)");
    println!("{:-<60}", "");

    for result in results {
        println!("{:>8} {:>12.2} {:>12.2} {:>12.1}",
            result.quality,
            result.psnr_db,
            result.bitrate_kbps,
            result.encode_time_ms
        );
    }

    Ok(())
}
```

---

## Troubleshooting

### Common Issues

**"Dimensions must be divisible by 16"**

ZVC69 requires frame dimensions to be multiples of 16. Use `align_dimensions()`:

```rust
let (width, height) = zvd::codec::zvc69::align_dimensions(1920, 1080);
// Returns (1920, 1088)
```

**"ModelNotFound"**

Ensure ONNX model files are present:
- `encoder.onnx`
- `decoder.onnx`
- `hyperprior_enc.onnx`
- `hyperprior_dec.onnx`

**"TensorRT engine incompatible"**

TensorRT engines are GPU-specific. Clear the engine cache when changing GPUs:

```rust
let trt_config = TensorRTConfig::default()
    .with_cache_dir("/path/to/cache");  // Delete this directory
```

**"Out of GPU memory"**

Reduce batch size or workspace:

```rust
let trt_config = TensorRTConfig::default()
    .with_max_batch_size(1)
    .with_workspace_size(512 << 20);  // 512 MB
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12 | Initial release with I-frame support |

---

## References

- DCVC-RT: Real-time neural video compression (CVPR 2025)
- Learned Image Compression with Hyperprior (Balle et al., ICLR 2018)
- Asymmetric Numeral Systems (ANS) entropy coding
- NVIDIA TensorRT Optimization Guide
