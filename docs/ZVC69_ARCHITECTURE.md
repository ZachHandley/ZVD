# ZVC69 Neural Video Codec - Technical Architecture

## Table of Contents

1. [Neural Codec Overview](#neural-codec-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Core Components](#core-components)
4. [Optimization Techniques](#optimization-techniques)
5. [Bitstream Format Specification](#bitstream-format-specification)

---

## Neural Codec Overview

### How Neural Codecs Differ from Traditional Codecs

Traditional video codecs (H.264, H.265, AV1) use hand-crafted algorithms developed over decades:

| Component | Traditional Codec | Neural Codec (ZVC69) |
|-----------|-------------------|----------------------|
| Transform | Fixed DCT/DST | Learned neural network |
| Motion Estimation | Block matching | Neural optical flow |
| Entropy Coding | Fixed probability models | Learned conditional distributions |
| Quality Control | Quantization tables | Learned rate-distortion optimization |
| Optimization Target | Hand-tuned RD metrics | End-to-end learned from data |

### The Learned Compression Paradigm

ZVC69 follows the **transform coding with hyperprior** architecture:

```
                                 ENCODER
    +--------+     +-----------+     +------------+     +---------+
    | Input  | --> | Analysis  | --> | Quantize   | --> | Entropy |
    | Frame  |     | Transform |     | (Round)    |     | Encode  |
    +--------+     +-----------+     +------------+     +---------+
                         |                                   |
                         v                                   |
                   +-----------+                             |
                   | Hyperprior| --> Entropy Parameters ---->+
                   | Encoder   |                             |
                   +-----------+                             v
                                                      [Bitstream]

                                 DECODER
                                                      [Bitstream]
                                                            |
    +--------+     +-----------+     +------------+     +---------+
    | Output | <-- | Synthesis | <-- | Dequantize | <-- | Entropy |
    | Frame  |     | Transform |     |            |     | Decode  |
    +--------+     +-----------+     +------------+     +---------+
                                                            ^
                   +-----------+                            |
                   | Hyperprior| --> Entropy Parameters ----+
                   | Decoder   |
                   +-----------+
```

### Key Advantages

1. **End-to-end Optimization**: The entire pipeline (transform, quantization, entropy) is jointly optimized
2. **Content Adaptation**: Networks learn content-specific compression strategies
3. **Perceptual Quality**: Can directly optimize for perceptual metrics (SSIM, LPIPS)
4. **Flexible Rate-Distortion**: Quality levels map to learned rate-distortion curves

---

## Pipeline Architecture

### High-Level Encode/Decode Flow

```
                           ENCODING PIPELINE

    Input Frame (YUV420P)
           |
           v
    +------------------+
    | Color Conversion |  Convert to RGB, normalize to [-1, 1]
    +------------------+
           |
           v
    +------------------+
    | Analysis Network |  ga: RGB -> Latent [N, 192, H/16, W/16]
    |   (Encoder CNN)  |
    +------------------+
           |
           +------------> Motion Estimation (P/B frames only)
           |                    |
           v                    v
    +------------------+  +------------------+
    | Quantization     |  | Motion Compress  |
    | y = round(y_hat) |  | (Neural MV)      |
    +------------------+  +------------------+
           |                    |
           v                    v
    +------------------+  +------------------+
    | Hyperprior Enc   |  | Residual Coding  |
    | ha: y -> z       |  | (P-frame only)   |
    +------------------+  +------------------+
           |                    |
           v                    |
    +------------------+        |
    | Hyper-Quantize   |        |
    | z = round(z_hat) |        |
    +------------------+        |
           |                    |
           v                    v
    +------------------+  +------------------+
    | Entropy Encode   |  | Motion/Residual  |
    | (ANS Coding)     |  | Entropy Encode   |
    +------------------+  +------------------+
           |                    |
           v                    v
    +------------------+  +------------------+
    | Bitstream        |  | Bitstream        |
    | (Main Latents)   |  | (Motion/Residual)|
    +------------------+  +------------------+
                    \        /
                     \      /
                      v    v
               +----------------+
               | Frame Bitstream|
               +----------------+

                           DECODING PIPELINE

               +----------------+
               | Frame Bitstream|
               +----------------+
                     |
                     v
    +------------------+  +------------------+
    | Entropy Decode   |  | Motion/Residual  |
    | (ANS)            |  | Entropy Decode   |
    +------------------+  +------------------+
           |                    |
           v                    v
    +------------------+  +------------------+
    | Hyperprior Dec   |  | Motion Decomp    |
    | hs: z_hat -> psi |  | (Warp Reference) |
    +------------------+  +------------------+
           |                    |
           v                    |
    +------------------+        |
    | Entropy Params   |        |
    | (mean, scale)    |        |
    +------------------+        |
           |                    |
           v                    |
    +------------------+        |
    | Dequantize       |        |
    | y_hat = float(y) |        |
    +------------------+        |
           |                    |
           v                    v
    +------------------+  +------------------+
    | Synthesis Net    |  | Add Residual     |
    | gs: y_hat -> x   |  | (P-frame only)   |
    +------------------+  +------------------+
           |                    |
           v                    v
    +------------------+
    | Color Conversion |  Convert RGB to YUV420P
    +------------------+
           |
           v
    Output Frame (YUV420P)
```

### Frame Type Processing

#### I-Frames (Keyframes)

```
Input Frame -> Analysis -> Quantize -> Hyperprior -> Entropy Code -> Bitstream
                                              |
                                    Entropy Parameters
```

No temporal dependencies. Full frame compressed independently.

#### P-Frames (Predicted)

```
Current Frame ----+
                  |
                  v
Reference Frame --+--> Motion Estimation --> Motion Vectors
                  |           |
                  |           v
                  |    Motion Compensation --> Predicted Frame
                  |           |
                  +-----------|
                              v
                        Residual = Current - Predicted
                              |
                              v
                        Residual Coding (or Skip)
```

#### B-Frames (Bidirectional)

```
Forward Ref ----+
                +--> Bidirectional Motion --> Two Motion Fields
Backward Ref ---+           |
                            v
                      Blend Predictions --> Combined Prediction
                            |
                            v
                      Residual Coding
```

---

## Core Components

### 1. Analysis/Synthesis Transforms

The analysis transform `ga` and synthesis transform `gs` are convolutional neural networks that replace the DCT/IDCT in traditional codecs.

#### Analysis Transform (Encoder)

```
                    ANALYSIS NETWORK (ga)

    Input: [N, 3, H, W] RGB normalized image
    Output: [N, 192, H/16, W/16] latent representation

    Architecture:
    +------------------+
    | Conv2d 3->192    |  stride=2, kernel=5x5
    | GDN              |  Generalized Divisive Normalization
    +------------------+
           |
           v
    +------------------+
    | Conv2d 192->192  |  stride=2, kernel=5x5
    | GDN              |
    +------------------+
           |
           v
    +------------------+
    | Conv2d 192->192  |  stride=2, kernel=5x5
    | GDN              |
    +------------------+
           |
           v
    +------------------+
    | Conv2d 192->192  |  stride=2, kernel=5x5
    +------------------+
           |
           v
    Latent: [N, 192, H/16, W/16]
```

#### Synthesis Transform (Decoder)

```
                    SYNTHESIS NETWORK (gs)

    Input: [N, 192, H/16, W/16] dequantized latents
    Output: [N, 3, H, W] reconstructed RGB image

    Architecture (mirror of analysis):
    +----------------------+
    | TransConv 192->192   |  stride=2, kernel=5x5
    | IGDN                 |  Inverse GDN
    +----------------------+
           |
           v
    +----------------------+
    | TransConv 192->192   |  stride=2
    | IGDN                 |
    +----------------------+
           |
           v
    +----------------------+
    | TransConv 192->192   |  stride=2
    | IGDN                 |
    +----------------------+
           |
           v
    +----------------------+
    | TransConv 192->3     |  stride=2
    +----------------------+
           |
           v
    Output: [N, 3, H, W]
```

### 2. Hyperprior Entropy Model

The hyperprior provides **side information** that helps predict the entropy parameters for main latents.

```
                    HYPERPRIOR ARCHITECTURE

    Main Latents y
         |
         v
    +------------------+
    | Hyper-Analysis   |  ha: [N, 192, H/16, W/16] -> [N, 128, H/64, W/64]
    | (3 conv layers)  |
    +------------------+
         |
         v
    Quantize z_hat = round(z)
         |
         v
    [Entropy Encode z_hat using Factorized Prior]
         |
         v
    +------------------+
    | Hyper-Synthesis  |  hs: [N, 128, H/64, W/64] -> [N, 384, H/16, W/16]
    | (3 deconv layers)|
    +------------------+
         |
         v
    Split into [mean, scale] each [N, 192, H/16, W/16]
         |
         v
    [Use (mean, scale) to entropy code main latents y]
```

#### Factorized Prior

Used for hyperprior latents `z` (no conditioning):

```rust
pub struct FactorizedPrior {
    num_channels: usize,     // 128 for hyperprior
    cdfs: Vec<Vec<u32>>,     // Per-channel learned CDFs
    min_symbol: i32,         // -128
    max_symbol: i32,         // 127
}
```

#### Gaussian Conditional

Used for main latents `y` (conditioned on hyperprior):

```rust
pub struct GaussianConditional {
    scale_table: Vec<f32>,   // 64 quantized scale levels
    min_symbol: i32,         // -128
    max_symbol: i32,         // 127
}
```

### 3. Motion Estimation (P-frames)

Neural optical flow estimation for motion prediction:

```
                    MOTION ESTIMATION

    Reference Frame [N, 3, H, W]
           +
    Current Frame [N, 3, H, W]
           |
           v
    +------------------+
    | Flow Network     |  Concatenate inputs, predict flow
    | (Encoder-Decoder)|
    +------------------+
           |
           v
    Motion Field [N, 2, H, W]  (dx, dy per pixel)
           |
           v
    +------------------+
    | Motion Compress  |  Downsample + Quantize
    +------------------+
           |
           v
    Compressed Motion [N, 2, H/4, W/4]
```

#### Motion Field Structure

```rust
pub struct MotionField {
    /// Flow vectors [N, 2, H, W] - (dx, dy) per pixel
    pub flow: Array4<f32>,
    /// Confidence/occlusion mask [N, 1, H, W]
    pub confidence: Option<Array4<f32>>,
    /// Motion precision level
    pub precision: MotionPrecision,
}

pub enum MotionPrecision {
    FullPel,    // Integer pixel accuracy
    HalfPel,    // 1/2 pixel accuracy
    QuarterPel, // 1/4 pixel accuracy (default)
}
```

### 4. Residual Coding

For P-frames, the residual (prediction error) is coded:

```
                    RESIDUAL CODING

    Current Frame ----+
                      +---> Residual = Current - Predicted
    Predicted Frame --+          |
                                 v
                          +------------------+
                          | Skip Detection   |
                          | (threshold check)|
                          +------------------+
                                 |
                    +------------+------------+
                    |                         |
                Skip (near-zero)         Code Residual
                    |                         |
                    v                         v
              [0-bit flag]           +------------------+
                                     | Residual Encoder |
                                     | (Neural Network) |
                                     +------------------+
                                           |
                                           v
                                     [Entropy Coded Residual]
```

#### Adaptive Quantization

```rust
// Per-block adaptive quantization based on content
pub fn adaptive_quantize_blockwise(
    residual: &Array4<f32>,
    base_scale: f32,
    motion_magnitude: &Array4<f32>,
    block_size: usize,
) -> Array4<f32>;

// Skip detection for near-zero residuals
pub fn should_skip_residual(
    residual: &Array4<f32>,
    threshold: f32,
) -> bool;
```

---

## Optimization Techniques

### 1. TensorRT Acceleration

TensorRT provides 2-4x speedup through:

- **Layer Fusion**: Combining Conv+BN+ReLU into single kernel
- **Precision Reduction**: FP16/INT8 inference with minimal quality loss
- **Kernel Auto-tuning**: Selecting optimal CUDA kernels for GPU
- **Memory Optimization**: Efficient tensor memory allocation

```
                    TENSORRT OPTIMIZATION

    ONNX Model                      Optimized Engine
    +---------+                     +---------+
    | Conv    |                     | Fused   |
    +---------+                     | Conv+   |
    | BN      |  ---- TensorRT -->  | BN+ReLU |
    +---------+      Builder        +---------+
    | ReLU    |                     | FP16    |
    +---------+                     +---------+
```

#### Precision Modes

| Mode | Speedup | Quality Impact | Memory |
|------|---------|----------------|--------|
| FP32 | 1x (baseline) | None | 100% |
| FP16 | ~2x | <0.1 dB PSNR | 50% |
| INT8 | ~4x | 0.3-0.5 dB PSNR | 25% |

### 2. Parallel Entropy Coding

Entropy coding is parallelized by partitioning latent tensors into horizontal slices:

```
                    PARALLEL ENTROPY CODING

    Latent Tensor [N, C, H, W]
           |
           v
    +------+------+------+------+
    |Slice0|Slice1|Slice2|Slice3|  Partition along H
    +------+------+------+------+
       |      |      |      |
       v      v      v      v
    [ANS]  [ANS]  [ANS]  [ANS]    Parallel encode (Rayon)
       |      |      |      |
       v      v      v      v
    +------+------+------+------+
    | Bits0| Bits1| Bits2| Bits3|
    +------+------+------+------+
           |
           v
    Merge into parallel-format bitstream (ZVCP header)
```

#### Parallel Bitstream Format

```
+----------+---+---+---+------------------+------------------+
|  ZVCP    |Ver|Cnt|Res| Slice Headers    | Slice Data       |
| (4 bytes)| 1 | 1 | 2 | (N x 16 bytes)   | (concatenated)   |
+----------+---+---+---+------------------+------------------+
```

### 3. Memory Pooling

Pre-allocated buffer pools eliminate allocation overhead:

```rust
pub struct FramePool {
    buffers: Mutex<VecDeque<PooledBuffer>>,
    buffer_size: usize,
    max_buffers: usize,
    stats: PoolStats,
}

// Zero-copy buffer reuse with RAII
pub struct PooledBuffer {
    data: Vec<u8>,
    pool: Arc<FramePool>,  // Return on drop
}
```

### 4. Pipelined Processing

Overlapping CPU and GPU operations for sustained throughput:

```
                    PIPELINE ARCHITECTURE

    Time -->

    Frame 0:  [Preproc]----[GPU Infer]----[Entropy]
    Frame 1:          [Preproc]----[GPU Infer]----[Entropy]
    Frame 2:                  [Preproc]----[GPU Infer]----[Entropy]
    Frame 3:                          [Preproc]----[GPU Infer]----[Entropy]

    Throughput = 1 frame / max(Preproc, Infer, Entropy) time
```

```rust
pub struct PipelineConfig {
    pub queue_depth: usize,           // Input buffer count
    pub num_workers: usize,           // CPU worker threads
    pub enable_async_transfers: bool, // Overlap CPU/GPU
    pub target_latency_ms: f64,       // Latency target
}

impl PipelineConfig {
    pub fn realtime_720p() -> Self;
    pub fn realtime_1080p() -> Self;
    pub fn low_latency() -> Self;
    pub fn high_throughput() -> Self;
}
```

### 5. CUDA Stream Management

Multiple CUDA streams enable concurrent GPU operations:

```rust
pub struct CudaStreamManager {
    streams: Vec<CudaStream>,
    roles: HashMap<StreamRole, usize>,
}

pub enum StreamRole {
    Inference,     // Neural network execution
    MemoryH2D,     // Host to Device transfer
    MemoryD2H,     // Device to Host transfer
    PostProcess,   // Output formatting
}
```

---

## Bitstream Format Specification

### File Structure

```
+------------------------+
|  FILE HEADER (64B)     |  Global metadata
+------------------------+
|  INDEX TABLE (opt)     |  Random access seek points
+------------------------+
|  FRAME 0 (I-Frame)     |  First frame (always keyframe)
+------------------------+
|  FRAME 1               |  Subsequent frames
+------------------------+
|  ...                   |
+------------------------+
|  FRAME N               |  Last frame
+------------------------+
|  TRAILER (opt)         |  Checksums, statistics
+------------------------+
```

### File Header (64 bytes)

```
Offset  Size  Field                Description
------  ----  -----                -----------
0x00    6     magic                "ZVC69\0"
0x06    1     version_major        Bitstream major version (1)
0x07    1     version_minor        Bitstream minor version (0)
0x08    4     flags                Feature flags (see below)
0x0C    2     width                Video width in pixels
0x0E    2     height               Video height in pixels
0x10    2     framerate_num        Framerate numerator
0x12    2     framerate_den        Framerate denominator
0x14    4     total_frames         Total frame count (0 = unknown)
0x18    1     gop_size             I-frame interval
0x19    1     quality_level        Quality setting (0-100)
0x1A    1     color_space          0=YUV420, 1=YUV444, 2=RGB
0x1B    1     bit_depth            8, 10, or 12 bits
0x1C    4     model_hash           First 4 bytes of model SHA-256
0x20    8     index_offset         Byte offset to index table
0x28    4     latent_channels      Number of latent channels (192)
0x2C    4     hyperprior_channels  Number of hyperprior channels (128)
0x30    16    reserved             Reserved (must be zero)
```

#### File Header Flags

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | HAS_INDEX | Index table present for random access |
| 1 | HAS_B_FRAMES | B-frames present in stream |
| 2 | HAS_TRAILER | Trailer with checksums present |
| 3 | INTEGERIZED | Model uses integer arithmetic |
| 4 | GMM_ENTROPY | Uses Gaussian Mixture entropy |
| 5 | CHECKERBOARD | Checkerboard autoregressive context |

### Frame Header (32 bytes)

```
Offset  Size  Field                Description
------  ----  -----                -----------
0x00    4     magic                "ZVC1"
0x04    1     frame_type           0=I, 1=P, 2=B, 3=BBi
0x05    1     quality_level        Frame quality (0-100)
0x06    2     reserved1            Reserved
0x08    4     frame_size           Size of frame data in bytes
0x0C    4     latent_height        Latent tensor height
0x10    4     latent_width         Latent tensor width
0x14    8     pts                  Presentation timestamp
0x1C    4     checksum             CRC-16 of frame data
```

### Frame Data Layout

#### I-Frame

```
+----------------------+
| Hyperprior Data      |  Entropy-coded z_hat
+----------------------+
| Main Latent Data     |  Entropy-coded y_hat
+----------------------+
```

#### P-Frame

```
+----------------------+
| Motion Data          |  Compressed motion field
+----------------------+
| Residual Data        |  Entropy-coded residuals (or skip flag)
+----------------------+
```

#### B-Frame

```
+----------------------+
| Forward Motion       |  Motion to forward reference
+----------------------+
| Backward Motion      |  Motion to backward reference
+----------------------+
| Blend Weights        |  Per-pixel blend weights
+----------------------+
| Residual Data        |  Entropy-coded residuals
+----------------------+
```

### Index Table

For random access seeking to keyframes:

```
Index Table Header (8 bytes):
  - entry_count: u32    Number of entries
  - reserved: u32       Reserved

Index Entry (16 bytes each):
  - frame_number: u32   Frame index
  - byte_offset: u64    Offset from file start
  - flags: u32          IS_KEYFRAME, IS_REFERENCE
```

### Entropy Coded Data Format

#### Parallel Format (ZVCP)

```
+----------+---+---+---+---------------------------+----------------+
| Magic    |Ver|Cnt|Res| Slice Headers             | Slice Data     |
| "ZVCP"   | 1 | N | 0 | (N x 16 bytes)            | (concatenated) |
+----------+---+---+---+---------------------------+----------------+

Slice Header (16 bytes):
  - start_row: u32      Starting row in latent tensor
  - num_rows: u32       Number of rows in slice
  - data_offset: u32    Offset in slice data section
  - data_size: u32      Size of slice's compressed data
```

#### Legacy Sequential Format

Raw ANS-coded bitstream without parallel header (for backwards compatibility).

### Motion Data Format

```
+-------------------+
| Motion Header     |
|  - precision: u8  |
|  - width: u16     |
|  - height: u16    |
+-------------------+
| Compressed Flow   |  ANS-coded motion vectors
+-------------------+
```

---

## Constants Reference

### Spatial Factors

| Component | Factor | Description |
|-----------|--------|-------------|
| Main Latent | 16x | H/16, W/16 relative to input |
| Hyperprior | 64x | H/64, W/64 relative to input |
| Motion | 4x | H/4, W/4 relative to input |
| Residual Latent | 8x | H/8, W/8 relative to input |

### Channel Counts

| Latent Type | Channels |
|-------------|----------|
| Main Latent | 192 |
| Hyperprior | 128 |
| Residual Latent | 64 |
| Residual Hyperprior | 32 |

### Symbol Ranges

| Parameter | Value |
|-----------|-------|
| MIN_SYMBOL | -128 |
| MAX_SYMBOL | 127 |
| CDF_PRECISION | 16 bits |
| SCALE_BOUND | 0.11 |
| NUM_SCALES | 64 |

### Dimension Requirements

| Parameter | Value |
|-----------|-------|
| MIN_DIMENSION | 64 |
| MAX_DIMENSION | 8192 |
| ALIGNMENT | 16 (must be divisible) |

---

## References

1. Balle, J., et al. "Variational image compression with a scale hyperprior." ICLR 2018.
2. Minnen, D., et al. "Joint autoregressive and hierarchical priors for learned image compression." NeurIPS 2018.
3. Lu, G., et al. "DVC: An end-to-end deep video compression framework." CVPR 2019.
4. Li, J., et al. "DCVC: Deep contextual video compression." NeurIPS 2021.
5. Sheng, X., et al. "DCVC-RT: Real-time neural video compression at 125fps." CVPR 2025.
6. Duda, J. "Asymmetric numeral systems: entropy coding combining speed of Huffman with compression rate of arithmetic coding." arXiv 2013.
7. NVIDIA TensorRT Developer Guide, 2025.
