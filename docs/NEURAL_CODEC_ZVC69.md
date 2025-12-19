# ZVC69: Neural Video Codec Research Document

> **Version**: 2.0 | **Status**: Research Complete | **Last Updated**: December 9, 2025

> **Project Goal**: Build a neural video codec that beats AV1/H.265 by 20%+ with real-time 1080p encoding/decoding, implemented in pure Rust, and royalty-free.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [State of the Art (December 2025)](#2-state-of-the-art-december-2025)
3. [Architecture Overview](#3-architecture-overview)
4. [Key Technical Decisions](#4-key-technical-decisions)
5. [Rust ML Framework Options](#5-rust-ml-framework-options)
6. [Bitstream Format](#6-bitstream-format-zvc69)
7. [Training Requirements](#7-training-requirements)
8. [Performance Targets](#8-performance-targets)
9. [Implementation Phases](#9-implementation-phases)
10. [Performance Optimization](#10-performance-optimization)
11. [Open Source Resources](#11-open-source-resources)
12. [Patent Considerations](#12-patent-considerations)
13. [File Structure](#13-file-structure)
14. [Technical Deep Dives](#14-technical-deep-dives)
15. [Risk Assessment](#15-risk-assessment)
16. [Quick Start Implementation Guide](#16-quick-start-implementation-guide)
17. [Development Roadmap](#17-development-roadmap)
18. [Team Requirements](#18-team-requirements)
19. [Conclusion](#19-conclusion)

---

## 1. Executive Summary

### Vision

ZVC69 is a next-generation neural video codec designed to:

- **Outperform traditional codecs**: Target 20%+ bitrate savings over AV1 and H.265/HEVC at equivalent quality
- **Achieve real-time performance**: 1080p30 encoding and 1080p60 decoding on consumer GPUs (RTX 3060+)
- **Pure Rust implementation**: Leverage Rust's safety, performance, and ecosystem for production deployment
- **Royalty-free**: Avoid the patent minefields of traditional video codecs (H.264, H.265, VVC)

### State of the Art Summary (December 2025)

Neural video codecs have reached a critical inflection point. **Microsoft's DCVC-RT achieves 21% better compression than VVC (H.266)** - the latest traditional codec standard - while delivering **125 fps encoding at 1080p** on modern GPUs. This is not theoretical research; it is production-ready technology with Apache 2.0 licensing.

Key benchmarks that validate ZVC69's viability:

| Codec | vs H.264 Bitrate | Real-Time 1080p? | License |
|-------|------------------|------------------|---------|
| H.265/HEVC | -40% | Yes (HW) | Patent minefield |
| AV1 | -50% | Yes (HW decode) | RF with caveats |
| VVC/H.266 | -50% | No | Patent pools forming |
| **DCVC-RT** | **-65%** | **Yes (125fps GPU)** | **Apache 2.0** |
| **ZVC69 Target** | **-60%** | **Yes** | **Apache 2.0** |

### Technical Approach

ZVC69 will be built on three core technology decisions validated through this research:

1. **Inference Runtime**: `ort` (ONNX Runtime 2.0) with TensorRT backend
   - Production-proven, 315k+ downloads/month
   - FP16/INT8 quantization for optimal GPU utilization
   - Cross-platform via CUDA, DirectML, CoreML, ROCm

2. **Entropy Coding**: `constriction` library for rANS + Range coding
   - Purpose-built for neural compression (MIT/Apache/BSL licensed)
   - Near-optimal compression with 6.1 ns/symbol decode speed
   - Direct integration with learned probability models

3. **Training Framework**: PyTorch + CompressAI, then ONNX export
   - CompressAI provides production-quality baseline models (BSD 3-Clause)
   - DCVC-RT architecture as reference (Apache 2.0)
   - Export to ONNX for Rust inference via ort

### Patent Landscape Warning

**Critical Development**: InterDigital acquired Deep Render (AI compression startup) in October 2025. This consolidates significant neural codec IP under a company with aggressive licensing history. ZVC69 must:

- Design architecturally distinct approaches where possible
- Document prior art extensively for all components
- Use Apache 2.0 licensed code for explicit patent grants
- Budget for FTO (Freedom to Operate) analysis before commercial deployment (~$50-150K)

CompressAI's BSD 3-Clause Clear license **explicitly does NOT grant patent rights**. Using their code does not protect against InterDigital patent claims.

### Why ZVC69 is Viable

1. **The technology works**: DCVC-RT proves neural codecs beat traditional codecs at real-time speeds
2. **Open foundations exist**: Apache 2.0 reference implementations from Microsoft
3. **Rust ecosystem is ready**: ort 2.0, burn 0.19, candle 0.9 are production-quality
4. **Market timing**: AV2 won't deploy until 2026+; H.267 not until 2034-2036
5. **GPU ubiquity**: NVIDIA CUDA is everywhere; Tensor Cores make neural inference practical

### Key Differentiators

| Aspect | Traditional Codecs | ZVC69 (Neural) |
|--------|-------------------|----------------|
| Design | Hand-crafted transforms | Learned end-to-end |
| Patents | Massive thicket (MPEG-LA, HEVC Advance) | Novel architecture, cleaner IP |
| Improvement | Diminishing returns | Rapid progress continues |
| GPU Utilization | Minimal | Native, highly parallelizable |
| Adaptation | Fixed algorithms | Learnable, content-aware |

### Expected Outcomes

With 10-13 weeks of focused development:

- **Phase 1 (Weeks 1-3)**: I-frame codec matching VVC quality at 30% lower bitrate
- **Phase 2 (Weeks 4-7)**: P-frame codec achieving 20%+ improvement over AV1
- **Phase 3 (Weeks 8-10)**: TensorRT optimization for real-time 1080p
- **Phase 4 (Weeks 11-13)**: Production hardening, rate control, API polish

### Next Steps

See [Section 16: Quick Start Implementation Guide](#16-quick-start-implementation-guide) for immediate action items, and [Section 17: Development Roadmap](#17-development-roadmap) for the complete timeline.

---

## 2. State of the Art (December 2025)

> **Last Updated**: December 2025. This section reflects the latest developments in neural video compression as of late 2025.

### Neural Codec Landscape

#### DCVC-RT (Microsoft, CVPR 2025)
The **first neural video codec achieving practical real-time performance** at high resolutions.

- **Compression**: 21% bitrate savings vs H.266/VTM at equivalent quality
- **DCVC-RT-Large**: Extended variant achieving **30.8% bitrate savings** vs VTM
- **Speed**: 125.2 fps encoding / 112.8 fps decoding at 1080p on NVIDIA A100
- **4K Support**: First NVC with real-time 4K encoding/decoding capability
- **License**: Apache 2.0 (open source, commercially usable)
- **Key Innovations**:
  - **Implicit temporal modeling**: Eliminates complex explicit motion modules
  - **Single low-resolution latents**: Replaces progressive downsampling
  - **Memory I/O optimization**: Identifies bandwidth (not MACs) as primary bottleneck
  - **Model integerization**: Ensures cross-device consistency with deterministic integer calculations
- **Practical Features**:
  - Continuous bitrate control via quantization parameters
  - Single model supports wide bitrate range
  - YUV 4:2:0 optimized with RGB adaptation support
- **Intra-Frame Performance**: DCVC-RT-Intra achieves 11.1% bitrate reduction vs VTM with 10x faster decoding than previous learned image codecs (40.7/44.2 fps encode/decode at 1080p)
- **Reference**: [github.com/microsoft/DCVC](https://github.com/microsoft/DCVC) | [CVPR 2025 Paper](https://arxiv.org/abs/2502.20762)

#### Newer NVC Frameworks (Late 2025)
A unified intra/inter NVC framework has been proposed that **outperforms DCVC-RT by 12.1% BD-rate** while maintaining real-time performance and providing more stable per-frame bitrate/quality.

#### DCVC-FM (Microsoft, 2024)
- **Compression**: 25.5% better than VTM under intra-period -1 setting
- **vs DCVC-DC**: 29.7% bitrate savings with 16% reduced MACs
- **Key Innovation**: Feature modulation for adaptive quality control

#### DCVC-MIP (Microsoft, 2024)
- **Compression**: 12.9% bitrate savings over DCVC-HEM
- **Key Innovation**: Motion Information Propagation (MIP) enabling bidirectional motion/frame interactions

#### Google DeepMind C3 (CVPR 2024)
Overfits a small model to each image/video for extremely low decoding complexity.

- **Compression**: Matches VTM (H.266 reference) at <3k MACs/pixel for images
- **Video Performance**: Matches Video Compression Transformer at <5k MACs/pixel
- **Key Innovation**: Builds on COOL-CHIC with 3D latents and novel context selection for efficient entropy coding
- **Limitation**: High encoding time due to per-content model fitting
- **Recent Development**: NVRC-Lite (2025) outperforms C3 with 21-23% BD-rate savings and 8.4x faster encoding
- **Reference**: [github.com/google-deepmind/c3_neural_compression](https://github.com/google-deepmind/c3_neural_compression)

### Traditional Codec Comparison (Updated December 2025)

| Codec | vs H.264 | Speed | Status |
|-------|----------|-------|--------|
| H.264/AVC | Baseline | Fast | Ubiquitous, core patents expiring 2023-2027 |
| H.265/HEVC | ~40% better | Medium | Patent fragmentation continues |
| AV1 | ~30% better | Slow encode | **30% of Netflix streaming** (Dec 2025) |
| VVC/H.266 | ~50% better | Very slow | ECM shows 27% improvement over VVC |
| **AV2** | **~40% better than AV1** | TBD | **Spec release end of 2025** |
| **DCVC-RT** | **~65% better** | **Real-time (GPU)** | **Apache 2.0, CVPR 2025** |

### AV2: Next-Generation Open Codec (Coming Late 2025)

AOMedia announced the **year-end 2025 release of AV2** on their 10th anniversary:

- **Compression**: 30-40% lower bitrate than AV1 at same quality
- **Decoder Complexity**: Target <2x AV1 decoder complexity
- **New Features**:
  - Enhanced AR/VR support
  - Split-screen delivery for multiple programs
  - Improved screen content handling
  - Wider visual quality range
- **Status**: Draft spec circulating, all core tools finalized, high-level syntax in progress
- **Not AI-Native**: Uses "data-driven tools" but remains a conventional hybrid codec
- **Adoption Timeline**: 53% of AOMedia members plan adoption within 12 months of finalization; 88% within 2 years
- **Reference**: [AOMedia AV2 Announcement](https://aomedia.org/press%20releases/AOMedia-Announces-Year-End-Launch-of-Next-Generation-Video-Codec-AV2-on-10th-Anniversary/)

### Industry Standardization (JVET/MPEG 2025)

**ITU-T/ISO Workshop on Future Video Coding** (January 2025, Geneva):
- Joint Video Experts Team (JVET) developing Enhanced Compression Model (ECM) and NNVC hybrids
- **ECM Performance**: 27.06% improvement over VVC (July 2025 report)
- **NNVC Progress**: Algorithm description v12 and software v14 (JVET-AM02019, July 2025)
- **H.267 Timeline**: Projected finalization July-October 2028; deployment ~2034-2036

**JVET Neural Network Integration Approaches**:
1. Neural tools within hybrid framework (replacing/augmenting existing tools)
2. Neural in-loop filtering for potential future standards
3. Full neural codec pipelines (research stage)

### Real-World Deployments (December 2025)

#### Netflix AV1 Production Deployment
- **Coverage**: AV1 now powers **30% of all Netflix viewing** (December 2025)
- **HDR**: AV1 HDR streaming launched March 2025 with HDR10+ dynamic metadata
- **Film Grain Synthesis**: Productized July 2025 - strips grain before encoding, reconstructs on playback - **~66% bitrate reduction** with better-looking cinematic grain
- **Device Support**: 88% of certified large-screen devices (2021-2025) support AV1
- **Future Plans**: Evaluating AV1 for live events and cloud gaming; looking at AV2 as next step

#### Meta AV1 Integration
- Deployed across **Messenger, Instagram, and WhatsApp**
- Enhances video quality under low-bandwidth conditions
- Challenges remain with Android device diversity

#### YouTube
- Continues as AOMedia stalwart with extensive AV1 deployment
- Along with Netflix, driving smart TV hardware AV1 support faster than mobile SoCs

### Hardware Support for Neural Codecs (2025)

#### Current State: No Dedicated Neural Codec Hardware
**Important**: As of December 2025, **no consumer GPUs have dedicated neural video codec acceleration**. Neural codecs run on general-purpose GPU compute (CUDA cores, Tensor cores).

#### NVIDIA Video Codec SDK 13.0 (January 2025)
- **Blackwell Architecture Support**: Enhanced NVENC/NVDEC performance
- **AV1 Ultra-High Quality Mode**: Extended from HEVC to AV1
- **MV-HEVC**: Hardware-accelerated stereo encoding for AR/VR headsets
- **New Codecs**: 422 H.264, 422 HEVC, 422i/420i H.264, multi-view HEVC
- **Baseline**: 8 simultaneous encoding streams (since January 2024)
- **LCEVC Enhancement**: 2-4x cheaper than CPU x265 for low-latency encoding

#### Intel Quick Sync Video (QSV)
- **AV1 Support**: Hardware encode/decode on Gen12+ (Tiger Lake and newer)
- **QSV AV1 Encoder**: Merged to mainline FFmpeg (av1_qsv)
- **VP9 12-bit**: Hardware decode on Tiger Lake and newer

#### AMD VCN (Video Core Next)
- Supports H.264, HEVC, AV1 (on RDNA2+)
- Less ecosystem integration than NVENC/QSV

#### GPU Performance for Neural Codecs
| GPU | DCVC-RT 1080p (est.) | Notes |
|-----|---------------------|-------|
| RTX 3090 | 125 fps encode | Original DCVC-RT benchmark |
| A100 | 125/113 fps enc/dec | Production benchmark |
| RTX 4090 | ~150+ fps (est.) | ~60% faster than 3090 in general compute |

### VR/AR and Cloud Gaming (2025)

#### Neural Codecs for Cloud Gaming
- **GameCodec**: First neural codec specifically for cloud gaming
  - Leverages rendering information (camera/object motion decomposition)
  - Addresses extreme motion and visual effects in games
- **Enhanced NVC for Gaming**: End-to-end neural compression designed for gaming video characteristics

#### VR/AR Streaming
- **NVIDIA CloudXR**: Specialized for VR/AR latency requirements
- **MV-HEVC in SDK 13.0**: Hardware stereo encoding for AR/VR headsets
- **2025 Trend**: VR headsets now lighter, wireless, with eye/hand tracking enabling foveated rendering

#### Cloud Gaming Landscape (2025)
- Technology becoming "truly large-scale and feasible"
- Processing power shifting from player devices to cloud
- Modern GPUs deliver real-time ray tracing and neural frame generation

### Neural Codec Startups & Acquisitions

#### WaveOne (Acquired by Apple, March 2023)
- Founded 2016 by Lubomir Bourdev (ex-Meta AI Research)
- **Innovation**: Content-aware compression prioritizing faces/important regions
- **Claims**: Up to 50% file size reduction with better quality in complex scenes
- Raised $9M before acquisition

#### Deep Render (Active, UK)
- London-based Imperial College spin-out (founded 2017)
- **Claims**: Up to 5x video size reduction mimicking human visual perception
- **Funding**: $15M total (including $9M Series A, March 2023)
- Self-identifies as "only AI-based compression startup" (vs traditional techniques)

#### Industry Outlook
- AI and conventional codecs working in tandem for near-term
- Neural codec startups likely acquisition targets
- Full neural codec deployment still years away from mainstream

### Latest Benchmark Data (2025)

#### DCVC Family vs Traditional Codecs (BD-Rate PSNR)

| Codec | vs VVC/VTM | vs HEVC | vs AV1 |
|-------|-----------|---------|--------|
| DCVC-RT | -21.3% | -42.1% | -23.8% |
| DCVC-RT-Large | -30.8% | ~-50% | ~-33% |
| DCVC-FM | -25.5% | N/A | N/A |
| ECM (Hybrid) | -27.06% | N/A | N/A |

#### VVC Performance at High Resolutions

| Resolution | VVC vs H.264 | AV1 vs H.264 | VVC vs AV1 |
|------------|--------------|--------------|------------|
| 4K UHD | ~78% savings | ~63% savings | ~15% better |
| 8K | ~59% savings | ~22% savings | ~46% better |

#### Key Findings
- **ECM** (Enhanced Compression Model) offers best coding performance but extreme complexity
- Higher resolutions show greater efficiency gains for newer codecs
- VVC/AV2 specifically optimized for 4K/8K content
- Neural codecs outperform all traditional codecs but require GPU inference

### The Paradigm Shift

```
Traditional Codec Pipeline:
  Motion Estimation -> Transform (DCT) -> Quantization -> Entropy Coding
  [Each stage hand-optimized over decades]

Neural Codec Pipeline:
  Analysis Network -> Latent Space -> Entropy Model -> Arithmetic Coding
  [End-to-end learned, jointly optimized]

2025 Hybrid Approach (JVET/ECM):
  Traditional Pipeline + Neural In-Loop Filtering + Data-Driven Tools
  [Combines best of both worlds for standardization]
```

Neural codecs don't just replace components - they learn an entirely new representation optimized for compression. However, the industry path forward appears to be **hybrid approaches** that integrate neural tools into conventional codec frameworks for easier standardization and hardware support.

---

## 3. Architecture Overview

> **Last Updated**: December 2025. This section reflects the latest neural video codec architectures including DCVC-RT, transformer-based approaches, diffusion models, and INR-based methods.

### State-of-the-Art Architectures (December 2025)

#### DCVC-RT: Current Production Benchmark

**DCVC-RT** (CVPR 2025) remains the gold standard for practical real-time neural video compression:

| Aspect | DCVC-RT | DCVC-RT-Large |
|--------|---------|---------------|
| BD-Rate vs VVC | -21.3% | -30.8% |
| 1080p Encode | 125.2 fps | ~80 fps |
| 1080p Decode | 112.8 fps | ~70 fps |
| 4K Support | Real-time | Near real-time |

**Key Architectural Innovations**:
- **Implicit temporal modeling**: Eliminates complex explicit motion modules
- **Single low-resolution latents**: Replaces progressive downsampling pipelines
- **Memory I/O optimization**: Identifies bandwidth (not MACs) as primary bottleneck
- **Model integerization**: Ensures cross-device consistency with deterministic integer calculations

#### UI2C: Unified Intra-Inter Coding (November 2025)

A newer framework that **outperforms DCVC-RT by 12.1% BD-rate** while retaining real-time performance:
- Single model handles both intra and inter coding adaptively
- Simultaneous two-frame compression exploits interframe redundancy bidirectionally
- More stable per-frame bitrate/quality than DCVC-RT

#### NVRC-Lite: INR-Based Approach (2025)

For offline/high-quality scenarios, **Implicit Neural Representation (INR)** codecs offer superior RD:
- **21-23% BD-rate savings vs Google C3** (another INR codec)
- **8.4x faster encoding, 2.5x faster decoding** than C3
- Per-video optimization (not suitable for real-time streaming)
- Outperforms VVC VTM in Random Access configuration

#### Transformer-Based Video Compression (2025)

Current transformer approaches in neural video compression:

```
Hybrid CNN-Transformer Architecture:
┌─────────────────────────────────────────────────────────────┐
│  Input Frame                                                 │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ConvNeXt/ViT-Inspired Encoder Blocks                │    │
│  │ (CNN backbone + Self-Attention for spatial context) │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Temporal Context Resampler (Hierarchical Attention) │    │
│  │ - Patchless sliding windows for uniform receptive   │    │
│  │ - Bidirectional context from masked transformers    │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  Latent Representation → Entropy Model → Bitstream          │
└─────────────────────────────────────────────────────────────┘
```

**Key Transformer Innovations**:
- **Masked transformers** (M2T, VCT): Predict PMFs by attending to tokens in all directions
- **Bidirectional transformers**: Extract richer contexts for improved compression
- **Spatiotemporal attention modules**: Replace patch-based representations with patchless sliding windows

#### Diffusion-Based Video Compression (2025)

Emerging paradigm achieving exceptional perceptual quality at ultra-low bitrates:

| Method | Key Innovation | Performance |
|--------|---------------|-------------|
| **GLVC** | Pretrained continuous tokenizer + perceptual latent space | Reduces flickering artifacts |
| **DIRAC** | Smooth R-D-P tradeoff traversal at test time | Competitive with GAN methods |
| **OneDC** | One-step diffusion decoding | 40% bitrate reduction, 20x faster than multi-step |
| **Conditional VG** | Motion/edge/flow conditioning | 15-30% improvement over H.264/H.265/H.266 |

**Diffusion Codec Characteristics**:
- Achieve **ultra-low bitrates (<0.02 bpp)** with comparative visual quality
- Trade bits for powerful generative priors
- Best for perceptual quality metrics (FVD, LPIPS) rather than PSNR
- Higher computational cost than autoencoder-based codecs

#### GIViC: Generative Implicit Video Compression (ICCV 2025)

First INR-based codec to outperform VTM:
- **15.94% BD-rate savings vs VVC VTM**
- **22.46% BD-rate savings vs DCVC-FM**
- Exploits long-term dependencies similar to LLMs and diffusion models

### High-Level Architecture (ZVC69)

```
┌─────────────────────────────────────────────────────────────────┐
│                         ZVC69 CODEC                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐       │
│  │   ENCODER    │    │  BITSTREAM  │    │   DECODER    │       │
│  │              │───▶│             │───▶│              │       │
│  │  Analysis    │    │  Arithmetic │    │  Synthesis   │       │
│  │  Transform   │    │   Coded     │    │  Transform   │       │
│  └──────────────┘    └─────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐       │
│  │   Motion     │    │  Entropy    │    │   Motion     │       │
│  │  Estimation  │    │   Model     │    │Compensation  │       │
│  └──────────────┘    └─────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Neural Network Components

#### 1. Analysis Transform (Encoder CNN)
Converts input frames to latent representation.

```
Architecture: 3 → 64 → 128 → 192 → 192 channels

Layer Stack:
  Conv2d(3, 64, 5x5, stride=2) + GDN
  Conv2d(64, 128, 5x5, stride=2) + GDN
  Conv2d(128, 192, 5x5, stride=2) + GDN
  Conv2d(192, 192, 5x5, stride=2)

Output: H/16 x W/16 x 192 latent tensor
```

**GDN (Generalized Divisive Normalization)**: Critical for image/video compression - normalizes activations based on local statistics, mimicking visual perception.

#### 2. Synthesis Transform (Decoder CNN)
Reconstructs frames from latent representation.

```
Architecture: 192 → 192 → 128 → 64 → 3 channels

Layer Stack:
  ConvTranspose2d(192, 192, 5x5, stride=2) + IGDN
  ConvTranspose2d(192, 128, 5x5, stride=2) + IGDN
  ConvTranspose2d(128, 64, 5x5, stride=2) + IGDN
  ConvTranspose2d(64, 3, 5x5, stride=2)

Output: H x W x 3 reconstructed frame
```

#### 3. Hyperprior Network
Provides side information to improve entropy model accuracy.

```
Hyperprior Encoder:
  Conv2d(192, 128, 3x3) + ReLU
  Conv2d(128, 128, 5x5, stride=2) + ReLU
  Conv2d(128, 128, 5x5, stride=2)

Hyperprior Decoder:
  ConvTranspose2d(128, 128, 5x5, stride=2) + ReLU
  ConvTranspose2d(128, 192, 5x5, stride=2) + ReLU
  Conv2d(192, 384, 3x3)  # Outputs mean and scale

Purpose: Predicts entropy model parameters (μ, σ) for main latents
```

#### 4. Motion Estimation Network
Estimates optical flow between frames.

```
Architecture: 6 → 64 → 128 → 64 → 2 channels

Input: Concatenated [current_frame, previous_frame] (6 channels)
Output: Optical flow (dx, dy) per pixel

Layer Stack:
  Conv2d(6, 64, 7x7, stride=2) + LeakyReLU
  Conv2d(64, 128, 5x5, stride=2) + LeakyReLU
  Conv2d(128, 64, 5x5, stride=2) + LeakyReLU
  Conv2d(64, 2, 5x5, stride=1)

  # Upsampling to full resolution
  Bilinear interpolate 8x
```

#### 5. Residual Coding Network
Encodes prediction residuals (difference between predicted and actual frame).

```
Reduced version of image codec:
  Analysis: 3 → 32 → 64 → 96 channels
  Synthesis: 96 → 64 → 32 → 3 channels

Smaller because residuals have lower entropy than full frames
```

### Model Size

| Component | Parameters | Notes |
|-----------|------------|-------|
| Analysis Transform | ~1.2M | Main encoder |
| Synthesis Transform | ~1.2M | Main decoder |
| Hyperprior Network | ~0.8M | Side information |
| Motion Network | ~0.5M | Optical flow |
| Residual Network | ~0.3M | Reduced codec |
| **Total** | **~4M** | Fits in GPU memory easily |

### Frame Type Pipelines

#### I-Frame Pipeline (Intra-coded)
No temporal dependencies - compressed like an image.

```
Input Frame
    │
    ▼
┌───────────────┐
│ Analysis Net  │
│  (Encoder)    │
└───────┬───────┘
        │ Latent y (H/16 x W/16 x 192)
        ▼
┌───────────────┐
│   Quantize    │ ŷ = round(y)
└───────┬───────┘
        │
        ├────────────────────────┐
        │                        │
        ▼                        ▼
┌───────────────┐        ┌───────────────┐
│  Hyperprior   │        │   Entropy     │
│   Encoder     │        │    Model      │
└───────┬───────┘        │  (uses μ, σ)  │
        │                └───────┬───────┘
        ▼                        │
┌───────────────┐                │
│  Hyperprior   │                │
│   Decoder     │────────────────┘
│  (μ, σ)       │
└───────────────┘
        │
        ▼
┌───────────────┐
│  Arithmetic   │
│   Encoder     │
└───────┬───────┘
        │
        ▼
   BITSTREAM
```

#### P-Frame Pipeline (Predictive)
Uses previous reconstructed frame as reference.

```
Current Frame ───────┬──────────────────────────────────┐
                     │                                   │
Previous Frame ──────┤                                   │
                     │                                   │
                     ▼                                   │
              ┌─────────────┐                           │
              │ Motion Net  │                           │
              └──────┬──────┘                           │
                     │ Optical Flow                      │
                     ▼                                   │
              ┌─────────────┐                           │
              │   Warp      │                           │
              │  Previous   │                           │
              └──────┬──────┘                           │
                     │ Predicted Frame                   │
                     │                                   │
                     ├───────────────────────────────────┤
                     │                                   │
                     ▼                                   ▼
              ┌─────────────┐                    ┌─────────────┐
              │  Subtract   │◀───────────────────│   Current   │
              │  (Residual) │                    │   Frame     │
              └──────┬──────┘                    └─────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Residual   │
              │   Codec     │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Motion +   │
              │  Residual   │
              │  Bitstream  │
              └─────────────┘
```

---

## 4. Key Technical Decisions

### Training Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | **PyTorch** | Best ecosystem for video compression research |
| Library | **CompressAI** | Production-quality learned compression, BSD license |
| Reference | **DCVC-RT** | State-of-the-art architecture, Apache 2.0 |

### Inference Stack (Rust)

| Component | Primary Choice | Alternative | Rationale |
|-----------|---------------|-------------|-----------|
| Neural Inference | **ort** (ONNX Runtime) | candle | TensorRT support, production-tested |
| Entropy Coding | **constriction** | arcode | ANS + Range coding, neural compression focused |
| Tensor Ops | **ndarray** | nalgebra | Good balance of performance and ergonomics |

### Entropy Coding Deep Dive (December 2025)

> **Last Updated**: December 2025. This section covers state-of-the-art entropy coding techniques for neural video codecs.

#### Entropy Coding Algorithms Comparison

Neural codecs require entropy coders that can interface with learned probability models. The two dominant approaches are:

| Algorithm | Type | Compression | Speed | GPU-Friendly | Best For |
|-----------|------|-------------|-------|--------------|----------|
| **Arithmetic Coding** | Range-based | Optimal | Moderate | No | Baseline, research |
| **rANS** | ANS variant | Near-optimal | Fast | Yes | Production neural codecs |
| **Range Coding** | Arithmetic variant | Near-optimal | Fast | Moderate | Autoregressive models |
| **tANS** | Table-based ANS | Near-optimal | Very fast | Limited | Fixed distributions |

**Why rANS Dominates Neural Compression (2025)**:
- **Near-optimal compression**: <0.1% overhead vs theoretical entropy
- **Stack semantics**: LIFO encoding enables bits-back coding
- **Parallelizable**: Can batch-process independent symbols
- **No renormalization overhead**: Unlike arithmetic coding
- **GPU implementations**: Facebook's DietGPU achieves 250-410 GB/s on A100

#### Rust Entropy Coding Libraries

**constriction** (Recommended) - [GitHub](https://github.com/bamler-lab/constriction) | [Docs](https://docs.rs/constriction)
```toml
[dependencies]
constriction = "0.4"
```

| Aspect | Details |
|--------|---------|
| Version | **0.4.1** (October 2024) |
| Algorithms | ANS, Range Coding, Huffman, Chain Coding |
| License | MIT / Apache-2.0 / BSL-1.0 |
| Python Bindings | Yes (pyo3) |
| Neural Compression | Designed for ML research |

**Performance Benchmarks** (from constriction docs):
| Coder | Bit Rate Overhead | Encode Speed | Decode Speed |
|-------|-------------------|--------------|--------------|
| ANS (default) | 0.0015% | 24.2 ns/symbol | 6.1 ns/symbol |
| Range Coding | 0.0237% | 16.6 ns/symbol | 14.3 ns/symbol |
| Arithmetic (baseline) | 0.0004% | 43.2 ns/symbol | 85.6 ns/symbol |

```rust
use constriction::stream::{stack::DefaultAnsCoder, model::DefaultLeakyQuantizer};
use probability::distribution::Gaussian;

// Encoding with learned Gaussian parameters
fn encode_latents(
    latents: &[i32],
    means: &[f64],
    scales: &[f64],
) -> Vec<u32> {
    let mut coder = DefaultAnsCoder::new();

    // Encode in reverse order (ANS is LIFO)
    for ((&y, &mu), &sigma) in latents.iter()
        .zip(means.iter())
        .zip(scales.iter())
        .rev()
    {
        let distribution = Gaussian::new(mu, sigma);
        let quantizer = DefaultLeakyQuantizer::new(-128..=127);
        let model = quantizer.quantize(distribution);

        coder.encode_symbol(y, model).unwrap();
    }

    coder.into_compressed().unwrap()
}

// Decoding with same parameters
fn decode_latents(
    compressed: Vec<u32>,
    means: &[f64],
    scales: &[f64],
    num_symbols: usize,
) -> Vec<i32> {
    let mut coder = DefaultAnsCoder::from_compressed(compressed).unwrap();
    let mut latents = Vec::with_capacity(num_symbols);

    for (&mu, &sigma) in means.iter().zip(scales.iter()) {
        let distribution = Gaussian::new(mu, sigma);
        let quantizer = DefaultLeakyQuantizer::new(-128..=127);
        let model = quantizer.quantize(distribution);

        latents.push(coder.decode_symbol(model).unwrap());
    }

    latents
}
```

**arcode** - [Crates.io](https://crates.io/crates/arcode)
```toml
[dependencies]
arcode = "0.2"
```

| Aspect | Details |
|--------|---------|
| Version | **0.2.4** (June 2022) |
| Algorithm | Arithmetic coding only |
| License | MIT |
| Maturity | Stable, less actively maintained |
| Best For | Simple use cases, educational |

**YAECL** (C++/Python, for reference)
- Unified library for neural compression research
- Supports both AC and rANS
- Batch operations: 1x1, 1xn, nxn symbol/CDF combinations
- Direct hyperprior model integration

**DietGPU** (Facebook Research, GPU)
- GPU-based rANS achieving 250-410 GB/s on A100
- Designed for distributed training compression
- Batch-oriented API for PyTorch integration
- First public GPU generalized ANS implementation

#### Neural Entropy Models

**1. Fully Factorized Prior**
Assumes spatial independence within channels. Simple but limited compression.
```
p(y) = prod_i p_i(y_i)
```

**2. Scale Hyperprior (Balle 2018)**
Conditions entropy model on side information z:
```
p(y|z) = prod_i N(y_i; 0, sigma_i^2)
where sigma = h_s(z)
```

**3. Mean-Scale Hyperprior (Minnen 2018)**
Adds mean prediction for better modeling:
```
p(y|z) = prod_i N(y_i; mu_i, sigma_i^2)
where (mu, sigma) = h_s(z)
```

**4. Autoregressive Context Model (Joint Prior)**
Uses masked convolutions to capture spatial dependencies:
```
p(y|z) = prod_i p(y_i | y_{<i}, z)
```

The context model processes already-decoded neighbors using **masked convolutions**:
- **Raster scan order**: Decode left-to-right, top-to-bottom (H*W steps)
- **Checkerboard pattern**: Two-pass decode (anchor + non-anchor), 2 steps, parallelizable

**5. Gaussian Mixture Models (Cheng 2020)**
Uses K-component GMM for more flexible distributions:
```
p(y_i|z) = sum_{k=1}^K w_k * N(y_i; mu_k, sigma_k^2)
```
- Typically K=3 components
- 9 parameters per latent element (3 weights, 3 means, 3 scales)
- ~10-15% BD-rate improvement over single Gaussian

**FlashGMM (2025)**: Accelerates GMM entropy coding by ~90x using SIMD optimizations.

#### Interfacing Neural Probabilities with Entropy Coders

The key challenge is converting continuous neural network outputs to discrete probability distributions for entropy coding:

```python
# PyTorch/CompressAI approach
class GaussianConditional:
    def forward(self, y, scales, means=None):
        # y: quantized latents
        # scales: predicted sigma from hyperprior
        # means: predicted mu (optional)

        # Compute PMF for each symbol
        half = 0.5
        values = y - means if means else y

        # CDF of Gaussian evaluated at bin edges
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)

        # PMF = CDF(upper) - CDF(lower)
        likelihood = upper - lower

        # Rate = -log2(likelihood)
        bits = -torch.log2(likelihood)
        return bits.sum()
```

For entropy coding, we need to pass CDFs to the coder:
```rust
// Convert Gaussian parameters to discrete CDF
fn gaussian_to_cdf(mu: f64, sigma: f64, min_val: i32, max_val: i32) -> Vec<u32> {
    let scale = (1u32 << 16) as f64;  // 16-bit precision
    let mut cdf = Vec::with_capacity((max_val - min_val + 2) as usize);
    cdf.push(0);

    for i in min_val..=max_val {
        let x = (i as f64 + 0.5 - mu) / sigma;
        let prob = 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
        cdf.push((prob * scale) as u32);
    }

    cdf
}
```

#### Quantization Techniques for Training

Neural codecs face a fundamental challenge: quantization (rounding) is non-differentiable. Three approaches:

**1. Additive Uniform Noise (AUN)**
During training, replace quantization with uniform noise U(-0.5, 0.5):
```python
# Training
y_tilde = y + torch.rand_like(y) - 0.5

# Inference
y_hat = torch.round(y)
```
- **Pros**: Smooth gradients, variational interpretation, expressive latents
- **Cons**: Train-test mismatch

**2. Straight-Through Estimator (STE)**
Use hard rounding in forward pass, identity gradient in backward:
```python
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Identity gradient
```
- **Pros**: No train-test mismatch
- **Cons**: Biased gradients, less expressive latents

**3. Soft-to-Hard Annealing**
Gradually transition from soft to hard quantization:
```python
# Temperature annealing: tau -> 0 during training
y_soft = y + tau * torch.tanh(y / tau)
```
- **Pros**: Smooth transition
- **Cons**: Unstable at low temperatures

**2025 Best Practice**: Start with AUN for initial training, fine-tune with STE for deployment consistency.

**4. Learned Quantization**
Train quantization step sizes or use vector quantization:
```python
# Scalar quantization with learned step size
step = nn.Parameter(torch.ones(channels))
y_hat = torch.round(y / step) * step
```

**5. Rate-Adaptive Quantization**
Single model supporting variable bitrates via quality parameter:
```python
# Scale latents before quantization
quality = 0.5  # 0=low bitrate, 1=high quality
y_scaled = y * quality_to_scale(quality)
y_hat = torch.round(y_scaled)
```
Recent work (2025) enables training-free bitrate adaptation.

### Why ONNX Runtime (ort)?

1. **TensorRT backend**: Automatic optimization, kernel fusion, FP16/INT8 quantization
2. **Battle-tested**: Used in production by Microsoft, Meta, etc.
3. **Multiple backends**: CUDA, DirectML, CoreML, CPU
4. **Model export**: PyTorch → ONNX → Optimized is well-established pipeline

### Why Consider Candle?

1. **Pure Rust**: No external dependencies, easier packaging
2. **Growing fast**: Hugging Face backing, active development
3. **Training support**: Can fine-tune models in Rust
4. **Control**: Full control over memory, no foreign runtime

### Decision Matrix

```
For ZVC69 v1.0: Use ort (ONNX Runtime)
  - Faster time to production
  - TensorRT gives immediate performance wins
  - Can swap to candle later if needed

For ZVC69 v2.0: Consider candle
  - If pure Rust becomes critical
  - If we need Rust-native training
  - If Metal performance on macOS is important
```

---

## 5. Rust ML Framework Options (December 2025)

> **Last Updated**: December 2025. This section reflects the latest versions and capabilities of Rust ML frameworks.

### Framework Landscape Overview

The Rust ML ecosystem has matured significantly in 2025. Enterprise adoption is accelerating as Fortune 500 companies recognize Rust's advantages for AI infrastructure, with Microsoft, Google, Meta, and Amazon all announcing major Rust initiatives for core infrastructure components.

### Detailed Comparison

#### ort (ONNX Runtime bindings) - [GitHub](https://github.com/pykeio/ort) | [Docs](https://ort.pyke.io/)
```toml
[dependencies]
ort = { version = "2.0.0-rc.10", features = ["cuda", "tensorrt"] }
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Version | **2.0.0-rc.10** | Production-ready RC, wraps ONNX Runtime 1.22 |
| GPU Support | Excellent | CUDA 11.6+, TensorRT 8.4+, DirectML (DX12), CoreML, ROCm, QNN |
| ONNX Support | Native | First-class ONNX support |
| Training | Yes | Training support added in 2.x series |
| Maturity | High | 315k+ downloads/month, used by SurrealDB, Wasmtime, Google Magika |
| Pure Rust | No | Bindings to C++ runtime (dynamic loading available) |
| **Best For** | | Production inference with maximum performance |

**2025 Updates**:
- **v2.0 Release Candidate**: Major API redesign, production-ready but API stabilizing
- **Expanded Execution Providers**: DirectML, QNN (Qualcomm), NNAPI (Android), MIGraphX (AMD), ROCm, ArmNN, XNNPACK (WASM)
- **Dynamic Loading**: `load-dynamic` feature avoids shared library issues via dlopen()
- **tract/candle backends**: Can use pure-Rust backends via `ort-tract` for WASM deployment
- **Notable Users**: Bloop (semantic code search), SurrealDB, Google Magika, Wasmtime WASI-NN

#### candle (Hugging Face) - [GitHub](https://github.com/huggingface/candle) | [Docs](https://huggingface.github.io/candle/)
```toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
candle-onnx = "0.9"        # ONNX model support
candle-transformers = "0.9"
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Version | **0.9.1** (candle-onnx) | Rapid development, ~0.12.1 for core (May 2025) |
| GPU Support | Excellent | CUDA, cuDNN, Metal (Apple Silicon), NCCL multi-GPU |
| ONNX Support | Good | Via candle-onnx crate (requires protoc) |
| Training | Yes | Full autodiff, distributed training on 2025 roadmap |
| Maturity | High | Hugging Face backed, production deployments growing |
| Pure Rust | Yes | No C++ dependencies (kernels in CUDA/Metal) |
| **Best For** | | Pure Rust inference, LLM deployment, serverless |

**2025 Updates**:
- **Model Zoo**: LLaMA/2/3, Mistral-7B, Mixtral-8x7B, Phi-1.5/2/3, BERT, T5, Yi, StarCoder
- **Vision Models**: Stable Diffusion 1.5/2.1/XL/Turbo, DINOv2, SAM, YOLO-v3/v8, ViT, EfficientNet
- **Quantization**: Out-of-box q4k, q8k, GGML-style schemes (4-8x model compression)
- **Metal Performance**: ~120ms first token latency on M2 MacBook Air, 3x real-time Whisper
- **WASM/WebGPU**: Full browser support, runs LLaMA2/Whisper in browser tabs
- **Ecosystem**: candle-vllm (OpenAI-compatible server), candle-lora, kalosm meta-framework
- **Performance**: 35-47% faster inference than PyTorch integration on benchmarks

#### burn - [Website](https://burn.dev/) | [GitHub](https://github.com/tracel-ai/burn)
```toml
[dependencies]
burn = { version = "0.19", features = ["wgpu"] }
# Or with CUDA:
burn = { version = "0.19", features = ["cuda-jit"] }
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Version | **0.19.1** (stable), 0.20.0-pre.4 (prerelease) | Active development |
| GPU Support | Excellent | CUDA, Metal, ROCm, Vulkan, DirectX 11/12, WebGPU via CubeCL |
| ONNX Support | Excellent | Imports ONNX → native Rust code, full optimization |
| Training | Yes | Distributed multi-GPU, INT4/INT8 quantization |
| Maturity | High | Premier Rust DL framework, strong community |
| Pure Rust | Yes | CubeCL-powered cross-platform GPU kernels |
| **Best For** | | Cross-platform training/inference, WebGPU, production deployment |

**2025 Updates (v0.19.0 - October 2025)**:
- **Distributed Training**: True multi-GPU with concurrent stream execution and gradient sync
- **Quantization**: Comprehensive INT4/INT8 support for memory-efficient models
- **LLVM Backend**: New CPU backend with SIMD optimization
- **CubeCL Integration**: Cross-platform GPU via CUDA, Metal, ROCm, Vulkan, WebGPU
- **Matrix Multiplication Engine**: Rivals cuBLAS/CUTLASS performance with tensor cores
- **PyTorch Interop**: Load weights from PyTorch/Safetensors formats
- **ONNX Import**: Converts to native Rust code with automatic kernel fusion
- **WebAssembly**: Multiple WASM-compatible backends (Candle, NdArray, WGPU)

#### tch-rs (PyTorch bindings) - [GitHub](https://github.com/LaurentMazare/tch-rs)
```toml
[dependencies]
tch = "0.17"  # Requires LibTorch 2.6.0+
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Version | **0.17.x** | Requires LibTorch v2.6.0 - v2.9.0 |
| GPU Support | Excellent | Full PyTorch GPU (CUDA, MPS on macOS) |
| ONNX Support | Via TorchScript | Export then load |
| Training | Yes | Full PyTorch capabilities |
| Maturity | High | Mature bindings, stable API |
| Pure Rust | No | Requires libtorch (static linking available) |
| **Best For** | | PyTorch parity, research, complex training |

**2025 Status**:
- **LibTorch 2.9.0**: Latest PyTorch version supported
- **burn-tch**: Backend for Burn using tch-rs (CPU, CUDA, MPS)
- **Static Linking**: `LIBTORCH_STATIC=1` for self-contained builds
- **Windows**: MSVC toolchain recommended (MinGW compatibility issues)

#### tract (Sonos) - [GitHub](https://github.com/sonos/tract)
```toml
[dependencies]
tract-onnx = "0.22"
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Version | **0.22.0** (August 2025) | Active maintenance |
| GPU Support | CPU only | No GPU acceleration (by design) |
| ONNX Support | Excellent | 85% ONNX backend tests pass, all "real life" models work |
| Training | No | Inference only |
| Maturity | Very High | Production use at Sonos, proven at scale |
| Pure Rust | Yes | Zero C/C++ dependencies |
| **Best For** | | Edge/embedded, CPU inference, WASM deployment |

**2025 Updates**:
- **v0.22.0**: Released August 2025 with continued ONNX compatibility improvements
- **ort Integration**: Can be used as ort backend via `ort_tract::api()`
- **Model Compatibility**: Forward/backward compatible model serialization
- **NNEF Support**: Alternative to ONNX for model exchange
- **Limitations**: No Tensor Sequences/Optional Tensors (structural limitation)

### New/Notable Frameworks (2025)

#### WONNX - [GitHub](https://github.com/webonnx/wonnx)
```toml
[dependencies]
wonnx = "0.5"
```
WebGPU-accelerated ONNX inference, 100% Rust. Good for non-NVIDIA GPUs where CUDA isn't available.
- **Version**: 0.5.1 (stable but less actively maintained)
- **Backends**: Vulkan, Metal, DirectX 12 via wgpu

#### dfdx - [GitHub](https://github.com/coreylowman/dfdx)
```toml
[dependencies]
dfdx = "0.5"
```
Shape-checked tensors with compile-time verification. Functional/declarative style.
- **Status**: Pre-alpha, experimental
- **Features**: Autodiff, CUDA backend, compile-time shape checking

#### cudarc - [GitHub](https://github.com/coreylowman/cudarc)
```toml
[dependencies]
cudarc = "0.12"
```
Safe Rust wrappers around CUDA toolkit (driver, nvrtc, cublas, curand, nccl).
- **2025**: Rust CUDA project rebooted after 3+ year dormancy
- **Interop**: Being explored for integration with Rust-CUDA kernel compilation

#### Rust-GPU - [GitHub](https://github.com/Rust-GPU/rust-gpu) | [Blog](https://rust-gpu.github.io/blog/)
Write GPU shaders/compute kernels in Rust, compiled to SPIR-V.
- **2025 Milestone**: Cross-platform demo running same Rust code on CUDA, SPIR-V, Metal, DX12, WebGPU, CPU
- **AI Porting**: Used AI to port Vulkan samples 30x faster than manual work
- **Status**: Transitioned from Embark Studios to community ownership

### Summary Table (December 2025)

| Framework | Version | GPU Backends | ONNX | Training | Pure Rust | Production Ready |
|-----------|---------|--------------|------|----------|-----------|------------------|
| **ort** | 2.0.0-rc.10 | CUDA, TensorRT, DirectML, ROCm, CoreML | Native | Yes | No | **Yes** |
| **candle** | 0.9.x | CUDA, Metal, WebGPU | Good | Yes | Yes | **Yes** |
| **burn** | 0.19.1 | CUDA, Metal, ROCm, Vulkan, DX, WebGPU | Excellent | Yes | Yes | **Yes** |
| **tch-rs** | 0.17.x | CUDA, MPS | Via TS | Yes | No | Yes |
| **tract** | 0.22.0 | CPU only | Excellent | No | Yes | **Yes** |
| **wonnx** | 0.5.1 | WebGPU (Vulkan/Metal/DX12) | Good | No | Yes | Limited |
| **dfdx** | 0.5.x | CUDA | No | Yes | Yes | Experimental |

### Recommendation for ZVC69 (Updated December 2025)

**Phase 1 (MVP)**: Use `ort` 2.0 with TensorRT
- Fastest path to real-time performance with ONNX Runtime 1.22
- TensorRT EP gives immediate NVIDIA optimization (FP16/INT8)
- Well-documented PyTorch → ONNX export pipeline
- Can achieve 1080p60 decoding on RTX 3060+

**Phase 2 (Cross-Platform)**: Integrate `burn` 0.19+
- CubeCL provides unified GPU backend across CUDA/Metal/ROCm/Vulkan
- Native ONNX import with automatic kernel fusion
- Distributed training for model fine-tuning
- WebGPU support for browser deployment

**Phase 3 (Pure Rust / Edge)**: Evaluate `candle`
- Smallest binary size for serverless deployment
- Best Metal performance for macOS (M1/M2/M3)
- Quantization support for edge devices
- Browser deployment via WASM + WebGPU

**Alternative Considerations**:
- **tract**: If CPU-only embedded deployment is needed
- **tch-rs**: If tight PyTorch integration for research is required
- **Rust-GPU**: For custom GPU kernels without CUDA dependency

---

## 6. Bitstream Format (ZVC69)

> **Last Updated**: December 2025. This section details the ZVC69 bitstream format with emphasis on entropy coding integration, random access, and seeking mechanisms.

### Design Principles

Neural codec bitstreams differ significantly from traditional codecs:

| Aspect | Traditional (H.264/AV1) | Neural (ZVC69) |
|--------|------------------------|----------------|
| Syntax | Complex, many flags | Simple, latent-focused |
| Entropy Data | CABAC/ANS symbols | rANS-coded latents |
| Metadata | Extensive headers | Minimal, model-defined |
| Random Access | NAL units, IDR frames | I-frame index table |
| Parallelism | Slice/tile based | Latent tensor parallelism |

### File Structure Overview

```
┌─────────────────────────────────────────────────────┐
│                  FILE HEADER (64B)                   │
├─────────────────────────────────────────────────────┤
│              INDEX TABLE (Optional)                  │
│          (For random access / seeking)               │
├─────────────────────────────────────────────────────┤
│                                                      │
│              FRAME 0 (I-Frame)                       │
│                                                      │
├─────────────────────────────────────────────────────┤
│              FRAME 1 (P-Frame)                       │
├─────────────────────────────────────────────────────┤
│              FRAME 2 (P-Frame)                       │
├─────────────────────────────────────────────────────┤
│                    ...                               │
├─────────────────────────────────────────────────────┤
│              FRAME N-1                               │
├─────────────────────────────────────────────────────┤
│              TRAILER (Optional)                      │
│          (Checksum, metadata)                        │
└─────────────────────────────────────────────────────┘
```

### File Header (Extended)

```
Offset  Size  Field              Description
────────────────────────────────────────────────────────────────
0x00    6     Magic              "ZVC69\0" (null-terminated)
0x06    1     Version_Major      Bitstream major version
0x07    1     Version_Minor      Bitstream minor version
0x08    4     Flags              Feature flags (see below)
0x0C    2     Width              Video width in pixels (big-endian)
0x0E    2     Height             Video height in pixels (big-endian)
0x10    2     Framerate_Num      Framerate numerator
0x12    2     Framerate_Den      Framerate denominator
0x14    4     Total_Frames       Total frame count (0 = unknown/streaming)
0x18    1     GOP_Size           I-frame interval (e.g., 10 = every 10th)
0x19    1     Quality_Level      0-100 quality setting
0x1A    1     Color_Space        0=YUV420, 1=YUV444, 2=RGB
0x1B    1     Bit_Depth          8, 10, or 12 bits per component
0x1C    4     Model_Hash         First 4 bytes of model SHA-256 (for verification)
0x20    8     Index_Offset       Byte offset to index table (0 = no index)
0x28    4     Latent_Channels    Number of latent channels (default: 192)
0x2C    4     Hyperprior_Channels Number of hyperprior channels (default: 128)
0x30    16    Reserved           Future use (must be zero)
────────────────────────────────────────────────────────────────
Total: 64 bytes
```

**Flag Bits**:
```
Bit 0: HAS_INDEX       - Index table present for random access
Bit 1: HAS_B_FRAMES    - B-frames present in stream
Bit 2: HAS_TRAILER     - Trailer with checksums present
Bit 3: INTEGERIZED     - Model uses integer arithmetic (deterministic)
Bit 4: GMM_ENTROPY     - Uses Gaussian Mixture Model entropy
Bit 5: CHECKERBOARD    - Checkerboard autoregressive context
Bit 6-31: Reserved
```

### Index Table (For Random Access / Seeking)

Essential for seeking in long videos. Located at `Index_Offset` from file start.

```
┌─────────────────────────────────────────────────────┐
│              INDEX TABLE HEADER                      │
├─────────────────────────────────────────────────────┤
│   Entry_Count (4B)    Number of index entries        │
│   Entry_Size (2B)     Size per entry (default: 16)   │
│   Reserved (2B)                                      │
├─────────────────────────────────────────────────────┤
│              INDEX ENTRIES                           │
│   [Frame_Number (4B), Byte_Offset (8B), Flags (4B)] │
│   [Frame_Number (4B), Byte_Offset (8B), Flags (4B)] │
│   ...                                                │
└─────────────────────────────────────────────────────┘
```

**Index Entry Flags**:
```
Bit 0: IS_KEYFRAME     - This is an I-frame (random access point)
Bit 1: IS_REFERENCE    - Frame is used as reference
Bit 2-31: Reserved
```

**Seeking Algorithm**:
```rust
fn seek_to_frame(index: &IndexTable, target_frame: u32) -> SeekResult {
    // Find nearest keyframe at or before target
    let keyframe = index.entries
        .iter()
        .filter(|e| e.frame_number <= target_frame && e.is_keyframe())
        .max_by_key(|e| e.frame_number)
        .ok_or(SeekError::NoKeyframe)?;

    SeekResult {
        seek_offset: keyframe.byte_offset,
        frames_to_decode: target_frame - keyframe.frame_number,
    }
}
```

### Frame Header (Extended)

```
Offset  Size  Field              Description
────────────────────────────────────────────────────────────────
0x00    1     Frame_Type         0=I, 1=P, 2=B (forward), 3=B (bi)
0x01    1     Temporal_Layer     Hierarchical B-frame layer (0-7)
0x02    2     Reference_Flags    Which reference frames are used
0x04    4     Frame_Size         Total compressed frame size in bytes
0x08    8     PTS                Presentation timestamp (90kHz, 64-bit)
0x10    8     DTS                Decode timestamp (90kHz, 64-bit)
0x18    2     QP_Offset          Quality parameter offset from base
0x1A    2     Checksum           CRC-16 of frame data (optional)
0x1C    4     Reserved           Future use
────────────────────────────────────────────────────────────────
Total: 32 bytes
```

### I-Frame Structure (Detailed)

```
┌─────────────────────────────────────────────────────┐
│              FRAME HEADER (32B)                      │
├─────────────────────────────────────────────────────┤
│              HYPERPRIOR SECTION                      │
│   ├─ Hyperprior_Size (4B)                           │
│   ├─ Entropy_State (4B) - Initial rANS state        │
│   └─ rANS_Coded_Hyperprior (variable)               │
├─────────────────────────────────────────────────────┤
│              MAIN LATENT SECTION                     │
│   ├─ Latent_Size (4B)                               │
│   ├─ Context_Mode (1B) - 0=factorized, 1=checker    │
│   ├─ Entropy_State (4B) - Initial rANS state        │
│   └─ rANS_Coded_Latent (variable)                   │
│       ├─ [Anchor symbols if checkerboard]           │
│       └─ [Non-anchor symbols if checkerboard]       │
└─────────────────────────────────────────────────────┘
```

**Hyperprior Latent (z)**:
- Shape: ceil(H/64) x ceil(W/64) x 128
- Quantized to integers in range [-128, 127]
- Encoded with **factorized prior** (learned non-parametric CDFs)
- rANS coded in raster scan order

**Main Latent (y)**:
- Shape: ceil(H/16) x ceil(W/16) x 192
- Quantized to integers in range [-128, 127]
- Encoded with **Gaussian conditional** (mu, sigma from hyperprior)
- Support for autoregressive context (checkerboard or serial)

### P-Frame Structure (Detailed)

```
┌─────────────────────────────────────────────────────┐
│              FRAME HEADER (32B)                      │
├─────────────────────────────────────────────────────┤
│              TEMPORAL CONTEXT                        │
│   ├─ Context_Size (4B)                              │
│   └─ rANS_Coded_Context (variable)                  │
│       (Implicit motion/temporal features)            │
├─────────────────────────────────────────────────────┤
│              RESIDUAL HYPERPRIOR                     │
│   ├─ Hyperprior_Size (4B)                           │
│   └─ rANS_Coded_Hyperprior (variable)               │
├─────────────────────────────────────────────────────┤
│              RESIDUAL LATENT                         │
│   ├─ Latent_Size (4B)                               │
│   └─ rANS_Coded_Residual (variable)                 │
└─────────────────────────────────────────────────────┘
```

**DCVC-RT Style (Implicit Motion)**:
- No explicit motion vectors transmitted
- Temporal context learned implicitly
- Single low-resolution latent representation
- More efficient than explicit motion + residual

### B-Frame Structure (Hierarchical)

```
┌─────────────────────────────────────────────────────┐
│              FRAME HEADER (32B)                      │
│   Reference_Flags indicates forward/backward refs   │
├─────────────────────────────────────────────────────┤
│              BIDIRECTIONAL CONTEXT                   │
│   ├─ Forward_Weight (2B)  - Interpolation weight    │
│   ├─ Backward_Weight (2B)                           │
│   └─ rANS_Coded_Context (variable)                  │
├─────────────────────────────────────────────────────┤
│              RESIDUAL (same as P-frame)              │
└─────────────────────────────────────────────────────┘
```

**Hierarchical B-Frame Coding**:
```
GOP Structure (GOP_Size=8):
Frame:    0   1   2   3   4   5   6   7   8
Type:     I   B   B   B   P   B   B   B   I
Layer:    0   3   2   3   1   3   2   3   0

Decode Order: 0, 4, 2, 1, 3, 8, 6, 5, 7
Display:      0, 1, 2, 3, 4, 5, 6, 7, 8
```

### Entropy Coding Integration (rANS)

```rust
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::model::DefaultLeakyQuantizer;

/// Encode I-frame latents using hyperprior-derived parameters
pub fn encode_iframe_latents(
    y: &Tensor,           // Main latent [B, C, H, W]
    z: &Tensor,           // Hyperprior latent
    hyperprior_cdf: &[u32], // Factorized prior CDF for z
    means: &Tensor,       // Predicted means from h_s(z)
    scales: &Tensor,      // Predicted scales from h_s(z)
) -> BitstreamSegment {
    let mut segment = BitstreamSegment::new();

    // 1. Encode hyperprior with factorized prior
    let mut z_coder = DefaultAnsCoder::new();
    let z_flat = z.flatten();

    for &symbol in z_flat.iter().rev() {  // Reverse for ANS
        let model = FactorizedPriorModel::from_cdf(hyperprior_cdf);
        z_coder.encode_symbol(symbol, model).unwrap();
    }

    segment.hyperprior = z_coder.into_compressed().unwrap();
    segment.hyperprior_state = z_coder.state();

    // 2. Encode main latent with Gaussian conditional
    let mut y_coder = DefaultAnsCoder::new();
    let y_flat = y.flatten();
    let mu_flat = means.flatten();
    let sigma_flat = scales.flatten();

    for ((&symbol, &mu), &sigma) in y_flat.iter()
        .zip(mu_flat.iter())
        .zip(sigma_flat.iter())
        .rev()
    {
        let quantizer = DefaultLeakyQuantizer::new(-128..=127);
        let gaussian = probability::distribution::Gaussian::new(mu, sigma);
        let model = quantizer.quantize(gaussian);

        y_coder.encode_symbol(symbol, model).unwrap();
    }

    segment.main_latent = y_coder.into_compressed().unwrap();
    segment.main_latent_state = y_coder.state();

    segment
}

/// Decode I-frame latents
pub fn decode_iframe_latents(
    segment: &BitstreamSegment,
    hyperprior_cdf: &[u32],
    shape_z: [usize; 4],
    shape_y: [usize; 4],
    hyperprior_decoder: &HyperpriorDecoder,
) -> (Tensor, Tensor) {
    // 1. Decode hyperprior
    let mut z_coder = DefaultAnsCoder::from_compressed(
        segment.hyperprior.clone()
    ).unwrap();

    let z_count = shape_z.iter().product();
    let mut z_symbols = Vec::with_capacity(z_count);

    for _ in 0..z_count {
        let model = FactorizedPriorModel::from_cdf(hyperprior_cdf);
        z_symbols.push(z_coder.decode_symbol(model).unwrap());
    }

    let z = Tensor::from_slice(&z_symbols).reshape(shape_z);

    // 2. Run hyperprior decoder to get mu, sigma
    let (means, scales) = hyperprior_decoder.forward(&z);

    // 3. Decode main latent with predicted parameters
    let mut y_coder = DefaultAnsCoder::from_compressed(
        segment.main_latent.clone()
    ).unwrap();

    let y_count = shape_y.iter().product();
    let mut y_symbols = Vec::with_capacity(y_count);

    let mu_flat = means.flatten();
    let sigma_flat = scales.flatten();

    for (&mu, &sigma) in mu_flat.iter().zip(sigma_flat.iter()) {
        let quantizer = DefaultLeakyQuantizer::new(-128..=127);
        let gaussian = probability::distribution::Gaussian::new(mu, sigma);
        let model = quantizer.quantize(gaussian);

        y_symbols.push(y_coder.decode_symbol(model).unwrap());
    }

    let y = Tensor::from_slice(&y_symbols).reshape(shape_y);

    (y, z)
}
```

### Autoregressive Context Encoding (Checkerboard)

For improved compression with manageable decode speed:

```rust
/// Checkerboard autoregressive encoding
pub fn encode_with_checkerboard(
    y: &Tensor,
    base_means: &Tensor,
    base_scales: &Tensor,
    context_model: &ContextModel,
) -> (Vec<u32>, Vec<u32>) {
    let [_, c, h, w] = y.shape();

    // Pass 1: Encode anchor positions (checkerboard pattern)
    let anchors = extract_checkerboard(y, /*anchor=*/true);
    let anchor_stream = encode_factorized(&anchors, base_means, base_scales);

    // Pass 2: Decode anchors, predict non-anchors
    let decoded_anchors = decode_factorized(&anchor_stream, base_means, base_scales);
    let (refined_means, refined_scales) = context_model.predict(
        &decoded_anchors,
        base_means,
        base_scales,
    );

    // Pass 3: Encode non-anchors with refined parameters
    let non_anchors = extract_checkerboard(y, /*anchor=*/false);
    let non_anchor_stream = encode_conditional(
        &non_anchors,
        &refined_means,
        &refined_scales,
    );

    (anchor_stream, non_anchor_stream)
}
```

### Bitstream Writer/Reader Implementation

```rust
use std::io::{Write, Read, Seek, SeekFrom};

pub struct ZVC69Writer<W: Write> {
    writer: W,
    header: FileHeader,
    index: Vec<IndexEntry>,
    frame_count: u32,
}

impl<W: Write + Seek> ZVC69Writer<W> {
    pub fn new(writer: W, header: FileHeader) -> Self {
        Self {
            writer,
            header,
            index: Vec::new(),
            frame_count: 0,
        }
    }

    pub fn write_header(&mut self) -> io::Result<()> {
        // Write magic
        self.writer.write_all(b"ZVC69\0")?;

        // Write version
        self.writer.write_all(&[self.header.version_major])?;
        self.writer.write_all(&[self.header.version_minor])?;

        // Write flags
        self.writer.write_all(&self.header.flags.to_be_bytes())?;

        // Write dimensions
        self.writer.write_all(&self.header.width.to_be_bytes())?;
        self.writer.write_all(&self.header.height.to_be_bytes())?;

        // ... continue for all header fields

        Ok(())
    }

    pub fn write_frame(&mut self, frame: &EncodedFrame) -> io::Result<()> {
        let offset = self.writer.stream_position()?;

        // Add to index if keyframe
        if frame.is_keyframe() {
            self.index.push(IndexEntry {
                frame_number: self.frame_count,
                byte_offset: offset,
                flags: INDEX_FLAG_KEYFRAME,
            });
        }

        // Write frame header
        self.writer.write_all(&[frame.frame_type as u8])?;
        self.writer.write_all(&[frame.temporal_layer])?;
        self.writer.write_all(&frame.reference_flags.to_be_bytes())?;
        self.writer.write_all(&(frame.data.len() as u32).to_be_bytes())?;
        self.writer.write_all(&frame.pts.to_be_bytes())?;
        self.writer.write_all(&frame.dts.to_be_bytes())?;
        // ... remaining header fields

        // Write frame data
        self.writer.write_all(&frame.data)?;

        self.frame_count += 1;
        Ok(())
    }

    pub fn finalize(mut self) -> io::Result<()> {
        if self.header.flags & FLAG_HAS_INDEX != 0 {
            let index_offset = self.writer.stream_position()?;

            // Write index table
            self.writer.write_all(&(self.index.len() as u32).to_be_bytes())?;
            self.writer.write_all(&16u16.to_be_bytes())?;  // Entry size
            self.writer.write_all(&[0u8; 2])?;  // Reserved

            for entry in &self.index {
                self.writer.write_all(&entry.frame_number.to_be_bytes())?;
                self.writer.write_all(&entry.byte_offset.to_be_bytes())?;
                self.writer.write_all(&entry.flags.to_be_bytes())?;
            }

            // Update header with index offset
            self.writer.seek(SeekFrom::Start(0x20))?;
            self.writer.write_all(&index_offset.to_be_bytes())?;
        }

        Ok(())
    }
}
```

### Comparison with Other Neural Codec Bitstreams

| Feature | ZVC69 | DCVC-RT | VVC (Traditional) |
|---------|-------|---------|-------------------|
| Header Size | 64B | ~40B | ~100-500B |
| Frame Header | 32B | ~20B | Variable (NALU) |
| Random Access | Index table | GOP-based | IDR + SEI |
| Entropy Coder | rANS | Arithmetic | CABAC |
| Seeking | Frame-accurate | GOP-accurate | NALU-based |
| Parallelism | Tensor-level | Tile-level | Slice/Tile |

### Bitstream Example (Updated)

```
Hex dump of ZVC69 v2 file (1080p30, Quality 80):

# File Header (64 bytes)
5A 56 43 36 39 00        # "ZVC69\0"
02 00                    # Version 2.0
00 00 00 03              # Flags: HAS_INDEX | HAS_TRAILER
07 80 04 38              # 1920x1080
00 1E 00 01              # 30/1 fps
00 00 03 84              # 900 frames (30 seconds)
0A                       # GOP size: 10
50                       # Quality: 80
00 08 00                 # YUV420, 8-bit
A3 F2 1B 9C              # Model hash
00 00 00 00 00 01 2C 40  # Index at offset 0x12C40
00 C0 00 80              # 192 latent, 128 hyperprior channels
00 00 00 00 00 00 00 00  # Reserved
00 00 00 00 00 00 00 00  # Reserved

# Frame 0 (I-frame, 32B header + data)
00 00 00 00              # Type: I, Layer: 0, Refs: none
00 00 F4 24              # Size: 62,500 bytes
00 00 00 00 00 01 51 80  # PTS: 86400 (1 sec @ 90kHz)
00 00 00 00 00 01 51 80  # DTS: same
00 00 00 00              # QP offset: 0, Checksum: 0
00 00 00 00              # Reserved

# Hyperprior section
00 00 04 00              # Hyperprior size: 1024 bytes
12 34 56 78              # Initial rANS state
[1024 bytes rANS data]   # Coded hyperprior z

# Main latent section
00 00 F0 00              # Latent size: 61,440 bytes
01                       # Context mode: checkerboard
AB CD EF 01              # Initial rANS state
[61440 bytes rANS data]  # Coded latent y
  [30720 bytes anchors]
  [30720 bytes non-anchors]

# Frame 1 (P-frame)
01 03 00 01              # Type: P, Layer: 3, Refs: frame 0
00 00 20 00              # Size: 8,192 bytes
...
```

---

## 7. Training Requirements

> **Last Updated**: December 2025. This section reflects the latest training techniques, datasets, loss functions, and hardware requirements for neural video codecs.

### Training Datasets (2025)

#### Primary: Vimeo-90K (Still Relevant)
- **Size**: 89,800 video clips (septuplet format - 7 frames each)
- **Content**: Diverse real-world scenes with dynamic and realistic motion
- **Resolution**: Various, typically 448x256
- **Use**: Primary training dataset - remains the standard baseline for fair comparison
- **Download**: [toflow.csail.mit.edu](http://toflow.csail.mit.edu/)
- **License**: MIT

**Note**: Most researchers "adopt the same training dataset Vimeo-90K Septuplet as the previous method for a fair comparison."

#### BVI-DVC (Recommended for Long Sequences)
- **Size**: High-quality video sequences designed for deep video compression
- **Content**: Diverse scenes optimized for codec training
- **Resolution**: Multiple resolutions including HD
- **Use**: Fine-tuning with longer sequences (10+ frames)
- **Key Insight**: "All three approaches benefit from long-sequence training" - fine-tuning models originally trained on Vimeo-90K with 5-frame sequences using BVI-DVC with 10-frame sequences shows significant improvement

#### Kinetics-600 (Large-Scale Alternative)
- **Size**: ~600K video clips from YouTube
- **Content**: Human actions, diverse motion patterns
- **Duration**: ~10 seconds per video, various resolutions and frame rates
- **Use**: Training motion-heavy codecs (e.g., MOVI-Codec)
- **Advantage**: Much larger scale than Vimeo-90K with more diverse motion

#### Additional Datasets

| Dataset | Size | Resolution | Use Case |
|---------|------|------------|----------|
| **REDS** | 30K sequences | 720p | Dynamic/realistic scenes |
| **BVI-AOM** | Multiple sequences | Various | AOMedia codec evaluation |
| **BVI-HOMTex** | Texture-focused | Various | Texture compression |
| **YouTube-NT** | Large-scale | Various | Alternative to closed datasets |

#### Evaluation/Test Datasets

| Dataset | Size | Resolution | Notes |
|---------|------|------------|-------|
| **UVG** | 16 sequences | 1080p/4K | Standard benchmark (Bosphorus, HoneyBee, etc.) |
| **MCL-JCV** | 30 sequences | 1080p | Joint quality evaluation |
| **HEVC Class B/C/D** | Multiple | Various | Traditional codec test sequences |
| **Kodak** | 24 images | 768x512 | Image codec evaluation |

### Loss Functions (2025 State-of-the-Art)

#### Rate-Distortion-Perception Tradeoff

Modern neural codecs optimize a three-way tradeoff:

```
L = λ_rate * R + λ_dist * D + λ_perc * P

Where:
  R = Rate (estimated bits from entropy model)
  D = Distortion (pixel-level fidelity)
  P = Perceptual quality (learned metrics)
  λ = Lagrange multipliers for each component
```

#### Distortion Loss Options

```python
import torch
import torch.nn.functional as F
from torchvision.models import vgg19
from pytorch_msssim import ms_ssim

# 1. MSE (Mean Squared Error) - PSNR-optimized
D_mse = F.mse_loss(reconstructed, original)

# 2. MS-SSIM (Multi-Scale SSIM) - Structural similarity
D_msssim = 1 - ms_ssim(reconstructed, original, data_range=1.0)

# 3. Charbonnier Loss - Smooth L1 variant, less sensitive to outliers
D_charbonnier = torch.sqrt((reconstructed - original)**2 + 1e-6).mean()

# 4. Combined Fidelity Loss (typical 2025 approach)
D_fidelity = 0.7 * D_mse + 0.3 * D_msssim
```

#### Perceptual Loss Options (2025)

```python
import lpips

# 1. LPIPS (Learned Perceptual Image Patch Similarity)
# - 'alex' is fastest and best as forward metric (default)
# - 'vgg' is closer to traditional perceptual loss for backprop
lpips_model = lpips.LPIPS(net='alex')  # or 'vgg' for training
D_lpips = lpips_model(reconstructed, original)

# 2. VGG Perceptual Loss (feature matching)
vgg = vgg19(pretrained=True).features[:36].eval()
D_vgg = F.mse_loss(vgg(reconstructed), vgg(original))

# 3. FVD (Frechet Video Distance) - for video temporal consistency
# Evaluated at test time, not typically used in training loss
```

#### GAN-Based Training for Perceptual Quality

```python
# HiFiC-style GAN training for neural codecs
# Produces realistic textures at low bitrates

class Discriminator(nn.Module):
    """PatchGAN discriminator for image/video compression"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)

# GAN Loss
def gan_loss(discriminator, reconstructed, original):
    # Generator wants discriminator to output 1 for reconstructed
    fake_pred = discriminator(reconstructed)
    real_pred = discriminator(original)

    # LSGAN loss (more stable than vanilla GAN)
    g_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    d_loss = 0.5 * (F.mse_loss(real_pred, torch.ones_like(real_pred)) +
                    F.mse_loss(fake_pred, torch.zeros_like(fake_pred)))

    return g_loss, d_loss
```

#### Multi-Scale Loss (2025 Best Practice)

```python
def multi_scale_loss(reconstructed, original, scales=[1.0, 0.5, 0.25]):
    """Compute loss at multiple resolutions for better gradient flow"""
    total_loss = 0
    for scale in scales:
        if scale < 1.0:
            size = (int(original.shape[2] * scale), int(original.shape[3] * scale))
            rec_scaled = F.interpolate(reconstructed, size, mode='bilinear')
            orig_scaled = F.interpolate(original, size, mode='bilinear')
        else:
            rec_scaled, orig_scaled = reconstructed, original

        total_loss += F.mse_loss(rec_scaled, orig_scaled) * scale

    return total_loss / len(scales)
```

#### Complete Training Loss (2025)

```python
def compute_training_loss(model, discriminator, original, lambda_rate=0.013,
                          lambda_lpips=0.1, lambda_gan=0.01):
    """
    Complete loss function combining R-D-P optimization.
    Based on HiFiC and GLVC approaches.
    """
    # Forward pass
    out = model(original)
    reconstructed = out['x_hat']

    # Rate loss (bits per pixel)
    bpp_loss = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out['likelihoods'].values()
    )

    # Distortion loss (MSE + MS-SSIM)
    mse_loss = F.mse_loss(reconstructed, original)
    msssim_loss = 1 - ms_ssim(reconstructed, original, data_range=1.0)
    distortion = 0.7 * mse_loss + 0.3 * msssim_loss

    # Perceptual loss (LPIPS)
    lpips_loss = lpips_model(reconstructed, original).mean()

    # GAN loss
    g_loss, d_loss = gan_loss(discriminator, reconstructed, original)

    # Total generator loss
    total_loss = (lambda_rate * bpp_loss +
                  distortion +
                  lambda_lpips * lpips_loss +
                  lambda_gan * g_loss)

    return {
        'total': total_loss,
        'bpp': bpp_loss,
        'mse': mse_loss,
        'msssim': msssim_loss,
        'lpips': lpips_loss,
        'gan_g': g_loss,
        'gan_d': d_loss,
    }
```

#### Rate Estimation

```python
# Bits from entropy model
def estimate_bits(latent, mean, scale):
    # Gaussian entropy model
    gaussian = torch.distributions.Normal(mean, scale)
    # Quantization via uniform noise (training)
    noisy_latent = latent + torch.rand_like(latent) - 0.5
    # Negative log-likelihood = bits
    bits = -gaussian.log_prob(noisy_latent)
    return bits.sum()
```

### Training Configuration (2025)

#### DCVC-Style Lambda Values

```python
# DCVC uses 4 models trained with different lambdas
lambda_values = {
    'mse': [256, 512, 1024, 2048],      # For PSNR-optimized models
    'ms_ssim': [8, 16, 32, 64],          # For MS-SSIM-optimized models
}

# DCVC-FM uses uniform quantization parameter sampling
# to improve harmonization of encoding and quantization
```

#### Modern Training Configuration

```python
# Hyperparameters (2025 best practices)
config = {
    'batch_size': 8,              # Per GPU
    'patch_size': 256,            # Random crop size
    'learning_rate': 1e-4,        # Initial LR
    'lambda_values': [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932],
    'total_iterations': 1_000_000,
    'optimizer': 'Adam',
    'betas': (0.9, 0.999),
    'scheduler': 'CosineAnnealingLR',
    'warmup_iterations': 10_000,
    'gradient_clip': 1.0,

    # Video-specific
    'num_frames': 7,              # Frames per training sample (Vimeo-90K)
    'gop_size': 10,               # For longer sequence training

    # Perceptual training (optional)
    'use_gan': True,
    'lambda_lpips': 0.1,
    'lambda_gan': 0.01,
    'discriminator_lr': 4e-4,
}
```

### Hardware Requirements (December 2025)

#### GPU Options and Pricing

| GPU | VRAM | Memory BW | TensorCores | Power | Cloud Cost/hr |
|-----|------|-----------|-------------|-------|---------------|
| RTX 3090 | 24GB | 936 GB/s | 3rd Gen | 350W | ~$0.50-1.00 |
| RTX 4090 | 24GB | 1008 GB/s | 4th Gen | 450W | ~$0.80-1.50 |
| A100 40GB | 40GB | 1555 GB/s | 3rd Gen | 400W | ~$1.50-3.00 |
| A100 80GB | 80GB | 2039 GB/s | 3rd Gen | 400W | ~$2.50-4.00 |
| H100 80GB | 80GB | 3350 GB/s | 4th Gen | 700W | ~$3.00-5.00 |
| H200 | 141GB | 4800 GB/s | 4th Gen | 700W | ~$5.00+ |

**Note**: A100 is EOL but remains cost-effective with mature software stack. H100 offers 2-3x faster training for most workloads. H200/B200 (Blackwell) are the current cutting edge.

#### Training Time Estimates

| Configuration | GPUs | Training Time | Best For |
|--------------|------|---------------|----------|
| **Minimum** | 1x RTX 3090 | 3-4 weeks | Prototyping, small models |
| **Budget** | 4x RTX 4090 | 1-2 weeks | Independent researchers |
| **Recommended** | 4x A100 40GB | 5-7 days | Production training |
| **Fast** | 8x A100 80GB | 2-3 days | Research labs |
| **Optimal** | 8x H100 | 1-2 days | Enterprise/SOTA |

#### Memory Requirements by Model Type

| Model Type | Min VRAM | Recommended VRAM | Notes |
|------------|----------|------------------|-------|
| Image codec (baseline) | 12GB | 24GB | CompressAI models |
| DCVC-style video codec | 24GB | 40GB | With motion networks |
| Transformer-hybrid | 40GB | 80GB | Attention layers need memory |
| Diffusion-based | 40GB+ | 80GB+ | Multi-step generation |
| Per-video INR (NVRC) | 8GB | 16GB | Single GPU sufficient |

### Training Stages (Multi-Stage Approach)

Modern neural video codecs use a **multi-stage training approach** (IP, PP, PP, ... then cascade):

#### Stage 1: Image Codec Pre-training (I-frames)
```python
# Train analysis/synthesis transforms + hyperprior
# Use pretrained CompressAI models (cheng2020-anchor for MSE, hyperprior for MS-SSIM)
# ~500K iterations on single images
```

#### Stage 2: Short Sequence Training
```python
# Train motion estimation with frozen image codec
# 5-7 frame sequences (Vimeo-90K septuplet)
# ~200K iterations
```

#### Stage 3: Long Sequence Training (Critical for Quality)
```python
# Unfreeze all components
# Train on 10+ frame sequences (BVI-DVC)
# Key insight: "The primary factor influencing error accumulation is
# the train-test mismatch in sequence lengths"
# ~300K-500K iterations
```

#### Stage 4: Cascade Training (DCVC-DC style)
```python
# Train on full GOP sequences: IPPPPP, PPPPPP, etc.
# Addresses error propagation over long prediction chains
# ~200K iterations
```

### Transfer Learning & Fine-Tuning (2025)

#### Starting from Pretrained Models

```python
from compressai.zoo import cheng2020_anchor

# Use pretrained image codec as starting point
base_model = cheng2020_anchor(quality=3, pretrained=True)

# Freeze early layers, fine-tune later layers
for name, param in base_model.named_parameters():
    if 'g_a.0' in name or 'g_a.1' in name:  # Early analysis layers
        param.requires_grad = False
```

#### Domain Adaptation Techniques

1. **Full Fine-Tuning**: Tune all parameters for new domain
2. **Lightweight Fine-Tuning**: Freeze early layers, tune deeper layers
3. **LoRA/Adapters**: Add small trainable modules (PEFT approach)
4. **Predictive Coding**: Hybrid method for on-device adaptation

```python
# Domain-specific adapters (AIM-style for video)
class SpatiotemporalAdapter(nn.Module):
    """Adapter for video domain adaptation"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, 1),
        )
        self.temporal = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, temporal_context=None):
        x = x + self.spatial(x)
        if temporal_context is not None:
            x = x + self.temporal(temporal_context)
        return x
```

### CompressAI (Version 1.2.8, June 2025)

#### Latest Features

- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **PyTorch Support**: 1.7+
- **License**: BSD 3-Clause Clear
- **Video Support**: SSF2020 (Scale-Space Flow) with 9 quality levels

#### Video Codec Support

```python
# CompressAI video codec (SSF2020)
from compressai.zoo import ssf2020

# Load pretrained video codec
model = ssf2020(quality=5, metric='mse')  # quality 1-9

# Video evaluation
# python3 -m compressai.utils.video.eval_model
```

#### Training Example (Updated 2025)

```python
from compressai.zoo import cheng2020_anchor
from compressai.optimizers import net_aux_optimizer
from compressai.losses import RateDistortionLoss
import lpips

# Load base model
model = cheng2020_anchor(quality=3, pretrained=True)
lpips_model = lpips.LPIPS(net='vgg').cuda()

# Configure optimizer
parameters = {
    "net": model.parameters(),
    "aux": model.aux_parameters(),
}
optimizer = net_aux_optimizer(parameters, lr=1e-4)

# Loss function with perceptual component
rd_criterion = RateDistortionLoss(lmbda=0.013)

# Training loop with 2025 best practices
for iteration, batch in enumerate(dataloader):
    optimizer.zero_grad()

    out = model(batch)

    # R-D loss
    rd_loss = rd_criterion(out, batch)

    # Perceptual loss (optional)
    lpips_loss = lpips_model(out['x_hat'], batch).mean()

    # Combined loss
    loss = rd_loss['loss'] + 0.1 * lpips_loss
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    model.update()  # Update quantization parameters

    # Learning rate scheduling
    if iteration % 100000 == 0 and iteration > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
```

### Distributed Training Setup

```python
# Multi-GPU training with PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

# Wrap model
local_rank = setup_distributed()
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Distributed sampler
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
```

### Training Resources Summary

| Resource | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| **GPU VRAM** | 24GB | 40GB | 80GB |
| **GPUs** | 1 | 4-8 | 8+ |
| **Dataset Size** | Vimeo-90K | + BVI-DVC | + Kinetics-600 |
| **Training Iterations** | 500K | 1M | 2M+ |
| **Training Time** | 2-3 weeks | 5-7 days | 1-2 days |
| **Sequence Length** | 5-7 frames | 10+ frames | 20+ frames |

---

## 8. Performance Targets

### Speed Targets

| Resolution | Encode Target | Decode Target | Target GPU |
|------------|--------------|---------------|------------|
| 720p30 | 60+ fps | 120+ fps | GTX 1060 |
| 1080p30 | 30+ fps | 60+ fps | RTX 3060 |
| 1080p60 | 60+ fps | 120+ fps | RTX 3080 |
| 4K30 | 30+ fps | 60+ fps | RTX 4090 |

### Quality Targets (vs bitrate)

| Bitrate | Resolution | PSNR Target | VMAF Target |
|---------|------------|-------------|-------------|
| 1 Mbps | 720p | 36+ dB | 85+ |
| 2 Mbps | 1080p | 35+ dB | 80+ |
| 4 Mbps | 1080p | 38+ dB | 90+ |
| 8 Mbps | 4K | 36+ dB | 85+ |

### Comparison Benchmarks

Target: Beat these by 20%+ at same quality:

| Codec | 1080p @ 35dB PSNR | Notes |
|-------|-------------------|-------|
| H.264 (x264) | ~4 Mbps | Baseline |
| H.265 (x265) | ~2.5 Mbps | -40% vs H.264 |
| AV1 (SVT-AV1) | ~2.2 Mbps | -45% vs H.264 |
| VVC (VVenC) | ~2 Mbps | -50% vs H.264 |
| **ZVC69 Target** | **<1.6 Mbps** | **-60% vs H.264** |

### Latency Targets

| Scenario | Encode Latency | Decode Latency |
|----------|----------------|----------------|
| Live streaming | <50ms | <20ms |
| Video conferencing | <100ms | <30ms |
| VoD encoding | N/A (offline) | <20ms |

### Memory Targets

| Component | Memory Budget |
|-----------|---------------|
| Encoder model | <500 MB GPU |
| Decoder model | <300 MB GPU |
| Frame buffer (1080p) | <50 MB |
| Total runtime | <1 GB GPU |

---

## 9. Implementation Phases

### Phase 1: Image Codec (I-Frames Only)

**Duration**: 4-6 weeks

**Goals**:
- Implement basic neural image codec
- Achieve 30%+ better compression than JPEG at same quality
- Real-time 1080p encoding/decoding

**Deliverables**:
1. Analysis transform (encoder)
2. Synthesis transform (decoder)
3. Hyperprior network
4. Arithmetic coding integration
5. Basic bitstream format
6. PSNR/SSIM benchmarks

**Success Criteria**:
- Encode 1080p image in <50ms
- Decode 1080p image in <20ms
- PSNR 35dB at 0.25 bpp

### Phase 2: Video Codec (P-Frames with Motion)

**Duration**: 6-8 weeks

**Goals**:
- Add temporal compression
- Implement motion estimation and compensation
- Achieve target 20% improvement over AV1

**Deliverables**:
1. Motion estimation network
2. Motion compensation (warping)
3. Residual coding network
4. GOP structure handling
5. Updated bitstream format
6. BD-rate benchmarks vs AV1

**Success Criteria**:
- 1080p30 real-time encoding
- 1080p60 real-time decoding
- 20%+ bitrate savings vs AV1 at same PSNR

### Phase 3: Optimization and Polish

**Duration**: 4-6 weeks

**Goals**:
- Optimize for production deployment
- Add advanced features
- Polish API and documentation

**Deliverables**:
1. TensorRT optimization
2. Multi-threading support
3. Quality presets (fast/medium/slow)
4. Rate control modes (CBR, VBR, CRF)
5. Comprehensive test suite
6. API documentation

**Success Criteria**:
- 4K30 encoding on RTX 4090
- <100ms encode latency at any resolution
- Production-quality API

### Milestone Timeline

```
Week 0-2:   Setup + Analysis Transform
Week 2-4:   Synthesis Transform + Hyperprior
Week 4-6:   Entropy Coding + I-Frame Pipeline
Week 6-8:   Motion Estimation Network
Week 8-10:  Motion Compensation + Warping
Week 10-12: Residual Coding + Full Pipeline
Week 12-14: TensorRT Optimization
Week 14-16: Rate Control + Presets
Week 16-18: Testing + Documentation
```

---

## 10. Performance Optimization

> **Last Updated**: December 2025. This section covers the latest techniques for achieving real-time neural video codec performance based on DCVC-RT research and industry best practices.

### Overview: The Performance Challenge

Neural video codecs face a fundamental challenge: achieving real-time performance while maintaining compression advantages over traditional codecs. DCVC-RT (CVPR 2025) demonstrated that **125 fps encoding at 1080p is achievable** on modern GPUs, making neural codecs practically deployable for the first time.

The key insight from DCVC-RT research: **Operational cost, not computational cost (MACs), is the primary bottleneck** to coding speed. This includes memory I/O, function call overhead, and entropy coding latency.

### 10.1 Real-Time Neural Codec Optimizations

#### 10.1.1 DCVC-RT's Approach to 125fps at 1080p

Microsoft's DCVC-RT achieved breakthrough real-time performance through several key optimizations:

**Operational Cost Reduction:**
```
Traditional NVC Bottlenecks:
1. Memory I/O bandwidth (PRIMARY)
2. Function call overhead
3. Progressive downsampling chains
4. Explicit motion estimation complexity

DCVC-RT Solutions:
1. Single low-resolution latents (1/8 scale vs progressive)
2. Implicit temporal modeling (eliminates motion modules)
3. Unified network architecture (fewer function calls)
4. Memory-efficient data layout
```

**Performance Comparison:**

| Codec | 1080p Encode | 1080p Decode | GPU | Key Factor |
|-------|--------------|--------------|-----|------------|
| DCVC-FM | 5.0 fps | 5.9 fps | A100 | High operational cost |
| DCVC-RT | 125.2 fps | 112.8 fps | A100 | Optimized memory I/O |
| DCVC-RT | 39.5 fps | 34.1 fps | RTX 2080Ti | Consumer GPU viable |

**Architectural Simplifications:**
- **Single-scale latents**: Uses fixed 1/8 resolution instead of progressive 1/2 -> 1/4 -> 1/8 downsampling
- **Implicit temporal modeling**: Replaces explicit optical flow with learned temporal context
- **Extended receptive field**: 1/8 scale provides sufficient receptive field for temporal redundancy reduction

#### 10.1.2 Model Integerization (INT16/INT8)

DCVC-RT implements **16-bit model integerization** for deterministic cross-platform inference:

```python
# DCVC-RT Integerization Approach
# Scaling factors for fixed-point arithmetic
K1 = 512   # Primary scale factor
K2 = 8192  # Secondary scale factor for precision

def integerize_weights(weight_fp32, scale=K1):
    """Convert FP32 weights to INT16 with scaling"""
    weight_int16 = torch.round(weight_fp32 * scale).to(torch.int16)
    return weight_int16

def integerize_activation(activation_fp32, scale=K2):
    """Convert FP32 activations to INT16"""
    activation_int16 = torch.round(activation_fp32 * scale).to(torch.int16)
    return activation_int16
```

**Precision Trade-offs:**

| Mode | Performance | Cross-Platform | Hardware Support |
|------|-------------|----------------|------------------|
| FP16 | Fastest (Tensor Cores) | Non-deterministic | Excellent (all modern GPUs) |
| INT16 | 4x slower than FP16 | Deterministic | Limited optimization |
| INT8 | Fast | Deterministic | Best (dedicated INT8 units) |

**Key Findings:**
- FP16 is **4x faster** than INT16 on A100 due to Tensor Core optimization
- INT16 provides **bit-exact reconstruction** across different platforms
- For production deployments requiring determinism, INT16 is necessary despite speed penalty
- INT8 offers best speed but requires careful quantization-aware training

#### 10.1.3 Memory I/O Optimization

Memory bandwidth is the primary bottleneck for neural codecs:

```
Memory Bandwidth Requirements (1080p):
- Input frame: 1920 x 1080 x 3 x 4 bytes = 24.9 MB (FP32)
- Latent (1/8): 240 x 135 x 192 x 4 bytes = 24.9 MB (FP32)
- Reference frame: 24.9 MB
- Total per frame: ~100-200 MB memory traffic

A100 Memory Bandwidth: 2039 GB/s (80GB model)
Theoretical max: ~10,000+ fps (memory limited)
Actual: ~125 fps (operational overhead)
```

**Optimization Strategies:**

1. **Single Low-Resolution Latents**
   - Reduces memory traffic by avoiding progressive downsampling
   - Single 1/8 scale representation vs multiple intermediate scales

2. **Fused Operations**
   - Combine multiple operations into single kernel launches
   - Reduce memory round-trips for intermediate results

3. **Memory Layout Optimization**
   - Use NHWC (channels-last) format for better memory coalescing
   - Align tensor dimensions to memory transaction sizes

4. **Activation Recomputation**
   - Trade compute for memory by recomputing activations during backward pass
   - Enables larger batch sizes and longer sequences

### 10.2 TensorRT Optimization for Neural Codecs

#### 10.2.1 TensorRT Workflow

```python
import tensorrt as trt
import torch
from torch2trt import torch2trt

def optimize_with_tensorrt(model, input_shape=(1, 3, 1080, 1920)):
    """
    Convert PyTorch neural codec to TensorRT for production deployment.
    """
    # Create sample input
    x = torch.randn(input_shape).cuda()

    # Convert to TensorRT with FP16
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        max_workspace_size=1 << 30,  # 1GB workspace
        max_batch_size=4,
    )

    return model_trt

# For INT8 quantization with calibration
def optimize_with_int8(model, calibration_data):
    """
    INT8 quantization with calibration dataset.
    """
    class CalibrationDataset:
        def __init__(self, data):
            self.data = data
            self.batch_size = 1
            self.index = 0

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    calibrator = trt.IInt8EntropyCalibrator2(
        CalibrationDataset(calibration_data),
        cache_file="codec_int8_calibration.cache"
    )

    model_trt = torch2trt(
        model,
        [calibration_data[0]],
        int8_mode=True,
        int8_calib_dataset=calibrator,
        max_workspace_size=1 << 30,
    )

    return model_trt
```

#### 10.2.2 FP16/INT8 Quantization Strategies

**Recommended Precision Hierarchy (NVIDIA Guidelines 2025):**

1. **FP8** (First choice on Ampere+): Minimal accuracy loss, strong performance
2. **INT8 SQ** (Smooth Quantization): Good for Ampere and earlier GPUs
3. **INT4 AWQ**: Maximum compression with acceptable accuracy loss
4. **FP16**: Safe fallback with good performance

**Per-Layer Quantization Strategy:**

```python
# TensorRT Model Optimizer configuration
quantization_config = {
    # Use per-tensor quantization for activations
    "activation_quantization": "per_tensor",

    # Use per-channel quantization for weights (better accuracy)
    "weight_quantization": "per_channel",

    # Enable FP16 for layers that don't benefit from INT8
    "mixed_precision": True,

    # Layers to keep in higher precision
    "fp16_layers": [
        "entropy_model",      # Entropy coding precision-sensitive
        "final_synthesis",    # Output quality critical
    ],
}
```

#### 10.2.3 Dynamic Batch Sizes

```python
# TensorRT dynamic shapes for variable resolution/batch
def build_engine_dynamic(onnx_path, output_path):
    """Build TensorRT engine with dynamic batch and resolution."""

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30  # 2GB
    config.set_flag(trt.BuilderFlag.FP16)

    # Define optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()

    # Input: [batch, channels, height, width]
    profile.set_shape(
        "input",
        min=(1, 3, 480, 640),      # Minimum: 480p single batch
        opt=(1, 3, 1080, 1920),    # Optimal: 1080p single batch
        max=(4, 3, 2160, 3840),    # Maximum: 4K batch of 4
    )

    config.add_optimization_profile(profile)

    # Build engine
    engine = builder.build_engine(network, config)

    # Serialize
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    return engine
```

#### 10.2.4 Latency vs Throughput Trade-offs

| Optimization | Latency Impact | Throughput Impact | Use Case |
|-------------|----------------|-------------------|----------|
| Larger batch size | +20-50ms | +2-4x | VOD encoding |
| FP16 mode | -30% | +60-80% | General use |
| INT8 mode | -40% | +100-150% | Edge deployment |
| CUDA Graphs | -10-20% | +20% | Low-latency streaming |
| Multi-stream | No change | +2-3x | Parallel encoding |

### 10.3 GPU Memory Optimization

#### 10.3.1 Activation Checkpointing

Activation checkpointing trades compute for memory by recomputing forward activations during backpropagation:

```python
import torch
from torch.utils.checkpoint import checkpoint

class MemoryEfficientEncoder(nn.Module):
    """
    Neural codec encoder with activation checkpointing.
    Reduces memory by 20-50% at cost of ~30% more compute.
    """

    def __init__(self, channels=192):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(3, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 192),
            EncoderBlock(192, channels),
        ])

        # Enable checkpointing for memory-intensive blocks
        self.checkpoint_blocks = [1, 2]  # Middle blocks

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i in self.checkpoint_blocks and self.training:
                # Checkpoint: don't save activations, recompute in backward
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

# Memory comparison
# Without checkpointing: ~24GB for 1080p, batch=4
# With checkpointing:    ~16GB for 1080p, batch=4
```

#### 10.3.2 Memory-Efficient Attention

For transformer-based neural codecs:

```python
import torch.nn.functional as F
from flash_attn import flash_attn_func

class MemoryEfficientAttention(nn.Module):
    """
    Flash Attention for neural video codecs.
    Reduces memory from O(N^2) to O(N) for sequence length N.
    """

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Flash Attention: O(N) memory instead of O(N^2)
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        )

        out = out.reshape(B, N, C)
        return self.proj(out)
```

#### 10.3.3 Tensor Memory Planning

```python
class TensorMemoryPlanner:
    """
    Plan tensor allocation to minimize peak memory usage.
    Based on liveness analysis of intermediate tensors.
    """

    def __init__(self, model):
        self.model = model
        self.tensor_schedule = {}

    def analyze_memory_usage(self, input_shape):
        """Profile memory usage and identify reuse opportunities."""

        # Track tensor lifetimes
        tensor_lifetimes = {}

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
        ) as prof:
            x = torch.randn(input_shape).cuda()
            _ = self.model(x)

        # Analyze memory events
        memory_events = prof.key_averages()
        peak_memory = max(e.cuda_memory_usage for e in memory_events)

        return {
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'tensor_count': len(memory_events),
        }

    def optimize_allocation(self):
        """
        Apply memory optimization strategies:
        1. In-place operations where safe
        2. Tensor fusion for consecutive operations
        3. Memory pool reuse
        """
        # Enable PyTorch memory efficient mode
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
```

#### 10.3.4 Multi-Frame Buffering Strategies

```python
class FrameBufferManager:
    """
    Efficient multi-frame buffer management for video codecs.
    Minimizes memory copies and GPU memory fragmentation.
    """

    def __init__(self, max_frames=10, resolution=(1080, 1920), device='cuda'):
        self.max_frames = max_frames
        self.resolution = resolution
        self.device = device

        # Pre-allocate frame buffers
        self.frame_buffer = torch.zeros(
            max_frames, 3, resolution[0], resolution[1],
            device=device, dtype=torch.float16
        )

        # Ring buffer index
        self.write_idx = 0
        self.read_idx = 0

        # Reference frame cache
        self.reference_frames = {}

    def add_frame(self, frame):
        """Add frame to buffer (zero-copy if possible)."""
        self.frame_buffer[self.write_idx].copy_(frame, non_blocking=True)
        self.write_idx = (self.write_idx + 1) % self.max_frames

    def get_reference_frames(self, count=4):
        """Get most recent reference frames for temporal prediction."""
        indices = [(self.write_idx - i - 1) % self.max_frames
                   for i in range(count)]
        return self.frame_buffer[indices]

    def clear_old_frames(self, keep_count=2):
        """Clear old frames while keeping recent references."""
        # Mark old frames as available for reuse
        pass
```

### 10.4 Multi-Threading and Parallelism

#### 10.4.1 Frame-Level Parallelism

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp

class ParallelVideoEncoder:
    """
    Frame-level parallel encoding with GPU stream management.
    """

    def __init__(self, model, num_workers=4):
        self.model = model
        self.num_workers = num_workers

        # Create CUDA streams for parallel execution
        self.streams = [torch.cuda.Stream() for _ in range(num_workers)]

        # Thread pool for CPU-bound entropy coding
        self.cpu_executor = ThreadPoolExecutor(max_workers=num_workers)

    async def encode_gop(self, frames):
        """
        Encode a GOP with frame-level parallelism.
        I-frame first, then P-frames can be partially parallelized.
        """
        results = []

        # Encode I-frame (blocking)
        i_frame_result = await self._encode_frame_async(frames[0], 'I', 0)
        results.append(i_frame_result)

        # P-frames: neural network inference in parallel with entropy coding
        tasks = []
        for i, frame in enumerate(frames[1:], 1):
            task = asyncio.create_task(
                self._encode_frame_async(frame, 'P', i)
            )
            tasks.append(task)

        # Wait for all P-frames
        p_results = await asyncio.gather(*tasks)
        results.extend(p_results)

        return results

    async def _encode_frame_async(self, frame, frame_type, stream_idx):
        """Encode single frame with dedicated CUDA stream."""
        stream = self.streams[stream_idx % self.num_workers]

        with torch.cuda.stream(stream):
            # Neural network inference (GPU)
            latent = self.model.encode(frame)

            # Quantization (GPU)
            quantized = self.model.quantize(latent)

        # Wait for GPU work to complete
        stream.synchronize()

        # Entropy coding (CPU) - run in thread pool
        loop = asyncio.get_event_loop()
        bitstream = await loop.run_in_executor(
            self.cpu_executor,
            self.model.entropy_encode,
            quantized.cpu()
        )

        return {
            'frame_type': frame_type,
            'bitstream': bitstream,
            'size_bytes': len(bitstream),
        }
```

#### 10.4.2 Tile-Based Parallelism

```python
class TileBasedEncoder:
    """
    Tile-based parallel encoding for large resolutions (4K/8K).
    Splits frame into independent tiles that can be encoded in parallel.
    """

    def __init__(self, model, tile_size=(540, 960)):
        self.model = model
        self.tile_size = tile_size

    def split_into_tiles(self, frame):
        """Split frame into non-overlapping tiles."""
        B, C, H, W = frame.shape
        tile_h, tile_w = self.tile_size

        tiles = []
        positions = []

        for y in range(0, H, tile_h):
            for x in range(0, W, tile_w):
                tile = frame[:, :, y:y+tile_h, x:x+tile_w]
                tiles.append(tile)
                positions.append((y, x))

        return tiles, positions

    def encode_tiles_parallel(self, tiles):
        """Encode tiles in parallel using multiple CUDA streams."""
        num_tiles = len(tiles)
        streams = [torch.cuda.Stream() for _ in range(min(num_tiles, 8))]

        results = [None] * num_tiles

        for i, tile in enumerate(tiles):
            stream = streams[i % len(streams)]

            with torch.cuda.stream(stream):
                results[i] = self.model.encode(tile)

        # Synchronize all streams
        torch.cuda.synchronize()

        return results

    def merge_tiles(self, encoded_tiles, positions, output_shape):
        """Merge encoded tiles back into full frame."""
        B, C, H, W = output_shape
        result = torch.zeros(B, C, H, W, device=encoded_tiles[0].device)

        tile_h, tile_w = self.tile_size

        for tile, (y, x) in zip(encoded_tiles, positions):
            result[:, :, y:y+tile_h, x:x+tile_w] = tile

        return result
```

#### 10.4.3 Async Encoding/Decoding Pipeline

```python
class AsyncEncoderPipeline:
    """
    Asynchronous encoding pipeline with overlap of:
    1. CPU preprocessing (color conversion, scaling)
    2. GPU neural network inference
    3. CPU entropy coding
    4. I/O (writing to disk/network)
    """

    def __init__(self, model):
        self.model = model

        # Separate queues for each pipeline stage
        self.preprocess_queue = asyncio.Queue(maxsize=4)
        self.inference_queue = asyncio.Queue(maxsize=4)
        self.entropy_queue = asyncio.Queue(maxsize=4)

        # CUDA events for synchronization
        self.events = {}

    async def preprocess_worker(self, input_queue):
        """CPU preprocessing worker."""
        while True:
            frame_data = await input_queue.get()
            if frame_data is None:
                break

            # Color conversion, normalization (CPU)
            preprocessed = self._preprocess(frame_data)
            await self.preprocess_queue.put(preprocessed)

    async def inference_worker(self):
        """GPU inference worker."""
        while True:
            frame = await self.preprocess_queue.get()
            if frame is None:
                break

            # Move to GPU and encode
            frame_gpu = frame.cuda(non_blocking=True)

            with torch.cuda.stream(self.inference_stream):
                latent = self.model.encode(frame_gpu)
                quantized = self.model.quantize(latent)

            # Record event for synchronization
            event = torch.cuda.Event()
            event.record()

            await self.inference_queue.put((quantized, event))

    async def entropy_worker(self):
        """CPU entropy coding worker."""
        while True:
            item = await self.inference_queue.get()
            if item is None:
                break

            quantized, event = item

            # Wait for GPU work to complete
            event.synchronize()

            # Move to CPU and entropy encode
            quantized_cpu = quantized.cpu()
            bitstream = self.model.entropy_encode(quantized_cpu)

            await self.entropy_queue.put(bitstream)

    async def run_pipeline(self, frame_source):
        """Run full async pipeline."""
        tasks = [
            asyncio.create_task(self.preprocess_worker(frame_source)),
            asyncio.create_task(self.inference_worker()),
            asyncio.create_task(self.entropy_worker()),
        ]

        await asyncio.gather(*tasks)
```

#### 10.4.4 GPU Stream Management

```python
class CUDAStreamManager:
    """
    Manage CUDA streams for optimal parallelism.
    Based on DCVC-RT's parallel coding scheme.
    """

    def __init__(self, num_streams=4):
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0

        # Default stream for synchronization points
        self.default_stream = torch.cuda.default_stream()

    def get_next_stream(self):
        """Get next available stream (round-robin)."""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream

    def parallel_encode_decode(self, encoder, decoder, frames, references):
        """
        DCVC-RT style: Run encoder and decoder in parallel.
        Entropy coding on CPU doesn't block GPU inference.
        """
        results = []

        for frame, ref in zip(frames, references):
            encode_stream = self.get_next_stream()
            decode_stream = self.get_next_stream()

            # Encoder on stream 1
            with torch.cuda.stream(encode_stream):
                latent = encoder(frame)

            # Decoder on stream 2 (using previous frame's latent)
            if ref is not None:
                with torch.cuda.stream(decode_stream):
                    reconstructed = decoder(ref)

            results.append({
                'latent': latent,
                'encode_stream': encode_stream,
            })

        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()

        return results
```

### 10.5 Model Compression Techniques

#### 10.5.1 Pruning for Neural Codecs

```python
import torch.nn.utils.prune as prune

class PrunedNeuralCodec:
    """
    Structured pruning for neural video codecs.
    Channel pruning is more hardware-friendly than unstructured pruning.
    """

    def __init__(self, model, target_sparsity=0.3):
        self.model = model
        self.target_sparsity = target_sparsity

    def apply_channel_pruning(self):
        """
        Apply channel pruning to convolutional layers.
        Reduces model size by ~30% with ~0.25 dB PSNR drop.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # L1 structured pruning on output channels
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.target_sparsity,
                    n=1,  # L1 norm
                    dim=0,  # Output channels
                )

    def iterative_pruning(self, train_loader, epochs=10, prune_per_epoch=0.1):
        """
        Iterative pruning: prune 10% at a time with fine-tuning.
        Better preserves accuracy than one-shot pruning.
        """
        current_sparsity = 0

        while current_sparsity < self.target_sparsity:
            # Prune incrementally
            increment = min(prune_per_epoch, self.target_sparsity - current_sparsity)
            self._prune_increment(increment)
            current_sparsity += increment

            # Fine-tune after pruning
            self._fine_tune(train_loader, epochs=epochs)

            # Evaluate
            metrics = self._evaluate()
            print(f"Sparsity: {current_sparsity:.1%}, BD-rate: {metrics['bd_rate']:.2%}")

    def _prune_increment(self, amount):
        """Prune by specified amount."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, 'weight', amount=amount, n=1, dim=0)

    def remove_pruning(self):
        """Make pruning permanent (remove masks)."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.remove(module, 'weight')
```

**Pruning Results for Neural Image Codecs:**

| Method | Size Reduction | PSNR Change | Latency Change |
|--------|---------------|-------------|----------------|
| Channel pruning 30% | -30% | -0.25 dB | -40% |
| Unstructured 50% | -50% | -0.15 dB | Variable |
| Structured + Fine-tune | -30% | -0.10 dB | -35% |

#### 10.5.2 Knowledge Distillation for Codecs

```python
class CodecKnowledgeDistillation:
    """
    Knowledge distillation for neural codecs (KDIC framework).
    Teacher: Large, accurate model
    Student: Small, efficient model

    Results: 40% parameter reduction, 57% FLOP reduction,
             with only 0.25-0.57% BD-rate increase.
    """

    def __init__(self, teacher_model, student_model, temperature=4.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, input_frame, lambda_kd=0.5):
        """
        Combined loss: task loss + distillation loss.
        """
        # Student forward pass
        student_out = self.student(input_frame)
        student_latent = student_out['latent']
        student_recon = student_out['reconstruction']

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(input_frame)
            teacher_latent = teacher_out['latent']
            teacher_recon = teacher_out['reconstruction']

        # Task loss (rate-distortion)
        rd_loss = self._compute_rd_loss(student_out, input_frame)

        # Distillation loss (feature matching)
        latent_kd_loss = F.mse_loss(student_latent, teacher_latent)

        # Soft target distillation for entropy model
        soft_target_loss = self._soft_target_loss(
            student_out['likelihoods'],
            teacher_out['likelihoods'],
        )

        total_loss = rd_loss + lambda_kd * (latent_kd_loss + soft_target_loss)

        return total_loss

    def _soft_target_loss(self, student_probs, teacher_probs):
        """KL divergence between soft probability distributions."""
        T = self.temperature

        student_soft = F.log_softmax(student_probs / T, dim=-1)
        teacher_soft = F.softmax(teacher_probs / T, dim=-1)

        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
```

#### 10.5.3 Neural Architecture Search (NAS) for Codecs

```python
class NASCodecSearchSpace:
    """
    Neural Architecture Search space for efficient neural codecs.
    Based on NVRC and ENAS approaches.
    """

    # Search space definition
    SEARCH_SPACE = {
        'channels': [64, 96, 128, 192, 256],
        'kernel_sizes': [3, 5, 7],
        'num_layers': [3, 4, 5, 6],
        'attention_type': ['none', 'se', 'cbam', 'transformer'],
        'normalization': ['bn', 'gn', 'ln', 'gdn'],
        'activation': ['relu', 'gelu', 'swish'],
        'downsample_factor': [8, 16, 32],
    }

    def __init__(self, target_flops=None, target_params=None):
        self.target_flops = target_flops
        self.target_params = target_params

    def sample_architecture(self):
        """Sample random architecture from search space."""
        import random

        arch = {}
        for key, options in self.SEARCH_SPACE.items():
            arch[key] = random.choice(options)

        return arch

    def evaluate_architecture(self, arch, val_loader):
        """
        Evaluate architecture without full training.
        Uses proxy metrics: BD-rate estimate, latency, memory.
        """
        model = self._build_model(arch)

        # Quick evaluation (few iterations)
        metrics = self._quick_evaluate(model, val_loader)

        # Resource constraints
        flops = self._estimate_flops(model)
        params = sum(p.numel() for p in model.parameters())

        # Constraint penalties
        penalty = 0
        if self.target_flops and flops > self.target_flops:
            penalty += (flops / self.target_flops - 1) * 0.5
        if self.target_params and params > self.target_params:
            penalty += (params / self.target_params - 1) * 0.5

        score = metrics['bd_rate'] + penalty

        return {
            'score': score,
            'bd_rate': metrics['bd_rate'],
            'flops': flops,
            'params': params,
        }
```

#### 10.5.4 Depth-wise Separable Convolutions

```python
class EfficientConvBlock(nn.Module):
    """
    Efficient convolutional block using depth-wise separable convolutions.
    Reduces parameters by ~8-9x compared to standard convolutions.

    Standard Conv: C_in * C_out * K * K parameters
    Depthwise Sep: C_in * K * K + C_in * C_out parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()

        # Depthwise convolution (spatial filtering per channel)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,  # Key: groups = in_channels
            bias=False,
        )

        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class MobileNetStyleEncoder(nn.Module):
    """
    MobileNet-inspired encoder for neural codecs.
    ~80% fewer parameters than standard encoder with similar quality.
    """

    def __init__(self, latent_channels=192):
        super().__init__()

        self.layers = nn.Sequential(
            # Initial conv (standard)
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            # Depthwise separable blocks
            EfficientConvBlock(32, 64, stride=2),
            EfficientConvBlock(64, 128, stride=2),
            EfficientConvBlock(128, 192, stride=2),

            # Final projection
            nn.Conv2d(192, latent_channels, 1),
        )

    def forward(self, x):
        return self.layers(x)
```

### 10.6 Latency Optimization

#### 10.6.1 Single-Frame Latency Targets

| Use Case | Target Latency | Acceptable Range |
|----------|---------------|------------------|
| Live streaming | <50ms | 30-100ms |
| Video conferencing | <100ms | 50-150ms |
| Cloud gaming | <20ms | 10-40ms |
| VOD encoding | No limit | - |
| Real-time preview | <33ms | 16-50ms |

#### 10.6.2 Pipeline Optimization

```python
class LowLatencyPipeline:
    """
    Optimized pipeline for minimum single-frame latency.
    Target: <20ms for 1080p on RTX 4090.
    """

    def __init__(self, model):
        self.model = model

        # Pre-compiled model with CUDA Graphs
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

        # Pre-allocated buffers
        self._setup_buffers()

    def _setup_buffers(self):
        """Pre-allocate all buffers to avoid runtime allocation."""
        self.input_buffer = torch.zeros(
            1, 3, 1080, 1920,
            device='cuda',
            dtype=torch.float16
        )
        self.latent_buffer = torch.zeros(
            1, 192, 68, 120,
            device='cuda',
            dtype=torch.float16
        )

    def warmup(self, num_iterations=10):
        """
        Warmup to trigger JIT compilation and CUDA kernel caching.
        Critical for consistent low latency.
        """
        dummy_input = torch.randn(1, 3, 1080, 1920, device='cuda', dtype=torch.float16)

        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)

        torch.cuda.synchronize()

    def capture_cuda_graph(self):
        """
        Capture CUDA graph for minimal kernel launch overhead.
        Reduces per-frame overhead by 10-20%.
        """
        self.static_input = torch.randn(
            1, 3, 1080, 1920,
            device='cuda',
            dtype=torch.float16
        )

        # Warmup
        for _ in range(3):
            self.static_output = self.model(self.static_input)

        # Capture graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.model(self.static_input)

    def encode_frame_low_latency(self, frame):
        """
        Encode single frame with minimum latency.
        Uses CUDA graph replay if available.
        """
        if self.cuda_graph is not None:
            # Copy input to static buffer
            self.static_input.copy_(frame)

            # Replay graph (minimal overhead)
            self.cuda_graph.replay()

            return self.static_output.clone()
        else:
            # Standard inference
            with torch.no_grad():
                return self.model(frame)
```

#### 10.6.3 Warm-up Strategies

```python
class WarmupManager:
    """
    Manage model warmup for consistent latency.
    First few frames are always slower due to:
    1. CUDA kernel compilation (JIT)
    2. Memory allocation
    3. cuDNN autotuning
    """

    def __init__(self, model, target_resolution=(1080, 1920)):
        self.model = model
        self.resolution = target_resolution
        self.is_warmed_up = False

    def warmup(self, warmup_frames=20):
        """
        Comprehensive warmup procedure.
        """
        print("Starting warmup...")

        # 1. Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = True

        # 2. Create dummy input at target resolution
        dummy = torch.randn(
            1, 3, self.resolution[0], self.resolution[1],
            device='cuda',
            dtype=torch.float16
        )

        # 3. Run warmup iterations
        latencies = []
        for i in range(warmup_frames):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _ = self.model(dummy)
            end.record()

            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
            latencies.append(latency)

        # 4. Report warmup statistics
        avg_warmup = sum(latencies[:5]) / 5
        avg_steady = sum(latencies[-5:]) / 5

        print(f"Warmup complete:")
        print(f"  Initial latency: {latencies[0]:.2f}ms")
        print(f"  Steady-state latency: {avg_steady:.2f}ms")
        print(f"  Speedup from warmup: {avg_warmup / avg_steady:.2f}x")

        self.is_warmed_up = True

        return {
            'initial_latency': latencies[0],
            'steady_latency': avg_steady,
            'all_latencies': latencies,
        }
```

#### 10.6.4 Memory Pre-allocation

```python
class MemoryPreallocator:
    """
    Pre-allocate all GPU memory needed for inference.
    Eliminates runtime allocation overhead.
    """

    def __init__(self, model, max_resolution=(2160, 3840), max_batch=4):
        self.model = model
        self.max_resolution = max_resolution
        self.max_batch = max_batch

        self.buffers = {}
        self._preallocate()

    def _preallocate(self):
        """Pre-allocate all required buffers."""
        H, W = self.max_resolution
        B = self.max_batch

        # Input buffer
        self.buffers['input'] = torch.zeros(
            B, 3, H, W,
            device='cuda',
            dtype=torch.float16
        )

        # Latent buffer (assuming 1/16 downsampling)
        self.buffers['latent'] = torch.zeros(
            B, 192, H // 16, W // 16,
            device='cuda',
            dtype=torch.float16
        )

        # Hyperprior buffer
        self.buffers['hyperprior'] = torch.zeros(
            B, 128, H // 64, W // 64,
            device='cuda',
            dtype=torch.float16
        )

        # Reconstruction buffer
        self.buffers['reconstruction'] = torch.zeros(
            B, 3, H, W,
            device='cuda',
            dtype=torch.float16
        )

        # Reference frame buffer (for temporal prediction)
        self.buffers['reference'] = torch.zeros(
            B, 3, H, W,
            device='cuda',
            dtype=torch.float16
        )

        print(f"Pre-allocated {self._total_memory_mb():.1f} MB of GPU memory")

    def _total_memory_mb(self):
        """Calculate total pre-allocated memory."""
        total_bytes = sum(
            buf.numel() * buf.element_size()
            for buf in self.buffers.values()
        )
        return total_bytes / (1024 * 1024)

    def get_buffer(self, name, shape):
        """Get a view of pre-allocated buffer with specified shape."""
        buf = self.buffers[name]
        return buf[:shape[0], :shape[1], :shape[2], :shape[3]]
```

### 10.7 Performance Benchmarking

#### Benchmark Results Summary

**DCVC-RT Real-World Performance:**

| Configuration | Encode FPS | Decode FPS | Notes |
|--------------|------------|------------|-------|
| A100, FP16, 1080p | 125.2 | 112.8 | Paper benchmark |
| A100, INT16, 1080p | ~30 | ~28 | 4x slower than FP16 |
| RTX 2080Ti, FP16, 1080p | 39.5 | 34.1 | Consumer GPU |
| Production (multi-stream) | 45-65 | - | With background processes |

**Optimization Impact:**

| Technique | Latency Reduction | Throughput Gain | Memory Reduction |
|-----------|------------------|-----------------|------------------|
| TensorRT FP16 | -30% | +60-80% | -50% |
| TensorRT INT8 | -40% | +100-150% | -75% |
| CUDA Graphs | -10-20% | +20% | 0% |
| Channel pruning 30% | -35% | +40% | -30% |
| Activation checkpointing | 0% | 0% | -30% |
| Flash Attention | -20% | +30% | -60% (attention) |

### 10.8 References

**DCVC-RT and Real-Time Neural Codecs:**
- [DCVC-RT CVPR 2025 Paper](https://arxiv.org/abs/2502.20762) - "Towards Practical Real-Time Neural Video Compression"
- [DCVC-RT Project Page](https://dcvccodec.github.io/)
- [Streaming Learning Center: DCVC-RT Evaluation](https://streaminglearningcenter.com/articles/evaluating-dcvc-rt-a-real-time-neural-video-codec-that-delivers-on-speed-and-compression.html)
- [SimaLabs: DCVC-RT vs HEVC Benchmark](https://www.simalabs.ai/resources/dcvc-rt-vs-hevc-1080p-benchmark-analysis-125-fps)

**TensorRT and Quantization:**
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [TensorRT Model Optimizer](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html)
- [INT8 Quantization with TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
- [FP8 vs INT8 Comparison](https://arxiv.org/pdf/2303.17951)

**Model Compression for Neural Codecs:**
- [Knowledge Distillation for Learned Image Compression (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Knowledge_Distillation_for_Learned_Image_Compression_ICCV_2025_paper.pdf)
- [Lightweight FPGA Deployment with KD and Hybrid Quantization](https://arxiv.org/html/2503.04832)
- [Quantized Decoder for Deterministic Reconstruction](https://arxiv.org/html/2312.11209)

**GPU Optimization:**
- [CUDA Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Advanced CUDA Kernel Optimization](https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/)
- [Efficient Parallel Video Processing on GPU](https://pmc.ncbi.nlm.nih.gov/articles/PMC3976889/)

**Memory Optimization:**
- [NeuZip: Memory-Efficient Training and Inference](https://arxiv.org/html/2410.20650)
- [Checkmate: Optimal Tensor Rematerialization](https://arxiv.org/pdf/1910.02653)
- [HybridServe: Activation Checkpointing for LLM Inference](https://arxiv.org/abs/2501.01792)

**Depth-wise Separable Convolutions:**
- [MobileNet with Depthwise Separable Convolutions](https://www.sciencedirect.com/topics/computer-science/depthwise-separable-convolution)
- [Optimizing Depthwise Separable Convolutions on GPUs](https://ieeexplore.ieee.org/document/9444208/)

---

## 11. Open Source Resources

### Neural Codec Implementations

#### Microsoft DCVC (Apache 2.0)
- **Repository**: [github.com/microsoft/DCVC](https://github.com/microsoft/DCVC)
- **Contains**: DCVC-RT, DCVC-DC, DCVC-FM
- **Language**: Python/PyTorch
- **Use**: Reference architecture, training code

#### CompressAI (BSD 3-Clause)
- **Repository**: [github.com/InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI)
- **Contains**: Multiple image codec architectures, training framework
- **Language**: Python/PyTorch
- **Use**: Base models, training utilities

#### OpenDVC (MIT)
- **Repository**: [github.com/RenYang-home/OpenDVC](https://github.com/RenYang-home/OpenDVC)
- **Contains**: Simple video codec implementation
- **Language**: Python/TensorFlow
- **Use**: Educational reference

### Rust ML Frameworks (December 2025)

#### candle (Apache 2.0 / MIT)
- **Repository**: [github.com/huggingface/candle](https://github.com/huggingface/candle)
- **Version**: 0.9.x (candle-onnx 0.9.1)
- **Use**: Pure Rust inference, LLM deployment, Metal/CUDA/WebGPU

#### ort (Apache 2.0 / MIT)
- **Repository**: [github.com/pykeio/ort](https://github.com/pykeio/ort)
- **Version**: 2.0.0-rc.10 (wraps ONNX Runtime 1.22)
- **Use**: ONNX Runtime bindings with TensorRT/CUDA/DirectML

#### burn (Apache 2.0 / MIT)
- **Repository**: [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
- **Version**: 0.19.1
- **Use**: Cross-platform training/inference, ONNX import, CubeCL GPU

#### tract (Apache 2.0 / MIT)
- **Repository**: [github.com/sonos/tract](https://github.com/sonos/tract)
- **Version**: 0.22.0
- **Use**: Pure Rust CPU inference, edge/embedded deployment

### Entropy Coding

#### arcode (MIT)
- **Crate**: [crates.io/crates/arcode](https://crates.io/crates/arcode)
- **Version**: 0.3.x
- **Use**: Arithmetic coding

#### range-coder (MIT)
- **Crate**: [crates.io/crates/range-coder](https://crates.io/crates/range-coder)
- **Version**: 0.1.x
- **Use**: Alternative entropy coder

### Utilities

#### ndarray (MIT / Apache 2.0)
- **Crate**: [crates.io/crates/ndarray](https://crates.io/crates/ndarray)
- **Use**: N-dimensional arrays

#### image (MIT)
- **Crate**: [crates.io/crates/image](https://crates.io/crates/image)
- **Use**: Image loading/saving

---

## 12. Patent Considerations

> **Last Updated**: December 2025
> **Disclaimer**: This section provides educational analysis only and does not constitute legal advice. Commercial deployment requires formal freedom-to-operate (FTO) analysis by qualified patent counsel.

### 12.1 Traditional Codec Patent Landscape (December 2025)

#### H.264/AVC - Approaching Patent Expiration

| Aspect | Details |
|--------|---------|
| **Patent Pool** | [Via LA](https://www.via-la.com/licensing-2/avc-h-264/) (formerly MPEG-LA, acquired April 2023) |
| **Total Patents** | Extensive portfolio from 42 patent owners |
| **Expiration Status** | Core patents expiring 2023-2027, some extend to 2030 |

**Key Expiration Dates** (Source: [Wikimedia Patent Tracking](https://meta.wikimedia.org/wiki/Have_the_patents_for_H.264_MPEG-4_AVC_expired_yet%3F)):
- **2025-12-20**: Orange SA patents (CN 101120591, CN 102065295B, EP 1839442, JP 5042856)
- **2025-12-13**: Siemens AG patents (CN 101099388, EP 1836852)
- **2027**: Last US patents in MPEG LA list expire
- **2030-11-10**: Last Brazilian patent (BR PI0304568-4) expires

**Royalty-Free Provisions**: Via LA/MPEG-LA provides perpetual royalty-free licensing for:
- Internet video that is free to end users ("Internet Broadcast AVC Video")
- Cisco's OpenH264 binary distributions (Cisco pays royalties on behalf of users)

#### H.265/HEVC - The "Patent Nightmare"

| Pool | Patents | Coverage | 2025 Status |
|------|---------|----------|-------------|
| **[Access Advance (HEVC Advance)](https://accessadvance.com/hevc-advance-patent-pool-general-pool-terms/)** | 29,000+ | 75-80% of SEPs | Major pool, pricing through 2030 announced |
| **Via LA (HEVC)** | Significant | Overlaps with Access Advance | Active licensing |
| **Velos Media** | Varies | Holdouts from other pools | Fragmented |

**Critical 2025 Developments**:
- [Access Advance July 2025 announcement](https://accessadvance.com/2025/07/21/access-advance-announces-hevc-advance-and-vvc-advance-pricing-through-2030/): Licensees joining before Dec 31, 2025 lock in current rates through 2030
- Beginning January 1, 2026: HEVC rates align with VVC pricing structure (25% increase for new licensees)
- [Xiaomi joined](https://accessadvance.com/2025/11/18/access-advance-welcomes-xiaomi-to-hevc-advance-and-vvc-advance-patent-pools-as-a-licensor-and-licensee/) as both licensor and licensee (November 2025)
- Huawei expanded to VVC Advance as licensee

**Risk Assessment**: **VERY HIGH** - Multiple overlapping pools, litigation history, unpredictable total costs

#### VVC/H.266 - The Next Generation Problem

| Aspect | Details |
|--------|---------|
| **Patent Pool** | [Access Advance (VVC Advance)](https://accessadvance.com/vvc-advance-patent-pool-general-pool-terms/) |
| **Patents** | 4,500+ essential patents |
| **Growth** | 3 new licensors, 9 new licensees in 2025 |

**Licensing Terms**:
- Five-year non-terminable license increments
- Current licenses auto-renew Dec 31, 2025 through Dec 31, 2030
- Rates locked through 2030 for existing licensees

**Risk Assessment**: **HIGH** - Still maturing, but inherits HEVC's fragmentation legacy

#### AV1 - "Royalty-Free" With Asterisks

**[Alliance for Open Media (AOMedia)](https://aomedia.org/)**:
| Aspect | Details |
|--------|---------|
| **Members** | Google, Apple, Microsoft, Netflix, Amazon, Meta, Cisco, Intel, AMD, NVIDIA, Mozilla, Samsung, and 40+ others |
| **License** | [AOMedia Patent License 1.0](https://aomedia.org/license/patent-license/) |
| **Model** | Reciprocal royalty-free: use freely, but grant same rights back |

**How AOMedia Achieves Royalty-Free Status**:
1. **Reciprocal Licensing**: All members license their AV1-essential patents royalty-free to anyone
2. **W3C-Style Rules**: Technology contributions require patent clearance
3. **Dual Verification**: No feature adopted without two independent parties confirming no third-party patent infringement
4. **Defensive Termination**: Suing over AV1 patents = losing all AV1 patent rights from all members
5. **Patent Defense Fund**: Financial support for smaller members/licensees if sued

**Third-Party Patent Claims** ([Sisvel Pool](https://www.sisvel.com/licensing-programmes/audio-and-video-coding-decoding/video-coding-platform-av1/)):
- [Sisvel announced AV1 patent pool](https://www.streamingmedia.com/Articles/ReadArticle.aspx?ArticleID=139636) in 2019
- Claims ~2,000 patents from Philips, GE, NTT, Ericsson, Dolby, Toshiba
- Google/AOMedia dispute Sisvel's essentiality claims
- **EU Investigation (2022)**: Antitrust regulators examining if AOMedia licensing restricts innovation

**[Avanci Video Pool](https://www.avanci.com/video/)** (October 2023):
- Launched to license streaming services
- Covers: H.265 (HEVC), H.266 (VVC), VP9, **AV1**, MPEG-DASH
- 32+ licensors as of 2025
- [Ericsson joined April 2025](https://www.avanci.com/2025/04/08/ericsson-joins-avanci-video-as-a-licensor/)

**[Access Advance VDP Pool](https://accessadvance.com/2025/01/16/access-advance-announces-video-distribution-patent-pool-in-response-to-market-demand/)** (January 2025):
- Covers HEVC, VVC, AV1, VP9 in single license
- Tiered pricing based on MAU, subscribers, revenue
- ByteDance, Kuaishou, Tencent joined as licensors/licensees
- Past royalty waiver for signups before June 30, 2025

**Risk Assessment**: **MEDIUM** - Royalty-free for members, but streaming services face third-party claims

---

### 12.2 Neural Codec Patent Landscape (December 2025)

#### Major Patent Holders and Their Positions

##### 1. InterDigital (Deep Render Acquisition)

**[InterDigital acquired Deep Render](https://ir.interdigital.com/news-events/press-releases/news-details/2025/InterDigital-acquires-AI-startup-Deep-Render/default.aspx)** on October 30, 2025, making this the most significant neural codec IP event of the year.

| Aspect | Details |
|--------|---------|
| **Deep Render Patents** | [30+ patents filed](https://patents.justia.com/assignee/deep-render-ltd), 60 innovations in progress |
| **Technology Focus** | AI-based lossy image/video compression, mobile/edge deployment |
| **Patent Coverage** | Neural network encoding, latent representation, flow-based motion, entropy coding |
| **Prior Position** | Aggressive patent portfolio, focused on commercialization |
| **InterDigital Strategy** | Market-leading video portfolio, HEVC/VVC licensing experience |

**Risk Assessment**: **HIGH** - InterDigital has history of aggressive patent licensing. Deep Render patents specifically cover neural codec architectures. Their [patent portfolio](https://deeprender.ai/blog/patents-defining-innovation) includes:
- Computer-implemented lossy image/video compression using trained neural networks
- Latent representation quantization and entropy encoding
- Flow-based motion estimation between frames
- Neural network-based reconstruction

##### 2. Google/DeepMind

| Aspect | Details |
|--------|---------|
| **Portfolio** | Extensive learned compression patents |
| **Posture** | Generally defensive, focused on open standards |
| **Key Assets** | Transformer architecture patent (foundational for many AI applications) |

**Risk Assessment**: **LOW-MEDIUM** - Google's defensive posture and AOMedia membership suggest they won't aggressively enforce against open implementations

##### 3. Microsoft

| Aspect | Details |
|--------|---------|
| **Key Project** | [DCVC Family](https://github.com/microsoft/DCVC) - Deep Contextual Video Compression |
| **Open Source** | Yes, multiple licenses |
| **Performance** | DCVC-RT: 100+ FPS 1080p, 21% bitrate savings vs H.266/VTM |

**Risk Assessment**: **LOW** - Open source release with permissive licensing signals intent not to enforce

##### 4. Qualcomm

| Aspect | Details |
|--------|---------|
| **Focus** | Mobile neural codecs, edge deployment |
| **Achievement** | [First inter-frame neural decoder on commercial phone](https://www.qualcomm.com/news/onq/2021/07/how-ai-research-enabling-next-gen-codecs) |
| **Licensing Model** | Expected FRAND (Fair, Reasonable, And Non-Discriminatory) |

**Risk Assessment**: **MEDIUM** - Will likely license rather than block, but expect royalties for mobile/embedded

##### 5. Apple

| Aspect | Details |
|--------|---------|
| **Focus** | Internal use, device optimization |
| **AOMedia Status** | Governing member |
| **Assertion History** | Typically non-aggressive on compression patents |

**Risk Assessment**: **LOW** - Unlikely to assert against third parties

##### 6. Academic/Research Institutions

Many neural codec innovations originate in academia, creating prior art:
- ETH Zurich (Hyperprior models)
- Tsinghua University
- Various university labs publishing at NeurIPS, CVPR, ICLR

---

### 12.3 Open Source Neural Codec Licenses

#### CompressAI (InterDigital)

| Aspect | Details |
|--------|---------|
| **Repository** | [github.com/InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI) |
| **License** | **BSD 3-Clause Clear License** |
| **Critical Note** | BSD 3-Clause Clear **explicitly does NOT grant patent rights** |

**License Text Implication**: You can use the code, but InterDigital (now owner of Deep Render patents) retains all patent rights. Using CompressAI does not protect you from patent claims.

#### DCVC (Microsoft)

| Aspect | Details |
|--------|---------|
| **Repository** | [github.com/microsoft/DCVC](https://github.com/microsoft/DCVC) |
| **License** | Multiple: Apache 2.0 (from CompressAI basis), BSD 3-Clause Clear, MIT |
| **Components** | Different components have different licenses |

**Key Point**: Apache 2.0 includes explicit patent grant; BSD 3-Clause Clear does not. Check which license applies to which component.

#### License Comparison for Neural Codec Use

| License | Copyright Grant | Patent Grant | Commercial Use | ZVC69 Suitability |
|---------|-----------------|--------------|----------------|-------------------|
| **Apache 2.0** | Yes | **Yes (explicit)** | Yes | **Best choice** |
| **MIT** | Yes | No (implicit?) | Yes | Good, patent unclear |
| **BSD 3-Clause** | Yes | No | Yes | Caution needed |
| **BSD 3-Clause Clear** | Yes | **No (explicit)** | Yes | **Higher risk** |
| **GPL v3** | Yes | Yes | Copyleft | Only for GPL projects |

**Recommendation**: Prefer Apache 2.0 licensed components for explicit patent protection. When using BSD 3-Clause Clear licensed code (like CompressAI), understand that you have no patent protection from the licensor.

---

### 12.4 Standardization Landscape (December 2025)

#### JVET Neural Network Video Coding (NNVC)

**[Workshop on Future Video Coding](https://www.mpeg.org/workshop-on-future-video-coding-advanced-signal-processing-ai-and-standards/)** - Geneva, January 2025

| Aspect | Details |
|--------|---------|
| **Timeline** | H.267 projected finalization: July-October 2028 |
| **Deployment** | Meaningful deployment expected 2034-2036 |
| **Approaches** | NN-based tools, super-resolution, end-to-end NN coding |
| **Current Status** | Neural Compression Software (NCS) version 1.0 with NN in-loop filtering |
| **Performance** | NCS-1.0 shows 8.71-9.44% improvement in random access scenarios |

**[Enhanced Compression Model (ECM) 17](https://streaminglearningcenter.com/codecs/ai-video-compression-standards-whos-doing-what-and-when.html)** - April 2025:
- JVET ECM delivers best overall coding performance
- 11% coding gain over DCVC-FM
- 16.1% gain over AVM in low-delay configurations

#### MPAI (Moving Picture, Audio, and Data Coding by AI)

| Project | Description | Status |
|---------|-------------|--------|
| **MPAI-EVC** | AI-enhanced modules in traditional pipelines | Active development |
| **MPAI-EEV** | AI-based End-to-End Video Coding | Version 4 reference model |
| **Availability** | Both available as open source | |

**Note**: MPAI is most aggressive on AI-native codec standardization

#### AOMedia

Currently focused on AV1 with AI-enhanced encoder tuning and scene analysis. No AI-native codec efforts announced as of December 2025.

---

### 12.5 Forming a Patent-Free Neural Codec

#### How AOMedia Achieved Royalty-Free Status (Model to Follow)

1. **Industry Coalition**: Major tech companies with overlapping patent portfolios joined forces
2. **Reciprocal Licensing**: All members contribute patents royalty-free
3. **Defensive Termination**: Litigation triggers loss of all member patent rights
4. **Due Diligence**: Each feature requires independent verification of non-infringement
5. **Defense Fund**: Financial support for members facing third-party claims
6. **W3C-Style Process**: Borrowed from web standards community

#### Could a Neural Codec Consortium Form?

**Potential Structure**:
- Coalition of implementers (cloud providers, device makers, streaming services)
- Reciprocal royalty-free commitment for neural codec patents
- Joint defense against third-party claims
- Open reference implementation

**Challenges**:
- InterDigital/Deep Render unlikely to join (licensing-focused business model)
- Academic patents may have been licensed to commercial entities
- No existing organization equivalent to AOMedia for neural codecs
- MPAI closest equivalent, but lacks major industry commitment

#### Defensive Publication Strategy

**What It Is**: Publishing technical innovations to create prior art, preventing others from patenting similar ideas.

**How to Execute** (Based on [IP.com](https://ip.com/innovation-power-suite/defensive-publishing/) and [Questel](https://www.questel.com/patent/patent-strategy-and-administration/defensive-publication/) guidance):

1. **Prior Art Databases**:
   - [IP.com Prior Art Database](https://ip.com/)
   - [Research Disclosure](https://www.researchdisclosure.com/)
   - arXiv.org (for academic-style publications)

2. **Publication Requirements**:
   - Detailed technical descriptions with diagrams
   - Clear, precise language (no vagueness)
   - Enabling disclosure (someone could implement from description)
   - Timestamped and indexed for patent examiner discovery

3. **Timing**: Publish early and often. Prior art databases publish in minutes vs. 14 weeks for patent office processing.

4. **Limitations**:
   - Once published, you cannot patent the innovation
   - Competitors gain insight into your technology
   - Does not protect against existing patents

---

### 12.6 ZVC69 Patent Strategy

#### Risk Assessment Matrix

| Risk Factor | Probability | Impact | Current Status |
|-------------|-------------|--------|----------------|
| InterDigital/Deep Render assertion | Medium-High | Very High | Acquired major neural codec portfolio Oct 2025 |
| Traditional codec contamination | Low | High | Using novel architecture, not H.264/HEVC techniques |
| Third-party unknown patents | Medium | High | Common risk, requires FTO analysis |
| Academic patent landmines | Low-Medium | Medium | Many innovations published as prior art |

#### Recommended Mitigation Strategies

##### 1. Architectural Independence

**Goal**: Design architecture that is demonstrably different from patented approaches

| Component | Avoid (Higher Risk) | Prefer (Lower Risk) |
|-----------|---------------------|---------------------|
| **Analysis Transform** | Patented attention mechanisms (specific architectures) | Standard CNN, novel attention variants with prior art |
| **Entropy Model** | Specific autoregressive patterns from patents | Hyperprior + factorized (Ball, et al. well-documented) |
| **Motion Estimation** | Proprietary flow architectures | Standard optical flow, published flow networks |
| **Quantization** | Learned quantization matrices from specific papers | Simple uniform quantization, well-known schemes |
| **Frame Prediction** | Specific flow-warping patents | Novel approaches, documented prior art |

##### 2. Prior Art Documentation

**For Each ZVC69 Component**:
- Document academic papers that precede relevant patents
- Note open source implementations that constitute prior art
- Track publication dates vs. patent priority dates

**Key Prior Art Sources**:
- NeurIPS, CVPR, ICLR compression papers (2016-present)
- arXiv preprints with timestamps
- Open source implementations (CompressAI models, etc.)

##### 3. Defensive Publication (If Novel)

If ZVC69 develops genuinely novel techniques:
- Publish to prior art databases immediately
- Consider academic publication for credibility
- Timestamp all design documents

##### 4. License Selection

**For ZVC69 Implementation**:
- Use **Apache 2.0** license for our code (explicit patent grant to users)
- Require contributor patent grants (Apache CLA model)
- Document all third-party code provenance and licenses

##### 5. Freedom to Operate (FTO) Analysis

**Before Commercial Deployment**:
- Engage qualified patent counsel
- Conduct formal FTO analysis covering:
  - InterDigital/Deep Render portfolio
  - Google learned compression patents
  - Microsoft DCVC-related patents
  - Qualcomm mobile neural codec patents
- Budget: $50,000-150,000 for comprehensive analysis
- Timeline: 3-6 months

#### Patent Insurance Options

For commercial deployment, consider:

| Coverage Type | Purpose | Typical Cost |
|---------------|---------|--------------|
| **Patent Defense Insurance** | Cover litigation defense costs | $25,000-100,000/year |
| **IP Indemnification Insurance** | Cover contractual indemnity obligations | Varies by revenue |
| **Comprehensive IP Insurance** | Both defense and indemnification | $50,000-200,000/year |

Average patent litigation defense cost: **$2.9 million** for moderate-size case. Insurance provides deterrence against NPEs (Non-Practicing Entities) seeking quick settlements.

---

### 12.7 Commercial Deployment Considerations

#### What Companies Need to License (December 2025)

| Use Case | Required Licenses | Estimated Annual Cost |
|----------|-------------------|----------------------|
| **Streaming Service (AV1 only)** | Potentially Sisvel AV1, Avanci Video | Variable, contact pools |
| **Streaming Service (multi-codec)** | Access Advance VDP or individual pools | Tiered by MAU/revenue |
| **Device Manufacturer (HEVC)** | Access Advance HEVC, Via LA HEVC | $0.20-2.00 per device |
| **Device Manufacturer (VVC)** | Access Advance VVC | Similar to HEVC |
| **Neural Codec (ZVC69)** | Unknown - no established pools | FTO analysis required |

#### Streaming Service Patent Pool Comparison

| Pool | Codecs Covered | Launched | Key Licensors |
|------|----------------|----------|---------------|
| **[Avanci Video](https://www.avanci.com/video/)** | HEVC, VVC, VP9, AV1, MPEG-DASH | Oct 2023 | 32+ licensors including Ericsson |
| **[Access Advance VDP](https://accessadvance.com/2025/01/16/access-advance-announces-video-distribution-patent-pool-in-response-to-market-demand/)** | HEVC, VVC, AV1, VP9 | Jan 2025 | ByteDance, Kuaishou, Tencent |
| **[Sisvel AV1](https://www.sisvel.com/licensing-programmes/audio-and-video-coding-decoding/video-coding-platform-av1/)** | AV1, VP9 | 2020 | Philips, Dolby, NTT, Ericsson, Toshiba |

---

### 12.8 Legal Disclaimer

> **IMPORTANT**: This section provides educational information about the patent landscape for neural video codecs as of December 2025. It is not legal advice and should not be relied upon for making business or legal decisions.
>
> **Before any commercial deployment of ZVC69 or similar neural codec technology**:
> 1. Engage qualified patent counsel familiar with video compression IP
> 2. Conduct formal Freedom to Operate (FTO) analysis
> 3. Consider patent insurance or indemnification arrangements
> 4. Monitor ongoing patent pool developments and litigation
>
> The patent landscape for neural codecs is rapidly evolving. The October 2025 InterDigital/Deep Render acquisition significantly changed the risk profile for AI-based video compression.

---

### 12.9 Key Sources and References

**Patent Pools and Licensing**:
- [Access Advance](https://accessadvance.com/) - HEVC Advance, VVC Advance, VDP Pool
- [Via LA](https://www.via-la.com/) - H.264/AVC, HEVC licensing
- [Avanci Video](https://www.avanci.com/video/) - Multi-codec streaming license
- [Sisvel](https://www.sisvel.com/) - AV1/VP9 patent pool

**Alliance and Standards**:
- [Alliance for Open Media](https://aomedia.org/) - AV1 development and licensing
- [MPEG](https://www.mpeg.org/) - Traditional codec standards
- [Streaming Media Global](https://www.streamingmediaglobal.com/) - Industry news and analysis

**Open Source Neural Codecs**:
- [CompressAI (InterDigital)](https://github.com/InterDigitalInc/CompressAI) - BSD 3-Clause Clear
- [DCVC (Microsoft)](https://github.com/microsoft/DCVC) - Multiple licenses

**Patent Research**:
- [Justia Patents - Deep Render](https://patents.justia.com/assignee/deep-render-ltd)
- [Wikimedia H.264 Patent Tracking](https://meta.wikimedia.org/wiki/Have_the_patents_for_H.264_MPEG-4_AVC_expired_yet%3F)

**News and Developments**:
- [InterDigital Deep Render Acquisition](https://ir.interdigital.com/news-events/press-releases/news-details/2025/InterDigital-acquires-AI-startup-Deep-Render/default.aspx) - October 30, 2025
- [Streaming Learning Center - AI Codec Standards](https://streaminglearningcenter.com/codecs/ai-video-compression-standards-whos-doing-what-and-when.html)

---

## 13. File Structure

### Proposed Module Organization

```
src/codec/zvc69/
├── mod.rs              # Module root, public API
├── encoder.rs          # ZVC69Encoder implementation
├── decoder.rs          # ZVC69Decoder implementation
├── model.rs            # Neural network model definitions
├── entropy.rs          # Arithmetic coding integration
├── bitstream.rs        # Bitstream read/write
├── motion.rs           # Motion estimation/compensation
├── residual.rs         # Residual coding
├── quantize.rs         # Quantization utilities
├── config.rs           # Configuration and presets
└── tests/
    ├── mod.rs
    ├── encoder_tests.rs
    ├── decoder_tests.rs
    └── bitstream_tests.rs
```

### Module Responsibilities

#### mod.rs
```rust
//! ZVC69 Neural Video Codec
//!
//! A next-generation neural video codec achieving 20%+ better
//! compression than AV1/H.265 with real-time performance.

pub mod encoder;
pub mod decoder;
pub mod model;
pub mod entropy;
pub mod bitstream;
pub mod motion;
pub mod residual;
pub mod quantize;
pub mod config;

pub use encoder::ZVC69Encoder;
pub use decoder::ZVC69Decoder;
pub use config::{ZVC69Config, Quality, Preset};
```

#### encoder.rs
```rust
//! ZVC69 Encoder
//!
//! Handles frame encoding pipeline:
//! - I-frame: analysis transform → quantize → entropy code
//! - P-frame: motion estimate → warp → residual → entropy code

pub struct ZVC69Encoder {
    config: ZVC69Config,
    model: EncoderModel,
    entropy_coder: ArithmeticEncoder,
    frame_buffer: FrameBuffer,
}

impl ZVC69Encoder {
    pub fn new(config: ZVC69Config) -> Result<Self>;
    pub fn encode_frame(&mut self, frame: &VideoFrame) -> Result<EncodedFrame>;
    pub fn flush(&mut self) -> Result<Vec<EncodedFrame>>;
}
```

#### decoder.rs
```rust
//! ZVC69 Decoder
//!
//! Handles frame decoding pipeline:
//! - I-frame: entropy decode → dequantize → synthesis transform
//! - P-frame: entropy decode → motion compensate → add residual

pub struct ZVC69Decoder {
    config: ZVC69Config,
    model: DecoderModel,
    entropy_decoder: ArithmeticDecoder,
    reference_frame: Option<VideoFrame>,
}

impl ZVC69Decoder {
    pub fn new() -> Result<Self>;
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<VideoFrame>;
}
```

#### model.rs
```rust
//! Neural Network Models
//!
//! ONNX model loading and inference using ort.

pub struct AnalysisTransform { /* ... */ }
pub struct SynthesisTransform { /* ... */ }
pub struct HyperpriorEncoder { /* ... */ }
pub struct HyperpriorDecoder { /* ... */ }
pub struct MotionEstimator { /* ... */ }
pub struct ResidualCodec { /* ... */ }

pub struct EncoderModel {
    analysis: AnalysisTransform,
    hyperprior_enc: HyperpriorEncoder,
    motion: MotionEstimator,
    residual: ResidualCodec,
}

pub struct DecoderModel {
    synthesis: SynthesisTransform,
    hyperprior_dec: HyperpriorDecoder,
    residual: ResidualCodec,
}
```

#### entropy.rs
```rust
//! Arithmetic Coding Integration
//!
//! Wraps arcode for latent encoding/decoding.

use arcode::{ArithmeticEncoder, ArithmeticDecoder};

pub fn encode_latent(
    latent: &Tensor,
    mean: &Tensor,
    scale: &Tensor,
) -> Result<Vec<u8>>;

pub fn decode_latent(
    data: &[u8],
    mean: &Tensor,
    scale: &Tensor,
    shape: &[usize],
) -> Result<Tensor>;
```

#### bitstream.rs
```rust
//! Bitstream Format
//!
//! ZVC69 bitstream reading and writing.

pub struct BitstreamWriter { /* ... */ }
pub struct BitstreamReader { /* ... */ }

pub struct FileHeader {
    pub version: u8,
    pub width: u16,
    pub height: u16,
    pub framerate: Rational,
    pub gop_size: u8,
    pub quality: u8,
}

pub struct FrameHeader {
    pub frame_type: FrameType,
    pub size: u32,
    pub pts: u64,
    pub dts: u64,
}

pub enum FrameType {
    I = 0,
    P = 1,
    B = 2,
}
```

---

## 14. Technical Deep Dives

### GDN (Generalized Divisive Normalization)

GDN is critical for neural image/video compression. It normalizes activations in a way that models human visual perception.

```python
class GDN(nn.Module):
    """Generalized Divisive Normalization"""

    def __init__(self, channels: int, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.eye(channels))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        _, C, _, _ = x.shape

        # Compute normalization denominator
        # beta + sum_j(gamma_ij * x_j^2)
        x_sq = x ** 2
        denom = self.beta + F.conv2d(
            x_sq,
            self.gamma.view(C, C, 1, 1)
        )
        denom = torch.sqrt(denom)

        if self.inverse:
            return x * denom  # IGDN
        else:
            return x / denom  # GDN
```

**Why GDN?**
- Models lateral inhibition in visual cortex
- Gaussianizes activations (better for entropy coding)
- Learnable parameters adapt to data

### Optical Flow Warping

Warping uses optical flow to predict the current frame from the previous frame.

```python
def warp(frame: Tensor, flow: Tensor) -> Tensor:
    """
    Warp frame using optical flow.

    Args:
        frame: (B, C, H, W) reference frame
        flow: (B, 2, H, W) optical flow (dx, dy)

    Returns:
        (B, C, H, W) warped frame
    """
    B, C, H, W = frame.shape

    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=frame.device),
        torch.arange(W, device=frame.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

    # Add flow to grid
    new_grid = grid + flow

    # Normalize to [-1, 1] for grid_sample
    new_grid[:, 0] = 2 * new_grid[:, 0] / (W - 1) - 1
    new_grid[:, 1] = 2 * new_grid[:, 1] / (H - 1) - 1

    # Permute for grid_sample: (B, H, W, 2)
    new_grid = new_grid.permute(0, 2, 3, 1)

    # Bilinear sampling
    warped = F.grid_sample(
        frame,
        new_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return warped
```

### Entropy Model Details

The entropy model predicts probability distributions for arithmetic coding.

```python
class GaussianConditional(nn.Module):
    """
    Gaussian entropy model with learned scale parameters.
    """

    def __init__(self, channels: int, scale_bound: float = 0.11):
        super().__init__()
        self.scale_bound = scale_bound

        # Learned scale parameters for tail mass
        self.scale_table = nn.Parameter(
            torch.exp(torch.linspace(-5, 5, 64))
        )

    def forward(self, y: Tensor, scales: Tensor):
        """
        Args:
            y: Latent tensor to encode
            scales: Predicted scales from hyperprior

        Returns:
            y_hat: Quantized latent
            likelihoods: Probability of each symbol (for rate loss)
        """
        # Clamp scales
        scales = torch.clamp(scales, min=self.scale_bound)

        # Add uniform noise during training (STE for quantization)
        if self.training:
            y_hat = y + torch.rand_like(y) - 0.5
        else:
            y_hat = torch.round(y)

        # Compute likelihoods
        # P(y_hat) = CDF(y_hat + 0.5) - CDF(y_hat - 0.5)
        gaussian = torch.distributions.Normal(0, scales)
        likelihood = gaussian.cdf(y_hat + 0.5) - gaussian.cdf(y_hat - 0.5)

        return y_hat, likelihood
```

### TensorRT Optimization

Converting PyTorch model to optimized TensorRT engine:

```python
import torch
import onnx
from onnxruntime.transformers import optimizer

# Step 1: Export to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 1080, 1920).cuda()

torch.onnx.export(
    model,
    dummy_input,
    "zvc69_encoder.onnx",
    input_names=['input'],
    output_names=['latent', 'hyperprior'],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'latent': {0: 'batch', 2: 'height', 3: 'width'},
    },
    opset_version=17,
)

# Step 2: Optimize ONNX
optimized = optimizer.optimize_model(
    "zvc69_encoder.onnx",
    model_type='bert',  # Use generic optimization
    opt_level=2,
)
optimized.save_model_to_file("zvc69_encoder_opt.onnx")
```

Loading in Rust with ort:

```rust
use ort::{Environment, SessionBuilder, GraphOptimizationLevel};

let environment = Environment::builder()
    .with_name("ZVC69")
    .with_execution_providers([
        ort::CUDAExecutionProvider::default().build(),
        ort::TensorrtExecutionProvider::default()
            .with_fp16(true)
            .build(),
    ])
    .build()?;

let session = SessionBuilder::new(&environment)?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_model_from_file("zvc69_encoder_opt.onnx")?;

// Run inference
let outputs = session.run(ort::inputs!["input" => input_tensor]?)?;
```

---

## 15. Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Performance not meeting targets | Medium | High | Start with DCVC-RT baseline, optimize iteratively |
| Training instability | Medium | Medium | Use CompressAI pretrained models, careful hyperparameters |
| ONNX export issues | Low | Medium | Use TorchScript as fallback |
| Memory constraints | Medium | Medium | Quantization, model pruning |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Patent infringement | Medium | Very High | FTO analysis, novel architecture |
| Competition (open source) | High | Low | Focus on Rust integration, not beating Google |
| Ecosystem adoption | Medium | Medium | Good documentation, easy integration |

### Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Training takes longer than expected | Medium | High | Use pretrained models, cloud compute burst |
| Optimization phase overruns | Medium | Medium | Ship MVP first, optimize later |
| Integration issues with ZVD | Low | Medium | Design API early, validate integration points |

### Contingency Plans

1. **If performance targets not met**:
   - Lower resolution targets (720p instead of 1080p)
   - Increase latency budget
   - Use more powerful GPU as baseline

2. **If patent issues arise**:
   - Redesign problematic components
   - License necessary patents
   - Restrict to non-commercial use

3. **If training is too slow**:
   - Use pretrained CompressAI models
   - Reduce model size
   - Focus on inference optimization only

---

## Appendix A: Reference BD-Rate Results

### DCVC-RT vs Traditional Codecs

From Microsoft DCVC-RT paper:

| Sequence | vs VVC | vs HEVC | vs AV1 |
|----------|--------|---------|--------|
| UVG | -21.3% | -42.1% | -23.8% |
| MCL-JCV | -18.7% | -38.5% | -20.2% |
| HEVC-B | -19.8% | -40.3% | -21.5% |
| **Average** | **-19.9%** | **-40.3%** | **-21.8%** |

### ZVC69 Target Performance

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|----------------|----------------|----------------|
| vs AV1 | -5% | -15% | -20% |
| 1080p30 encode | 15 fps | 25 fps | 35 fps |
| 1080p30 decode | 30 fps | 50 fps | 70 fps |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **BD-rate** | Bjontegaard Delta Rate - measures bitrate savings at same quality |
| **bpp** | Bits per pixel |
| **GDN** | Generalized Divisive Normalization |
| **GOP** | Group of Pictures (I-frame interval) |
| **Hyperprior** | Side information that improves entropy model |
| **Latent** | Compressed representation in neural codec |
| **MS-SSIM** | Multi-Scale Structural Similarity Index |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) |
| **Rate-Distortion** | Tradeoff between bitrate and quality |
| **VMAF** | Video Multi-Method Assessment Fusion (perceptual metric) |

---

## Appendix C: Quick Start Checklist

- [ ] Set up PyTorch + CompressAI environment
- [ ] Download Vimeo-90K dataset
- [ ] Train basic image codec (hyperprior model)
- [ ] Export to ONNX
- [ ] Integrate with ort in Rust
- [ ] Implement arithmetic coding with arcode
- [ ] Build I-frame pipeline
- [ ] Add motion estimation network
- [ ] Implement P-frame pipeline
- [ ] Optimize with TensorRT
- [ ] Benchmark against AV1

---

*Document Version: 2.0*
*Status: Research Complete*
*Last Updated: December 9, 2025*
*Authors: ZVD Team*

---

## Appendix D: Research Sources (December 2025 Update)

### Section 3 & 7 Sources (Architecture & Training - December 2025)

**Neural Video Codec Architectures:**
- [DCVC-RT CVPR 2025 Paper](https://arxiv.org/abs/2502.20762) - "Towards Practical Real-Time Neural Video Compression"
- [DCVC-RT Project Page](https://dcvccodec.github.io/)
- [UI2C: Unified Intra-Inter Coding](https://arxiv.org/html/2510.14431) - Outperforms DCVC-RT by 12.1% BD-rate
- [NVRC-Lite 2025](https://arxiv.org/html/2512.04019) - INR-based codec with 21-23% BD-rate savings vs C3
- [GIViC ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_GIViC_Generative_Implicit_Video_Compression_ICCV_2025_paper.pdf)
- [Learned Video Compression Survey IJCAI 2025](https://dl.acm.org/doi/10.24963/ijcai.2025/1165)

**Transformer-Based Video Compression:**
- [Video Compression Transformer (VCT)](https://www.emergentmind.com/topics/video-compression-transformer-vct)
- [NeuralMDC with Masked Transformer](https://arxiv.org/html/2412.07922)
- [DCVC-FM Feature Modulation](https://arxiv.org/html/2402.17414v1)

**Diffusion-Based Video Compression:**
- [Extreme Video Compression with Diffusion](https://arxiv.org/html/2402.08934v1)
- [OneDC One-Step Diffusion Codec](https://onedc-codec.github.io/)
- [GLVC Generative Latent Video Compression](https://www.researchgate.net/publication/396457949_Generative_Latent_Video_Compression)
- [Conditional Video Generation for Compression](https://arxiv.org/html/2507.15269v1)
- [DIRAC Diffusion-based Residual Augmentation Codec](https://arxiv.org/abs/2301.05489)

**Training Datasets:**
- [Vimeo-90K Dataset](http://toflow.csail.mit.edu/)
- [BVI-DVC Training Database](https://www.researchgate.net/publication/354296087_BVI-DVC_A_Training_Database_for_Deep_Video_Compression)
- [I2VC Intra-Inter Framework](https://arxiv.org/html/2405.14336v1)
- [Instance-Adaptive Video Compression](https://arxiv.org/html/2111.10302)

**Loss Functions & Perceptual Quality:**
- [LPIPS Perceptual Similarity](https://github.com/richzhang/PerceptualSimilarity)
- [Controllable Distortion-Perception Tradeoff](https://arxiv.org/html/2412.11379)
- [Fidelity-preserving Learned Image Compression](https://arxiv.org/html/2403.11241)
- [Exploiting Latent Properties for Neural Codecs](https://arxiv.org/html/2501.01231)

**Hardware & Training:**
- [NVIDIA GPU Use Cases Guide 2025](https://www.simcentric.com/hong-kong-dedicated-server/nvidia-gpu-use-cases-ultimate-classification-guide-2025/)
- [H100 vs A100 Comparison](https://gcore.com/blog/nvidia-h100-a100)
- [NVIDIA Blackwell vs Hopper](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)

**CompressAI & Training Frameworks:**
- [CompressAI GitHub](https://github.com/InterDigitalInc/CompressAI)
- [CompressAI PyPI](https://pypi.org/project/compressai/)
- [CompressAI-Vision](https://github.com/InterDigitalInc/CompressAI-Vision)
- [SSF2020 Scale-Space Flow Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.pdf)

**Transfer Learning & Domain Adaptation:**
- [Predictive Coding for Domain Adaptation](https://arxiv.org/abs/2509.20269)
- [Parameter-Efficient Fine-Tuning Survey](https://arxiv.org/html/2402.02242v6)
- [SlimVC Slimmable Video Codec](https://www.sciencedirect.com/science/article/pii/S0925231224012967)
- [Learned Image/Video Compression Collection](https://github.com/cshw2021/Learned-Image-Video-Compression)

### Rust ML Framework Sources (Section 5)

**ort (ONNX Runtime Rust Bindings)**:
- [ort Documentation](https://ort.pyke.io/)
- [ort GitHub Repository](https://github.com/pykeio/ort)
- [ort crates.io](https://crates.io/crates/ort)

**candle (Hugging Face)**:
- [candle GitHub Repository](https://github.com/huggingface/candle)
- [candle Documentation](https://huggingface.github.io/candle/)
- [candle Changelog](https://github.com/huggingface/candle/blob/main/CHANGELOG.md)
- [Candle: A Minimalist Rust ML Framework](https://www.blog.brightcoding.dev/2025/09/29/candle-a-minimalist-rust-ml-framework-with-fast-demos-like-whisper-and-llama2/)

**burn**:
- [burn Official Website](https://burn.dev/)
- [burn GitHub Repository](https://github.com/tracel-ai/burn)
- [burn 0.19.0 Release Blog](https://burn.dev/blog/release-0.19.0/)
- [burn Cross-Platform GPU Backend](https://burn.dev/blog/cross-platform-gpu-backend/)

**tract (Sonos)**:
- [tract GitHub Repository](https://github.com/sonos/tract)
- [tract-onnx crates.io](https://crates.io/crates/tract-onnx)

**tch-rs (PyTorch Bindings)**:
- [tch-rs GitHub Repository](https://github.com/LaurentMazare/tch-rs)

**Rust GPU Ecosystem**:
- [Rust-GPU GitHub](https://github.com/Rust-GPU/rust-gpu)
- [Rust-GPU Blog](https://rust-gpu.github.io/blog/)
- [Rust CUDA Project Reboot](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot/)
- [cudarc GitHub](https://github.com/coreylowman/cudarc)
- [wonnx GitHub](https://github.com/webonnx/wonnx)
- [dfdx GitHub](https://github.com/coreylowman/dfdx)

**General Rust ML Resources**:
- [Rust for Machine Learning 2025 Comparison](https://markaicode.com/rust-machine-learning-framework-comparison-2025/)
- [Rust AI Libraries Comparison](https://markaicode.com/rust-ai-libraries-comparison/)
- [Are We Learning Yet?](https://www.arewelearningyet.com/)
- [Awesome Rust Machine Learning](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning)

### Primary Sources

**DCVC-RT / Neural Video Codecs:**
- [Microsoft DCVC GitHub](https://github.com/microsoft/DCVC)
- [DCVC-RT CVPR 2025 Paper](https://arxiv.org/abs/2502.20762)
- [DCVC-RT Project Page](https://dcvccodec.github.io/)
- [Streaming Learning Center: DCVC-RT Evaluation](https://streaminglearningcenter.com/articles/evaluating-dcvc-rt-a-real-time-neural-video-codec-that-delivers-on-speed-and-compression.html)

**Google DeepMind C3:**
- [C3 GitHub Repository](https://github.com/google-deepmind/c3_neural_compression)
- [C3 Project Page](https://c3-neural-compression.github.io/)

**AV2 Development:**
- [AOMedia AV2 Announcement](https://aomedia.org/press%20releases/AOMedia-Announces-Year-End-Launch-of-Next-Generation-Video-Codec-AV2-on-10th-Anniversary/)
- [CNX Software: AV2 Release Details](https://www.cnx-software.com/2025/11/21/aomedia-av2-open-video-codec-release-nears-delivers-around-40-bandwidth-reduction/)

**Industry Deployments:**
- [Netflix AV1 Adoption (December 2025)](https://www.broadbandtvnews.com/2025/12/05/netflix-says-av1-now-powers-30-of-its-streaming/)
- [Streaming Media: State of Video Codec Market 2025](https://www.streamingmedia.com/Articles/Editorial/Featured-Articles/The-State-of-the-Video-Codec-Market-2025-168628.aspx)

**Standardization:**
- [MPEG Workshop on Future Video Coding (January 2025)](https://www.mpeg.org/workshop-on-future-video-coding-advanced-signal-processing-ai-and-standards/)
- [AI Video Compression Standards Overview](https://streaminglearningcenter.com/codecs/ai-video-compression-standards-whos-doing-what-and-when.html)

**Hardware:**
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)
- [Intel Media Capabilities](https://www.intel.com/content/www/us/en/docs/onevpl/developer-reference-media-intel-hardware/1-1/overview.html)

---

## 16. Quick Start Implementation Guide

> This section provides step-by-step instructions for getting started with ZVC69 development.

### 16.1 Prerequisites

#### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | GTX 1060 6GB | RTX 3080+ 10GB |
| **VRAM** | 6 GB | 12+ GB |
| **RAM** | 16 GB | 32+ GB |
| **Storage** | 100 GB SSD | 500 GB NVMe |
| **CPU** | 4 cores | 8+ cores |

#### Software Requirements

```bash
# Core tools
rustc >= 1.75.0
cargo >= 1.75.0
python >= 3.10
cuda >= 11.8 (12.x recommended)
cudnn >= 8.6

# Verify CUDA installation
nvidia-smi
nvcc --version
```

### 16.2 Environment Setup

#### Step 1: Create Python Training Environment

```bash
# Create isolated environment
python -m venv zvc69_train
source zvc69_train/bin/activate  # Linux/Mac
# or: zvc69_train\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install CompressAI and dependencies
pip install compressai==1.2.8
pip install lpips pytorch-msssim tensorboard

# Install DCVC-RT (for reference architecture)
git clone https://github.com/microsoft/DCVC.git
cd DCVC && pip install -e .
```

#### Step 2: Create Rust Project Structure

```bash
# Initialize ZVC69 codec crate
cargo new --lib zvc69
cd zvc69

# Add dependencies to Cargo.toml
cat >> Cargo.toml << 'EOF'

[dependencies]
# Neural inference
ort = { version = "2.0.0-rc.10", features = ["cuda", "tensorrt"] }

# Entropy coding
constriction = "0.4"

# Tensor operations
ndarray = "0.16"

# Image/Video handling
image = "0.25"

# Utilities
thiserror = "1.0"
anyhow = "1.0"
byteorder = "1.5"

[dev-dependencies]
criterion = "0.5"
EOF
```

#### Step 3: Download Training Data

```bash
# Download Vimeo-90K Septuplet (primary training dataset)
mkdir -p data/vimeo90k
cd data/vimeo90k
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
unzip vimeo_septuplet.zip

# Download test sequences (UVG)
mkdir -p data/uvg
cd data/uvg
# Download from https://ultravideo.fi/#testsequences
```

### 16.3 Minimal Viable Implementation

#### Phase 1A: Image Codec (I-Frame) - Week 1

The minimal implementation uses a pretrained CompressAI model exported to ONNX:

```python
# train/export_baseline.py
"""Export CompressAI baseline model to ONNX for Rust inference."""
import torch
from compressai.zoo import cheng2020_anchor

def export_image_codec():
    # Load pretrained model (quality level 3 = medium)
    model = cheng2020_anchor(quality=3, pretrained=True)
    model.eval()

    # Create dummy input (batch=1, channels=3, height=256, width=256)
    # Note: actual resolution is variable, but export with representative size
    dummy_input = torch.randn(1, 3, 256, 256)

    # Export encoder (analysis transform + hyperprior encoder)
    torch.onnx.export(
        model.g_a,  # Analysis transform
        dummy_input,
        "models/encoder.onnx",
        input_names=["input"],
        output_names=["latent"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "latent": {0: "batch", 2: "h16", 3: "w16"}},
        opset_version=17,
    )

    # Export decoder (synthesis transform)
    latent_shape = (1, 192, 16, 16)  # 256/16 = 16
    dummy_latent = torch.randn(latent_shape)

    torch.onnx.export(
        model.g_s,  # Synthesis transform
        dummy_latent,
        "models/decoder.onnx",
        input_names=["latent"],
        output_names=["output"],
        dynamic_axes={"latent": {0: "batch", 2: "h16", 3: "w16"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
    )

    print("Exported encoder.onnx and decoder.onnx")
    return model

if __name__ == "__main__":
    export_image_codec()
```

#### Phase 1B: Rust Inference - Week 2

```rust
// src/lib.rs
//! ZVC69 Neural Video Codec

pub mod encoder;
pub mod decoder;
pub mod entropy;
pub mod bitstream;

pub use encoder::ZVC69Encoder;
pub use decoder::ZVC69Decoder;
```

```rust
// src/encoder.rs
//! ZVC69 Encoder implementation

use ort::{Session, SessionBuilder, GraphOptimizationLevel};
use ndarray::{Array4, ArrayView4};
use anyhow::Result;

pub struct ZVC69Encoder {
    analysis_session: Session,
    hyperprior_session: Session,
}

impl ZVC69Encoder {
    pub fn new(model_path: &str) -> Result<Self> {
        let analysis_session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(format!("{}/encoder.onnx", model_path))?;

        let hyperprior_session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(format!("{}/hyperprior_enc.onnx", model_path))?;

        Ok(Self {
            analysis_session,
            hyperprior_session,
        })
    }

    pub fn encode_frame(&self, frame: ArrayView4<f32>) -> Result<EncodedFrame> {
        // Run analysis transform
        let inputs = ort::inputs!["input" => frame.to_owned()]?;
        let outputs = self.analysis_session.run(inputs)?;
        let latent = outputs["latent"].extract_tensor::<f32>()?;

        // Quantize latent
        let quantized = quantize_latent(&latent.view());

        // Run hyperprior encoder for entropy model parameters
        let hp_inputs = ort::inputs!["latent" => latent.to_owned()]?;
        let hp_outputs = self.hyperprior_session.run(hp_inputs)?;

        // Extract mean and scale for entropy coding
        let mean = hp_outputs["mean"].extract_tensor::<f32>()?;
        let scale = hp_outputs["scale"].extract_tensor::<f32>()?;

        // Entropy encode using constriction
        let bitstream = entropy_encode(&quantized, &mean, &scale)?;

        Ok(EncodedFrame {
            frame_type: FrameType::I,
            data: bitstream,
        })
    }
}

fn quantize_latent(latent: &ArrayView4<f32>) -> Array4<i16> {
    latent.mapv(|x| x.round() as i16)
}
```

### 16.4 First Milestone Targets

#### Milestone 1: I-Frame Codec (End of Week 2)

| Metric | Target | Validation |
|--------|--------|------------|
| PSNR @ 0.25 bpp | > 32 dB | Test on Kodak dataset |
| Encode time (1080p) | < 100 ms | Single RTX 3060 |
| Decode time (1080p) | < 50 ms | Single RTX 3060 |
| Model size | < 100 MB | Combined encoder + decoder |

#### Milestone 2: P-Frame Codec (End of Week 6)

| Metric | Target | Validation |
|--------|--------|------------|
| BD-rate vs AV1 | > -10% | Test on UVG sequences |
| GOP handling | I + 9P | 10-frame GOP |
| Encode fps (1080p30) | > 20 fps | RTX 3080 |

#### Milestone 3: Real-Time (End of Week 10)

| Metric | Target | Validation |
|--------|--------|------------|
| BD-rate vs AV1 | > -20% | Full UVG + MCL-JCV |
| Encode fps (1080p30) | > 35 fps | RTX 3080 |
| Decode fps (1080p60) | > 70 fps | RTX 3080 |

### 16.5 Dependencies Installation Summary

```toml
# Cargo.toml - Complete dependencies
[package]
name = "zvc69"
version = "0.1.0"
edition = "2021"

[dependencies]
# Neural inference (choose one primary)
ort = { version = "2.0.0-rc.10", features = ["cuda", "tensorrt"] }

# Entropy coding
constriction = "0.4"

# Tensor operations
ndarray = { version = "0.16", features = ["rayon"] }

# Image handling
image = "0.25"

# Video frame handling (optional, for testing)
# ffmpeg-next = "7.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# I/O
byteorder = "1.5"

# Parallelism
rayon = "1.10"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"

[features]
default = ["cuda"]
cuda = ["ort/cuda"]
tensorrt = ["ort/tensorrt"]
```

```bash
# Python requirements.txt for training
torch>=2.0.0
torchvision>=0.15.0
compressai>=1.2.8
lpips>=0.1.4
pytorch-msssim>=1.0.0
tensorboard>=2.12.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
onnx>=1.14.0
onnxruntime-gpu>=1.15.0
```

---

## 17. Development Roadmap

> Detailed timeline for ZVC69 development from research to production MVP.

### 17.1 Phase Overview

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 1** | 2-3 weeks | Image Codec (I-frames) | Working I-frame encoder/decoder |
| **Phase 2** | 3-4 weeks | Video Codec (P-frames) | Temporal prediction, full GOP |
| **Phase 3** | 2-3 weeks | Optimization | TensorRT, real-time performance |
| **Phase 4** | 2-3 weeks | Production Hardening | API polish, rate control, testing |
| **Total** | **10-13 weeks** | | **Production MVP** |

### 17.2 Phase 1: Image Codec (I-Frames) - Weeks 1-3

#### Week 1: Foundation

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Set up development environment | Python training env + Rust project |
| 3-4 | Export CompressAI baseline to ONNX | encoder.onnx, decoder.onnx, hyperprior.onnx |
| 5 | Implement basic Rust inference with ort | Load and run ONNX models |

**Deliverables:**
- Working ONNX export pipeline
- Basic Rust inference skeleton
- Environment documentation

#### Week 2: Entropy Coding Integration

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Implement entropy coding with constriction | Encode/decode latents to bitstream |
| 3-4 | Implement bitstream format (file header, frame header) | ZVC69 file format v1 |
| 5 | End-to-end I-frame encode/decode | Complete I-frame pipeline |

**Deliverables:**
- Working entropy encoder/decoder
- ZVC69 bitstream format implementation
- I-frame roundtrip test passing

#### Week 3: I-Frame Quality & Testing

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Benchmark against JPEG, WebP, AVIF | BD-rate curves |
| 3-4 | Implement quality metrics (PSNR, MS-SSIM) | Automated quality testing |
| 5 | Documentation and code cleanup | Phase 1 complete |

**Success Criteria:**
- [ ] PSNR > 32 dB at 0.25 bpp on Kodak
- [ ] Encode 1080p < 100ms on RTX 3060
- [ ] Decode 1080p < 50ms on RTX 3060
- [ ] Bitstream format documented

### 17.3 Phase 2: Video Codec (P-Frames) - Weeks 4-7

#### Week 4: Temporal Context

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Study DCVC-RT implicit motion approach | Architecture design document |
| 3-4 | Export temporal context model to ONNX | temporal_context.onnx |
| 5 | Implement reference frame buffer | Frame management in Rust |

**Deliverables:**
- Temporal context network exported
- Reference frame buffer implementation
- P-frame architecture design

#### Week 5: P-Frame Pipeline

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Implement P-frame encoder | Temporal prediction + residual |
| 3-4 | Implement P-frame decoder | Temporal reconstruction |
| 5 | GOP structure handling | I-P-P-P... sequences |

**Deliverables:**
- Working P-frame encoder/decoder
- GOP management (configurable I-frame interval)
- P-frame roundtrip test passing

#### Week 6: Residual Coding

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Implement residual analysis network | Residual encoder |
| 3-4 | Implement residual synthesis network | Residual decoder |
| 5 | Integrate with P-frame pipeline | Complete P-frame codec |

**Deliverables:**
- Residual coding network integrated
- Reduced bitrate for P-frames
- Full GOP encode/decode working

#### Week 7: Video Quality & Benchmarking

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Benchmark on UVG dataset | BD-rate vs AV1, HEVC |
| 3-4 | Implement VMAF scoring | Perceptual quality metrics |
| 5 | Address error propagation issues | Quality stabilization |

**Success Criteria:**
- [ ] BD-rate > -10% vs AV1 on UVG
- [ ] 1080p30 encoding > 15 fps on RTX 3080
- [ ] No visible error accumulation over 10+ frames
- [ ] VMAF > 80 at 2 Mbps 1080p

### 17.4 Phase 3: Optimization - Weeks 8-10

#### Week 8: TensorRT Integration

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Convert ONNX models to TensorRT engines | FP16 optimized engines |
| 3-4 | Implement dynamic batch/resolution | Flexible inference |
| 5 | Benchmark TensorRT vs vanilla ONNX | Performance comparison |

**Deliverables:**
- TensorRT-optimized inference
- 2-3x speedup over vanilla ONNX
- FP16 inference validated

#### Week 9: Memory & Latency Optimization

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Implement CUDA graphs for low latency | Reduced kernel overhead |
| 3-4 | Memory pre-allocation and pooling | Consistent memory usage |
| 5 | Pipeline optimization (overlap CPU/GPU) | Improved throughput |

**Deliverables:**
- CUDA graph capture for hot paths
- Memory allocation optimized
- Async pipeline implementation

#### Week 10: Performance Targets

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Final performance tuning | Meet encode/decode targets |
| 3-4 | Multi-resolution support (720p, 1080p, 4K) | Resolution flexibility |
| 5 | Comprehensive benchmarking | Performance documentation |

**Success Criteria:**
- [ ] 1080p30 encoding > 35 fps (RTX 3080)
- [ ] 1080p60 decoding > 70 fps (RTX 3080)
- [ ] < 50ms single-frame latency
- [ ] < 1 GB GPU memory usage

### 17.5 Phase 4: Production Hardening - Weeks 11-13

#### Week 11: Rate Control

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Implement CRF (Constant Rate Factor) mode | Quality-targeted encoding |
| 3-4 | Implement CBR (Constant Bitrate) mode | Bandwidth-constrained encoding |
| 5 | Implement VBR (Variable Bitrate) mode | Optimal quality distribution |

**Deliverables:**
- Three rate control modes
- Bitrate targeting accuracy < 5% error
- Quality consistency across modes

#### Week 12: API & Integration

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Design and implement public API | Clean, documented API |
| 3-4 | Integrate with ZVD main crate | Codec registry integration |
| 5 | Implement quality presets (ultrafast, fast, medium, slow) | User-friendly presets |

**Deliverables:**
- Stable public API
- ZVD integration complete
- Quality presets implemented

#### Week 13: Testing & Documentation

| Day | Task | Outcome |
|-----|------|---------|
| 1-2 | Comprehensive test suite | Unit + integration tests |
| 3-4 | Fuzz testing for bitstream robustness | Security hardening |
| 5 | Final documentation and examples | Production-ready docs |

**Success Criteria:**
- [ ] Test coverage > 80%
- [ ] No crashes on malformed bitstreams
- [ ] API documentation complete
- [ ] Example applications working

### 17.6 Risk Mitigation Timeline

| Risk | Detection Point | Mitigation |
|------|-----------------|------------|
| Model export issues | Week 1-2 | Fallback to TorchScript |
| Performance not meeting targets | Week 8 | Reduce model complexity, lower resolution targets |
| Patent concerns identified | Any phase | Redesign affected components |
| Training divergence | Pre-Phase 1 | Use pretrained CompressAI models only |

### 17.7 Post-MVP Roadmap (Months 4-6)

| Feature | Priority | Estimated Time |
|---------|----------|----------------|
| B-frame support | High | 3-4 weeks |
| INT8 quantization | High | 2 weeks |
| WebGPU/WASM support | Medium | 4-6 weeks |
| Custom training pipeline | Medium | 4-6 weeks |
| Hardware encoder integration | Low | 4-6 weeks |
| Adaptive streaming (HLS/DASH) | Low | 2-3 weeks |

---

## 18. Team Requirements

> Skills and team composition recommendations for ZVC69 development.

### 18.1 Required Skills

#### Core Technical Skills

| Skill | Importance | Description |
|-------|------------|-------------|
| **Rust systems programming** | Critical | Memory management, unsafe code, FFI |
| **Deep learning fundamentals** | Critical | CNN architectures, training, optimization |
| **Video codec concepts** | High | I/P/B frames, motion estimation, entropy coding |
| **GPU programming (CUDA)** | High | Kernel optimization, memory management |
| **PyTorch** | High | Model training, ONNX export |
| **Signal processing** | Medium | DCT, wavelets, quantization theory |
| **Performance optimization** | Medium | Profiling, benchmarking, low-latency systems |

#### Domain Knowledge

| Area | Importance | Notes |
|------|------------|-------|
| Neural image compression | Critical | Hyperprior models, GDN, entropy coding |
| Traditional video codecs | High | H.264/H.265 architecture understanding |
| Rate-distortion optimization | High | Loss function design, Lagrange multipliers |
| Arithmetic/ANS coding | High | Entropy coding implementation |
| ONNX ecosystem | Medium | Model export, runtime optimization |
| TensorRT | Medium | FP16/INT8 optimization |

### 18.2 Recommended Team Composition

#### Minimum Viable Team (2 people)

| Role | Responsibilities | Time Allocation |
|------|------------------|-----------------|
| **Lead Engineer** | Architecture, Rust implementation, GPU optimization | 100% |
| **ML Engineer** | Model training, ONNX export, quality benchmarking | 100% |

**Pros:** Low coordination overhead, fast iteration
**Cons:** Single points of failure, limited parallelization

#### Optimal Team (4 people)

| Role | Responsibilities | Time Allocation |
|------|------------------|-----------------|
| **Tech Lead** | Architecture, code review, technical decisions | 80% |
| **Rust Engineer** | Core codec implementation, bitstream format | 100% |
| **ML Engineer** | Model architecture, training, export pipeline | 100% |
| **Performance Engineer** | TensorRT optimization, benchmarking, profiling | 100% |

**Pros:** Parallel development, specialized expertise
**Cons:** Coordination overhead, higher cost

#### Extended Team (6+ people)

Add these roles for faster development or additional features:

| Role | Responsibilities |
|------|------------------|
| **QA Engineer** | Test suite development, fuzzing, CI/CD |
| **Documentation Writer** | API docs, tutorials, examples |
| **DevOps Engineer** | Build systems, deployment, cloud infrastructure |

### 18.3 Solo Developer Path

For a single developer with all required skills, the following approach is recommended:

#### Week-by-Week Focus

```
Weeks 1-3:   Focus entirely on I-frame codec
             Use pretrained CompressAI models (no custom training)

Weeks 4-7:   Focus on P-frame implementation
             Use DCVC-RT pretrained models if available

Weeks 8-10:  Focus on TensorRT optimization
             Accept lower performance targets if needed

Weeks 11-13: Focus on API and documentation
             Prioritize stability over features
```

#### Time Estimates (Solo)

| Phase | Team of 4 | Solo Developer |
|-------|-----------|----------------|
| Phase 1: I-Frame | 2-3 weeks | 3-4 weeks |
| Phase 2: P-Frame | 3-4 weeks | 5-6 weeks |
| Phase 3: Optimization | 2-3 weeks | 3-4 weeks |
| Phase 4: Hardening | 2-3 weeks | 3-4 weeks |
| **Total** | **10-13 weeks** | **14-18 weeks** |

#### Solo Developer Shortcuts

1. **Use pretrained models only** - Skip custom training entirely
2. **Target single resolution** - 1080p only, no dynamic resolution
3. **Skip B-frames** - I-P-P-P pattern only
4. **Single rate control mode** - CRF only, skip CBR/VBR
5. **Minimal API** - Focus on core encode/decode, skip presets

### 18.4 Hiring Recommendations

#### Where to Find Candidates

| Skill | Sources |
|-------|---------|
| Rust experts | Rust community, Ferrous Systems alumni, Mozilla alumni |
| Neural compression | CVPR/NeurIPS compression workshop attendees, InterDigital/Google alumni |
| Video codec experience | Netflix, YouTube, Twitch engineering teams |
| GPU optimization | NVIDIA Developer Program, game engine companies |

#### Interview Focus Areas

1. **Rust proficiency test**: Implement a simple entropy coder
2. **ML understanding**: Explain rate-distortion tradeoff, hyperprior models
3. **Systems thinking**: Discuss latency vs throughput optimization
4. **Video codec knowledge**: Explain I/P/B frame dependencies

### 18.5 External Resources

#### Consultants/Contractors

| Area | When to Engage |
|------|----------------|
| Patent attorney | Before commercial deployment (~$50-150K) |
| Video codec consultant | Architecture review (one-time) |
| TensorRT specialist | Optimization phase (2-4 weeks) |
| Security auditor | Pre-production (1-2 weeks) |

#### Training Resources

| Resource | Use Case |
|----------|----------|
| NVIDIA Deep Learning Institute | TensorRT optimization courses |
| CompressAI tutorials | Neural compression fundamentals |
| Rust async book | Tokio/async patterns |
| DCVC-RT paper/code | Reference architecture study |

---

## 19. Conclusion

### 19.1 Why ZVC69 is Viable

After comprehensive research into the state of neural video compression as of December 2025, ZVC69 represents a viable and timely project for the following reasons:

#### Technical Viability

1. **Proven Performance**: Microsoft's DCVC-RT demonstrates that neural codecs can achieve 21% better compression than VVC while running at 125 fps on modern GPUs. This is not theoretical - it is published, benchmarked, and open-source (Apache 2.0).

2. **Mature Rust Ecosystem**: The Rust ML ecosystem has reached production quality:
   - `ort` 2.0 provides battle-tested ONNX Runtime bindings with TensorRT support
   - `constriction` offers purpose-built entropy coding for neural compression
   - `burn` 0.19 and `candle` 0.9 provide pure-Rust alternatives for edge deployment

3. **GPU Ubiquity**: NVIDIA GPUs with Tensor Cores are now standard in gaming PCs, cloud instances, and professional workstations. The hardware to run neural codecs at real-time speeds is already deployed at scale.

#### Market Timing

1. **Traditional Codec Stagnation**: VVC offers only incremental improvements over HEVC, and H.267 won't deploy until 2034-2036. Neural codecs can leapfrog traditional approaches now.

2. **AV2 Gap**: AOMedia's AV2 is targeting late 2025 release, but widespread deployment will take 2-3 years. This creates a window for neural codec adoption.

3. **Streaming Demand**: Netflix reports AV1 now powers 30% of its streaming (December 2025). The industry is actively seeking better compression, and neural codecs offer the next leap.

### 19.2 Key Technical Differentiators

ZVC69's architecture provides several advantages over existing solutions:

| Differentiator | Benefit |
|----------------|---------|
| **Pure Rust implementation** | Memory safety, no GC pauses, single-binary deployment |
| **ONNX Runtime + TensorRT** | Maximum GPU utilization, cross-platform support |
| **constriction entropy coding** | Native Rust, optimized for neural compression |
| **Implicit temporal modeling** | Simpler architecture than explicit motion estimation |
| **Apache 2.0 licensing** | Explicit patent grant, commercial-friendly |

### 19.3 Risk Summary

| Risk | Severity | Mitigation Status |
|------|----------|-------------------|
| **InterDigital/Deep Render patents** | High | Requires FTO analysis before commercial deployment |
| **Performance targets** | Medium | DCVC-RT proves targets are achievable |
| **Rust ecosystem maturity** | Low | ort/burn/candle are production-ready |
| **Training complexity** | Low | Using pretrained CompressAI/DCVC models |

### 19.4 Expected Outcomes

With the proposed 10-13 week development timeline:

| Outcome | Confidence | Validation |
|---------|------------|------------|
| 20%+ bitrate savings vs AV1 | High | DCVC-RT achieves 24% vs AV1 |
| Real-time 1080p30 encoding | High | DCVC-RT achieves 125 fps on A100 |
| Real-time 1080p60 decoding | High | DCVC-RT achieves 113 fps on A100 |
| Production-quality API | Medium | Depends on team/timeline |
| Patent-safe deployment | Medium | Requires legal review |

### 19.5 Next Steps

#### Immediate Actions (Week 0)

1. **Environment Setup**
   - Install CUDA 12.x, cuDNN 8.x
   - Set up Python training environment with CompressAI
   - Initialize Rust project with ort dependencies

2. **Model Acquisition**
   - Download CompressAI pretrained models (cheng2020-anchor)
   - Clone DCVC repository for reference architecture
   - Export baseline models to ONNX

3. **Team Assembly** (if applicable)
   - Identify ML engineer with compression experience
   - Secure access to training GPUs (4x RTX 4090 or cloud equivalent)

#### Decision Points

| Decision | Timeline | Options |
|----------|----------|---------|
| Training approach | Week 1 | Pretrained only vs. custom training |
| Primary ML framework | Week 1 | ort (recommended) vs. burn vs. candle |
| Target GPU tier | Week 1 | RTX 3060 (consumer) vs. RTX 4090 (prosumer) |
| Patent strategy | Pre-commercial | FTO analysis vs. defensive publication |

### 19.6 Final Assessment

ZVC69 is not just viable - it represents a strategic opportunity. The convergence of:

- **Breakthrough neural codec research** (DCVC-RT, 2025)
- **Mature Rust ML ecosystem** (ort 2.0, burn 0.19)
- **GPU hardware ubiquity** (Tensor Cores in consumer GPUs)
- **Traditional codec stagnation** (VVC deployment slow, H.267 decade away)

...creates a unique window for a new video codec that can deliver meaningful compression improvements while the industry waits for the next generation of traditional standards.

The research is complete. The technology is proven. The path forward is clear.

**ZVC69: The future of video compression, written in Rust.**

---

*Document Version: 2.0*
*Status: Research Complete*
*Last Updated: December 9, 2025*
*Authors: ZVD Team*

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-01 | Initial research document |
| 1.1 | 2025-11-15 | Added Section 5 (Rust ML Frameworks) |
| 1.2 | 2025-11-20 | Added Section 10 (Performance Optimization) |
| 1.3 | 2025-12-01 | Updated Section 2 (State of the Art) |
| 1.4 | 2025-12-05 | Added Section 12 (Patent Considerations) |
| **2.0** | **2025-12-09** | **Research Complete: Added Executive Summary update, Quick Start Guide (Section 16), Development Roadmap (Section 17), Team Requirements (Section 18), Conclusion (Section 19)** |
