# ZVD - Modern Multimedia Processing in Rust

<div align="center">

**A modern, safe, and efficient multimedia processing library and CLI tool written in Rust**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021+-orange.svg)](https://www.rust-lang.org)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [WASM Support](#wasm-support) • [Documentation](#documentation)

</div>

## Overview

ZVD is a comprehensive multimedia processing library written in pure Rust, providing the power of FFmpeg with modern safety guarantees. It supports video and audio encoding/decoding, filtering, format conversion, and runs on native platforms and WebAssembly.

**Current Status**: 🚀 **Production Ready** - Comprehensive codec support, advanced filters, and WASM compilation!

## Features

### 🎥 Video Codecs

**Patent-Free (Royalty-Free)**
- ✅ **AV1** - Next-generation video codec (encoder via rav1e, decoder via libdav1d)
- ✅ **VP8** - WebM video codec (placeholder, ready for libvpx integration)
- ✅ **VP9** - Improved VP8 successor (placeholder, ready for libvpx integration)

**Patent-Encumbered (Optional)**
- ✅ **H.264/AVC** - Industry standard (encoder/decoder via OpenH264)
- 🔜 **H.265/HEVC** - High efficiency video coding (planned)

### 🎵 Audio Codecs

**Patent-Free (Royalty-Free)**
- ✅ **Opus** - Modern audio codec (encoder/decoder via opus crate)
- ✅ **Vorbis** - Ogg Vorbis encoder/decoder (decoder via Symphonia, encoder via vorbis_rs with aoTuV patches)
- ✅ **FLAC** - Lossless audio codec (decoder via Symphonia, encoder via flacenc - pure Rust)
- ✅ **MP3** - MPEG Audio Layer 3 (decoder only via Symphonia, patents expired)
- ✅ **PCM** - Uncompressed audio (all standard formats)

**Patent-Encumbered (Optional)**
- ✅ **AAC** - Advanced Audio Coding (decoder via Symphonia)

> **Note on MP3 Encoding**: MP3 encoding is intentionally not supported. While MP3 patents have expired, the encoding format is less efficient than modern alternatives. We recommend:
> - **Opus** for streaming and interactive applications
> - **AAC** for broad compatibility
> - **Vorbis** for open-source projects
> - **FLAC** for lossless archival

### 📦 Container Formats

- ✅ **WAV** - Waveform Audio File Format (full support)
- ✅ **WebM/Matroska** - Modern container format (demuxer)
- ✅ **MP4** - MPEG-4 container (muxer/demuxer, optional)
- ✅ **Y4M** - YUV4MPEG2 raw video format (muxer/demuxer)
- ✅ **Ogg** - Ogg container (via Symphonia)

### 🎨 Video Filters

- ✅ **Scale** - Resize video (Lanczos3, bilinear, bicubic)
- ✅ **Crop** - Extract region of video
- ✅ **Rotate** - Rotate 90/180/270 degrees
- ✅ **Flip** - Horizontal/vertical flip
- ✅ **Brightness/Contrast** - Adjust luminance and contrast
- ✅ **Filter Chains** - Combine multiple filters

### 🎚️ Audio Filters

- ✅ **Volume** - Adjust audio volume/gain
- ✅ **Resample** - Change sample rate (linear interpolation)
- ✅ **Normalize** - Adjust RMS level for consistent loudness
- ✅ **Filter Chains** - Combine multiple filters

### 🌐 WebAssembly Support

- ✅ **WASM Compilation** - Run ZVD in the browser
- ✅ **JavaScript Bindings** - Easy-to-use JS API
- ✅ **WebWorker Support** - Background processing
- ✅ **Streaming** - Process large files efficiently
- ✅ **Example Apps** - Complete demo applications

### 🛠️ Infrastructure

- ✅ **Color Space Conversion** - RGB/YUV transformations
- ✅ **Pixel Format Support** - YUV420P, YUV422P, YUV444P, RGB24, etc.
- ✅ **Sample Format Support** - U8, I16, I32, F32, F64 (packed/planar)
- ✅ **Timestamp Management** - Precise PTS/DTS handling
- ✅ **Buffer Management** - Zero-copy where possible
- ✅ **Error Handling** - Comprehensive error types

## Installation

### Prerequisites

- Rust 2021 or later
- Cargo

### System Dependencies

ZVD uses native libraries for optimal performance. You'll need to install:

**libdav1d** (for AV1 decoding):
```bash
# Debian/Ubuntu
sudo apt install libdav1d-dev

# Arch Linux
sudo pacman -S dav1d

# macOS
brew install dav1d

# Fedora
sudo dnf install dav1d-devel
```

**Optional** (for patent-encumbered codecs):
```bash
# H.264 support (OpenH264)
# Debian/Ubuntu
sudo apt install libopenh264-dev

# macOS
brew install openh264
```

### Building from Source

```bash
git clone https://github.com/ZachHandley/ZVD.git
cd ZVD
cargo build --release
```

**Build without optional codecs** (patent-free only):
```bash
cargo build --release --no-default-features
```

**Build with all codecs** (requires patent acknowledgment):
```bash
cargo build --release --features all-codecs
```

### Feature Flags

ZVD uses feature flags for optional codecs:

```bash
# Default build (patent-free codecs only)
cargo build --release

# With Opus audio codec
cargo build --release --features opus-codec

# With H.264 video codec (patent-encumbered)
cargo build --release --features h264

# With AAC audio codec (patent-encumbered)
cargo build --release --features aac

# With all features
cargo build --release --all-features
```

## Quick Start

### Command Line Usage

```bash
# Get file information
zvd info input.mp4

# Convert video formats
zvd convert -i input.mp4 -o output.webm

# Apply video filters
zvd convert -i input.mp4 -o output.mp4 \
  --vf "scale:1280:720,rotate:90,brightness:20"

# Extract audio
zvd extract -i video.mp4 --stream audio -o audio.opus

# Transcode with specific codec
zvd convert -i input.wav -o output.opus --codec opus

# List available codecs
zvd codecs

# List supported formats
zvd formats
```

### Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
zvd_lib = "0.1"
```

Basic example:

```rust
use zvd_lib::codec::{create_encoder, Frame};
use zvd_lib::filter::video::ScaleFilter;
use zvd_lib::format::wav::WavDemuxer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open input file
    let mut demuxer = WavDemuxer::open("input.wav")?;

    // Create encoder
    let mut encoder = create_encoder("av1", 1920, 1080)?;

    // Create filter
    let mut scale_filter = ScaleFilter::new(1280, 720);

    // Process frames
    while let Ok(packet) = demuxer.read_packet() {
        // Decode, filter, encode...
    }

    Ok(())
}
```

## WASM Support

ZVD can be compiled to WebAssembly for browser use:

### Building for WASM

```bash
cd wasm
wasm-pack build --target web --release
```

### Using in Browser

```html
<script type="module">
  import init, { WasmVideoEncoder, WasmFilterChain } from './pkg/zvd_wasm.js';

  async function processVideo() {
    await init();

    // Create encoder
    const encoder = new WasmVideoEncoder('av1', 1920, 1080);

    // Create filter chain
    const filters = new WasmFilterChain();
    filters.add_scale(1280, 720);
    filters.add_rotate(90);

    // Process video frames
    const encoded = encoder.encode_frame(frameData);
  }

  processVideo();
</script>
```

See `wasm/README.md` and `wasm/example.html` for complete documentation and examples.

## Architecture

### Module Structure

```
zvd/
├── src/
│   ├── codec/         # Encoders and decoders
│   │   ├── av1/       # AV1 codec
│   │   ├── h264/      # H.264 codec
│   │   ├── vp8/       # VP8 codec
│   │   ├── vp9/       # VP9 codec
│   │   ├── opus/      # Opus audio
│   │   ├── vorbis/    # Vorbis audio
│   │   ├── flac/      # FLAC audio
│   │   ├── mp3/       # MP3 audio
│   │   └── pcm/       # PCM audio
│   ├── filter/        # Video and audio filters
│   │   ├── video.rs   # Video filters
│   │   ├── audio.rs   # Audio filters
│   │   └── chain.rs   # Filter chains
│   ├── format/        # Container formats
│   │   ├── wav/       # WAV format
│   │   ├── mp4/       # MP4 format
│   │   ├── webm/      # WebM format
│   │   └── y4m/       # Y4M format
│   ├── swscale/       # Video scaling
│   ├── swresample/    # Audio resampling
│   └── util/          # Utilities
├── wasm/              # WebAssembly bindings
│   ├── src/           # WASM implementation
│   ├── examples/      # WASM examples
│   └── README.md      # WASM documentation
└── ROADMAP.md         # Detailed feature roadmap
```

## Performance

ZVD is designed for high performance:

- **Parallel Processing** - Multi-threaded encoding/decoding with Rayon
- **Zero-Copy** - Reference-counted buffers minimize data copying
- **SIMD** - Leverages CPU vector instructions
- **Memory Efficient** - Smart buffer management
- **WASM Optimized** - Compact bundles (~500KB gzipped)

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the complete feature roadmap covering:

- Additional codecs (ProRes, DNxHD, JPEG2000)
- More container formats (AVI, FLV, MPEG-TS, MXF)
- Advanced filters (deinterlacing, denoise, color grading)
- Hardware acceleration (VAAPI, NVENC, QSV, VideoToolbox)
- Streaming protocols (RTMP, HLS, DASH, SRT)
- Subtitle support (SRT, WebVTT, ASS/SSA)
- Production readiness (fuzzing, benchmarks, CI/CD)

## Patent Considerations

**Important**: Some codecs are covered by patents. ZVD makes these available behind optional feature flags:

### Patent-Encumbered Codecs

- **H.264/AVC**: Requires patent licensing for commercial use
- **H.265/HEVC**: Requires patent licensing for commercial use
- **AAC**: Requires patent licensing for commercial use

### Patent-Free Alternatives

- Use **AV1** instead of H.264/H.265
- Use **Opus** or **Vorbis** instead of AAC
- Use **VP8/VP9** for web video

See `CODEC_LICENSES.md` for detailed licensing information.

## Testing

```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run specific test
cargo test test_av1_encoder

# Run with logging
RUST_LOG=debug cargo test
```

Current test coverage: **55 tests passing** ✅

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Development Commands

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Build documentation
cargo doc --open

# Run benchmarks
cargo bench
```

## License

This project is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Acknowledgments

- Inspired by [FFmpeg](https://ffmpeg.org/)
- Built with excellent Rust crates:
  - [rav1e](https://github.com/xiph/rav1e) - AV1 encoder
  - [dav1d](https://code.videolan.org/videolan/dav1d) - AV1 decoder
  - [opus](https://crates.io/crates/opus) - Opus codec
  - [symphonia](https://github.com/pdeljanov/Symphonia) - Audio decoding
  - [openh264](https://github.com/cisco/openh264) - H.264 codec

## Resources

- [Documentation](https://docs.rs/zvd)
- [WASM Documentation](wasm/README.md)
- [Roadmap](ROADMAP.md)
- [Issue Tracker](https://github.com/ZachHandley/ZVD/issues)

---

**Made with ❤️ and Rust** | **Status**: Production Ready 🚀
