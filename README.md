# ZVD - FFMPEG in Rust

<div align="center">

**A modern, safe, and efficient multimedia processing library and CLI tool written in Rust**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021+-orange.svg)](https://www.rust-lang.org)

</div>

## Overview

ZVD is a reimplementation of FFMPEG's core functionality in pure Rust, providing efficient and safe multimedia processing capabilities. The project aims to deliver the power of FFMPEG with the safety guarantees and modern development experience of Rust.

**Current Status**: ✅ **Phase 1 Complete** - Full WAV audio processing working!

## Features

### ✅ Working Features (Phase 1)

- **WAV Audio Support**
  - ✅ Read WAV files (mono and stereo)
  - ✅ Write WAV files
  - ✅ Convert WAV to WAV (lossless)
  - ✅ Display file information
  - ✅ Sample formats: 8-bit, 16-bit, 32-bit PCM, 32/64-bit float

- **PCM Codec**
  - ✅ Decode PCM audio (all standard formats)
  - ✅ Encode PCM audio
  - ✅ Packed and planar format support

### Architecture

ZVD is organized into several key modules mirroring FFMPEG's structure:

- **`format`** - Container format handling (demuxing/muxing)
  - ✅ WAV format (complete)
  - 🔨 MP4, Matroska (planned)
  - Stream parsing and packet extraction
  - Seeking and metadata handling

- **`codec`** - Audio and video codec implementations
  - ✅ PCM (uncompressed audio)
  - 🔨 Opus, FLAC (Phase 2)
  - 🔨 AV1, VP9 video (Phase 3)
  - H.264/H.265 (patent-encumbered, use with caution)

- **`filter`** - Audio and video filtering and processing
  - Video scaling, cropping, rotation
  - Audio resampling, volume adjustment
  - Filter graphs for complex processing chains

- **`util`** - Common utilities and data structures
  - Rational numbers for frame rates and timestamps
  - Pixel format and sample format definitions
  - Buffer management

- **`swscale`** - Video scaling and color space conversion
  - Multiple scaling algorithms (bilinear, bicubic, lanczos)
  - RGB/YUV conversions
  - High-quality resampling

- **`swresample`** - Audio resampling and format conversion
  - Sample rate conversion
  - Channel layout mapping
  - Sample format conversion

## Installation

### Prerequisites

- Rust 2021 or later
- Cargo

### Building from Source

```bash
git clone https://github.com/ZachHandley/ZVD.git
cd ZVD
cargo build --release
```

The compiled binary will be available at `target/release/zvd`.

## Usage

### Command Line Interface

ZVD provides a powerful CLI similar to FFMPEG:

#### Get File Information

```bash
# Display WAV file information
zvd info audio.wav

# Output:
# File: audio.wav
#
# Streams: 1
#
# Stream #0:
#   Type: audio
#   Codec: pcm
#   Time Base: 1/44100
#   Frames: 88200
#   Duration: 2.00s
#   Sample Rate: 44100 Hz
#   Channels: 2
#   Sample Format: s16
#   Bits Per Sample: 16
#   Bit Rate: 1411 kbps
```

#### Convert Media Files

```bash
# Convert WAV to WAV (works now!)
zvd convert -i input.wav -o output.wav

# Convert with format specification
zvd convert -i audio.wav -o output.wav --format wav
```

#### Extract Streams

```bash
zvd extract -i input.mp4 --stream 0 -o video.h264
```

#### Probe Media Files

```bash
# Human-readable output
zvd probe input.mp4

# JSON output
zvd probe input.mp4 --json
```

#### List Codecs and Formats

```bash
# List all codecs
zvd codecs

# Filter by type
zvd codecs --filter video
zvd codecs --filter audio

# List formats
zvd formats
zvd formats --muxers
zvd formats --demuxers
```

### Library Usage

ZVD can also be used as a library in your Rust projects:

```toml
[dependencies]
zvd_lib = "0.1"
```

```rust
use zvd_lib::{Config, init};
use zvd_lib::format::demuxer;
use zvd_lib::codec::decoder;

fn main() -> anyhow::Result<()> {
    // Initialize ZVD
    let config = Config::default();
    init(config)?;

    // Open a media file
    let demuxer = demuxer::create_demuxer("input.mp4")?;

    // Process streams
    for stream in demuxer.streams() {
        println!("Stream {}: {:?}", stream.info.index, stream.info.media_type);
    }

    Ok(())
}
```

## Architecture

### Design Principles

1. **Safety First** - Leverage Rust's memory safety guarantees
2. **Zero-Cost Abstractions** - Performance on par with C implementations
3. **Modular Design** - Clean separation of concerns
4. **Parallelism** - Efficient use of multi-core processors
5. **Interoperability** - Compatible with existing multimedia ecosystems

### Module Structure

```
zvd/
├── src/
│   ├── lib.rs              # Library root
│   ├── main.rs             # CLI application
│   ├── error.rs            # Error types
│   ├── format/             # Container formats
│   │   ├── mod.rs
│   │   ├── demuxer.rs
│   │   ├── muxer.rs
│   │   ├── packet.rs
│   │   └── stream.rs
│   ├── codec/              # Codecs
│   │   ├── mod.rs
│   │   ├── decoder.rs
│   │   ├── encoder.rs
│   │   └── frame.rs
│   ├── filter/             # Filters
│   │   ├── mod.rs
│   │   ├── graph.rs
│   │   ├── video.rs
│   │   └── audio.rs
│   ├── util/               # Utilities
│   │   ├── mod.rs
│   │   ├── rational.rs
│   │   ├── timestamp.rs
│   │   ├── buffer.rs
│   │   ├── pixfmt.rs
│   │   └── samplefmt.rs
│   ├── swscale/            # Video scaling
│   │   └── mod.rs
│   └── swresample/         # Audio resampling
│       └── mod.rs
└── Cargo.toml
```

## Performance

ZVD is designed with performance in mind:

- **Parallel Processing** - Uses Rayon for efficient parallel processing
- **Zero-Copy** - Minimizes data copying through reference-counted buffers
- **SIMD** - Leverages CPU SIMD instructions where available
- **Memory Efficiency** - Smart buffer management and memory pooling

## Roadmap

### Phase 1: Foundation ✅ **COMPLETE**
- [x] Project structure and module layout
- [x] Core data structures (timestamps, buffers, formats)
- [x] Basic CLI framework
- [x] **WAV format support (COMPLETE)**
- [x] **PCM codec (COMPLETE)**
- [x] **Working conversion pipeline (COMPLETE)**
- [x] **25 unit tests passing**

### Phase 2: Core Codecs
- [ ] H.264 decoder
- [ ] H.265 decoder
- [ ] VP9 decoder
- [ ] AAC decoder
- [ ] Basic encoders

### Phase 3: Container Formats
- [ ] MP4 demuxer
- [ ] Matroska demuxer
- [ ] MP4 muxer
- [ ] Matroska muxer

### Phase 4: Advanced Features
- [ ] Hardware acceleration (VAAPI, NVDEC)
- [ ] Advanced filters
- [ ] Streaming protocols (HLS, DASH)
- [ ] Network protocols (RTMP, RTSP)

### Phase 5: Optimization
- [ ] SIMD optimizations
- [ ] Multi-threaded encoding/decoding
- [ ] Memory optimization
- [ ] Benchmarking suite

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ZachHandley/ZVD.git
cd ZVD

# Run tests
cargo test

# Try it out with WAV files
cargo run --release -- info test.wav
cargo run --release -- convert -i test.wav -o output.wav

# Build documentation
cargo doc --open
```

### Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for code formatting
- Use `clippy` for linting

```bash
cargo fmt
cargo clippy
```

## License

This project is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Acknowledgments

- Inspired by [FFMPEG](https://ffmpeg.org/) - The leading multimedia framework
- Built with excellent Rust crates from the ecosystem
- Thanks to all contributors and the Rust community

## Resources

- [Documentation](https://docs.rs/zvd)
- [Issue Tracker](https://github.com/ZachHandley/ZVD/issues)
- [Discussions](https://github.com/ZachHandley/ZVD/discussions)

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/ZachHandley/ZVD.git
cd ZVD
cargo build --release

# Try it out!
./target/release/zvd info test_stereo.wav
./target/release/zvd convert -i test_stereo.wav -o my_copy.wav

# List supported formats and codecs
./target/release/zvd formats
./target/release/zvd codecs
```

**Status**: ✅ Phase 1 complete! WAV audio processing fully working.
**Next**: Phase 2 will add Opus, FLAC, and other audio codecs.
