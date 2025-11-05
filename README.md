# ZVD - FFMPEG in Rust

<div align="center">

**A modern, safe, and efficient multimedia processing library and CLI tool written in Rust**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021+-orange.svg)](https://www.rust-lang.org)

</div>

## Overview

ZVD is a reimplementation of FFMPEG's core functionality in pure Rust, providing efficient and safe multimedia processing capabilities. The project aims to deliver the power of FFMPEG with the safety guarantees and modern development experience of Rust.

## Features

### Current Architecture

ZVD is organized into several key modules mirroring FFMPEG's structure:

- **`format`** - Container format handling (demuxing/muxing)
  - Support for MP4, Matroska, AVI, and more
  - Stream parsing and packet extraction
  - Seeking and metadata handling

- **`codec`** - Audio and video codec implementations
  - H.264/AVC, H.265/HEVC support
  - VP8, VP9 codecs
  - AAC, MP3, Opus audio codecs

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
zvd info input.mp4
```

#### Convert Media Files

```bash
# Basic conversion
zvd convert -i input.mp4 -o output.webm

# Specify codecs and bitrates
zvd convert -i input.mp4 -o output.mkv \
  --vcodec h265 \
  --acodec opus \
  --vbitrate 2M \
  --abitrate 128k
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

### Phase 1: Foundation (Current)
- [x] Project structure and module layout
- [x] Core data structures (timestamps, buffers, formats)
- [x] Basic CLI framework
- [ ] WAV format support
- [ ] Raw video format support

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

# Run with verbose logging
cargo run -- -v info test.mp4

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

**Note**: ZVD is currently in early development. Many features are not yet implemented. Check the roadmap above for current progress and planned features.
