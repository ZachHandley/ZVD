# ZVD Project Roadmap & TODO

## Executive Summary

ZVD is building a pure Rust multimedia processing framework. This document outlines our implementation strategy, focusing on **patent-free, royalty-free codecs** and leveraging the Rust ecosystem.

## Codec Strategy & Licensing

### ✅ Patent-Free & Safe Choices

| Codec | Type | Status | License | Notes |
|-------|------|--------|---------|-------|
| **AV1** | Video | ✅ Royalty-free | Open | Alliance for Open Media, successor to VP9 |
| **VP9** | Video | ✅ Royalty-free | BSD-like | Google, older but widely supported |
| **VP8** | Video | ✅ Royalty-free | BSD-like | Google, WebM standard |
| **Opus** | Audio | ✅ Royalty-free | BSD | Best-in-class audio codec |
| **Vorbis** | Audio | ✅ Royalty-free | BSD | OGG Vorbis, widely used |
| **FLAC** | Audio | ✅ Royalty-free | BSD | Lossless audio |
| **PCM/WAV** | Audio | ✅ Unencumbered | - | Uncompressed audio |

### ⚠️ Patent-Encumbered (Use with Caution)

| Codec | Type | Patent Status | Notes |
|-------|------|--------------|-------|
| **H.264/AVC** | Video | ⚠️ MPEG-LA | Heavily patented, licensing required for commercial use |
| **H.265/HEVC** | Video | ⚠️ MPEG-LA | Similar patents to H.264 |
| **AAC** | Audio | ⚠️ MPEG-LA | Patent license required |
| **MP3** | Audio | ✅ Patents expired | Most patents expired, safe to use |

### Container Formats

| Format | License | Notes |
|--------|---------|-------|
| **WebM** | Open | VP8/VP9 + Vorbis/Opus in Matroska |
| **Matroska (MKV)** | Open | Flexible container, supports any codec |
| **OGG** | Open | Vorbis/Opus container |
| **WAV** | Unencumbered | Simple PCM audio |
| **MP4** | ISO standard | Can contain any codec, but typically H.264/AAC |

## Rust Ecosystem Libraries

### Audio Processing
- **Symphonia** (MPL-2.0): Pure Rust audio decoding
  - Formats: WAV, FLAC, MP3, AAC, Vorbis, OGG, MP4, MKV
  - 100% safe Rust, no C bindings
  - Performance within 15% of FFmpeg

- **hound** (Apache-2.0): WAV encoding/decoding
  - Already in our dependencies
  - Simple, fast, pure Rust

- **lewton** (Apache-2.0/MIT): Pure Rust Vorbis decoder

### Video Processing
- **rav1e** (BSD-2): AV1 encoder in Rust
  - Production-ready, maintained by Xiph.org
  - Fast and memory-safe

- **rav1d** (BSD-2): AV1 decoder in Rust
  - Port of high-performance dav1d from C
  - Memory-safe with competitive performance

- **libvpx** (BSD): VP8/VP9 (C library, needs bindings)

### Container Formats
- **mp4parse** (MPL-2.0): MP4 demuxer
- **matroska** crate: MKV support

---

## Phase 1: WAV Audio Support (2-3 days)
**Goal**: Get basic audio processing working end-to-end

### 1.1 WAV Demuxer Implementation
- [ ] Create `src/format/wav/` module
- [ ] Implement RIFF parser
  - [ ] Parse WAV header
  - [ ] Read fmt chunk
  - [ ] Read data chunk
  - [ ] Handle LIST/INFO chunks for metadata
- [ ] Create `WavDemuxer` struct implementing `Demuxer` trait
- [ ] Parse PCM format parameters (sample rate, bit depth, channels)
- [ ] Stream audio packets from data chunk
- [ ] Add tests with sample WAV files

**Files to create:**
- `src/format/wav/mod.rs`
- `src/format/wav/demuxer.rs`
- `src/format/wav/header.rs`
- `tests/wav_demux_test.rs`

### 1.2 PCM Decoder Implementation
- [ ] Create `src/codec/pcm/` module
- [ ] Implement `PcmDecoder` struct
  - [ ] Handle u8, i16, i24, i32, f32 formats
  - [ ] Convert to internal audio frame format
  - [ ] Handle endianness
- [ ] Implement `Decoder` trait for PCM
- [ ] Add unit tests for all PCM formats

**Files to create:**
- `src/codec/pcm/mod.rs`
- `src/codec/pcm/decoder.rs`
- `tests/pcm_codec_test.rs`

### 1.3 PCM Encoder Implementation
- [ ] Implement `PcmEncoder` struct
  - [ ] Convert internal frames to PCM
  - [ ] Handle format conversions
  - [ ] Support all common bit depths
- [ ] Implement `Encoder` trait for PCM
- [ ] Add encoding tests

**Files to create:**
- `src/codec/pcm/encoder.rs`

### 1.4 WAV Muxer Implementation
- [ ] Create `WavMuxer` struct implementing `Muxer` trait
- [ ] Write RIFF/WAVE header
- [ ] Write fmt chunk
- [ ] Stream data chunks
- [ ] Update header with final sizes
- [ ] Add metadata support
- [ ] Add muxing tests

**Files to create:**
- `src/format/wav/muxer.rs`
- `tests/wav_mux_test.rs`

### 1.5 CLI Integration
- [ ] Wire up WAV demuxer in `format::create_demuxer()`
- [ ] Wire up WAV muxer in `format::create_muxer()`
- [ ] Wire up PCM codec in codec factories
- [ ] Implement `cmd_info()` for WAV files
- [ ] Implement `cmd_convert()` for WAV-to-WAV
- [ ] Add integration tests
- [ ] Update README with working examples

### 1.6 Basic Audio Processing
- [ ] Implement sample rate conversion in `swresample`
- [ ] Implement bit depth conversion
- [ ] Implement channel mixing (mono↔stereo)
- [ ] Add volume filter implementation

**Success Criteria:**
```bash
# These commands must work:
zvd info test.wav
zvd convert -i input.wav -o output.wav
zvd convert -i 44khz.wav -o 48khz.wav --sample-rate 48000
```

---

## Phase 2: Patent-Free Audio Codecs (3-5 days)
**Goal**: Support Opus and FLAC for compressed audio

### 2.1 Integrate Symphonia for Demuxing
- [ ] Add Symphonia dependency with careful feature selection
- [ ] Create adapter layer `src/format/symphonia_adapter.rs`
- [ ] Implement `SymphoniaDemuxer` wrapper
- [ ] Support FLAC, Vorbis, MP3 demuxing via Symphonia
- [ ] Add tests for each format

### 2.2 FLAC Support
- [ ] Integrate FLAC decoder from Symphonia
- [ ] Implement FLAC encoder (use `flac-sys` or pure Rust)
- [ ] Add FLAC muxer
- [ ] Support FLAC metadata (tags, cover art)
- [ ] Add comprehensive tests

### 2.3 Opus Support
- [ ] Integrate Opus decoder (`opus` crate or Symphonia)
- [ ] Integrate Opus encoder (`audiopus` crate)
- [ ] Implement OGG/Opus container support
- [ ] Add Opus configuration (bitrate, complexity)
- [ ] Test with various bitrates

### 2.4 Vorbis Support
- [ ] Use Symphonia's Vorbis decoder
- [ ] Add OGG container support
- [ ] Test OGG/Vorbis files

**Success Criteria:**
```bash
zvd info audio.flac
zvd convert -i input.wav -o output.opus --acodec opus --abitrate 128k
zvd convert -i input.flac -o output.wav
```

---

## Phase 3: Patent-Free Video Codecs (1-2 weeks)
**Goal**: Support AV1 and VP9 for video

### 3.1 Y4M Raw Video Support
- [ ] Implement Y4M (YUV4MPEG2) format parser
- [ ] Support YUV420p, YUV422p, YUV444p
- [ ] Implement Y4M muxer for testing
- [ ] Add pixel format conversions in `swscale`

### 3.2 AV1 Decoder Integration
- [ ] Add `rav1d` dependency
- [ ] Create `src/codec/av1/` module
- [ ] Implement `Av1Decoder` wrapper
- [ ] Handle AV1 codec parameters
- [ ] Add decoder tests with sample files

### 3.3 AV1 Encoder Integration
- [ ] Add `rav1e` dependency
- [ ] Implement `Av1Encoder` wrapper
- [ ] Configure encoding parameters (speed, quality)
- [ ] Add 2-pass encoding support
- [ ] Encoder performance tests

### 3.4 VP9 Support
- [ ] Investigate pure Rust VP9 (or use libvpx bindings)
- [ ] Implement VP9 decoder wrapper
- [ ] Implement VP9 encoder wrapper
- [ ] Add VP9 configuration options

### 3.5 WebM Container Support
- [ ] Implement WebM (Matroska subset) demuxer
- [ ] Support VP8/VP9 + Vorbis/Opus in WebM
- [ ] Implement WebM muxer
- [ ] Handle seeking in WebM
- [ ] Add metadata support

**Success Criteria:**
```bash
zvd info video.webm
zvd convert -i input.y4m -o output.webm --vcodec av1
zvd convert -i input.webm -o output.y4m
```

---

## Phase 4: Advanced Filters (1 week)
**Goal**: Implement useful video and audio filters

### 4.1 Video Filters
- [ ] Scale filter (resize video)
  - [ ] Bilinear interpolation
  - [ ] Bicubic interpolation
  - [ ] Lanczos resampling
- [ ] Crop filter
- [ ] Rotate/flip filter
- [ ] Color space conversion (RGB ↔ YUV)
- [ ] Deinterlace filter
- [ ] Denoise filter (basic)

### 4.2 Audio Filters
- [ ] Volume adjustment (already partially done)
- [ ] Audio mixing (multiple sources)
- [ ] Equalizer (basic)
- [ ] Compression/limiting
- [ ] Fade in/out
- [ ] Audio normalization

### 4.3 Filter Graph Improvements
- [ ] Implement proper graph traversal
- [ ] Support multiple inputs/outputs
- [ ] Add filter chaining syntax
- [ ] Parallel filter execution

**Success Criteria:**
```bash
zvd convert -i input.webm -o output.webm --scale 1280:720
zvd convert -i input.wav -o output.wav --volume 1.5 --normalize
```

---

## Phase 5: MP4 Container (1 week)
**Goal**: Support MP4 container with free codecs

### 5.1 MP4 Demuxer
- [ ] Use `mp4parse` or implement from scratch
- [ ] Parse MP4 atoms (ftyp, moov, mdat, etc.)
- [ ] Extract streams and codec parameters
- [ ] Support fragmented MP4
- [ ] Handle seeking
- [ ] Read metadata

### 5.2 MP4 Muxer
- [ ] Implement MP4 writer
- [ ] Write ftyp, moov, mdat atoms
- [ ] Support progressive MP4
- [ ] Support fragmented MP4
- [ ] Write metadata
- [ ] Optimize for streaming

### 5.3 Codec Combinations
- [ ] MP4 + AV1 + Opus
- [ ] MP4 + VP9 + Opus
- [ ] Test all combinations

**Success Criteria:**
```bash
zvd info video.mp4
zvd convert -i input.webm -o output.mp4 --vcodec av1 --acodec opus
```

---

## Phase 6: Matroska/MKV Support (3-5 days)
**Goal**: Full MKV support

### 6.1 MKV Demuxer
- [ ] Parse EBML structure
- [ ] Support all common codecs in MKV
- [ ] Handle chapters and attachments
- [ ] Support multiple audio/subtitle tracks

### 6.2 MKV Muxer
- [ ] Write EBML/Matroska structure
- [ ] Support multiple streams
- [ ] Add chapter support
- [ ] Attachment support (fonts, cover art)

---

## Phase 7: Hardware Acceleration (2 weeks)
**Goal**: GPU-accelerated encoding/decoding

### 7.1 VAAPI (Linux)
- [ ] Integrate VAAPI for decode
- [ ] VAAPI encode support
- [ ] Hardware surface management

### 7.2 NVDEC/NVENC (NVIDIA)
- [ ] NVDEC integration
- [ ] NVENC integration
- [ ] Memory transfer optimization

### 7.3 VideoToolbox (macOS)
- [ ] VideoToolbox decode
- [ ] VideoToolbox encode

### 7.4 Quick Sync (Intel)
- [ ] QSV decode support
- [ ] QSV encode support

---

## Phase 8: Streaming & Network (2 weeks)
**Goal**: Support streaming protocols

### 8.1 HLS (HTTP Live Streaming)
- [ ] HLS playlist parser
- [ ] Segment fetching
- [ ] Adaptive bitrate logic
- [ ] HLS playlist generator
- [ ] Segment creation

### 8.2 DASH
- [ ] DASH MPD parser
- [ ] Segment fetching
- [ ] Adaptive logic

### 8.3 RTMP (Optional)
- [ ] RTMP client
- [ ] RTMP server
- [ ] FLV container support

---

## Phase 9: Performance Optimization (Ongoing)
**Goal**: Match or exceed FFmpeg performance

### 9.1 SIMD Optimizations
- [ ] Identify hot paths via profiling
- [ ] Implement SIMD pixel conversions
- [ ] SIMD scaling algorithms
- [ ] SIMD audio processing

### 9.2 Multi-threading
- [ ] Parallel encoding (frame slicing)
- [ ] Parallel decoding
- [ ] Thread pool optimization
- [ ] Lock-free data structures where needed

### 9.3 Memory Optimization
- [ ] Buffer pooling
- [ ] Zero-copy pipelines
- [ ] Reduce allocations in hot paths

### 9.4 Benchmarking
- [ ] Comprehensive benchmark suite
- [ ] Compare against FFmpeg
- [ ] Profile and optimize bottlenecks
- [ ] Continuous performance monitoring

---

## Testing Strategy

### Unit Tests
- Every module must have unit tests
- Target 80%+ code coverage
- Test edge cases and error conditions

### Integration Tests
- End-to-end pipeline tests
- Real file format tests
- Cross-format conversion tests

### Fuzz Testing
- Fuzz demuxers with malformed files
- Fuzz codecs with random data
- Use cargo-fuzz

### Performance Tests
- Benchmark against FFmpeg
- Track performance regressions
- Use criterion for micro-benchmarks

### Sample Files
Create test suite with:
- Various sample rates, bit depths, channels
- Different resolutions and frame rates
- Edge cases (very long, very short, corrupt files)
- All supported formats

---

## Documentation Requirements

### User Documentation
- [ ] Comprehensive README
- [ ] Installation guide
- [ ] Usage examples for all features
- [ ] CLI reference
- [ ] Performance tuning guide

### Developer Documentation
- [ ] Architecture overview
- [ ] Module documentation
- [ ] Contributing guide
- [ ] Codec implementation guide
- [ ] API documentation (rustdoc)

### Tutorials
- [ ] Basic audio conversion
- [ ] Video transcoding
- [ ] Filter usage
- [ ] Hardware acceleration setup
- [ ] Streaming setup

---

## Future Considerations

### Additional Codecs (If Needed)
- H.264/H.265 via openh264 or x264/x265 (with licensing awareness)
- AAC via FDK-AAC (licensing required)

### Advanced Features
- Multi-pass encoding
- Scene detection
- Audio/video synchronization
- Subtitle support
- Color grading
- GPU filters

### Platform Support
- Windows optimization
- macOS optimization
- Linux optimization
- WebAssembly port
- Mobile platforms (Android/iOS)

---

## Success Metrics

### Phase 1 Complete
- Can read and write WAV files
- Can convert between WAV formats
- All tests passing
- Documentation complete

### Phase 2 Complete
- Support for Opus, FLAC, Vorbis
- Audio quality comparable to FFmpeg
- Performance within 20% of FFmpeg

### Phase 3 Complete
- AV1 and VP9 encode/decode working
- WebM container support
- Video quality comparable to FFmpeg

### Project Success
- ✅ Pure Rust implementation (90%+)
- ✅ Memory safe (no unsafe except in critical paths)
- ✅ Performance within 15% of FFmpeg
- ✅ Support all patent-free codecs
- ✅ Production-ready quality
- ✅ Comprehensive documentation
- ✅ 80%+ test coverage

---

## Development Workflow

1. **Branch Strategy**: Feature branches for each phase
2. **Code Review**: All PRs require review
3. **CI/CD**: Automated testing on all platforms
4. **Releases**: Semantic versioning, regular releases
5. **Issue Tracking**: GitHub issues for bugs/features

## Resources

- [Symphonia Documentation](https://github.com/pdeljanov/Symphonia)
- [rav1e Documentation](https://github.com/xiph/rav1e)
- [rav1d Documentation](https://github.com/memorysafety/rav1d)
- [FFmpeg as Reference](https://ffmpeg.org/documentation.html)
- [AV1 Specification](https://aomediacodec.github.io/av1-spec/)
- [Opus Specification](https://opus-codec.org/)
- [Matroska Specification](https://www.matroska.org/technical/specs/index.html)

---

**Last Updated**: 2025-11-05
**Current Phase**: Phase 1 - WAV Audio Support
**Next Milestone**: Working WAV conversion by end of week
