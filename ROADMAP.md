# ZVD Feature Roadmap to FFmpeg Parity

## Phase 7: Additional Patent-Free Video Codecs
- [ ] VP8 encoder/decoder (libvpx)
- [ ] VP9 encoder/decoder (libvpx)
- [ ] Theora decoder

## Phase 8: Additional Audio Codecs
- [ ] Opus encoder/decoder (libopus)
- [ ] Vorbis decoder (lewton/libvorbis)
- [ ] FLAC encoder/decoder (complete)
- [ ] MP3 decoder (minimp3/symphonia)

## Phase 9: Advanced Patent-Encumbered Codecs
- [ ] H.265/HEVC encoder (x265) - behind feature flag
- [ ] AAC encoder (fdk-aac) - behind feature flag
- [ ] AC-3/E-AC-3 decoder - behind feature flag
- [ ] MPEG-2 Video decoder - behind feature flag

## Phase 10: Professional Codecs
- [ ] ProRes decoder/encoder (behind feature flag)
- [ ] DNxHD/DNxHR decoder/encoder
- [ ] AVC-Intra support
- [ ] JPEG2000 support

## Phase 11: Additional Container Formats
- [ ] AVI muxer/demuxer
- [ ] FLV (Flash Video) muxer/demuxer
- [ ] MPEG-TS (Transport Stream) support
- [ ] MXF (Material Exchange Format)
- [ ] GIF support
- [ ] Image sequences (PNG, JPEG, TIFF)

## Phase 12: Advanced Filtering
- [ ] Complex filter graph support
- [ ] Video filters:
  - [ ] Deinterlacing (yadif, bwdif)
  - [ ] Denoise (hqdn3d, nlmeans)
  - [ ] Sharpen/Blur
  - [ ] Color correction/grading
  - [ ] Overlay/watermark
  - [ ] Crop/Pad
  - [ ] Transpose/Flip
- [ ] Audio filters:
  - [ ] Equalization
  - [ ] Compressor/Limiter
  - [ ] Echo/Reverb
  - [ ] Pitch shifting
  - [ ] Tempo adjustment

## Phase 13: Hardware Acceleration
- [ ] VAAPI (Linux Intel/AMD)
- [ ] NVENC/NVDEC (NVIDIA)
- [ ] Intel Quick Sync Video (QSV)
- [ ] VideoToolbox (macOS/iOS)
- [ ] AMF (AMD Media Framework)
- [ ] Vulkan compute

## Phase 14: Streaming Protocols
- [ ] RTMP input/output
- [ ] HLS (HTTP Live Streaming)
- [ ] DASH (Dynamic Adaptive Streaming)
- [ ] RTP/RTSP
- [ ] SRT (Secure Reliable Transport)
- [ ] RTCP support

## Phase 15: Subtitle Support
- [ ] SRT subtitle format
- [ ] WebVTT
- [ ] ASS/SSA
- [ ] Closed Captions (CEA-608/708)
- [ ] DVB subtitles
- [ ] Subtitle rendering

## Phase 16: WASM Support
- [ ] Core library WASM compilation
- [ ] JavaScript bindings
- [ ] Browser-compatible codec selection
- [ ] Web Worker support
- [ ] Streaming support in browser
- [ ] Example web application

## Phase 17: Advanced Features
- [ ] Scene detection
- [ ] Motion estimation
- [ ] Thumbnail generation
- [ ] Metadata extraction/editing
- [ ] Multi-pass encoding
- [ ] Codec parameter optimization
- [ ] Quality metrics (PSNR, SSIM, VMAF)

## Phase 18: Performance Optimizations
- [ ] SIMD optimizations (SSE, AVX, NEON)
- [ ] Multi-threaded encoding/decoding
- [ ] Memory pool management
- [ ] Zero-copy pipelines
- [ ] Parallel filter execution

## Phase 19: Developer Experience
- [ ] Comprehensive examples
- [ ] API documentation
- [ ] Performance benchmarks
- [ ] Codec comparison tools
- [ ] CLI improvements (progress bars, better error messages)

## Phase 20: Production Readiness
- [ ] Extensive test suite
- [ ] Fuzzing for security
- [ ] Memory leak detection
- [ ] Performance profiling
- [ ] CI/CD pipeline
- [ ] Release automation
- [ ] Package distribution (crates.io, npm for WASM)

---

## Priority Order for Implementation

1. **Phase 8**: Audio codecs (Opus, Vorbis, FLAC, MP3)
2. **Phase 7**: VP8/VP9 video codecs
3. **Phase 16**: WASM support (high value for web use cases)
4. **Phase 12**: Advanced filtering
5. **Phase 11**: Additional containers
6. **Phase 13**: Hardware acceleration
7. **Phase 14**: Streaming protocols
8. Rest based on user demand and use cases
