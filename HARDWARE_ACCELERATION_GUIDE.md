# Hardware Acceleration Implementation Guide

**Status**: Stubs in place, full implementation requires external dependencies
**Priority**: HIGH (User: "Hardware acceleration is crucial")
**Estimated Effort**: 5,000-8,000 lines + external dependencies

## Current State

ZVD has **stub implementations** for hardware acceleration in `src/hwaccel/`:
- ✅ Module structure and traits defined
- ✅ Device detection stubs (NVENC, VAAPI, QSV, VideoToolbox)
- ✅ Frame upload/download interface
- ❌ **No actual hardware acceleration** - all stubs return frame clones

## What's Missing

### 1. NVIDIA NVENC/NVDEC (Most Critical)

**Required Dependencies**:
```toml
# CUDA runtime bindings
cudarc = "0.12"  # OR cuda-sys = "0.3"

# NVIDIA Video Codec SDK (requires manual download)
# Download from: https://developer.nvidia.com/nvidia-video-codec-sdk
# Not available on crates.io - requires custom build script
```

**System Requirements**:
- NVIDIA GPU (GTX 600 series or newer)
- CUDA Toolkit 11.0+ installed
- NVIDIA drivers 470.x or newer
- Linux: `nvidia-smi` available
- Windows: `nvcuda.dll` in PATH

**Implementation Checklist** (~2,000-3,000 lines):
- [ ] CUDA context initialization
- [ ] CUDA memory allocation/deallocation
- [ ] Device memory transfers (host → device, device → host)
- [ ] NVENC encoder initialization
  - Codec selection (H.264, H.265)
  - Preset configuration (P1-P7)
  - Rate control modes (CBR, VBR, CQP)
  - B-frame configuration
- [ ] NVDEC decoder initialization
  - Format detection
  - Output surface management
- [ ] Encoding pipeline
  - Input surface preparation
  - Encoding loop with frame submission
  - Bitstream extraction
  - Asynchronous encoding support
- [ ] Decoding pipeline
  - Bitstream parsing
  - Frame decoding
  - Surface conversion
- [ ] Multi-GPU support
- [ ] Error handling for hardware failures
- [ ] Resource cleanup

**Key Challenges**:
- NVIDIA Video Codec SDK is **not open source** and requires manual installation
- No official Rust bindings - must create FFI wrapper
- Complex state management for CUDA contexts
- Platform-specific compilation (CUDA on Windows vs Linux)

---

### 2. VAAPI (Video Acceleration API) - Linux Intel/AMD

**Required Dependencies**:
```toml
# VA-API bindings
libva = "0.5"  # OR va-sys for lower-level access

# Optional: Display handling
libva-display = "0.5"
```

**System Requirements**:
- Linux only
- Intel GPU (HD Graphics 4000+) or AMD GPU (GCN 1.0+)
- `/dev/dri/renderD128` or `/dev/dri/card0` device node
- `libva` and `libva-drm` system libraries
- Intel: i965-va-driver or iHD driver
- AMD: mesa-va-drivers

**Implementation Checklist** (~1,500-2,000 lines):
- [ ] VAAPI display initialization (DRM or X11)
- [ ] VA context creation
- [ ] VA surface allocation
- [ ] VA buffer management
- [ ] Encoder configuration
  - Profile selection (H.264 Main, High, etc.)
  - Entrypoint selection (VAEntrypointEncSlice)
  - Rate control
- [ ] Decoder configuration
  - Bitstream buffer submission
  - Picture parameter setup
- [ ] Surface mapping and data transfer
- [ ] Format conversion (NV12, I420, etc.)
- [ ] Error handling

**Key Challenges**:
- Linux-only
- Different drivers for Intel vs AMD (capability differences)
- Display requirement for initialization
- Format negotiation complexity

---

### 3. Intel Quick Sync Video (QSV)

**Required Dependencies**:
```toml
# Intel Media SDK (deprecated, use oneVPL)
# OR
# oneVPL (Video Processing Library)
onevpl = { version = "1.0", optional = true }
```

**System Requirements**:
- Intel CPU with integrated graphics (2nd gen Core or newer)
- Linux: libmfx, libva
- Windows: Intel Graphics Driver

**Implementation Checklist** (~1,500-2,000 lines):
- [ ] MFX session initialization
- [ ] Device handle management
- [ ] Memory allocator setup
- [ ] Encoder initialization
  - Codec selection
  - Hardware vs software fallback
- [ ] Decoder initialization
- [ ] Surface pool management
- [ ] Synchronization primitives
- [ ] Error handling

**Key Challenges**:
- Intel Media SDK is deprecated (use oneVPL)
- oneVPL has limited Rust support
- Requires Intel hardware

---

### 4. Apple VideoToolbox (macOS/iOS)

**Required Dependencies**:
```toml
# Core Foundation and VideoToolbox frameworks
core-foundation = "0.9"
core-graphics = "0.23"

# VideoToolbox bindings (may need custom)
# No mature crates available - likely need manual FFI
```

**System Requirements**:
- macOS 10.8+ or iOS 8+
- Apple Silicon or Intel Mac with hardware encoder

**Implementation Checklist** (~1,500-2,000 lines):
- [ ] VideoToolbox session creation
- [ ] Compression session setup
- [ ] Decompression session setup
- [ ] CVPixelBuffer management
- [ ] Format description creation
- [ ] Encoding callback handling
- [ ] Decoding callback handling
- [ ] Sample buffer management
- [ ] Error handling

**Key Challenges**:
- macOS/iOS only
- Objective-C runtime integration
- Complex Core Foundation memory management
- Limited Rust bindings

---

### 5. AMD AMF (Advanced Media Framework)

**Required Dependencies**:
```toml
# No mature Rust crates - requires custom FFI wrapper
# Download AMF SDK from: https://github.com/GPUOpen-LibrariesAndSDKs/AMF
```

**System Requirements**:
- AMD GPU (GCN 1.0 or newer)
- AMD drivers with AMF support
- Windows: AMF runtime
- Linux: AMF with Vulkan or OpenGL

**Implementation Checklist** (~1,500-2,000 lines):
- [ ] AMF context initialization
- [ ] Component creation (encoder/decoder)
- [ ] Surface allocation
- [ ] Property configuration
- [ ] Encoding pipeline
- [ ] Decoding pipeline
- [ ] Vulkan/D3D11 interop
- [ ] Error handling

**Key Challenges**:
- No Rust bindings available
- AMD GPU required
- Less documentation than NVENC

---

## Recommended Implementation Strategy

### Phase 1: Choose One Platform (Best ROI)

**Option A: NVIDIA NVENC (Recommended)**
- **Pros**: Most widely used, best documentation, highest performance
- **Cons**: Requires NVIDIA GPU, proprietary SDK
- **Use Case**: Professional editing, streaming, gaming
- **Estimated Time**: 2-3 weeks for basic implementation

**Option B: VAAPI (Linux Focus)**
- **Pros**: Open source, works on Intel/AMD, widely supported on Linux
- **Cons**: Linux-only, driver-dependent behavior
- **Use Case**: Linux servers, Intel/AMD laptops
- **Estimated Time**: 1-2 weeks for basic implementation

### Phase 2: Abstraction Layer

Create a unified API that works across platforms:
```rust
pub trait HardwareEncoder {
    fn encode_frame(&mut self, frame: &VideoFrame) -> Result<Vec<u8>>;
    fn flush(&mut self) -> Result<Vec<Vec<u8>>>;
}

pub trait HardwareDecoder {
    fn decode_packet(&mut self, data: &[u8]) -> Result<VideoFrame>;
}
```

### Phase 3: Fallback Strategy

Always provide software fallback:
```rust
pub fn create_encoder(codec: &str, prefer_hw: bool) -> Box<dyn Encoder> {
    if prefer_hw {
        if let Ok(hw_encoder) = create_hw_encoder(codec) {
            return Box::new(hw_encoder);
        }
    }
    // Fall back to software encoder
    Box::new(create_sw_encoder(codec))
}
```

---

## Alternative: FFmpeg Hardware Acceleration

**Pragmatic Approach**: Use FFmpeg's hwaccel through bindings

```toml
ffmpeg-next = { version = "7.0", features = ["hwaccel"] }
```

**Pros**:
- Mature, battle-tested implementation
- Supports all platforms (NVENC, VAAPI, QSV, VideoToolbox, AMF)
- Automatic format negotiation
- Extensive documentation

**Cons**:
- Not pure Rust (requires FFmpeg installation)
- Larger binary size
- Less control over encoding parameters
- External dependency

**Implementation**: ~500-800 lines to wrap FFmpeg's hwaccel API

---

## Testing Requirements

Hardware acceleration is **difficult to test** in CI/CD:
- Requires actual GPU hardware
- Different GPUs have different capabilities
- Driver version dependencies
- Platform-specific behavior

**Testing Strategy**:
1. **Unit Tests**: Test device detection and initialization
2. **Integration Tests**: Test with mock hardware (stubs)
3. **Manual Testing**: Requires actual hardware for each platform
4. **Continuous Testing**: Use cloud instances with GPUs (expensive)

---

## Why This Is Challenging

1. **External Dependencies**: All platforms require C libraries
2. **System Requirements**: Requires specific hardware and drivers
3. **Platform-Specific**: Different code paths for each OS/GPU
4. **Binary Distribution**: Users must have drivers installed
5. **Testing Complexity**: Requires real hardware for validation
6. **Documentation**: Limited Rust examples, mostly C/C++ docs
7. **Error Handling**: Hardware failures, driver bugs, OOM conditions

---

## Current Status in ZVD

**What Works**:
- ✅ Device detection (checks for device files/drivers)
- ✅ Stub API that compiles
- ✅ Type-safe abstraction layer

**What Doesn't Work**:
- ❌ Actual hardware encoding
- ❌ Actual hardware decoding
- ❌ Memory transfers to/from GPU
- ❌ CUDA/VAAPI/QSV/VT initialization
- ❌ Format conversion for hardware surfaces

---

## Recommendation

Given the complexity, I recommend:

1. **Short Term**: Document the requirements (this file)
2. **Medium Term**: Implement ONE platform (VAAPI or NVENC) as proof-of-concept
3. **Long Term**: Either:
   - Build out all platforms (5,000-8,000 lines + external deps)
   - OR use FFmpeg bindings for faster time-to-market

**For Pure Rust Approach**:
- Start with VAAPI (Linux) - most open and documented
- Use existing Rust VAAPI bindings
- Implement H.264 encoding first
- Extend to other codecs and platforms

**For Pragmatic Approach**:
- Use `ffmpeg-next` with hwaccel feature
- Wrap FFmpeg's hardware acceleration
- Gets all platforms working quickly
- Focus ZVD's effort on other features

---

## Next Steps

1. **Decide on approach**: Pure Rust vs FFmpeg bindings
2. **Choose platform**: NVENC, VAAPI, or QSV first
3. **Add dependencies** to Cargo.toml
4. **Implement basic encoder** for chosen platform
5. **Add tests** (as much as possible without hardware)
6. **Document usage** and requirements for users

---

**Total Estimated Effort**:
- Pure Rust (all platforms): 5,000-8,000 lines, 4-6 weeks
- Pure Rust (single platform): 1,500-2,500 lines, 1-2 weeks
- FFmpeg bindings: 500-800 lines, 3-5 days

**User Impact**:
- 10-50x encoding speedup
- Real-time 4K/8K encoding
- Reduced CPU usage
- Battery efficiency on laptops
- Professional workflows enabled

---

**Status**: This guide provides everything needed to implement hardware acceleration.
For immediate progress, I'll now work on **WebM muxer** (more achievable) while this serves as a roadmap for hardware acceleration.
