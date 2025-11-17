# AV1 Decoder Implementation

## Overview

ZVD's AV1 decoder is built on **libdav1d**, the reference AV1 decoder developed by VideoLAN. It provides production-ready AV1 video decoding with hardware acceleration support and excellent performance.

## Implementation Details

### Library Choice: dav1d vs rav1d

We chose **dav1d-rs** (Rust bindings to libdav1d) over **rav1d** (pure Rust port) for the following reasons:

#### Why dav1d-rs?

**✅ Pros:**
- **Mature and Production-Ready**: libdav1d is the reference implementation used by VLC, FFmpeg, Chrome, and Firefox
- **Hardware Acceleration**: Supports SIMD optimizations (SSE, AVX2, NEON) for all platforms
- **Performance**: Highly optimized C code with assembly for critical paths
- **Public API**: Complete, documented Rust API via dav1d-rs crate
- **Active Development**: Maintained by VideoLAN with regular updates
- **Battle-Tested**: Used in production by millions of users daily

**❌ Cons:**
- Requires system library dependency (libdav1d)
- Larger binary size due to C library
- FFI boundary (though minimal overhead with safe bindings)

#### Why not rav1d?

**✅ Pros:**
- Pure Rust implementation (no C dependencies)
- Memory safety guarantees throughout
- Easier cross-compilation

**❌ Cons:**
- **No Public API**: Version 1.1.0 marks all types as `pub(crate)` - cannot be used from external crates
- Still ~5% slower than dav1d despite optimizations
- Less mature (still catching up to dav1d performance)
- No hardware acceleration paths yet

**Decision**: We prioritized a **working decoder NOW** over purity. The dav1d-rs crate provides complete functionality with excellent performance.

## System Requirements

### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install libdav1d-dev
```

### Linux (Arch)
```bash
sudo pacman -S dav1d
```

### Linux (Fedora)
```bash
sudo dnf install dav1d-devel
```

### macOS
```bash
brew install dav1d
```

### Windows
Use vcpkg:
```bash
vcpkg install dav1d
```

Or download pre-built binaries from [VideoLAN](https://code.videolan.org/videolan/dav1d).

## Usage

### Basic Decoding

```rust
use zvd_lib::codec::{Decoder, av1::Av1Decoder};
use zvd_lib::format::Packet;

// Create decoder with auto-detected thread count
let mut decoder = Av1Decoder::new()?;

// Or specify thread count explicitly
let mut decoder = Av1Decoder::with_threads(4)?;

// Send encoded packet
decoder.send_packet(&packet)?;

// Receive decoded frame
let frame = decoder.receive_frame()?;

println!("Decoded frame: {}x{}", frame.width, frame.height);
```

### Multi-Threaded Decoding

```rust
use zvd_lib::codec::av1::Av1Decoder;

// Use all available CPU cores
let num_cpus = num_cpus::get() as u32;
let mut decoder = Av1Decoder::with_threads(num_cpus)?;

// Decoder will automatically use frame threading
// for better throughput on multi-core systems
```

### Flushing the Decoder

```rust
// After sending all packets, flush to get remaining frames
decoder.flush()?;

// Retrieve all buffered frames
loop {
    match decoder.receive_frame() {
        Ok(frame) => process_frame(frame),
        Err(e) if e.is_try_again() => break,
        Err(e) => return Err(e),
    }
}
```

## Supported Pixel Formats

The decoder automatically converts from dav1d's internal formats to ZVD pixel formats:

| AV1 Format | Bit Depth | ZVD Format |
|------------|-----------|------------|
| I400 (Grayscale) | 8-bit | `GRAY8` |
| I400 (Grayscale) | 10/12-bit | `GRAY16` |
| I420 (4:2:0) | 8-bit | `YUV420P` |
| I420 (4:2:0) | 10-bit | `YUV420P10LE` |
| I422 (4:2:2) | 8-bit | `YUV422P` |
| I422 (4:2:2) | 10-bit | `YUV422P10LE` |
| I444 (4:4:4) | 8-bit | `YUV444P` |
| I444 (4:4:4) | 10-bit | `YUV444P10LE` |

## Performance Characteristics

### Thread Scaling

libdav1d uses two levels of parallelism:

1. **Tile Threading**: Decodes multiple tiles in parallel
2. **Frame Threading**: Decodes multiple frames in parallel

Recommended thread counts:
- **1-2 cores**: `set_n_threads(1)` - Single-threaded
- **4 cores**: `set_n_threads(2)` - Good balance
- **6+ cores**: `set_n_threads(4)` - Maximum throughput
- **Auto-detect**: `set_n_threads(0)` - Let libdav1d decide

### Memory Usage

Frame delay setting affects memory usage:
- **Low latency**: `set_max_frame_delay(1)` - Minimal buffering
- **High throughput**: `set_max_frame_delay(8)` - More buffering, better threading

Default: 1 for single-threaded, 8 for multi-threaded.

### SIMD Acceleration

libdav1d automatically uses the best available SIMD:
- **x86-64**: AVX2, SSSE3, SSE2
- **ARM**: NEON
- **Other**: Scalar fallback

No configuration needed - detected at runtime.

## Error Handling

The decoder returns descriptive errors:

```rust
match decoder.send_packet(&packet) {
    Ok(()) => { /* Success */ }
    Err(e) if e.is_try_again() => {
        // Decoder buffer full, retrieve frames first
        let frame = decoder.receive_frame()?;
    }
    Err(e) => {
        eprintln!("Decode error: {}", e);
    }
}
```

Common error types:
- **TryAgain**: Temporary condition, retry after retrieving frames
- **Codec**: Decoder error (invalid data, unsupported feature)
- **InvalidArgument**: Invalid API usage
- **NotEnoughMemory**: Out of memory

## Comparison with Other Decoders

| Feature | libdav1d | rav1d | libaom |
|---------|----------|-------|--------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory Safety | ⭐⭐⭐ (FFI) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (FFI) |
| Maturity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| API Availability | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Hardware Accel | ✅ | ❌ | ❌ |
| Pure Rust | ❌ | ✅ | ❌ |
| Binary Size | Large | Medium | Large |

**Verdict**: libdav1d provides the best balance of performance, maturity, and API availability.

## Future Considerations

When rav1d exposes a public API and reaches performance parity, we may consider:

1. Making decoder backend configurable via feature flags
2. Defaulting to rav1d for pure Rust builds
3. Using libdav1d as optional performance optimization

For now, **libdav1d is the production choice**.

## Troubleshooting

### Build Fails: "library `dav1d` required by crate `dav1d-sys` was not found"

Install libdav1d development files (see System Requirements above).

### Runtime Error: "cannot open shared object file: libdav1d.so"

Install the runtime library:
```bash
# Debian/Ubuntu
sudo apt install libdav1d6

# Fedora
sudo dnf install dav1d
```

### Performance Issues

1. Enable multi-threading: `Av1Decoder::with_threads(4)`
2. Increase frame delay: Settings can be configured in future API
3. Check CPU SIMD support: libdav1d logs SIMD features at startup

## References

- [libdav1d repository](https://code.videolan.org/videolan/dav1d)
- [dav1d-rs crate](https://github.com/rust-av/dav1d-rs)
- [AV1 specification](https://aomediacodec.github.io/av1-spec/)
- [rav1d repository](https://github.com/memorysafety/rav1d) (for future reference)
