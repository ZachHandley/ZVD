# Building ZVD

This document provides detailed build instructions for ZVD on all supported platforms.

## Quick Start

### 1. Install System Dependencies

ZVD requires libdav1d for AV1 decoding. Choose your platform:

<details>
<summary><b>Debian/Ubuntu</b></summary>

```bash
sudo apt update
sudo apt install libdav1d-dev
```

To verify installation:
```bash
pkg-config --modversion dav1d
# Should output: 1.3.0 or higher
```
</details>

<details>
<summary><b>Arch Linux</b></summary>

```bash
sudo pacman -Syu
sudo pacman -S dav1d
```

To verify installation:
```bash
pkg-config --modversion dav1d
```
</details>

<details>
<summary><b>Fedora/RHEL</b></summary>

```bash
sudo dnf install dav1d-devel
```

To verify installation:
```bash
pkg-config --modversion dav1d
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install dav1d
```

To verify installation:
```bash
pkg-config --modversion dav1d
```
</details>

<details>
<summary><b>Windows</b></summary>

Option 1: Using vcpkg
```powershell
vcpkg install dav1d
```

Option 2: Download pre-built binaries
1. Download from [VideoLAN](https://code.videolan.org/videolan/dav1d/-/releases)
2. Extract to a known location
3. Set environment variables:
   ```powershell
   $env:PKG_CONFIG_PATH = "C:\path\to\dav1d\lib\pkgconfig"
   ```

To verify installation:
```powershell
pkg-config --modversion dav1d
```
</details>

### 2. Build ZVD

```bash
git clone https://github.com/ZachHandley/ZVD.git
cd ZVD
cargo build --release
```

### 3. Run Tests

```bash
cargo test --lib
```

## Build Options

### Default Build (All Features)

```bash
cargo build --release
```

Includes:
- AV1 decoder (via libdav1d)
- AV1 encoder (via rav1e)
- All patent-free codecs (Opus, Vorbis, FLAC, MP3)

### Minimal Build (Patent-Free Only)

```bash
cargo build --release --no-default-features
```

Includes only:
- Core multimedia framework
- Patent-free codecs

### Optional Patent-Encumbered Codecs

**⚠️ Warning**: These codecs may require patent licenses in some jurisdictions. See [CODEC_LICENSES.md](CODEC_LICENSES.md).

```bash
# H.264 support (requires libopenh264)
cargo build --release --features h264

# AAC support
cargo build --release --features aac

# All codecs
cargo build --release --features all-codecs
```

## Troubleshooting

### Error: "library `dav1d` required by crate `dav1d-sys` was not found"

**Cause**: libdav1d development files are not installed or not in PKG_CONFIG_PATH.

**Solution**:
1. Install libdav1d (see platform instructions above)
2. Verify installation: `pkg-config --modversion dav1d`
3. If installed but not found, set PKG_CONFIG_PATH:
   ```bash
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

### Error: "pkg-config exited with status code 1"

**Cause**: pkg-config is not installed.

**Solution**:
```bash
# Debian/Ubuntu
sudo apt install pkg-config

# macOS
brew install pkg-config

# Fedora
sudo dnf install pkgconf-pkg-config
```

### Error: Runtime "cannot open shared object file: libdav1d.so"

**Cause**: Runtime library is installed but not in library path.

**Solution**:
```bash
# Temporary fix
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Permanent fix (Linux)
sudo ldconfig

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### Warning: "unused manifest key: workspace.lints"

**Cause**: Using older Rust version.

**Solution**: Update Rust:
```bash
rustup update stable
```

ZVD requires Rust 2021 edition (1.56.0+). Recommended: 1.70.0+

## Build Performance

### Optimized Release Build

For maximum performance:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This enables CPU-specific optimizations.

### Faster Development Builds

For faster compile times during development:
```bash
cargo build
```

Development builds include debug symbols and no optimizations.

### Build Time Comparison

On a typical system (4-core CPU, SSD):

| Build Type | Time (Clean) | Time (Incremental) |
|------------|--------------|-------------------|
| Debug | ~2 minutes | ~10 seconds |
| Release | ~5 minutes | ~30 seconds |
| Release (native) | ~6 minutes | ~30 seconds |

## Cross-Compilation

### Building for WebAssembly

See [wasm/README.md](wasm/README.md) for detailed WASM build instructions.

Quick start:
```bash
cd wasm
wasm-pack build --target web
```

### Building for ARM

```bash
# Add target
rustup target add aarch64-unknown-linux-gnu

# Install cross-compiler
sudo apt install gcc-aarch64-linux-gnu

# Build
cargo build --release --target aarch64-unknown-linux-gnu
```

Note: Cross-compiling requires cross-compiled libdav1d for the target platform.

## Docker Build

For reproducible builds:

```dockerfile
FROM rust:1.75

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libdav1d-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build ZVD
WORKDIR /app
COPY . .
RUN cargo build --release

# Result in /app/target/release/zvd
```

Build:
```bash
docker build -t zvd .
```

## CI/CD

ZVD includes GitHub Actions workflows:

- `.github/workflows/ci.yml`: Build and test on push/PR
- `.github/workflows/release.yml`: Create releases

Local CI testing:
```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run CI locally
act -j test
```

## Benchmarking

Run performance benchmarks:
```bash
cargo bench
```

This requires:
- libdav1d installed
- Test video files (generated automatically)

Results are saved to `target/criterion/`.

## Static Linking

To create a fully static binary (no runtime dependencies):

```bash
# Linux with musl
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

Note: This requires statically-linked libdav1d, which may need to be built separately.

## Verification

After building, verify the installation:

```bash
# Check binary
./target/release/zvd --version

# Run decoder test
cargo test --lib codec::av1::tests::test_av1_decoder_creation -- --exact

# Expected output:
# test codec::av1::tests::test_av1_decoder_creation ... ok
```

## Getting Help

If you encounter build issues:

1. Check this troubleshooting guide
2. Verify system dependencies: `pkg-config --modversion dav1d`
3. Update Rust: `rustup update`
4. Search existing issues: https://github.com/ZachHandley/ZVD/issues
5. Open a new issue with:
   - Your OS and version
   - Rust version: `rustc --version`
   - Full build output
   - Output of `pkg-config --modversion dav1d`

## Next Steps

After successful build:

- Read the [Quick Start Guide](README.md#quick-start)
- Explore [examples/](examples/)
- Review [API Documentation](https://docs.rs/zvd_lib)
- Check out [WASM support](wasm/README.md)
