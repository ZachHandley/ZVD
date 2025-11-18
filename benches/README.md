# ZVD Performance Benchmarks

This directory contains Criterion-based performance benchmarks for ZVD codecs and filters.

## Benchmark Suites

### Codec Benchmarks (`codec_benchmarks.rs`)

Measures encoding and decoding performance for:

- **AV1**: Encode/decode at 480p, 720p, 1080p
- **H.264**: Encode/decode at 480p, 720p, 1080p (requires `h264` feature)
- **VP8**: Encode at 480p, 720p (requires `vp8-codec` feature)
- **VP9**: Encode at 480p, 720p (requires `vp9-codec` feature)
- **Opus**: Encode/decode stereo 20ms frames (requires `opus-codec` feature)

### Filter Benchmarks (`filter_benchmarks.rs`)

Measures filter processing performance for:

**Video Filters:**
- **Scale**: 1080p → 720p/480p/4K
- **Crop**: 1080p → 720p region
- **Rotate**: 90°/180°/270° rotation
- **Flip**: Horizontal/vertical flip
- **Brightness/Contrast**: Adjustment on 1080p
- **Filter Chain**: Multiple filters combined

**Audio Filters:**
- **Volume**: Stereo 1024 samples
- **Resample**: Various sample rate conversions
- **Normalize**: RMS normalization
- **Filter Chain**: Multiple audio filters combined

## Running Benchmarks

### Run All Benchmarks (Patent-Free Codecs Only)

```bash
cargo bench --no-default-features
```

### Run All Benchmarks (With All Features)

```bash
cargo bench --all-features
```

### Run Specific Benchmark Suite

```bash
# Codec benchmarks only
cargo bench --bench codec_benchmarks --all-features

# Filter benchmarks only
cargo bench --bench filter_benchmarks
```

### Run Specific Benchmark

```bash
# Run only AV1 encoding benchmarks
cargo bench --bench codec_benchmarks av1_encode

# Run only scale filter benchmarks
cargo bench --bench filter_benchmarks scale_filter
```

### Generate Detailed Reports

Criterion automatically generates HTML reports in `target/criterion/`:

```bash
cargo bench --all-features
open target/criterion/report/index.html
```

## Benchmark Metrics

All benchmarks measure:

- **Throughput**: Pixels/samples processed per second
- **Latency**: Time per iteration
- **Statistical Analysis**: Mean, median, standard deviation

## Feature Requirements

Some benchmarks require specific features to be enabled:

- `h264`: H.264 codec benchmarks
- `vp8-codec`: VP8 codec benchmarks
- `vp9-codec`: VP9 codec benchmarks
- `opus-codec`: Opus codec benchmarks

## System Requirements

For accurate benchmarks:

- Run on a quiet system (minimal background processes)
- Ensure adequate cooling (thermal throttling affects results)
- Run benchmarks multiple times for consistency
- System dependencies must be installed (libdav1d, libvpx, etc.)

## Interpreting Results

### Throughput

Higher is better. Measured in elements/second (pixels for video, samples for audio).

Example:
```
av1_encode/1920x1080  time: [45.2 ms 45.5 ms 45.8 ms]
                      thrpt: [45.2M pixels/s 45.5M pixels/s 45.9M pixels/s]
```

This shows AV1 encoding processes ~45.5 million pixels per second at 1080p.

### Latency

Lower is better. Time to process one frame/packet.

### Comparison Baseline

After running benchmarks once, Criterion establishes a baseline. Subsequent runs compare against this baseline and show performance changes (faster/slower).

## Performance Tips

If benchmarks show suboptimal performance:

1. **Build with optimizations**: Always use `--release` (automatically used with `cargo bench`)
2. **Enable CPU features**: Set `RUSTFLAGS="-C target-cpu=native"`
3. **Adjust encoder settings**: Use faster presets for real-time needs
4. **Multi-threading**: Ensure codecs use multiple threads where supported

## Known Limitations

- Benchmarks require system dependencies to run (cannot run in all CI environments)
- Some codecs (AV1) are intentionally slower for better quality
- First benchmark run may be slower due to caching effects
- Results vary by CPU architecture and available SIMD instructions

## Adding New Benchmarks

To add new benchmarks:

1. Add benchmark function to appropriate file (`codec_benchmarks.rs` or `filter_benchmarks.rs`)
2. Use `criterion_group!` to include in benchmark suite
3. Follow existing patterns for throughput measurement
4. Document feature requirements if codec-specific

Example:

```rust
fn bench_new_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_codec");
    group.throughput(Throughput::Elements(1920 * 1080));

    group.bench_function("1080p", |b| {
        b.iter(|| {
            // Benchmark code here
        });
    });

    group.finish();
}
```

## Continuous Integration

To run benchmarks in CI:

```bash
# Quick benchmark (reduced sample size)
cargo bench --no-fail-fast -- --quick

# Full benchmark with reports
cargo bench --all-features
```

**Note**: CI environments may show inconsistent results due to shared resources. Use local benchmarks for accurate performance measurement.
