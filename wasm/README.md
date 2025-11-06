# ZVD WASM - Web Assembly Bindings

Browser-compatible multimedia processing using ZVD.

## Features

- Video encoding/decoding (AV1, VP8, VP9)
- Audio encoding/decoding (Opus, Vorbis, FLAC, MP3)
- Video filters (scale, crop, rotate, flip, brightness/contrast)
- Audio filters (volume, normalize, resample)
- Zero-copy operations where possible
- WebWorker support

## Building

### Prerequisites

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Or using cargo
cargo install wasm-pack
```

### Build for Web

```bash
# Build optimized WASM module
wasm-pack build --target web --release

# Build for Node.js
wasm-pack build --target nodejs --release

# Build for bundlers (webpack, rollup, etc.)
wasm-pack build --target bundler --release
```

## Usage

### In Browser (ES Modules)

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ZVD WASM Example</title>
</head>
<body>
    <h1>ZVD Multimedia Processing</h1>
    <video id="input" controls></video>
    <canvas id="output"></canvas>

    <script type="module">
        import init, {
            WasmVideoDecoder,
            WasmVideoEncoder,
            WasmFilterChain,
            list_codecs
        } from './pkg/zvd_wasm.js';

        async function main() {
            // Initialize WASM module
            await init();

            // List available codecs
            const codecs = list_codecs();
            console.log('Available codecs:', codecs);

            // Create video decoder
            const decoder = new WasmVideoDecoder('av1');

            // Create filter chain
            const filters = new WasmFilterChain();
            filters.add_scale(1280, 720);
            filters.add_rotate(90);

            // Process video frames
            // (actual video processing code here)
        }

        main();
    </script>
</body>
</html>
```

### With Webpack/Bundlers

```javascript
import init, { WasmVideoEncoder, WasmFilterChain } from 'zvd-wasm';

async function processVideo(videoFile) {
    await init();

    const encoder = new WasmVideoEncoder('av1', 1920, 1080);
    const filters = new WasmFilterChain();

    filters.add_scale(1280, 720);
    filters.add_crop(0, 0, 1280, 720);

    // Process video frames
    for (const frame of videoFrames) {
        const filtered = filters.apply(frame);
        const encoded = encoder.encode_frame(filtered);
        // ... handle encoded data
    }

    encoder.flush();
}
```

### WebWorker Support

```javascript
// worker.js
import init, { WasmVideoEncoder } from './pkg/zvd_wasm.js';

self.onmessage = async (e) => {
    await init();

    const encoder = new WasmVideoEncoder(e.data.codec, e.data.width, e.data.height);

    // Process frames in worker
    const encoded = encoder.encode_frame(e.data.frameData);
    self.postMessage({ encoded });
};
```

## API Reference

### Video Processing

#### `WasmVideoEncoder`
- `new(codec, width, height)` - Create video encoder
- `encode_frame(frameData)` - Encode a video frame
- `flush()` - Flush encoder and get remaining data

#### `WasmVideoDecoder`
- `new(codec)` - Create video decoder
- `decode_packet(packetData)` - Decode a video packet

### Audio Processing

#### `WasmAudioEncoder`
- `new(codec, sampleRate, channels)` - Create audio encoder
- `encode_samples(samples)` - Encode audio samples

#### `WasmAudioDecoder`
- `new(codec)` - Create audio decoder
- `decode_packet(packetData)` - Decode audio packet

### Filters

#### `WasmFilterChain`
- `new()` - Create filter chain
- `add_scale(width, height)` - Add scale filter
- `add_crop(x, y, width, height)` - Add crop filter
- `add_rotate(angle)` - Add rotate filter (90, 180, 270)
- `add_flip(horizontal, vertical)` - Add flip filter
- `apply(frameData)` - Apply all filters to frame

### Utilities

- `get_version()` - Get ZVD version
- `list_codecs()` - List available codecs

## Performance Tips

1. **Use WebWorkers** - Process video/audio in background threads
2. **Batch operations** - Process multiple frames before returning to JS
3. **Zero-copy** - Use TypedArrays and avoid unnecessary copying
4. **Optimize bundle** - Use `--release` builds and enable LTO
5. **Stream processing** - Process chunks instead of entire files

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 89+
- Safari 15+
- Modern browsers with WebAssembly support

## Size Optimization

The WASM module is optimized for size:
- Release builds use `opt-level = "z"`
- LTO (Link Time Optimization) enabled
- Dead code elimination
- Typical bundle size: ~500KB gzipped

## Limitations

- No hardware acceleration (uses CPU only)
- Some features require specific browser APIs
- Patent-encumbered codecs (H.264, AAC) not available in WASM build
- Large files may require streaming approach

## Examples

See `examples/` directory for complete examples:
- `video-transcoder/` - Video transcoding app
- `audio-processor/` - Audio processing app
- `filter-demo/` - Video filter demonstration
- `webcam-capture/` - Real-time webcam processing

## License

MIT OR Apache-2.0
