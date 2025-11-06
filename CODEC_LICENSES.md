# Codec Licenses and Patent Notice

ZVD supports multiple audio and video codecs, including both patent-free open codecs and codecs that may be subject to patent licensing requirements. This document outlines the licensing requirements and patent implications for each supported codec.

## Patent-Free Open Codecs (Recommended)

These codecs are free to use without patent licensing concerns:

### Video Codecs
- **AV1** - Royalty-free, open standard developed by Alliance for Open Media
  - License: BSD-style licenses (dav1d, rav1e)
  - No patent licensing fees
  - Recommended for new projects

### Audio Codecs
- **FLAC** - Free Lossless Audio Codec
  - License: BSD-style
  - No patent licensing fees

- **Vorbis** - Open audio codec
  - License: BSD-style
  - No patent licensing fees

- **Opus** - Modern low-latency audio codec
  - License: BSD-style
  - No patent licensing fees

## Codecs with Patent Considerations

**IMPORTANT**: The following codecs may be subject to patent licensing requirements. By using these codecs in ZVD, you acknowledge that:

1. You are responsible for obtaining any necessary patent licenses
2. You will comply with all applicable patent licensing terms
3. The ZVD project makes no warranties regarding patent compliance
4. You may need to pay licensing fees depending on your use case

### H.264/AVC (Advanced Video Coding)

**Patents**: Subject to MPEG LA patent pool
**Implementation**: OpenH264 library (Cisco provides binary with patent license)
**License**: BSD 2-Clause (source code), Cisco Binary License (compiled binary)

**Patent Licensing**:
- Cisco provides patent licensing coverage when using their pre-built binary
- If compiling from source without Cisco's binary, you may need your own patent license
- Commercial use may require licensing fees from MPEG LA
- See: https://www.openh264.org/BINARY_LICENSE.txt

**When to pay**:
- Commercial distribution of encoded video
- Streaming services
- Professional video production
- Check MPEG LA licensing terms: https://www.mpegla.com/programs/avc-h-264/

### H.265/HEVC (High Efficiency Video Coding)

**Patents**: Subject to multiple patent pools (MPEG LA, HEVC Advance, Velos Media)
**License**: Various depending on implementation

**Patent Licensing**:
- More complex patent situation than H.264
- Multiple patent pools require separate licenses
- Fees can be substantial for commercial use
- Check all three patent pools for requirements

### AAC (Advanced Audio Coding)

**Patents**: Subject to patent licensing
**Implementation**: Symphonia AAC decoder (via libfdk-aac)
**License**: Fraunhofer FDK AAC Codec Library (custom license)

**Patent Licensing**:
- May require licensing fees for commercial distribution
- Decoder use is generally free for personal use
- Encoder use may require licensing for commercial applications
- Via Licensing manages AAC patent pool: https://www.via-la.com/

## Recommendations by Use Case

### Personal/Educational Use
- All codecs available in ZVD can be used
- Patent licensing typically not required for personal use

### Open Source Projects
- **Recommended**: AV1, FLAC, Vorbis, Opus
- **Caution**: H.264, H.265, AAC (check project requirements)

### Commercial Applications
- **Recommended**: AV1, FLAC, Vorbis, Opus (no licensing fees)
- **Requires Review**: H.264, H.265, AAC
  - Consult with legal counsel
  - Obtain necessary patent licenses
  - Budget for licensing fees

### Streaming/Broadcasting
- **Best Choice**: AV1 + Opus (fully royalty-free)
- **Alternative**: H.264 + AAC (widely supported, licensing required)

## Enabling Codecs in ZVD

### Patent-Free Codecs (Enabled by Default)
These are enabled by default and require no additional configuration:
- AV1 (dav1d, rav1e)
- FLAC, Vorbis (Symphonia)

### Patent-Encumbered Codecs (Opt-In)
Enable these features in your Cargo.toml:

```toml
[dependencies]
zvd = { version = "0.1", features = ["h264", "h265", "aac"] }
```

Or build with specific features:
```bash
cargo build --features h264,aac
```

## Disclaimer

This document is provided for informational purposes only and does not constitute legal advice. Patent laws vary by jurisdiction, and licensing requirements may change over time. The ZVD project:

- Does not provide patent licenses
- Does not indemnify users against patent claims
- Is not responsible for users' compliance with patent licensing requirements
- Recommends consulting with legal counsel for commercial use

**By using patent-encumbered codecs in ZVD, you accept full responsibility for obtaining necessary licenses and complying with patent requirements.**

## Further Reading

- MPEG LA: https://www.mpegla.com/
- Cisco OpenH264: https://www.openh264.org/
- Alliance for Open Media: https://aomedia.org/
- Via Licensing (AAC): https://www.via-la.com/

## Last Updated

January 2025
