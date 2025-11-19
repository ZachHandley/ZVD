# ZVD Codec Licensing & Patent Information

**Last Updated**: 2025-11-18

This document provides detailed licensing and patent information for all codecs supported by ZVD. Understanding these requirements is crucial for commercial deployment.

---

## Quick Reference

| Codec | Patent Status | Commercial Use | License Type | Feature Flag |
|-------|---------------|----------------|--------------|--------------|
| **AV1** | ✅ Royalty-free | ✅ Free | Open source | Default |
| **VP8** | ✅ Royalty-free | ✅ Free | BSD-style | `vp8-codec` |
| **VP9** | ✅ Royalty-free | ✅ Free | BSD-style | `vp9-codec` |
| **H.264** | ⚠️ Patent-encumbered | ⚠️ License required | OpenH264 BSD | `h264` |
| **Opus** | ✅ Royalty-free | ✅ Free | BSD | `opus-codec` |
| **FLAC** | ✅ Royalty-free | ✅ Free | BSD-3 | Default |
| **Vorbis** | ✅ Royalty-free | ✅ Free | BSD-3 | Default |
| **MP3** | ✅ Patents expired | ✅ Free | MPL-2.0 | Default |
| **AAC** | ⚠️ Patent-encumbered | ⚠️ License required | MPL-2.0 | `aac` |
| **ProRes** | ⚠️ Apple proprietary | ⚠️ License required | Header parsing only | N/A |
| **DNxHD** | ⚠️ Avid proprietary | ⚠️ License required | Header parsing only | N/A |

---

## Patent-Free Codecs (Recommended)

These codecs are free to use for any purpose, including commercial applications.

### AV1 (Alliance for Open Media)

**Patent Status**: ✅ Royalty-free, patent-licensed
**Commercial Use**: ✅ Free for all uses

**Details**:
- Developed by the Alliance for Open Media (AOMedia)
- Patent pool with royalty-free licensing
- All AOMedia members have cross-licensed their patents
- No fees, no restrictions for deployment or distribution

**Implementation**:
- Encoder: rav1e (Pure Rust, BSD-2-Clause)
- Decoder: dav1d via dav1d-rs (ISC license)

**Recommended For**: Next-generation video, web delivery, streaming

---

### VP8 / VP9 (Google)

**Patent Status**: ✅ Royalty-free
**Commercial Use**: ✅ Free for all uses

**Details**:
- Developed by Google, open-sourced
- Google provides patent indemnification for VP8/VP9 users
- WebM Project provides additional legal protection
- No fees for encoding, decoding, or distribution

**Implementation**:
- Library: libvpx (BSD-3-Clause)
- System dependency: libvpx-dev required

**Recommended For**: WebM videos, YouTube, web conferencing

**Patent Indemnification**:
Google has pledged not to sue users of VP8/VP9 for patent infringement and will defend VP8/VP9 users in patent litigation.

---

### Opus (Xiph.Org + IETF)

**Patent Status**: ✅ Royalty-free, patent-licensed
**Commercial Use**: ✅ Free for all uses

**Details**:
- IETF standard (RFC 6716)
- Combines SILK (Skype) and CELT technologies
- All known patents licensed royalty-free
- Xiph.Org Foundation guarantees free use

**Implementation**:
- Library: libopus (BSD-3-Clause)
- Rust bindings: opus crate (BSD)

**Recommended For**: Voice calls, music streaming, low-latency audio

---

### FLAC (Xiph.Org Foundation)

**Patent Status**: ✅ Completely free
**Commercial Use**: ✅ Free for all uses

**Details**:
- Open source, developed by Xiph.Org Foundation
- No known patents
- Reference implementation is BSD-3-Clause
- Widely supported across all platforms

**Implementation**:
- Decoder: Symphonia (MPL-2.0)
- Pure Rust, no system dependencies

**Recommended For**: Lossless audio archival, music distribution

---

### Vorbis (Xiph.Org Foundation)

**Patent Status**: ✅ Patent-free
**Commercial Use**: ✅ Free for all uses

**Details**:
- Explicitly designed to be patent-free
- Xiph.Org Foundation provides legal guarantees
- Reference implementation: libvorbis (BSD-3-Clause)
- Used in Ogg container format

**Implementation**:
- Decoder: Symphonia (MPL-2.0)
- Pure Rust, no system dependencies

**Recommended For**: Open-source audio, gaming, Ogg streams

---

### MP3 (Fraunhofer IIS / Thomson)

**Patent Status**: ✅ Patents EXPIRED (2017)
**Commercial Use**: ✅ Free for all uses (as of 2017)

**Details**:
- Last patents expired April 16, 2017 (US)
- European patents expired 2012-2015
- Now completely free to use worldwide
- No licensing fees required

**Implementation**:
- Decoder: Symphonia (MPL-2.0)
- Pure Rust, no system dependencies

**Historical Note**:
Prior to 2017, MP3 required patent licensing from Fraunhofer IIS and Thomson. All patents have now expired.

**Recommended For**: Legacy compatibility, widespread support

---

## Patent-Encumbered Codecs (Requires Licensing)

These codecs may require patent licensing fees for commercial use.

### H.264 / AVC (MPEG LA)

**Patent Status**: ⚠️ Patent-encumbered
**Commercial Use**: ⚠️ License REQUIRED for most uses

**Details**:
- Covered by MPEG LA patent pool (multiple patent holders)
- ~2,000+ patents in the pool
- Free for personal use, LIMITED free commercial streaming
- Licensing fees apply for encoding products, paid distribution

**Free Use Cases** (as of last update):
- ✅ Personal, non-commercial use
- ✅ Free Internet streaming (with limitations)
- ✅ Internal use within companies

**Paid Licensing Required**:
- ❌ Selling video content
- ❌ Subscription video services (may apply)
- ❌ Selling encoding/decoding products
- ❌ Broadcasting

**Licensing**:
- MPEG LA: https://www.mpegla.com/programs/avc-h-264/
- Contact MPEG LA for commercial deployment

**Implementation**:
- Library: OpenH264 (Cisco, BSD-2-Clause)
- Cisco provides their own patent license for OpenH264 binaries
- System dependency: libopenh264 required

**Cisco OpenH264 Special Case**:
Cisco has paid MPEG LA fees and provides OpenH264 under BSD license with their patent coverage. However:
- Cisco's license covers OpenH264 binary use
- Does NOT cover H.264 in general
- Does NOT cover derived works or modifications
- Consult legal counsel for commercial use

**Recommended Alternative**: Use AV1, VP8, or VP9 instead

---

### AAC (Via Licensing, MPEG LA)

**Patent Status**: ⚠️ Patent-encumbered
**Commercial Use**: ⚠️ License REQUIRED

**Details**:
- Covered by Via Licensing and MPEG LA patent pools
- Hundreds of patents from multiple holders
- Licensing fees for encoders and decoders
- Fees vary by deployment scale

**Licensing Requirements**:
- Via Licensing: https://www.via-la.com/licensing-2/aac/
- MPEG LA: https://www.mpegla.com/programs/aac/

**License Fees** (approximate, verify with licensors):
- Decoders: Per-unit fees
- Encoders: Per-unit fees
- Broadcasting: Annual fees or per-stream fees
- May have royalty caps or free use thresholds

**ZVD AAC Support**:
- Decoder only (via Symphonia)
- LC-AAC profile only
- Feature-gated with `aac` flag
- User assumes patent licensing responsibility

**Recommended Alternatives**: 
- Use Opus for lossy audio (better quality, free)
- Use FLAC for lossless audio (free)

---

### ProRes (Apple Inc.)

**Patent Status**: ⚠️ Proprietary, likely patent-encumbered
**Commercial Use**: ⚠️ Apple license may be required

**Details**:
- Proprietary codec owned by Apple
- Patent status not fully public
- License terms unclear for third-party implementations
- Widely used in professional video production

**ZVD Support**:
- Header parsing only (format detection)
- Full encoding/decoding not implemented
- Requires FFmpeg integration for full support

**Commercial Use**:
- Using Apple's official encoders/decoders: Generally allowed
- Third-party implementations: Legal status unclear
- Consult Apple and legal counsel

**Licensing**:
- No public licensing program for third-party codec implementation
- Apple Final Cut Pro and related tools include ProRes support

**Recommended Approach**:
- For ProRes workflows, use Apple's official tools
- For codec development, use FFmpeg (LGPL with patent responsibility disclaimer)
- Consider alternatives: DNxHD, uncompressed, or AV1 for archival

---

### DNxHD / DNxHR (Avid Technology)

**Patent Status**: ⚠️ Proprietary, likely patent-encumbered
**Commercial Use**: ⚠️ Avid license may be required

**Details**:
- Proprietary codec owned by Avid Technology
- Patent status not fully disclosed
- License terms unclear for third-party implementations
- Standard in professional broadcast and post-production

**ZVD Support**:
- Header parsing only (format detection)
- Full encoding/decoding not implemented
- Requires FFmpeg integration for full support

**Commercial Use**:
- Using Avid's official tools: Generally allowed
- Third-party implementations: Legal status unclear
- Consult Avid and legal counsel

**Licensing**:
- No clear public licensing program
- Avid Media Composer and related tools include DNxHD/DNxHR

**Recommended Approach**:
- For DNxHD workflows, use Avid's official tools
- For codec development, use FFmpeg (LGPL with patent disclaimer)
- Consider alternatives: ProRes, uncompressed, or AV1 for archival

---

## Software License Information

### ZVD Library

**License**: MIT OR Apache-2.0 (dual-licensed)
- You may choose either license
- Both are permissive open-source licenses
- Commercial use allowed
- Patent grants included

**Important**: 
- ZVD's license does NOT grant patent rights for patent-encumbered codecs
- Users must obtain their own patent licenses for H.264, AAC, etc.
- Patent responsibility lies with the end user

---

### Third-Party Libraries

| Library | License | Patents | Notes |
|---------|---------|---------|-------|
| **rav1e** | BSD-2-Clause | Royalty-free | AV1 encoder |
| **dav1d** | ISC | Royalty-free | AV1 decoder |
| **libvpx** | BSD-3-Clause | Royalty-free | VP8/VP9 |
| **OpenH264** | BSD-2-Clause | See H.264 section | Cisco patent coverage |
| **libopus** | BSD-3-Clause | Royalty-free | Opus codec |
| **Symphonia** | MPL-2.0 | Varies by codec | Audio decoding |

**Note on MPL-2.0**:
- Mozilla Public License 2.0
- Weak copyleft (file-level)
- Commercial use allowed
- Must disclose modifications to MPL files
- Can combine with proprietary code

---

## Deployment Recommendations

### For Open-Source Projects
✅ **Use**: AV1, VP8, VP9, Opus, FLAC, Vorbis, MP3
- All royalty-free
- No licensing fees
- Maximum compatibility with FOSS licenses

### For Commercial Web Streaming
✅ **Use**: AV1, VP9, Opus
- Modern codecs with best quality
- No patent licensing fees
- Wide browser support

⚠️ **H.264 Consideration**:
- Free for certain Internet streaming uses (verify with MPEG LA)
- Broader device compatibility
- Check current MPEG LA licensing terms

### For Professional Video Production
⚠️ **ProRes/DNxHD**:
- Use official tools (Apple, Avid)
- Patent licensing handled by tool vendor
- ZVD provides format detection only

### For Consumer Products
⚠️ **Patent-Encumbered Codecs**:
- Must obtain licenses from MPEG LA, Via Licensing, etc.
- Budget for per-unit fees
- Consult patent attorneys

---

## Legal Disclaimer

**IMPORTANT**: This document provides general information only and is not legal advice.

- Patent laws vary by jurisdiction
- Patent licensing terms change over time
- This information may be outdated
- Consult qualified patent attorneys for commercial deployment

**ZVD Developers**:
- Do NOT provide patent indemnification
- Do NOT guarantee patent safety
- Do NOT act as legal advisors

**Your Responsibility**:
- Research current patent status
- Obtain necessary licenses
- Comply with all applicable laws
- Seek professional legal advice

---

## Frequently Asked Questions

### Q: Can I use ZVD commercially?
**A**: Yes, ZVD's software license (MIT/Apache-2.0) allows commercial use. However, you must obtain separate patent licenses for patent-encumbered codecs (H.264, AAC, etc.).

### Q: Is AV1 really free?
**A**: Yes, AV1 is royalty-free. AOMedia members have cross-licensed all patents and committed to royalty-free licensing.

### Q: What about H.265/HEVC?
**A**: Not currently supported by ZVD. HEVC has complex patent licensing from multiple pools (MPEG LA, HEVC Advance, Velos Media). Recommend using AV1 instead.

### Q: Can I use H.264 for free streaming?
**A**: MPEG LA has provisions for free Internet broadcast streaming, but with limitations. Review current MPEG LA terms and consult legal counsel.

### Q: Do I need a license for Opus in VoIP?
**A**: No, Opus is completely royalty-free for all uses, including VoIP applications.

### Q: What's the safest codec choice?
**A**: For maximum legal safety: AV1 (video) + Opus (audio). Both are modern, high-quality, and completely royalty-free.

---

## Resources

### Patent Licensing Authorities
- **MPEG LA**: https://www.mpegla.com/
- **Via Licensing**: https://www.via-la.com/
- **AOMedia (AV1)**: https://aomedia.org/
- **Xiph.Org (Opus, Vorbis, FLAC)**: https://xiph.org/

### Codec Information
- **AV1**: https://aomedia.org/av1/
- **VP8/VP9**: https://www.webmproject.org/
- **Opus**: https://opus-codec.org/
- **H.264 Info**: https://www.mpegla.com/programs/avc-h-264/

### Legal Information
- **Mozilla Patent Policy**: https://www.mozilla.org/en-US/MPL/2.0/FAQ/
- **Apache-2.0 License**: https://www.apache.org/licenses/LICENSE-2.0
- **MIT License**: https://opensource.org/licenses/MIT

---

**Document Version**: 1.0
**Last Reviewed**: 2025-11-18

For updates and corrections, please file an issue at:
https://github.com/ZachHandley/ZVD/issues
