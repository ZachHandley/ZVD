//! MP4 container format support
//!
//! This module provides MP4/MPEG-4 Part 14 container format support for ZVD.
//!
//! # Patent Notice
//!
//! **IMPORTANT**: MP4 containers often contain patent-encumbered codecs (H.264, AAC).
//! By using this module, you acknowledge that you are responsible for:
//! - Obtaining necessary patent licenses for commercial use
//! - Complying with all applicable codec licensing terms
//! - Understanding the patent implications in your jurisdiction
//!
//! See `CODEC_LICENSES.md` in the project root for detailed licensing information.
//!
//! # Features
//!
//! This module is only available when the `mp4-support` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! zvd = { version = "0.1", features = ["mp4-support"] }
//! ```
//!
//! Or to enable all patent-encumbered codecs:
//!
//! ```toml
//! [dependencies]
//! zvd = { version = "0.1", features = ["all-codecs"] }
//! ```

pub mod demuxer;
pub mod muxer;

pub use demuxer::Mp4Demuxer;
pub use muxer::Mp4Muxer;
