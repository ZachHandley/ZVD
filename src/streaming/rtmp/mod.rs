//! RTMP (Real-Time Messaging Protocol) streaming support
//!
//! RTMP is Adobe's protocol for streaming audio, video, and data over the Internet.
//! Commonly used for live streaming to platforms like YouTube, Twitch, Facebook Live.
//!
//! ## Features
//!
//! - RTMP handshake and connection establishment
//! - AMF0 encoding for commands and metadata
//! - Chunked message protocol
//! - Live stream publishing
//! - FLV tag format support
//!
//! ## Usage
//!
//! ```rust,no_run
//! use zvd::streaming::rtmp::RtmpClient;
//!
//! let mut client = RtmpClient::new();
//! client.connect("rtmp://live.example.com/app", "stream_key")?;
//!
//! // Send video/audio packets
//! client.send_video_packet(&video_data, timestamp, is_keyframe)?;
//! client.send_audio_packet(&audio_data, timestamp)?;
//!
//! client.disconnect()?;
//! ```

pub mod protocol;
pub mod amf0;
pub mod client;

pub use protocol::{RtmpHandshake, ChunkStream, RtmpMessage, MessageType};
pub use amf0::{Amf0Value, Amf0Encoder};
pub use client::RtmpClient;
