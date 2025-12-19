//! Streaming protocol support
//!
//! This module provides support for various streaming protocols:
//!
//! - **RTMP** - Real-Time Messaging Protocol (live streaming)
//! - **HLS** - HTTP Live Streaming (Apple)
//! - **DASH** - Dynamic Adaptive Streaming over HTTP (MPEG-DASH)
//! - **RTP/RTSP** - Real-time Transport Protocol / Real Time Streaming Protocol
//! - **SRT** - Secure Reliable Transport
//! - **WebRTC** - Web Real-Time Communication

pub mod dash;
pub mod hls;
pub mod rtmp;
pub mod rtp;
pub mod srt;

// Additional streaming modules
pub mod ll_hls; // Low-latency HLS
pub mod pipeline; // Streaming pipeline
pub mod segmenter; // Media segmenter

use crate::error::Result;
use crate::format::Packet;
use std::time::Duration;

/// Streaming protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamProtocol {
    RTMP,
    HLS,
    DASH,
    RTP,
    RTSP,
    SRT,
    WebRTC,
}

/// Streaming quality profile
#[derive(Debug, Clone)]
pub struct QualityProfile {
    pub name: String,
    pub bitrate: u32,
    pub width: u32,
    pub height: u32,
    pub framerate: u32,
}

impl QualityProfile {
    /// Create a quality profile
    pub fn new(name: &str, bitrate: u32, width: u32, height: u32, framerate: u32) -> Self {
        QualityProfile {
            name: name.to_string(),
            bitrate,
            width,
            height,
            framerate,
        }
    }

    /// Common quality profiles
    pub fn profiles() -> Vec<QualityProfile> {
        vec![
            QualityProfile::new("240p", 400_000, 426, 240, 30),
            QualityProfile::new("360p", 800_000, 640, 360, 30),
            QualityProfile::new("480p", 1_200_000, 854, 480, 30),
            QualityProfile::new("720p", 2_500_000, 1280, 720, 30),
            QualityProfile::new("1080p", 5_000_000, 1920, 1080, 30),
            QualityProfile::new("1440p", 9_000_000, 2560, 1440, 30),
            QualityProfile::new("4K", 20_000_000, 3840, 2160, 30),
        ]
    }
}

/// Streaming session
pub trait StreamSession: Send {
    /// Connect to streaming server
    fn connect(&mut self, url: &str) -> Result<()>;

    /// Send packet to stream
    fn send_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Receive packet from stream
    fn receive_packet(&mut self) -> Result<Packet>;

    /// Disconnect from streaming server
    fn disconnect(&mut self) -> Result<()>;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Get protocol type
    fn protocol(&self) -> StreamProtocol;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_profiles() {
        let profiles = QualityProfile::profiles();
        assert!(!profiles.is_empty());
        assert_eq!(profiles[0].name, "240p");
        assert_eq!(profiles[6].name, "4K");
    }

    #[test]
    fn test_quality_profile_creation() {
        let profile = QualityProfile::new("test", 1000000, 1920, 1080, 60);
        assert_eq!(profile.name, "test");
        assert_eq!(profile.bitrate, 1000000);
    }
}
