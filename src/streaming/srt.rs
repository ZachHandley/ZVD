//! SRT (Secure Reliable Transport) support
//!
//! SRT is an open source video transport protocol that optimizes streaming
//! performance across unpredictable networks with secure streams.

use super::{StreamProtocol, StreamSession};
use crate::error::{Error, Result};
use crate::format::Packet;

/// SRT mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtMode {
    /// Caller mode (initiates connection)
    Caller,
    /// Listener mode (accepts connections)
    Listener,
    /// Rendezvous mode (both sides connect)
    Rendezvous,
}

/// SRT session
pub struct SrtSession {
    url: String,
    connected: bool,
    mode: SrtMode,
    latency_ms: u32,
    passphrase: Option<String>,
}

impl SrtSession {
    /// Create a new SRT session
    pub fn new(mode: SrtMode) -> Self {
        SrtSession {
            url: String::new(),
            connected: false,
            mode,
            latency_ms: 120, // Default 120ms latency
            passphrase: None,
        }
    }

    /// Set latency in milliseconds
    pub fn set_latency(&mut self, latency_ms: u32) {
        self.latency_ms = latency_ms;
    }

    /// Set encryption passphrase
    pub fn set_passphrase(&mut self, passphrase: &str) {
        self.passphrase = Some(passphrase.to_string());
    }

    /// Get mode
    pub fn mode(&self) -> SrtMode {
        self.mode
    }
}

impl StreamSession for SrtSession {
    fn connect(&mut self, url: &str) -> Result<()> {
        // Placeholder for SRT connection
        // Would use srt-rs or similar for actual implementation
        self.url = url.to_string();
        self.connected = true;
        Ok(())
    }

    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        if !self.connected {
            return Err(Error::invalid_state("Not connected to SRT"));
        }
        // Placeholder - would send over SRT
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if !self.connected {
            return Err(Error::invalid_state("Not connected to SRT"));
        }
        // Placeholder - would receive from SRT
        Err(Error::TryAgain)
    }

    fn disconnect(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn protocol(&self) -> StreamProtocol {
        StreamProtocol::SRT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_session_creation() {
        let session = SrtSession::new(SrtMode::Caller);
        assert!(!session.is_connected());
        assert_eq!(session.mode(), SrtMode::Caller);
        assert_eq!(session.latency_ms, 120);
    }

    #[test]
    fn test_srt_set_latency() {
        let mut session = SrtSession::new(SrtMode::Listener);
        session.set_latency(200);
        assert_eq!(session.latency_ms, 200);
    }
}
