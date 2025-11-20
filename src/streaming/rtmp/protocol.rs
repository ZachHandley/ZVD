//! RTMP protocol implementation
//!
//! Core RTMP protocol components including handshake, chunking, and messages

use crate::error::{Error, Result};
use std::io::{Read, Write};

/// RTMP handshake version
pub const RTMP_VERSION: u8 = 3;

/// Default chunk size
pub const DEFAULT_CHUNK_SIZE: u32 = 128;

/// RTMP message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    SetChunkSize = 1,
    Abort = 2,
    Acknowledgement = 3,
    UserControl = 4,
    WindowAckSize = 5,
    SetPeerBandwidth = 6,
    Audio = 8,
    Video = 9,
    DataAmf3 = 15,
    SharedObjectAmf3 = 16,
    CommandAmf3 = 17,
    DataAmf0 = 18,
    SharedObjectAmf0 = 19,
    CommandAmf0 = 20,
    Aggregate = 22,
}

impl MessageType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(MessageType::SetChunkSize),
            2 => Some(MessageType::Abort),
            3 => Some(MessageType::Acknowledgement),
            4 => Some(MessageType::UserControl),
            5 => Some(MessageType::WindowAckSize),
            6 => Some(MessageType::SetPeerBandwidth),
            8 => Some(MessageType::Audio),
            9 => Some(MessageType::Video),
            15 => Some(MessageType::DataAmf3),
            16 => Some(MessageType::SharedObjectAmf3),
            17 => Some(MessageType::CommandAmf3),
            18 => Some(MessageType::DataAmf0),
            19 => Some(MessageType::SharedObjectAmf0),
            20 => Some(MessageType::CommandAmf0),
            22 => Some(MessageType::Aggregate),
            _ => None,
        }
    }
}

/// RTMP message
#[derive(Debug, Clone)]
pub struct RtmpMessage {
    pub timestamp: u32,
    pub message_length: u32,
    pub message_type: MessageType,
    pub message_stream_id: u32,
    pub payload: Vec<u8>,
}

impl RtmpMessage {
    pub fn new(message_type: MessageType, stream_id: u32, payload: Vec<u8>) -> Self {
        RtmpMessage {
            timestamp: 0,
            message_length: payload.len() as u32,
            message_type,
            message_stream_id: stream_id,
            payload,
        }
    }
}

/// RTMP chunk format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkFormat {
    Type0 = 0, // 11 bytes header
    Type1 = 1, // 7 bytes header
    Type2 = 2, // 3 bytes header
    Type3 = 3, // 0 bytes header (continuation)
}

/// RTMP chunk header
#[derive(Debug, Clone)]
pub struct ChunkHeader {
    pub format: ChunkFormat,
    pub chunk_stream_id: u32,
    pub timestamp: Option<u32>,
    pub message_length: Option<u32>,
    pub message_type: Option<MessageType>,
    pub message_stream_id: Option<u32>,
}

impl ChunkHeader {
    /// Create a Type 0 chunk header (full header)
    pub fn type0(chunk_stream_id: u32, message: &RtmpMessage) -> Self {
        ChunkHeader {
            format: ChunkFormat::Type0,
            chunk_stream_id,
            timestamp: Some(message.timestamp),
            message_length: Some(message.message_length),
            message_type: Some(message.message_type),
            message_stream_id: Some(message.message_stream_id),
        }
    }

    /// Encode chunk header to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Basic header
        let fmt = (self.format as u8) << 6;
        if self.chunk_stream_id < 64 {
            bytes.push(fmt | (self.chunk_stream_id as u8));
        } else if self.chunk_stream_id < 320 {
            bytes.push(fmt);
            bytes.push((self.chunk_stream_id - 64) as u8);
        } else {
            bytes.push(fmt | 1);
            let id = self.chunk_stream_id - 64;
            bytes.push((id & 0xFF) as u8);
            bytes.push(((id >> 8) & 0xFF) as u8);
        }

        // Message header (depends on format type)
        match self.format {
            ChunkFormat::Type0 => {
                // 11 bytes: timestamp (3) + message length (3) + message type (1) + message stream id (4)
                let ts = self.timestamp.unwrap_or(0);
                bytes.extend_from_slice(&ts.to_be_bytes()[1..4]); // 3 bytes

                let len = self.message_length.unwrap_or(0);
                bytes.extend_from_slice(&len.to_be_bytes()[1..4]); // 3 bytes

                bytes.push(self.message_type.unwrap() as u8); // 1 byte

                let stream_id = self.message_stream_id.unwrap_or(0);
                bytes.extend_from_slice(&stream_id.to_le_bytes()); // 4 bytes (little-endian!)
            }
            ChunkFormat::Type1 => {
                // 7 bytes: timestamp delta (3) + message length (3) + message type (1)
                let ts = self.timestamp.unwrap_or(0);
                bytes.extend_from_slice(&ts.to_be_bytes()[1..4]);

                let len = self.message_length.unwrap_or(0);
                bytes.extend_from_slice(&len.to_be_bytes()[1..4]);

                bytes.push(self.message_type.unwrap() as u8);
            }
            ChunkFormat::Type2 => {
                // 3 bytes: timestamp delta (3)
                let ts = self.timestamp.unwrap_or(0);
                bytes.extend_from_slice(&ts.to_be_bytes()[1..4]);
            }
            ChunkFormat::Type3 => {
                // 0 bytes: no additional header
            }
        }

        bytes
    }
}

/// RTMP handshake handler
pub struct RtmpHandshake;

impl RtmpHandshake {
    /// Perform RTMP handshake (simple version without encryption)
    pub fn perform_simple_handshake<S: Read + Write>(stream: &mut S) -> Result<()> {
        // C0: Send version byte
        stream.write_all(&[RTMP_VERSION])
            .map_err(|e| Error::Io(e))?;

        // C1: Send 1536 bytes (timestamp + zero + random data)
        let mut c1 = vec![0u8; 1536];
        // Timestamp (4 bytes) - can be 0
        c1[0..4].copy_from_slice(&0u32.to_be_bytes());
        // Zero (4 bytes)
        c1[4..8].copy_from_slice(&[0, 0, 0, 0]);
        // Random data (1528 bytes) - use zeros for simplicity
        // In production, would use random data

        stream.write_all(&c1)
            .map_err(|e| Error::Io(e))?;
        stream.flush()
            .map_err(|e| Error::Io(e))?;

        // S0: Read server version
        let mut s0 = [0u8; 1];
        stream.read_exact(&mut s0)
            .map_err(|e| Error::Io(e))?;

        if s0[0] != RTMP_VERSION {
            return Err(Error::format(format!(
                "Unsupported RTMP version: {}",
                s0[0]
            )));
        }

        // S1: Read server handshake (1536 bytes)
        let mut s1 = vec![0u8; 1536];
        stream.read_exact(&mut s1)
            .map_err(|e| Error::Io(e))?;

        // C2: Echo S1 back to server
        stream.write_all(&s1)
            .map_err(|e| Error::Io(e))?;
        stream.flush()
            .map_err(|e| Error::Io(e))?;

        // S2: Read server's echo (1536 bytes)
        let mut s2 = vec![0u8; 1536];
        stream.read_exact(&mut s2)
            .map_err(|e| Error::Io(e))?;

        // Handshake complete!
        Ok(())
    }

    /// Perform RTMP handshake with separate reader and writer
    pub fn perform_simple_handshake_split<W: Write, R: Read>(writer: &mut W, reader: &mut R) -> Result<()> {
        // C0: Send version byte
        writer.write_all(&[RTMP_VERSION])
            .map_err(|e| Error::Io(e))?;

        // C1: Send 1536 bytes (timestamp + zero + random data)
        let mut c1 = vec![0u8; 1536];
        // Timestamp (4 bytes) - can be 0
        c1[0..4].copy_from_slice(&0u32.to_be_bytes());
        // Zero (4 bytes)
        c1[4..8].copy_from_slice(&[0, 0, 0, 0]);
        // Random data (1528 bytes) - use zeros for simplicity
        // In production, would use random data

        writer.write_all(&c1)
            .map_err(|e| Error::Io(e))?;
        writer.flush()
            .map_err(|e| Error::Io(e))?;

        // S0: Read server version
        let mut s0 = [0u8; 1];
        reader.read_exact(&mut s0)
            .map_err(|e| Error::Io(e))?;

        if s0[0] != RTMP_VERSION {
            return Err(Error::format(format!(
                "Unsupported RTMP version: {}",
                s0[0]
            )));
        }

        // S1: Read server handshake (1536 bytes)
        let mut s1 = vec![0u8; 1536];
        reader.read_exact(&mut s1)
            .map_err(|e| Error::Io(e))?;

        // C2: Echo S1 back to server
        writer.write_all(&s1)
            .map_err(|e| Error::Io(e))?;
        writer.flush()
            .map_err(|e| Error::Io(e))?;

        // S2: Read server's echo (1536 bytes)
        let mut s2 = vec![0u8; 1536];
        reader.read_exact(&mut s2)
            .map_err(|e| Error::Io(e))?;

        // Handshake complete!
        Ok(())
    }
}

/// RTMP chunk stream
pub struct ChunkStream {
    pub chunk_size: u32,
    pub window_ack_size: u32,
    pub peer_bandwidth: u32,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

impl ChunkStream {
    pub fn new() -> Self {
        ChunkStream {
            chunk_size: DEFAULT_CHUNK_SIZE,
            window_ack_size: 2500000,
            peer_bandwidth: 2500000,
            bytes_read: 0,
            bytes_written: 0,
        }
    }

    /// Write a message as chunks
    pub fn write_message<W: Write>(
        &mut self,
        writer: &mut W,
        chunk_stream_id: u32,
        message: &RtmpMessage,
    ) -> Result<()> {
        let payload = &message.payload;
        let mut offset = 0;
        let mut is_first = true;

        while offset < payload.len() {
            let chunk_size = std::cmp::min(self.chunk_size as usize, payload.len() - offset);

            // Create chunk header
            let header = if is_first {
                ChunkHeader::type0(chunk_stream_id, message)
            } else {
                // Type 3 for continuation chunks
                ChunkHeader {
                    format: ChunkFormat::Type3,
                    chunk_stream_id,
                    timestamp: None,
                    message_length: None,
                    message_type: None,
                    message_stream_id: None,
                }
            };

            // Write header
            let header_bytes = header.encode();
            writer.write_all(&header_bytes)
                .map_err(|e| Error::Io(e))?;

            // Write chunk data
            writer.write_all(&payload[offset..offset + chunk_size])
                .map_err(|e| Error::Io(e))?;

            offset += chunk_size;
            is_first = false;
            self.bytes_written += (header_bytes.len() + chunk_size) as u64;
        }

        writer.flush()
            .map_err(|e| Error::Io(e))?;

        Ok(())
    }
}

impl Default for ChunkStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(MessageType::from_u8(1), Some(MessageType::SetChunkSize));
        assert_eq!(MessageType::from_u8(8), Some(MessageType::Audio));
        assert_eq!(MessageType::from_u8(9), Some(MessageType::Video));
        assert_eq!(MessageType::from_u8(20), Some(MessageType::CommandAmf0));
        assert_eq!(MessageType::from_u8(99), None);
    }

    #[test]
    fn test_chunk_header_encoding() {
        let message = RtmpMessage::new(MessageType::Audio, 1, vec![1, 2, 3, 4]);
        let header = ChunkHeader::type0(3, &message);
        let bytes = header.encode();

        // Should have basic header + Type 0 message header
        assert!(!bytes.is_empty());
        assert_eq!(bytes[0] & 0xC0, 0x00); // Format 0
    }

    #[test]
    fn test_chunk_stream_creation() {
        let stream = ChunkStream::new();
        assert_eq!(stream.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(stream.bytes_read, 0);
        assert_eq!(stream.bytes_written, 0);
    }
}
