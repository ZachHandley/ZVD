//! RTMP client for live streaming

use super::amf0::{Amf0Encoder, Amf0Value};
use super::protocol::{ChunkStream, MessageType, RtmpHandshake, RtmpMessage};
use crate::error::{Error, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, SystemTime};

/// RTMP streaming client
pub struct RtmpClient {
    stream: Option<BufWriter<TcpStream>>,
    reader: Option<BufReader<TcpStream>>,
    chunk_stream: ChunkStream,
    connected: bool,
    stream_id: u32,
    transaction_id: f64,
}

impl RtmpClient {
    /// Create a new RTMP client
    pub fn new() -> Self {
        RtmpClient {
            stream: None,
            reader: None,
            chunk_stream: ChunkStream::new(),
            connected: false,
            stream_id: 0,
            transaction_id: 1.0,
        }
    }

    /// Connect to RTMP server and start publishing
    pub fn connect(&mut self, rtmp_url: &str, stream_key: &str) -> Result<()> {
        // Parse RTMP URL
        let (host, port, app) = Self::parse_rtmp_url(rtmp_url)?;

        // Connect TCP
        let tcp = TcpStream::connect(format!("{}:{}", host, port))
            .map_err(|e| Error::Io(e))?;

        // Set timeouts
        tcp.set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| Error::Io(e))?;
        tcp.set_write_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| Error::Io(e))?;

        // Clone for reader
        let tcp_read = tcp.try_clone().map_err(|e| Error::Io(e))?;

        let mut writer = BufWriter::new(tcp);
        let mut reader = BufReader::new(tcp_read);

        // Perform RTMP handshake
        RtmpHandshake::perform_simple_handshake_split(&mut writer, &mut reader)?;

        // Send connect command
        let tc_url = rtmp_url.to_string();
        let connect_data = Amf0Encoder::encode_connect(
            self.transaction_id,
            &app,
            "FMLE/3.0 (compatible; ZVD)",
            &tc_url,
        );
        self.transaction_id += 1.0;

        let connect_msg = RtmpMessage {
            timestamp: 0,
            message_length: connect_data.len() as u32,
            message_type: MessageType::CommandAmf0,
            message_stream_id: 0,
            payload: connect_data,
        };

        self.chunk_stream.write_message(&mut writer, 3, &connect_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        // Read _result response
        // In a full implementation, we'd parse the server's response
        // For now, we'll wait a moment and continue
        std::thread::sleep(Duration::from_millis(100));

        // Send releaseStream
        let release_data = Amf0Encoder::encode_release_stream(self.transaction_id, stream_key);
        self.transaction_id += 1.0;

        let release_msg = RtmpMessage {
            timestamp: 0,
            message_length: release_data.len() as u32,
            message_type: MessageType::CommandAmf0,
            message_stream_id: 0,
            payload: release_data,
        };

        self.chunk_stream.write_message(&mut writer, 3, &release_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        // Send FCPublish
        let fc_publish_data = Amf0Encoder::encode_fc_publish(self.transaction_id, stream_key);
        self.transaction_id += 1.0;

        let fc_publish_msg = RtmpMessage {
            timestamp: 0,
            message_length: fc_publish_data.len() as u32,
            message_type: MessageType::CommandAmf0,
            message_stream_id: 0,
            payload: fc_publish_data,
        };

        self.chunk_stream.write_message(&mut writer, 3, &fc_publish_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        // Send createStream
        let create_stream_data = Amf0Encoder::encode_create_stream(self.transaction_id);
        self.transaction_id += 1.0;

        let create_stream_msg = RtmpMessage {
            timestamp: 0,
            message_length: create_stream_data.len() as u32,
            message_type: MessageType::CommandAmf0,
            message_stream_id: 0,
            payload: create_stream_data,
        };

        self.chunk_stream.write_message(&mut writer, 3, &create_stream_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        // Wait for stream ID response
        std::thread::sleep(Duration::from_millis(100));
        self.stream_id = 1; // Typically the server returns stream ID 1

        // Send publish command
        let publish_data = Amf0Encoder::encode_publish(0.0, stream_key, "live");

        let publish_msg = RtmpMessage {
            timestamp: 0,
            message_length: publish_data.len() as u32,
            message_type: MessageType::CommandAmf0,
            message_stream_id: self.stream_id,
            payload: publish_data,
        };

        self.chunk_stream.write_message(&mut writer, 3, &publish_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        // Store connections
        self.stream = Some(writer);
        self.reader = Some(reader);
        self.connected = true;

        Ok(())
    }

    /// Send video packet
    pub fn send_video_packet(&mut self, data: &[u8], timestamp: u32, is_keyframe: bool) -> Result<()> {
        if !self.connected {
            return Err(Error::InvalidState("Not connected to RTMP server".to_string()));
        }

        let writer = self.stream.as_mut()
            .ok_or_else(|| Error::InvalidState("No writer available".to_string()))?;

        // FLV video tag format:
        // Byte 0: Frame type (4 bits) + Codec ID (4 bits)
        // For H.264: Codec ID = 7
        // Frame type: 1 = keyframe, 2 = inter frame
        let frame_type = if is_keyframe { 1 } else { 2 };
        let codec_id = 7; // H.264/AVC
        let video_header = (frame_type << 4) | codec_id;

        // Build payload: header byte + video data
        let mut payload = Vec::with_capacity(data.len() + 1);
        payload.push(video_header);
        payload.extend_from_slice(data);

        let video_msg = RtmpMessage {
            timestamp,
            message_length: payload.len() as u32,
            message_type: MessageType::Video,
            message_stream_id: self.stream_id,
            payload,
        };

        self.chunk_stream.write_message(writer, 6, &video_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        Ok(())
    }

    /// Send audio packet
    pub fn send_audio_packet(&mut self, data: &[u8], timestamp: u32) -> Result<()> {
        if !self.connected {
            return Err(Error::InvalidState("Not connected to RTMP server".to_string()));
        }

        let writer = self.stream.as_mut()
            .ok_or_else(|| Error::InvalidState("No writer available".to_string()))?;

        // FLV audio tag format:
        // Byte 0: Sound format (4 bits) + Sound rate (2 bits) + Sound size (1 bit) + Sound type (1 bit)
        // For AAC: Sound format = 10
        let sound_format = 10; // AAC
        let sound_rate = 3;    // 44 kHz
        let sound_size = 1;    // 16-bit samples
        let sound_type = 1;    // Stereo
        let audio_header = (sound_format << 4) | (sound_rate << 2) | (sound_size << 1) | sound_type;

        // Build payload: header byte + audio data
        let mut payload = Vec::with_capacity(data.len() + 1);
        payload.push(audio_header);
        payload.extend_from_slice(data);

        let audio_msg = RtmpMessage {
            timestamp,
            message_length: payload.len() as u32,
            message_type: MessageType::Audio,
            message_stream_id: self.stream_id,
            payload,
        };

        self.chunk_stream.write_message(writer, 4, &audio_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        Ok(())
    }

    /// Send metadata (video/audio configuration)
    pub fn send_metadata(&mut self, metadata: Vec<(&str, Amf0Value)>) -> Result<()> {
        if !self.connected {
            return Err(Error::InvalidState("Not connected to RTMP server".to_string()));
        }

        let writer = self.stream.as_mut()
            .ok_or_else(|| Error::InvalidState("No writer available".to_string()))?;

        // Build @setDataFrame metadata
        let mut payload = Vec::new();

        // First element: command name "@setDataFrame"
        payload.extend_from_slice(&Amf0Value::String("@setDataFrame".to_string()).encode());

        // Second element: data stream type "onMetaData"
        payload.extend_from_slice(&Amf0Value::String("onMetaData".to_string()).encode());

        // Third element: metadata object
        let metadata_obj = Amf0Value::command_object(metadata);
        payload.extend_from_slice(&metadata_obj.encode());

        let metadata_msg = RtmpMessage {
            timestamp: 0,
            message_length: payload.len() as u32,
            message_type: MessageType::DataAmf0,
            message_stream_id: self.stream_id,
            payload,
        };

        self.chunk_stream.write_message(writer, 3, &metadata_msg)?;
        writer.flush().map_err(|e| Error::Io(e))?;

        Ok(())
    }

    /// Disconnect from RTMP server
    pub fn disconnect(&mut self) -> Result<()> {
        if let Some(mut writer) = self.stream.take() {
            writer.flush().map_err(|e| Error::Io(e))?;
        }

        self.reader = None;
        self.connected = false;
        self.stream_id = 0;

        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Parse RTMP URL into components
    fn parse_rtmp_url(url: &str) -> Result<(String, u16, String)> {
        // Format: rtmp://host:port/app or rtmp://host/app
        if !url.starts_with("rtmp://") {
            return Err(Error::InvalidInput("Invalid RTMP URL format".to_string()));
        }

        let url_without_scheme = &url[7..]; // Remove "rtmp://"
        let parts: Vec<&str> = url_without_scheme.splitn(2, '/').collect();

        if parts.len() < 2 {
            return Err(Error::InvalidInput("Invalid RTMP URL format".to_string()));
        }

        let host_port = parts[0];
        let app = parts[1].to_string();

        // Parse host and port
        let (host, port) = if host_port.contains(':') {
            let hp: Vec<&str> = host_port.splitn(2, ':').collect();
            let port = hp[1].parse::<u16>()
                .map_err(|_| Error::InvalidInput("Invalid port number".to_string()))?;
            (hp[0].to_string(), port)
        } else {
            (host_port.to_string(), 1935) // Default RTMP port
        };

        Ok((host, port, app))
    }
}

impl Default for RtmpClient {
    fn default() -> Self {
        Self::new()
    }
}

// Implement StreamSession trait
impl crate::streaming::StreamSession for RtmpClient {
    fn connect(&mut self, url: &str) -> Result<()> {
        // Extract stream key from URL (last path component)
        let parts: Vec<&str> = url.rsplitn(2, '/').collect();
        let stream_key = if parts.len() == 2 {
            parts[0]
        } else {
            return Err(Error::InvalidInput("No stream key in URL".to_string()));
        };

        // Reconstruct base URL without stream key
        let base_url = if parts.len() == 2 {
            &url[0..url.len() - stream_key.len() - 1]
        } else {
            url
        };

        self.connect(base_url, stream_key)
    }

    fn send_packet(&mut self, packet: &crate::format::Packet) -> Result<()> {
        use crate::codec::CodecType;

        match packet.codec_type {
            CodecType::Video => {
                // Determine if keyframe based on packet flags
                let is_keyframe = packet.flags & 0x01 != 0;
                self.send_video_packet(&packet.data, packet.pts as u32, is_keyframe)
            }
            CodecType::Audio => {
                self.send_audio_packet(&packet.data, packet.pts as u32)
            }
            _ => Ok(()), // Ignore other packet types
        }
    }

    fn receive_packet(&mut self) -> Result<crate::format::Packet> {
        // RTMP client is typically send-only for publishing
        // Receiving would require implementing a full RTMP message parser
        Err(Error::Unsupported("RTMP client does not support receiving packets".to_string()))
    }

    fn disconnect(&mut self) -> Result<()> {
        self.disconnect()
    }

    fn is_connected(&self) -> bool {
        self.is_connected()
    }

    fn protocol(&self) -> crate::streaming::StreamProtocol {
        crate::streaming::StreamProtocol::RTMP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtmp_client_creation() {
        let client = RtmpClient::new();
        assert!(!client.is_connected());
    }

    #[test]
    fn test_rtmp_url_parsing() {
        let (host, port, app) = RtmpClient::parse_rtmp_url("rtmp://live.example.com/live").unwrap();
        assert_eq!(host, "live.example.com");
        assert_eq!(port, 1935);
        assert_eq!(app, "live");

        let (host, port, app) = RtmpClient::parse_rtmp_url("rtmp://localhost:1936/app/stream").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 1936);
        assert_eq!(app, "app/stream");
    }

    #[test]
    fn test_send_without_connect() {
        let mut client = RtmpClient::new();
        let result = client.send_video_packet(&[0u8; 10], 0, true);
        assert!(result.is_err());
    }
}
