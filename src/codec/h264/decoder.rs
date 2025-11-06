//! H.264 decoder using OpenH264

use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::Timestamp;
use openh264::decoder::Decoder as OpenH264Decoder;

/// H.264 decoder wrapping OpenH264
pub struct H264Decoder {
    decoder: OpenH264Decoder,
    frame_count: u64,
}

impl H264Decoder {
    /// Create a new H.264 decoder
    pub fn new() -> Result<Self> {
        let decoder = OpenH264Decoder::new()
            .map_err(|e| Error::codec(format!("Failed to create H.264 decoder: {:?}", e)))?;

        Ok(H264Decoder {
            decoder,
            frame_count: 0,
        })
    }

    /// Convert OpenH264 YUV data to VideoFrame
    /// Note: This is a placeholder implementation. The actual openh264 YUVBuffer API
    /// differs from the expected interface. A full implementation would need to:
    /// 1. Access the YUVBuffer's dimensions and strides
    /// 2. Extract Y, U, V plane data
    /// 3. Convert to our VideoFrame format
    #[allow(dead_code)]
    fn yuv_to_video_frame(&self, _yuv: &openh264::formats::YUVBuffer, _pts: Timestamp) -> Result<VideoFrame> {
        // Placeholder - would need proper YUVBuffer API access
        Err(Error::unsupported(
            "YUV to VideoFrame conversion needs proper openh264 API integration"
        ))
    }
}

impl Decoder for H264Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Decode the packet
        let data = packet.data.as_slice();

        // OpenH264 expects NAL units
        // The packet data should already be in NAL unit format
        match self.decoder.decode(data) {
            Ok(Some(_yuv)) => {
                // Frame decoded successfully
                // In a real implementation, we'd store this for receive_frame()
                self.frame_count += 1;
                Ok(())
            }
            Ok(None) => {
                // No frame yet, need more data
                Ok(())
            }
            Err(e) => Err(Error::codec(format!("H.264 decoding failed: {:?}", e))),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // OpenH264 returns decoded frames immediately in decode() call
        // This is a simplified implementation - in a real scenario, we'd need
        // to buffer the decoded frames and return them here

        // For now, return TryAgain to indicate no frame is immediately available
        // A more complete implementation would buffer frames from decode()
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // OpenH264 decoder doesn't require explicit flushing
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h264_decoder_creation() {
        let decoder = H264Decoder::new();
        assert!(decoder.is_ok());
    }
}
