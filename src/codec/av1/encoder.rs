//! AV1 encoder using rav1e

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use rav1e::prelude::*;

/// AV1 encoder wrapping rav1e
pub struct Av1Encoder {
    context: Context<u8>,
    width: usize,
    height: usize,
    frame_count: u64,
}

impl Av1Encoder {
    /// Create a new AV1 encoder with the given dimensions
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_speed(width, height, 6) // Default to speed 6 (balanced)
    }

    /// Create a new AV1 encoder with specified speed preset
    /// Speed: 0 (slowest/best) to 10 (fastest/worst)
    pub fn with_speed(width: u32, height: u32, speed: u8) -> Result<Self> {
        // Create encoder config with speed preset
        let mut enc = EncoderConfig::with_speed_preset(speed);

        // Set video dimensions
        enc.width = width as usize;
        enc.height = height as usize;

        // Set other basic parameters
        enc.time_base = Rational::new(1, 30); // 30 fps default
        enc.min_key_frame_interval = 12;
        enc.max_key_frame_interval = 240;
        enc.quantizer = 100; // Default quality (0-255, lower is better)

        // Create config with encoder config
        let cfg = Config::default()
            .with_encoder_config(enc)
            .with_threads(4); // Use 4 threads by default

        // Create the context
        let context = cfg
            .new_context()
            .map_err(|e| Error::codec(format!("Failed to create AV1 encoder: {:?}", e)))?;

        Ok(Av1Encoder {
            context,
            width: width as usize,
            height: height as usize,
            frame_count: 0,
        })
    }

    /// Set target bitrate in bits per second
    pub fn set_bitrate(&mut self, bitrate: u64) -> Result<()> {
        // Note: We can't change encoder config after context creation
        // This would require recreating the encoder
        // For now, just return an error
        Err(Error::unsupported(
            "Cannot change bitrate after encoder creation",
        ))
    }

    /// Set quantizer (0-255, lower is higher quality)
    pub fn set_quantizer(&mut self, quantizer: u8) -> Result<()> {
        // Note: We can't change encoder config after context creation
        Err(Error::unsupported(
            "Cannot change quantizer after encoder creation",
        ))
    }

    /// Convert our VideoFrame to rav1e Frame
    fn video_frame_to_rav1e(&self, video_frame: &VideoFrame) -> Result<rav1e::prelude::Frame<u8>> {
        // Ensure we have YUV420P format
        if video_frame.format != PixelFormat::YUV420P {
            return Err(Error::codec(format!(
                "Unsupported pixel format for AV1 encoding: {:?}. Only YUV420P is currently supported.",
                video_frame.format
            )));
        }

        // Create a new frame with rav1e
        let mut frame = self.context.new_frame();

        // Copy Y plane
        if video_frame.data.len() < 1 {
            return Err(Error::codec("Missing Y plane"));
        }
        let y_data = video_frame.data[0].as_slice();
        let y_stride = video_frame.linesize[0];
        let y_plane_stride = frame.planes[0].cfg.stride;
        for (y, row) in frame.planes[0].data_origin_mut().chunks_mut(y_plane_stride).enumerate() {
            if y >= self.height {
                break;
            }
            let src_start = y * y_stride;
            let src_end = (src_start + self.width).min(y_data.len());
            let dst_end = self.width.min(row.len());
            row[..dst_end].copy_from_slice(&y_data[src_start..src_end]);
        }

        // Copy U plane
        if video_frame.data.len() < 2 {
            return Err(Error::codec("Missing U plane"));
        }
        let u_data = video_frame.data[1].as_slice();
        let u_stride = video_frame.linesize[1];
        let u_width = self.width / 2;
        let u_height = self.height / 2;
        let u_plane_stride = frame.planes[1].cfg.stride;
        for (y, row) in frame.planes[1].data_origin_mut().chunks_mut(u_plane_stride).enumerate() {
            if y >= u_height {
                break;
            }
            let src_start = y * u_stride;
            let src_end = (src_start + u_width).min(u_data.len());
            let dst_end = u_width.min(row.len());
            row[..dst_end].copy_from_slice(&u_data[src_start..src_end]);
        }

        // Copy V plane
        if video_frame.data.len() < 3 {
            return Err(Error::codec("Missing V plane"));
        }
        let v_data = video_frame.data[2].as_slice();
        let v_stride = video_frame.linesize[2];
        let v_plane_stride = frame.planes[2].cfg.stride;
        for (y, row) in frame.planes[2].data_origin_mut().chunks_mut(v_plane_stride).enumerate() {
            if y >= u_height {
                break;
            }
            let src_start = y * v_stride;
            let src_end = (src_start + u_width).min(v_data.len());
            let dst_end = u_width.min(row.len());
            row[..dst_end].copy_from_slice(&v_data[src_start..src_end]);
        }

        Ok(frame)
    }
}

impl Encoder for Av1Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(video_frame) => {
                // Convert to rav1e frame
                let rav1e_frame = self.video_frame_to_rav1e(video_frame)?;

                // Send to encoder
                self.context
                    .send_frame(rav1e_frame)
                    .map_err(|e| match e {
                        EncoderStatus::Failure => Error::codec("Encoder failure"),
                        EncoderStatus::EnoughData => Error::codec("Encoder has enough data"),
                        _ => Error::codec(format!("Encoder error: {:?}", e)),
                    })?;

                self.frame_count += 1;
                Ok(())
            }
            Frame::Audio(_) => Err(Error::codec("AV1 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // Try to receive a packet from the encoder
        match self.context.receive_packet() {
            Ok(packet) => {
                // Convert rav1e packet to our Packet type
                let data = Buffer::from_vec(packet.data.to_vec());

                let mut zvd_packet = Packet::new(0, data);

                // Set packet metadata
                zvd_packet.pts = if let Some(pts) = packet.input_frameno.checked_mul(1) {
                    Timestamp::new(pts as i64)
                } else {
                    Timestamp::none()
                };

                zvd_packet.dts = zvd_packet.pts; // For now, DTS = PTS
                zvd_packet.duration = 1;
                zvd_packet.set_keyframe(packet.frame_type == FrameType::KEY);

                Ok(zvd_packet)
            }
            Err(e) => match e {
                EncoderStatus::Encoded => {
                    // This shouldn't happen with receive_packet
                    Err(Error::codec("Unexpected Encoded status"))
                }
                EncoderStatus::LimitReached => Err(Error::EndOfStream),
                EncoderStatus::NeedMoreData => Err(Error::TryAgain),
                EncoderStatus::EnoughData => Err(Error::codec("Encoder has enough data")),
                EncoderStatus::NotReady => Err(Error::TryAgain),
                EncoderStatus::Failure => Err(Error::codec("Encoder failure")),
            },
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Signal end of input
        self.context.flush();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av1_encoder_creation() {
        let encoder = Av1Encoder::new(640, 480);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_with_speed() {
        let encoder = Av1Encoder::with_speed(320, 240, 10); // Fastest
        assert!(encoder.is_ok());
    }
}
