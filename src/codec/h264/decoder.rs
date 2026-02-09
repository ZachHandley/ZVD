//! H.264 decoder using OpenH264
//!
//! This module provides H.264 decoding using the OpenH264 library. The decoder
//! converts H.264 NAL units to YUV420P video frames.
//!
//! ## Security Notes
//! This decoder requires openh264 >= 0.9.1 which includes fixes for CVE-2025-27091,
//! a heap overflow vulnerability in the decoding functions.

use crate::codec::frame::{PictureType, VideoFrame};
use crate::codec::{Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use openh264::decoder::{DecodedYUV, Decoder as OpenH264Decoder};
use openh264::formats::YUVSource;
use std::collections::VecDeque;

use super::nal::contains_keyframe;

/// H.264 decoder wrapping OpenH264
///
/// The decoder maintains an internal frame buffer to handle OpenH264's
/// decode timing where frames may not be immediately available after
/// sending a packet.
pub struct H264Decoder {
    /// The underlying OpenH264 decoder
    decoder: OpenH264Decoder,
    /// Queue of decoded frames waiting to be returned
    frame_queue: VecDeque<VideoFrame>,
    /// Frame count for statistics
    frame_count: u64,
    /// Current PTS being processed
    current_pts: Timestamp,
    /// Whether the decoder has been initialized with valid data
    initialized: bool,
}

impl H264Decoder {
    /// Create a new H.264 decoder
    ///
    /// # Returns
    /// A new H264Decoder instance or an error if initialization fails
    ///
    /// # Example
    /// ```ignore
    /// use zvd_lib::codec::h264::H264Decoder;
    /// let decoder = H264Decoder::new().expect("Failed to create decoder");
    /// ```
    pub fn new() -> Result<Self> {
        let decoder = OpenH264Decoder::new()
            .map_err(|e| Error::codec(format!("Failed to create H.264 decoder: {:?}", e)))?;

        Ok(H264Decoder {
            decoder,
            frame_queue: VecDeque::with_capacity(8),
            frame_count: 0,
            current_pts: Timestamp::none(),
            initialized: false,
        })
    }

    /// Convert OpenH264 DecodedYUV to our VideoFrame format
    ///
    /// This extracts Y, U, V plane data from the OpenH264 decoder output
    /// and creates a proper VideoFrame with correct strides and dimensions.
    fn decoded_yuv_to_video_frame(
        yuv: &DecodedYUV<'_>,
        pts: Timestamp,
        is_keyframe: bool,
    ) -> Result<VideoFrame> {
        let (width, height) = yuv.dimensions();

        // Get strides for each plane
        let (y_stride, u_stride, v_stride) = yuv.strides();

        // Get plane data
        let y_data = yuv.y();
        let u_data = yuv.u();
        let v_data = yuv.v();

        // For YUV420P, the U and V planes are half width and half height
        let uv_height = (height + 1) / 2;

        // Validate plane sizes
        let expected_y_size = y_stride * height;
        let expected_u_size = u_stride * uv_height;
        let expected_v_size = v_stride * uv_height;

        if y_data.len() < expected_y_size {
            return Err(Error::codec(format!(
                "Y plane too small: {} < {}",
                y_data.len(),
                expected_y_size
            )));
        }

        if u_data.len() < expected_u_size {
            return Err(Error::codec(format!(
                "U plane too small: {} < {}",
                u_data.len(),
                expected_u_size
            )));
        }

        if v_data.len() < expected_v_size {
            return Err(Error::codec(format!(
                "V plane too small: {} < {}",
                v_data.len(),
                expected_v_size
            )));
        }

        // Copy plane data, respecting strides
        // For Y plane: copy width bytes per row, skip stride-width padding
        let mut y_plane = Vec::with_capacity(width * height);
        for row in 0..height {
            let src_start = row * y_stride;
            let src_end = src_start + width;
            y_plane.extend_from_slice(&y_data[src_start..src_end]);
        }

        // For U plane
        let uv_width = (width + 1) / 2;
        let mut u_plane = Vec::with_capacity(uv_width * uv_height);
        for row in 0..uv_height {
            let src_start = row * u_stride;
            let src_end = src_start + uv_width;
            u_plane.extend_from_slice(&u_data[src_start..src_end]);
        }

        // For V plane
        let mut v_plane = Vec::with_capacity(uv_width * uv_height);
        for row in 0..uv_height {
            let src_start = row * v_stride;
            let src_end = src_start + uv_width;
            v_plane.extend_from_slice(&v_data[src_start..src_end]);
        }

        // Create the video frame
        let mut frame = VideoFrame::new(width as u32, height as u32, PixelFormat::YUV420P);
        frame.pts = pts;
        frame.keyframe = is_keyframe;
        frame.pict_type = if is_keyframe {
            PictureType::I
        } else {
            PictureType::P
        };

        // Set plane data
        frame.data.push(Buffer::from_vec(y_plane));
        frame.data.push(Buffer::from_vec(u_plane));
        frame.data.push(Buffer::from_vec(v_plane));

        // Set linesizes (compact, no padding after stripping)
        frame.linesize.push(width);
        frame.linesize.push(uv_width);
        frame.linesize.push(uv_width);

        Ok(frame)
    }
}

impl Decoder for H264Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let data = packet.data.as_slice();

        if data.is_empty() {
            return Ok(());
        }

        // Store the PTS for the decoded frame
        let pts = packet.pts;

        // Check if this packet contains a keyframe (IDR NAL unit)
        let is_keyframe = contains_keyframe(data);

        // Decode the packet - OpenH264 expects NAL units with start codes
        // We need to convert the frame first before adding to queue to avoid
        // borrow checker issues (decode returns a reference into self.decoder)
        let decode_result = self.decoder.decode(data);

        match decode_result {
            Ok(Some(yuv)) => {
                // Frame decoded successfully - convert to VideoFrame first
                let frame = Self::decoded_yuv_to_video_frame(&yuv, pts, is_keyframe)?;
                // Now we can mutate self to add frame to queue
                self.initialized = true;
                self.frame_queue.push_back(frame);
                self.frame_count += 1;
            }
            Ok(None) => {
                // No frame yet, need more data (buffering internally)
                // This is normal for B-frames or when decoder needs more NAL units
            }
            Err(e) => {
                // Log error but don't fail completely - may be recoverable
                return Err(Error::codec(format!("H.264 decoding failed: {:?}", e)));
            }
        }

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // Return queued frames first
        if let Some(frame) = self.frame_queue.pop_front() {
            return Ok(Frame::Video(frame));
        }

        // No frames available
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // Flush remaining frames from the decoder
        // In OpenH264 0.9.x, flush_remaining returns Vec<DecodedYUV>
        // We collect all frames first to avoid borrow checker issues
        let flush_result = self.decoder.flush_remaining();

        match flush_result {
            Ok(frames) => {
                // Convert all flushed frames to VideoFrames first
                let mut converted_frames = Vec::new();
                for yuv in &frames {
                    match Self::decoded_yuv_to_video_frame(yuv, Timestamp::none(), false) {
                        Ok(frame) => converted_frames.push(frame),
                        Err(e) => {
                            tracing::warn!("Error converting flushed H.264 frame: {:?}", e);
                        }
                    }
                }
                // Now add all frames to queue
                for frame in converted_frames {
                    self.frame_queue.push_back(frame);
                    self.frame_count += 1;
                }
            }
            Err(e) => {
                tracing::warn!("Error during H.264 decoder flush: {:?}", e);
            }
        }
        Ok(())
    }
}

impl Default for H264Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default H264Decoder")
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

    #[test]
    fn test_h264_decoder_default() {
        let decoder = H264Decoder::default();
        assert_eq!(decoder.frame_count, 0);
        assert!(decoder.frame_queue.is_empty());
    }

    #[test]
    fn test_h264_decoder_empty_packet() {
        let mut decoder = H264Decoder::new().unwrap();
        let packet = Packet::new(0, Buffer::empty());
        assert!(decoder.send_packet(&packet).is_ok());
    }

    #[test]
    fn test_h264_decoder_receive_empty() {
        let mut decoder = H264Decoder::new().unwrap();
        match decoder.receive_frame() {
            Err(Error::TryAgain) => {} // Expected
            other => panic!("Expected TryAgain, got {:?}", other),
        }
    }

    #[test]
    fn test_h264_decoder_flush_empty() {
        let mut decoder = H264Decoder::new().unwrap();
        assert!(decoder.flush().is_ok());
    }
}
