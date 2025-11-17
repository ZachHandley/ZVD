//! AV1 decoder using dav1d
//!
//! This module provides a complete AV1 decoder implementation using the dav1d library,
//! the reference AV1 decoder from VideoLAN. The dav1d-rs crate provides safe Rust
//! bindings to libdav1d.
//!
//! # System Requirements
//!
//! libdav1d must be installed on the system:
//! - Debian/Ubuntu: `apt install libdav1d-dev`
//! - Arch Linux: `pacman -S dav1d`
//! - macOS: `brew install dav1d`
//! - Fedora: `dnf install dav1d-devel`

use crate::codec::{Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::PixelFormat;
use dav1d::{Decoder as Dav1dDecoder, Picture, PixelLayout, PlanarImageComponent};
use std::sync::{Arc, Mutex};

/// AV1 decoder wrapping dav1d
///
/// This decoder uses libdav1d, the reference AV1 decoder from VideoLAN.
/// It provides hardware-accelerated decoding when available and is
/// highly optimized for performance.
pub struct Av1Decoder {
    /// Internal dav1d decoder instance
    decoder: Arc<Mutex<Dav1dDecoder>>,
    /// Buffered decoded pictures waiting to be retrieved
    picture_buffer: Arc<Mutex<Vec<Picture>>>,
}

impl Av1Decoder {
    /// Create a new AV1 decoder with default settings
    pub fn new() -> Result<Self> {
        Self::with_threads(0)
    }

    /// Create a new AV1 decoder with specified thread count
    ///
    /// # Arguments
    ///
    /// * `n_threads` - Number of threads to use (0 = auto-detect)
    pub fn with_threads(n_threads: u32) -> Result<Self> {
        let mut settings = dav1d::Settings::new();
        settings.set_n_threads(n_threads);
        // Set max frame delay for better throughput with multi-threading
        settings.set_max_frame_delay(if n_threads > 1 { 8 } else { 1 });

        let decoder = Dav1dDecoder::with_settings(&settings)
            .map_err(|e| Error::codec(format!("Failed to create dav1d decoder: {}", e)))?;

        Ok(Av1Decoder {
            decoder: Arc::new(Mutex::new(decoder)),
            picture_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Convert dav1d pixel layout and bit depth to our PixelFormat
    fn pixel_layout_to_format(layout: PixelLayout, bits_per_component: usize) -> PixelFormat {
        match (layout, bits_per_component) {
            (PixelLayout::I400, 8) => PixelFormat::GRAY8,
            (PixelLayout::I400, _) => PixelFormat::GRAY16,
            (PixelLayout::I420, 8) => PixelFormat::YUV420P,
            (PixelLayout::I420, 10) => PixelFormat::YUV420P10LE,
            (PixelLayout::I420, _) => PixelFormat::YUV420P,
            (PixelLayout::I422, 8) => PixelFormat::YUV422P,
            (PixelLayout::I422, 10) => PixelFormat::YUV422P10LE,
            (PixelLayout::I422, _) => PixelFormat::YUV422P,
            (PixelLayout::I444, 8) => PixelFormat::YUV444P,
            (PixelLayout::I444, 10) => PixelFormat::YUV444P10LE,
            (PixelLayout::I444, _) => PixelFormat::YUV444P,
        }
    }

    /// Convert a dav1d Picture to our Frame type
    fn picture_to_frame(picture: &Picture) -> Result<Frame> {
        use crate::codec::frame::VideoFrame;
        use crate::util::{Buffer, Timestamp};

        let width = picture.width();
        let height = picture.height();
        let bits_per_component = picture
            .bits_per_component()
            .ok_or_else(|| Error::codec("Invalid bits per component"))?
            .0;
        let pixel_format = Self::pixel_layout_to_format(picture.pixel_layout(), bits_per_component);

        // Extract plane data and convert to Buffers
        let y_plane = picture.plane(PlanarImageComponent::Y);
        let y_buffer = Buffer::from_vec(y_plane.as_ref().to_vec());

        let (u_buffer, v_buffer) = match picture.pixel_layout() {
            PixelLayout::I400 => (Buffer::empty(), Buffer::empty()),
            _ => {
                let u_plane = picture.plane(PlanarImageComponent::U);
                let v_plane = picture.plane(PlanarImageComponent::V);
                (
                    Buffer::from_vec(u_plane.as_ref().to_vec()),
                    Buffer::from_vec(v_plane.as_ref().to_vec()),
                )
            }
        };

        // Get stride information
        let y_stride = picture.stride(PlanarImageComponent::Y) as usize;
        let uv_stride = if picture.pixel_layout() != PixelLayout::I400 {
            picture.stride(PlanarImageComponent::U) as usize
        } else {
            0
        };

        // Create video frame
        let mut video_frame = VideoFrame::new(width, height, pixel_format);

        // Set plane data
        video_frame.data = vec![y_buffer, u_buffer, v_buffer];
        video_frame.linesize = vec![y_stride, uv_stride, uv_stride];

        // Set timestamp if available
        if let Some(pts) = picture.timestamp() {
            video_frame.pts = Timestamp::new(pts);
        }

        // Note: dav1d Picture doesn't expose frame type/keyframe information in the safe API
        // This would need to be tracked separately from packet flags if needed
        // For now, we leave keyframe as false and pict_type as None
        // TODO: Track keyframe status from packet flags during decoding

        Ok(Frame::Video(video_frame))
    }

    /// Try to retrieve pending pictures from the decoder
    fn retrieve_pending_pictures(&mut self) -> Result<()> {
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| Error::codec("Failed to lock decoder"))?;

        let mut picture_buffer = self
            .picture_buffer
            .lock()
            .map_err(|_| Error::codec("Failed to lock picture buffer"))?;

        loop {
            match decoder.get_picture() {
                Ok(picture) => {
                    picture_buffer.push(picture);
                }
                Err(dav1d::Error::Again) => {
                    // No more pictures available
                    break;
                }
                Err(e) => {
                    return Err(Error::codec(format!("Failed to get picture: {}", e)));
                }
            }
        }

        Ok(())
    }
}

impl Default for Av1Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default AV1 decoder")
    }
}

impl Decoder for Av1Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| Error::codec("Failed to lock decoder"))?;

        // Convert PTS to Option<i64> for dav1d (only if valid timestamp)
        let pts = if packet.pts.is_valid() {
            Some(packet.pts.value)
        } else {
            None
        };
        let duration = if packet.duration > 0 {
            Some(packet.duration)
        } else {
            None
        };

        // Convert Buffer to Vec<u8> for dav1d
        let data = packet.data.as_slice().to_vec();

        match decoder.send_data(data, None, pts, duration) {
            Ok(()) => {
                // Data consumed successfully
                drop(decoder); // Release lock before retrieving pictures
                self.retrieve_pending_pictures()?;
                Ok(())
            }
            Err(dav1d::Error::Again) => {
                // Decoder has pending pictures that need to be retrieved first
                drop(decoder); // Release lock
                self.retrieve_pending_pictures()?;

                // Try again after retrieving pictures
                let mut decoder = self
                    .decoder
                    .lock()
                    .map_err(|_| Error::codec("Failed to lock decoder"))?;

                decoder
                    .send_pending_data()
                    .map_err(|e| Error::codec(format!("Failed to send pending data: {}", e)))?;

                drop(decoder);
                self.retrieve_pending_pictures()?;
                Ok(())
            }
            Err(e) => Err(Error::codec(format!("Failed to send packet: {}", e))),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // First try to retrieve any pending pictures
        self.retrieve_pending_pictures()?;

        // Get a picture from buffer
        let mut picture_buffer = self
            .picture_buffer
            .lock()
            .map_err(|_| Error::codec("Failed to lock picture buffer"))?;

        if picture_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        let picture = picture_buffer.remove(0);
        drop(picture_buffer); // Release lock

        Self::picture_to_frame(&picture)
    }

    fn flush(&mut self) -> Result<()> {
        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| Error::codec("Failed to lock decoder"))?;

        decoder.flush();

        // Clear picture buffer
        let mut picture_buffer = self
            .picture_buffer
            .lock()
            .map_err(|_| Error::codec("Failed to lock picture buffer"))?;

        picture_buffer.clear();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av1_decoder_creation() {
        let decoder = Av1Decoder::new();
        assert!(
            decoder.is_ok(),
            "Decoder creation failed. Make sure libdav1d is installed."
        );
    }

    #[test]
    fn test_av1_decoder_with_threads() {
        let decoder = Av1Decoder::with_threads(4);
        assert!(
            decoder.is_ok(),
            "Decoder creation with threads failed. Make sure libdav1d is installed."
        );
    }

    #[test]
    fn test_pixel_layout_conversion() {
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I420, 8),
            PixelFormat::YUV420P
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I420, 10),
            PixelFormat::YUV420P10LE
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I422, 8),
            PixelFormat::YUV422P
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I422, 10),
            PixelFormat::YUV422P10LE
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I444, 8),
            PixelFormat::YUV444P
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I444, 10),
            PixelFormat::YUV444P10LE
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I400, 8),
            PixelFormat::GRAY8
        );
    }

    #[test]
    fn test_flush() {
        let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
        // Flush should not error even on empty decoder
        assert!(decoder.flush().is_ok());
    }

    #[test]
    fn test_decoder_state() {
        let decoder = Av1Decoder::new().expect("Failed to create decoder");
        assert!(decoder.decoder.lock().is_ok());
        assert!(decoder.picture_buffer.lock().is_ok());
    }
}
