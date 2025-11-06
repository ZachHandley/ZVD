//! AV1 decoder using dav1d

use crate::codec::{Decoder, Frame, PictureType, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use dav1d::{Decoder as Dav1dDecoder, Picture, PixelLayout, PlanarImageComponent};

/// AV1 decoder wrapping libdav1d
pub struct Av1Decoder {
    decoder: Dav1dDecoder,
}

impl Av1Decoder {
    /// Create a new AV1 decoder with default settings
    pub fn new() -> Result<Self> {
        let decoder = Dav1dDecoder::new()
            .map_err(|e| Error::codec(format!("Failed to create AV1 decoder: {:?}", e)))?;

        Ok(Av1Decoder { decoder })
    }

    /// Convert dav1d PixelLayout to our PixelFormat
    fn pixel_layout_to_format(layout: PixelLayout) -> PixelFormat {
        match layout {
            PixelLayout::I400 => PixelFormat::GRAY8,
            PixelLayout::I420 => PixelFormat::YUV420P,
            PixelLayout::I422 => PixelFormat::YUV422P,
            PixelLayout::I444 => PixelFormat::YUV444P,
            _ => PixelFormat::YUV420P, // Default fallback
        }
    }

    /// Convert dav1d Picture to our VideoFrame
    fn picture_to_frame(picture: Picture, pts: Timestamp) -> VideoFrame {
        let width = picture.width();
        let height = picture.height();
        let pixel_layout = picture.pixel_layout();
        let format = Self::pixel_layout_to_format(pixel_layout);

        // Extract planes from picture
        let mut data = Vec::new();
        let mut linesize = Vec::new();

        // Get Y plane
        let y_plane = picture.plane(PlanarImageComponent::Y);
        data.push(Buffer::from_vec(y_plane.to_vec()));
        linesize.push(picture.stride(PlanarImageComponent::Y) as usize);

        // Get U plane (if present)
        let u_plane = picture.plane(PlanarImageComponent::U);
        if !u_plane.is_empty() {
            data.push(Buffer::from_vec(u_plane.to_vec()));
            linesize.push(picture.stride(PlanarImageComponent::U) as usize);
        }

        // Get V plane (if present)
        let v_plane = picture.plane(PlanarImageComponent::V);
        if !v_plane.is_empty() {
            data.push(Buffer::from_vec(v_plane.to_vec()));
            linesize.push(picture.stride(PlanarImageComponent::V) as usize);
        }

        // Note: The current dav1d-rs bindings (v0.10.x) don't expose frame_type()
        // Frame type information is available in the underlying dav1d library but not
        // exposed in the Rust bindings. Future versions may add this functionality.
        // For now, conservatively mark all frames as P-frames.
        // Workaround: Parse OBU headers manually or upgrade dav1d-rs when available.
        let keyframe = false;
        let pict_type = PictureType::P;

        VideoFrame {
            data,
            linesize,
            width,
            height,
            format,
            pts,
            duration: picture.duration(),
            keyframe,
            pict_type,
        }
    }
}

impl Default for Av1Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default AV1 decoder")
    }
}

impl Decoder for Av1Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let data = packet.data.as_slice().to_vec(); // dav1d needs owned data

        // Send data to dav1d
        let offset = Some(0i64);
        let timestamp = if packet.pts.is_valid() {
            Some(packet.pts.value)
        } else {
            None
        };
        let duration = Some(packet.duration);

        self.decoder
            .send_data(data, offset, timestamp, duration)
            .map_err(|e| match e {
                dav1d::Error::Again => Error::TryAgain,
                _ => Error::codec(format!("Failed to send data to AV1 decoder: {:?}", e)),
            })?;

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // Try to get a picture from the decoder
        let picture = self
            .decoder
            .get_picture()
            .map_err(|e| match e {
                dav1d::Error::Again => Error::TryAgain,
                _ => Error::codec(format!("Failed to get picture from AV1 decoder: {:?}", e)),
            })?;

        // Convert picture to our VideoFrame format
        // Use picture timestamp if available
        let pts = picture
            .timestamp()
            .map(|ts| Timestamp::new(ts))
            .unwrap_or(Timestamp::none());
        let frame = Self::picture_to_frame(picture, pts);

        Ok(Frame::Video(frame))
    }

    fn flush(&mut self) -> Result<()> {
        self.decoder.flush();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av1_decoder_creation() {
        let decoder = Av1Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_pixel_layout_conversion() {
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I420),
            PixelFormat::YUV420P
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I422),
            PixelFormat::YUV422P
        );
        assert_eq!(
            Av1Decoder::pixel_layout_to_format(PixelLayout::I444),
            PixelFormat::YUV444P
        );
    }
}
