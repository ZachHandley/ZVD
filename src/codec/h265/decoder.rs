//! H.265/HEVC decoder implementation

use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::PixelFormat;

use super::nal::{NalUnit, NalUnitType};
use super::headers::{Vps, Sps, Pps};

/// H.265/HEVC decoder
///
/// This is a pure Rust implementation of an H.265/HEVC decoder.
///
/// # Implementation Status
///
/// **Phase 8.1: Decoder Foundation** (Current)
/// - [x] Basic structure
/// - [ ] NAL unit processing
/// - [ ] Parameter set storage
/// - [ ] Frame buffering
///
/// **Future Phases**:
/// - Phase 8.2: Basic intra prediction
/// - Phase 8.3: Full intra decoding
/// - Phase 8.4: Inter prediction
///
/// # Example
///
/// ```rust,no_run
/// use zvd_lib::codec::h265::H265Decoder;
/// use zvd_lib::codec::Decoder;
///
/// let mut decoder = H265Decoder::new()?;
///
/// // Send NAL units
/// // decoder.send_packet(&nal_packet)?;
///
/// // Receive decoded frames
/// // let frame = decoder.receive_frame()?;
/// # Ok::<(), zvd_lib::error::Error>(())
/// ```
pub struct H265Decoder {
    /// Active Video Parameter Set
    vps: Option<Vps>,
    /// Active Sequence Parameter Set
    sps: Option<Sps>,
    /// Active Picture Parameter Set
    pps: Option<Pps>,

    /// Frame buffer for decoded frames
    frame_buffer: Vec<VideoFrame>,

    /// Current picture width
    width: u32,
    /// Current picture height
    height: u32,
    /// Current pixel format
    pixel_format: PixelFormat,
}

impl H265Decoder {
    /// Create a new H.265 decoder
    pub fn new() -> Result<Self> {
        Ok(H265Decoder {
            vps: None,
            sps: None,
            pps: None,
            frame_buffer: Vec::new(),
            width: 0,
            height: 0,
            pixel_format: PixelFormat::YUV420P,
        })
    }

    /// Process a NAL unit
    fn process_nal_unit(&mut self, nal: NalUnit) -> Result<()> {
        match nal.nal_type() {
            NalUnitType::VpsNut => {
                // Parse VPS
                let vps = Vps::parse(&nal.rbsp)?;
                self.vps = Some(vps);
                Ok(())
            }
            NalUnitType::SpsNut => {
                // Parse SPS
                let sps = Sps::parse(&nal.rbsp)?;

                // Update decoder state from SPS
                self.width = sps.pic_width_in_luma_samples;
                self.height = sps.pic_height_in_luma_samples;

                // Set pixel format based on chroma format
                self.pixel_format = match sps.chroma_format_idc {
                    1 => PixelFormat::YUV420P,
                    2 => PixelFormat::YUV422P,
                    3 => PixelFormat::YUV444P,
                    _ => return Err(Error::codec(format!("Unsupported chroma format: {}", sps.chroma_format_idc))),
                };

                self.sps = Some(sps);
                Ok(())
            }
            NalUnitType::PpsNut => {
                // Parse PPS
                let pps = Pps::parse(&nal.rbsp)?;
                self.pps = Some(pps);
                Ok(())
            }
            nal_type if nal_type.is_slice() => {
                // TODO: Decode slice
                // For now, just acknowledge we received a slice
                Err(Error::codec("Slice decoding not yet implemented"))
            }
            _ => {
                // Ignore other NAL unit types for now
                Ok(())
            }
        }
    }
}

impl Default for H265Decoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl Decoder for H265Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Parse NAL unit from packet data
        let nal = NalUnit::parse(packet.data.as_slice())?;

        // Process the NAL unit
        self.process_nal_unit(nal)?;

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // Check if we have a decoded frame in buffer
        if self.frame_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        // Return the first frame from buffer
        Ok(Frame::Video(self.frame_buffer.remove(0)))
    }

    fn flush(&mut self) -> Result<()> {
        // Clear frame buffer
        self.frame_buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h265_decoder_creation() {
        let decoder = H265Decoder::new();
        assert!(decoder.is_ok(), "Should create H.265 decoder");

        let decoder = decoder.unwrap();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
        assert!(decoder.vps.is_none());
        assert!(decoder.sps.is_none());
        assert!(decoder.pps.is_none());
    }

    #[test]
    fn test_h265_decoder_default() {
        let decoder = H265Decoder::default();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
    }

    #[test]
    fn test_h265_decoder_flush() {
        let mut decoder = H265Decoder::new().unwrap();
        assert!(decoder.flush().is_ok());
    }
}
