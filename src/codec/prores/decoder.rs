//! ProRes decoder implementation - Pure Rust!

use super::{ProResProfile, ProResFrameHeader};
use super::bitstream::ProResBitstreamReader;
use super::slice::{Slice, SliceDecoder};
use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

/// ProRes video decoder
pub struct ProResDecoder {
    width: u32,
    height: u32,
    profile: Option<ProResProfile>,
    pending_frame: Option<Frame>,
}

impl ProResDecoder {
    /// Create a new ProRes decoder
    pub fn new() -> Self {
        ProResDecoder {
            width: 0,
            height: 0,
            profile: None,
            pending_frame: None,
        }
    }

    /// Parse frame header
    fn parse_frame_header(&mut self, data: &[u8]) -> Result<ProResFrameHeader> {
        if data.len() < 20 {
            return Err(Error::invalid_input("Insufficient data for ProRes header"));
        }

        // Frame size
        let frame_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);

        // Frame identifier
        let mut frame_identifier = [0u8; 4];
        frame_identifier.copy_from_slice(&data[4..8]);

        if &frame_identifier != b"icpf" {
            return Err(Error::invalid_input("Invalid ProRes frame identifier"));
        }

        // Header size
        let header_size = u16::from_be_bytes([data[8], data[9]]);

        if data.len() < header_size as usize {
            return Err(Error::invalid_input("Incomplete ProRes header"));
        }

        // Version
        let version = data[10];

        // Encoder ID
        let mut encoder_id = [0u8; 4];
        encoder_id.copy_from_slice(&data[11..15]);

        // Dimensions
        let width = u16::from_be_bytes([data[15], data[16]]);
        let height = u16::from_be_bytes([data[17], data[18]]);

        self.width = width as u32;
        self.height = height as u32;

        // Chroma format
        let chroma_format = data[19];

        // Determine profile from chroma format (simplified)
        self.profile = Some(if chroma_format == 3 {
            ProResProfile::FourFourFourFour
        } else {
            ProResProfile::Standard
        });

        // Other fields (if available)
        let interlace_mode = if data.len() > 20 { data[20] } else { 0 };
        let aspect_ratio = if data.len() > 21 { data[21] } else { 0 };
        let framerate_code = if data.len() > 22 { data[22] } else { 0 };
        let color_primaries = if data.len() > 23 { data[23] } else { 1 };
        let transfer_characteristics = if data.len() > 24 { data[24] } else { 1 };
        let matrix_coefficients = if data.len() > 25 { data[25] } else { 1 };
        let alpha_info = if data.len() > 26 { data[26] } else { 0 };

        Ok(ProResFrameHeader {
            frame_size,
            frame_identifier,
            header_size,
            version,
            encoder_id,
            width,
            height,
            chroma_format,
            interlace_mode,
            aspect_ratio,
            framerate_code,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            alpha_info,
        })
    }

    /// Decode frame data using our pure Rust implementation!
    fn decode_frame_data(&self, data: &[u8], header: &ProResFrameHeader) -> Result<VideoFrame> {
        let width = self.width as usize;
        let height = self.height as usize;

        let profile = self.profile.unwrap_or(ProResProfile::Standard);

        // Determine pixel format and chroma subsampling
        let is_444 = header.chroma_format == 3 || header.alpha_info != 0;
        let pixel_format = if is_444 {
            PixelFormat::YUV444P
        } else {
            PixelFormat::YUV422P
        };

        let chroma_width = if is_444 { width } else { width / 2 };

        // Parse all slices from bitstream
        let mut reader = ProResBitstreamReader::new(data);
        let mut slices = Vec::new();

        while reader.has_more() {
            match Slice::parse(&mut reader) {
                Ok(slice) => slices.push(slice),
                Err(_) => break, // End of slices
            }
        }

        // If no slices parsed, create empty frame
        if slices.is_empty() {
            let mut frame = VideoFrame::new(self.width, self.height, pixel_format);
            frame.pts = Timestamp::new(0);
            return Ok(frame);
        }

        let qp = slices.first().map(|s| s.header.qp).unwrap_or(16);

        // Allocate planes
        let mut y_plane = vec![0i16; width * height];
        let mut u_plane = vec![0i16; chroma_width * height];
        let mut v_plane = vec![0i16; chroma_width * height];

        // Decode slices - they come in groups of 3 (Y, U, V) per slice row
        let slice_height = 16;
        let num_slice_rows = (height + slice_height - 1) / slice_height;

        for slice_row in 0..num_slice_rows {
            let base_idx = slice_row * 3;

            // Decode Y slice
            if base_idx < slices.len() {
                let y_decoder = SliceDecoder::new(profile, width, height, qp);
                y_decoder.decode_slice(&slices[base_idx], &mut y_plane)?;
            }

            // Decode U slice
            if base_idx + 1 < slices.len() {
                let u_decoder = SliceDecoder::new(profile, chroma_width, height, qp);
                u_decoder.decode_slice(&slices[base_idx + 1], &mut u_plane)?;
            }

            // Decode V slice
            if base_idx + 2 < slices.len() {
                let v_decoder = SliceDecoder::new(profile, chroma_width, height, qp);
                v_decoder.decode_slice(&slices[base_idx + 2], &mut v_plane)?;
            }
        }

        // Convert to VideoFrame
        let mut frame = VideoFrame::new(self.width, self.height, pixel_format);

        // Copy Y plane (convert i16 to u8)
        for i in 0..y_plane.len().min(frame.y_plane.len()) {
            let y_val = (y_plane[i] + 128).clamp(0, 255) as u8;
            frame.y_plane[i] = y_val;
        }

        // Upsample and copy U plane
        self.upsample_chroma(&u_plane, &mut frame.u_plane, width, height, is_444)?;

        // Upsample and copy V plane
        self.upsample_chroma(&v_plane, &mut frame.v_plane, width, height, is_444)?;

        frame.pts = Timestamp::new(0);
        Ok(frame)
    }

    /// Upsample chroma plane to output frame
    fn upsample_chroma(
        &self,
        input: &[i16],
        output: &mut [u8],
        width: usize,
        height: usize,
        is_444: bool,
    ) -> Result<()> {
        if is_444 {
            // 4:4:4 - no upsampling needed, same resolution as Y
            for i in 0..input.len().min(output.len()) {
                output[i] = (input[i] + 128).clamp(0, 255) as u8;
            }
        } else {
            // 4:2:2 - horizontal upsampling (double width)
            let chroma_width = width / 2;
            for y in 0..height {
                for x in 0..width {
                    let x_src = x / 2;
                    let idx_src = y * chroma_width + x_src;
                    let idx_dst = y * width + x;

                    if idx_src < input.len() && idx_dst < output.len() {
                        output[idx_dst] = (input[idx_src] + 128).clamp(0, 255) as u8;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for ProResDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for ProResDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Parse frame header
        let header = self.parse_frame_header(packet.data.as_slice())?;

        // Decode frame data
        let data_slice = packet.data.as_slice();
        let mut video_frame = self.decode_frame_data(
            &data_slice[header.header_size as usize..],
            &header,
        )?;

        video_frame.pts = packet.pts;

        self.pending_frame = Some(Frame::Video(video_frame));

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.pending_frame.take().ok_or_else(|| Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // ProRes has no delayed frames
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prores_decoder_creation() {
        let decoder = ProResDecoder::new();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
    }

    #[test]
    fn test_prores_decoder_parse_header() {
        let mut decoder = ProResDecoder::new();

        // Create minimal valid header
        let mut header_data = vec![0u8; 148];

        // Frame size
        header_data[0..4].copy_from_slice(&1000u32.to_be_bytes());

        // Frame identifier "icpf"
        header_data[4..8].copy_from_slice(b"icpf");

        // Header size
        header_data[8..10].copy_from_slice(&148u16.to_be_bytes());

        // Version
        header_data[10] = 0;

        // Encoder ID
        header_data[11..15].copy_from_slice(b"test");

        // Dimensions
        header_data[15..17].copy_from_slice(&1920u16.to_be_bytes());
        header_data[17..19].copy_from_slice(&1080u16.to_be_bytes());

        // Chroma format
        header_data[19] = 2; // 422

        let header = decoder.parse_frame_header(&header_data).unwrap();

        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(decoder.width, 1920);
        assert_eq!(decoder.height, 1080);
    }
}
