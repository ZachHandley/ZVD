//! VP8 video decoder using libvpx
//!
//! This module provides a complete VP8 decoder implementation using the libvpx library.
//! VP8 is a royalty-free video codec developed by On2 Technologies and released as
//! open source by Google.
//!
//! # System Requirements
//!
//! libvpx must be installed on the system:
//! - Debian/Ubuntu: `apt install libvpx-dev`
//! - Arch Linux: `pacman -S libvpx`
//! - macOS: `brew install libvpx`
//! - Fedora: `dnf install libvpx-devel`
//!
//! # Features
//!
//! Enable the `vp8-libvpx` feature in Cargo.toml to use this decoder:
//! ```toml
//! zvd = { version = "0.1", features = ["vp8-libvpx"] }
//! ```

use crate::codec::{Decoder, Frame, PictureType, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use std::collections::VecDeque;
use std::ptr;

#[cfg(feature = "vp8-libvpx")]
use libvpx_sys::*;

/// VP8 video decoder wrapping libvpx
///
/// This decoder uses libvpx to decode VP8 video streams. It provides
/// efficient decoding with support for multi-threading and handles
/// all VP8 profiles.
pub struct Vp8Decoder {
    /// Decoded frames waiting to be retrieved
    frame_buffer: VecDeque<VideoFrame>,
    /// Frame counter for PTS generation when not provided
    frame_count: u64,
    /// Last decoded width (updated from bitstream)
    width: u32,
    /// Last decoded height (updated from bitstream)
    height: u32,
    /// libvpx decoder context
    #[cfg(feature = "vp8-libvpx")]
    ctx: vpx_codec_ctx_t,
    /// Flag indicating if decoder has been initialized
    #[cfg(feature = "vp8-libvpx")]
    initialized: bool,
}

impl Vp8Decoder {
    /// Create a new VP8 decoder with default settings
    pub fn new() -> Result<Self> {
        #[cfg(feature = "vp8-libvpx")]
        {
            Self::with_threads(0)
        }
        #[cfg(not(feature = "vp8-libvpx"))]
        {
            Err(Error::unsupported(
                "VP8 decoding requires the 'vp8-libvpx' feature to be enabled",
            ))
        }
    }

    /// Create a new VP8 decoder with specified thread count
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads to use (0 = auto-detect based on CPU cores)
    #[cfg(feature = "vp8-libvpx")]
    pub fn with_threads(threads: u32) -> Result<Self> {
        // Initialize the codec context
        // Safety: We're initializing a zeroed context which is safe
        let mut ctx: vpx_codec_ctx_t = unsafe { std::mem::zeroed() };

        // Get VP8 decoder interface
        // Safety: vpx_codec_vp8_dx returns a valid interface pointer
        let iface = unsafe { vpx_codec_vp8_dx() };
        if iface.is_null() {
            return Err(Error::codec("Failed to get VP8 decoder interface"));
        }

        // Configure decoder
        let mut cfg: vpx_codec_dec_cfg_t = unsafe { std::mem::zeroed() };
        cfg.threads = threads;
        cfg.w = 0; // Will be determined from bitstream
        cfg.h = 0; // Will be determined from bitstream

        // Initialize the decoder
        // Safety: We pass valid pointers and the ABI version constant
        let res = unsafe {
            vpx_codec_dec_init_ver(
                &mut ctx,
                iface,
                &cfg,
                0, // flags
                VPX_DECODER_ABI_VERSION as i32,
            )
        };

        if res != vpx_codec_err_t::VPX_CODEC_OK {
            let err_msg = unsafe {
                let err_str = vpx_codec_error(&ctx);
                if err_str.is_null() {
                    "Unknown error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err_str)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(Error::codec(format!(
                "Failed to initialize VP8 decoder: {}",
                err_msg
            )));
        }

        Ok(Vp8Decoder {
            frame_buffer: VecDeque::new(),
            frame_count: 0,
            width: 0,
            height: 0,
            ctx,
            initialized: true,
        })
    }

    /// Convert a VPX image to our VideoFrame type
    #[cfg(feature = "vp8-libvpx")]
    fn vpx_image_to_frame(&self, img: &vpx_image_t, pts: Timestamp) -> Result<VideoFrame> {
        // Determine pixel format from VPX image format
        let format = match img.fmt {
            vpx_img_fmt::VPX_IMG_FMT_I420 | vpx_img_fmt::VPX_IMG_FMT_YV12 => PixelFormat::YUV420P,
            vpx_img_fmt::VPX_IMG_FMT_I422 => PixelFormat::YUV422P,
            vpx_img_fmt::VPX_IMG_FMT_I444 => PixelFormat::YUV444P,
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported VPX image format: {:?}",
                    img.fmt
                )));
            }
        };

        let width = img.d_w;
        let height = img.d_h;

        // Calculate plane sizes based on format
        let (uv_width_divisor, uv_height_divisor) = match format {
            PixelFormat::YUV420P => (2, 2),
            PixelFormat::YUV422P => (2, 1),
            PixelFormat::YUV444P => (1, 1),
            _ => unreachable!(),
        };

        // Get plane data
        // Safety: planes array is valid and contains pointers to plane data
        let y_plane_ptr = img.planes[0];
        let u_plane_ptr = img.planes[1];
        let v_plane_ptr = img.planes[2];

        let y_stride = img.stride[0] as usize;
        let u_stride = img.stride[1] as usize;
        let v_stride = img.stride[2] as usize;

        // Copy Y plane
        let y_height = height as usize;
        let y_width = width as usize;
        let mut y_data = Vec::with_capacity(y_height * y_stride);

        // Safety: We're copying from valid memory regions
        unsafe {
            for row in 0..y_height {
                let src = y_plane_ptr.add(row * y_stride);
                let slice = std::slice::from_raw_parts(src, y_stride);
                y_data.extend_from_slice(slice);
            }
        }

        // Copy U plane
        let uv_height = height as usize / uv_height_divisor;
        let uv_width = width as usize / uv_width_divisor;
        let mut u_data = Vec::with_capacity(uv_height * u_stride);
        let mut v_data = Vec::with_capacity(uv_height * v_stride);

        // Safety: We're copying from valid memory regions
        unsafe {
            for row in 0..uv_height {
                let u_src = u_plane_ptr.add(row * u_stride);
                let u_slice = std::slice::from_raw_parts(u_src, u_stride);
                u_data.extend_from_slice(u_slice);

                let v_src = v_plane_ptr.add(row * v_stride);
                let v_slice = std::slice::from_raw_parts(v_src, v_stride);
                v_data.extend_from_slice(v_slice);
            }
        }

        // Create video frame
        let mut video_frame = VideoFrame::new(width, height, format);
        video_frame.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];
        video_frame.linesize = vec![y_stride, u_stride, v_stride];
        video_frame.pts = pts;
        video_frame.duration = 1;

        Ok(video_frame)
    }

    /// Retrieve all pending decoded frames from the decoder
    #[cfg(feature = "vp8-libvpx")]
    fn retrieve_pending_frames(&mut self, pts: Timestamp) -> Result<()> {
        let mut iter: vpx_codec_iter_t = ptr::null();

        loop {
            // Safety: We pass a valid context and iterator pointer
            let img = unsafe { vpx_codec_get_frame(&mut self.ctx, &mut iter) };

            if img.is_null() {
                break;
            }

            // Safety: img is a valid pointer returned by vpx_codec_get_frame
            let img_ref = unsafe { &*img };

            // Update dimensions from decoded frame
            self.width = img_ref.d_w;
            self.height = img_ref.d_h;

            // Convert to VideoFrame
            let mut video_frame = self.vpx_image_to_frame(img_ref, pts)?;

            // Determine if this is a keyframe (first frame or frame after flush)
            // VP8 keyframes are indicated by the frame type in the bitstream
            // We can't easily determine this from vpx_image_t, so we mark based on frame count
            video_frame.keyframe = self.frame_count == 0;
            video_frame.pict_type = if video_frame.keyframe {
                PictureType::I
            } else {
                PictureType::P
            };

            self.frame_buffer.push_back(video_frame);
        }

        Ok(())
    }
}

impl Default for Vp8Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default VP8 decoder")
    }
}

impl Decoder for Vp8Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        #[cfg(feature = "vp8-libvpx")]
        {
            if !self.initialized {
                return Err(Error::InvalidState("Decoder not initialized".into()));
            }

            if packet.data.is_empty() {
                return Err(Error::InvalidInput("Empty VP8 packet".into()));
            }

            let data = packet.data.as_slice();

            // Decode the packet
            // Safety: We pass valid data pointer and size
            let res = unsafe {
                vpx_codec_decode(
                    &mut self.ctx,
                    data.as_ptr(),
                    data.len() as u32,
                    ptr::null_mut(), // user_priv
                    0,               // deadline (0 = best quality)
                )
            };

            if res != vpx_codec_err_t::VPX_CODEC_OK {
                let err_msg = unsafe {
                    let err_str = vpx_codec_error(&self.ctx);
                    if err_str.is_null() {
                        "Unknown error".to_string()
                    } else {
                        std::ffi::CStr::from_ptr(err_str)
                            .to_string_lossy()
                            .into_owned()
                    }
                };
                return Err(Error::codec(format!("VP8 decode error: {}", err_msg)));
            }

            // Retrieve decoded frames
            let pts = if packet.pts.is_valid() {
                packet.pts
            } else {
                Timestamp::new(self.frame_count as i64)
            };

            self.retrieve_pending_frames(pts)?;
            self.frame_count += 1;

            Ok(())
        }

        #[cfg(not(feature = "vp8-libvpx"))]
        {
            let _ = packet;
            Err(Error::unsupported(
                "VP8 decoding requires the 'vp8-libvpx' feature",
            ))
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.frame_buffer.pop_front() {
            Ok(Frame::Video(frame))
        } else {
            Err(Error::TryAgain)
        }
    }

    fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "vp8-libvpx")]
        {
            if !self.initialized {
                return Ok(());
            }

            // Send a null packet to flush the decoder
            // Safety: Passing null data with 0 size signals flush
            let res = unsafe {
                vpx_codec_decode(&mut self.ctx, ptr::null(), 0, ptr::null_mut(), 0)
            };

            if res != vpx_codec_err_t::VPX_CODEC_OK {
                // Ignore flush errors, just clear the buffer
            }

            // Retrieve any remaining frames
            let _ = self.retrieve_pending_frames(Timestamp::none());

            // Clear frame buffer
            self.frame_buffer.clear();
            self.frame_count = 0;
        }

        Ok(())
    }
}

#[cfg(feature = "vp8-libvpx")]
impl Drop for Vp8Decoder {
    fn drop(&mut self) {
        if self.initialized {
            // Safety: We only destroy an initialized context
            unsafe {
                vpx_codec_destroy(&mut self.ctx);
            }
        }
    }
}

// Ensure the decoder can be sent between threads (libvpx contexts are thread-safe)
#[cfg(feature = "vp8-libvpx")]
unsafe impl Send for Vp8Decoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_decoder_creation() {
        let decoder = Vp8Decoder::new();
        assert!(
            decoder.is_ok(),
            "Decoder creation failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_decoder_with_threads() {
        let decoder = Vp8Decoder::with_threads(4);
        assert!(
            decoder.is_ok(),
            "Decoder creation with threads failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_flush_empty_decoder() {
        let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");
        // Flush should not error even on empty decoder
        assert!(decoder.flush().is_ok());
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_receive_without_send() {
        let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");
        // Should return TryAgain when no frames are available
        match decoder.receive_frame() {
            Err(Error::TryAgain) => {}
            other => panic!("Expected TryAgain, got {:?}", other),
        }
    }

    #[test]
    fn test_vp8_decoder_creation_without_feature() {
        #[cfg(not(feature = "vp8-libvpx"))]
        {
            let decoder = Vp8Decoder::new();
            assert!(decoder.is_err(), "Should fail without vp8-libvpx feature");
        }
    }
}
