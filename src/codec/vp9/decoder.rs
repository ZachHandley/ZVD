//! VP9 video decoder using libvpx
//!
//! This module provides a complete VP9 decoder implementation using the libvpx library.
//! The vpx-sys crate provides FFI bindings to libvpx.
//!
//! # System Requirements
//!
//! libvpx must be installed on the system:
//! - Debian/Ubuntu: `apt install libvpx-dev`
//! - Arch Linux: `pacman -S libvpx`
//! - macOS: `brew install libvpx`
//! - Fedora: `dnf install libvpx-devel`

use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use std::ptr;

#[cfg(feature = "vp9-codec")]
use vpx_sys::*;

/// VP9 video decoder wrapping libvpx
///
/// This decoder uses libvpx's VP9 decoder interface to decode VP9 video streams.
/// VP9 is Google's successor to VP8, offering better compression efficiency.
/// It supports multiple profiles, higher bit depths, and advanced features like superframes.
#[cfg(feature = "vp9-codec")]
pub struct Vp9Decoder {
    /// libvpx decoder context
    ctx: vpx_codec_ctx_t,
    /// Whether the decoder has been initialized
    initialized: bool,
    /// Buffered decoded frames waiting to be retrieved
    frame_buffer: Vec<VideoFrame>,
    /// Frame counter for tracking
    frame_count: u64,
}

#[cfg(feature = "vp9-codec")]
impl Vp9Decoder {
    /// Create a new VP9 decoder
    pub fn new() -> Result<Self> {
        Self::with_threads(0)
    }

    /// Create a new VP9 decoder with specified thread count
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads to use (0 = auto-detect)
    pub fn with_threads(threads: u32) -> Result<Self> {
        unsafe {
            let mut ctx: vpx_codec_ctx_t = std::mem::zeroed();

            // Get VP9 decoder interface
            let iface = vpx_codec_vp9_dx();
            if iface.is_null() {
                return Err(Error::codec("Failed to get VP9 decoder interface"));
            }

            // Initialize decoder configuration
            let mut cfg: vpx_codec_dec_cfg_t = std::mem::zeroed();
            cfg.threads = threads;
            cfg.w = 0; // Will be set from stream
            cfg.h = 0; // Will be set from stream

            // Initialize the decoder
            let ret = vpx_codec_dec_init_ver(
                &mut ctx,
                iface,
                &cfg,
                0, // flags
                VPX_DECODER_ABI_VERSION as i32,
            );

            if ret != VPX_CODEC_OK {
                let error_str = if !ctx.err_detail.is_null() {
                    std::ffi::CStr::from_ptr(ctx.err_detail)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("Error code: {}", ret)
                };
                return Err(Error::codec(format!("Failed to initialize VP9 decoder: {}", error_str)));
            }

            Ok(Vp9Decoder {
                ctx,
                initialized: true,
                frame_buffer: Vec::new(),
                frame_count: 0,
            })
        }
    }

    /// Convert vpx image to our VideoFrame format
    fn vpx_image_to_frame(img: &vpx_image_t, pts: Timestamp) -> Result<VideoFrame> {
        let width = img.d_w;
        let height = img.d_h;

        // VP9 supports multiple pixel formats including high bit depth
        let pixel_format = match img.fmt {
            vpx_img_fmt::VPX_IMG_FMT_I420 => PixelFormat::YUV420P,
            vpx_img_fmt::VPX_IMG_FMT_I422 => PixelFormat::YUV422P,
            vpx_img_fmt::VPX_IMG_FMT_I444 => PixelFormat::YUV444P,
            vpx_img_fmt::VPX_IMG_FMT_I42016 => PixelFormat::YUV420P10LE,
            vpx_img_fmt::VPX_IMG_FMT_I42216 => PixelFormat::YUV422P10LE,
            vpx_img_fmt::VPX_IMG_FMT_I44416 => PixelFormat::YUV444P10LE,
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported VP9 pixel format: {:?}",
                    img.fmt
                )));
            }
        };

        // Determine if this is high bit depth
        let is_high_bit_depth = matches!(
            img.fmt,
            vpx_img_fmt::VPX_IMG_FMT_I42016
                | vpx_img_fmt::VPX_IMG_FMT_I42216
                | vpx_img_fmt::VPX_IMG_FMT_I44416
        );

        let bytes_per_sample = if is_high_bit_depth { 2 } else { 1 };

        // Extract plane data
        let y_plane = unsafe {
            if img.planes[0].is_null() {
                return Err(Error::codec("Y plane is null"));
            }
            std::slice::from_raw_parts(
                img.planes[0],
                (img.stride[0] as u32 * height * bytes_per_sample) as usize,
            )
        };

        let u_plane = unsafe {
            if img.planes[1].is_null() {
                return Err(Error::codec("U plane is null"));
            }
            let uv_height = match pixel_format {
                PixelFormat::YUV420P | PixelFormat::YUV420P10LE => height / 2,
                PixelFormat::YUV422P | PixelFormat::YUV422P10LE => height,
                PixelFormat::YUV444P | PixelFormat::YUV444P10LE => height,
                _ => height / 2,
            };
            std::slice::from_raw_parts(
                img.planes[1],
                (img.stride[1] as u32 * uv_height * bytes_per_sample) as usize,
            )
        };

        let v_plane = unsafe {
            if img.planes[2].is_null() {
                return Err(Error::codec("V plane is null"));
            }
            let uv_height = match pixel_format {
                PixelFormat::YUV420P | PixelFormat::YUV420P10LE => height / 2,
                PixelFormat::YUV422P | PixelFormat::YUV422P10LE => height,
                PixelFormat::YUV444P | PixelFormat::YUV444P10LE => height,
                _ => height / 2,
            };
            std::slice::from_raw_parts(
                img.planes[2],
                (img.stride[2] as u32 * uv_height * bytes_per_sample) as usize,
            )
        };

        // Create buffers
        let y_buffer = Buffer::from_vec(y_plane.to_vec());
        let u_buffer = Buffer::from_vec(u_plane.to_vec());
        let v_buffer = Buffer::from_vec(v_plane.to_vec());

        // Create video frame
        let mut video_frame = VideoFrame::new(width, height, pixel_format);
        video_frame.data = vec![y_buffer, u_buffer, v_buffer];
        video_frame.linesize = vec![
            img.stride[0] as usize,
            img.stride[1] as usize,
            img.stride[2] as usize,
        ];
        video_frame.pts = pts;

        Ok(video_frame)
    }

    /// Retrieve all available decoded frames from the decoder
    fn retrieve_frames(&mut self) -> Result<()> {
        unsafe {
            let mut iter: vpx_codec_iter_t = ptr::null_mut();

            loop {
                let img = vpx_codec_get_frame(&mut self.ctx, &mut iter);
                if img.is_null() {
                    // No more frames
                    break;
                }

                let img_ref = &*img;
                // Use frame count as PTS if not available from stream
                let pts = Timestamp::new(self.frame_count as i64);

                let video_frame = Self::vpx_image_to_frame(img_ref, pts)?;
                self.frame_buffer.push(video_frame);
                self.frame_count += 1;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "vp9-codec")]
impl Drop for Vp9Decoder {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                vpx_codec_destroy(&mut self.ctx);
            }
            self.initialized = false;
        }
    }
}

#[cfg(feature = "vp9-codec")]
impl Default for Vp9Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default VP9 decoder")
    }
}

#[cfg(feature = "vp9-codec")]
impl Decoder for Vp9Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Err(Error::codec("Empty VP9 packet"));
        }

        unsafe {
            let data_ptr = packet.data.as_slice().as_ptr();
            let data_len = packet.data.len();

            // Decode the packet
            let ret = vpx_codec_decode(
                &mut self.ctx,
                data_ptr,
                data_len as u32,
                ptr::null_mut(), // user_priv
                0,               // deadline (0 = best quality)
            );

            if ret != VPX_CODEC_OK {
                let error_str = if !self.ctx.err_detail.is_null() {
                    std::ffi::CStr::from_ptr(self.ctx.err_detail)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("Error code: {}", ret)
                };
                return Err(Error::codec(format!("VP9 decoding failed: {}", error_str)));
            }

            // Retrieve decoded frames
            self.retrieve_frames()?;
        }

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.frame_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        let video_frame = self.frame_buffer.remove(0);
        Ok(Frame::Video(video_frame))
    }

    fn flush(&mut self) -> Result<()> {
        self.frame_buffer.clear();
        self.frame_count = 0;
        Ok(())
    }
}

// Stub implementation when vp9-codec feature is not enabled
#[cfg(not(feature = "vp9-codec"))]
pub struct Vp9Decoder {
    _private: (),
}

#[cfg(not(feature = "vp9-codec"))]
impl Vp9Decoder {
    pub fn new() -> Result<Self> {
        Err(Error::unsupported(
            "VP9 codec support not enabled. Enable the 'vp9-codec' feature and ensure libvpx is installed."
        ))
    }
}

#[cfg(not(feature = "vp9-codec"))]
impl Decoder for Vp9Decoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "vp9-codec")]
    fn test_vp9_decoder_creation() {
        let decoder = Vp9Decoder::new();
        assert!(
            decoder.is_ok(),
            "Decoder creation failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp9-codec")]
    fn test_vp9_decoder_with_threads() {
        let decoder = Vp9Decoder::with_threads(4);
        assert!(
            decoder.is_ok(),
            "Decoder creation with threads failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp9-codec")]
    fn test_flush() {
        let mut decoder = Vp9Decoder::new().expect("Failed to create decoder");
        // Flush should not error even on empty decoder
        assert!(decoder.flush().is_ok());
    }

    #[test]
    #[cfg(not(feature = "vp9-codec"))]
    fn test_vp9_disabled() {
        let decoder = Vp9Decoder::new();
        assert!(decoder.is_err());
    }
}
