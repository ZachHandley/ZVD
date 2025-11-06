//! Video filters

use super::Filter;
use crate::codec::{Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::util::{Buffer, PixelFormat};
use image::{ImageBuffer, Luma};

/// Scale video filter
pub struct ScaleFilter {
    target_width: u32,
    target_height: u32,
}

impl ScaleFilter {
    /// Create a new scale filter
    pub fn new(width: u32, height: u32) -> Self {
        ScaleFilter {
            target_width: width,
            target_height: height,
        }
    }

    /// Scale a YUV420P frame
    fn scale_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let src_width = video_frame.width;
        let src_height = video_frame.height;

        // Scale Y plane
        let y_plane = video_frame.data[0].as_slice();
        let y_img = ImageBuffer::<Luma<u8>, &[u8]>::from_raw(src_width, src_height, y_plane)
            .ok_or_else(|| Error::filter("Failed to create Y plane image"))?;

        let y_scaled = image::imageops::resize(
            &y_img,
            self.target_width,
            self.target_height,
            image::imageops::FilterType::Lanczos3,
        );

        // Scale U plane (half resolution for 420)
        let u_plane = video_frame.data[1].as_slice();
        let u_width = src_width / 2;
        let u_height = src_height / 2;
        let u_img = ImageBuffer::<Luma<u8>, &[u8]>::from_raw(u_width, u_height, u_plane)
            .ok_or_else(|| Error::filter("Failed to create U plane image"))?;

        let u_scaled = image::imageops::resize(
            &u_img,
            self.target_width / 2,
            self.target_height / 2,
            image::imageops::FilterType::Lanczos3,
        );

        // Scale V plane (half resolution for 420)
        let v_plane = video_frame.data[2].as_slice();
        let v_img = ImageBuffer::<Luma<u8>, &[u8]>::from_raw(u_width, u_height, v_plane)
            .ok_or_else(|| Error::filter("Failed to create V plane image"))?;

        let v_scaled = image::imageops::resize(
            &v_img,
            self.target_width / 2,
            self.target_height / 2,
            image::imageops::FilterType::Lanczos3,
        );

        // Create output frame
        let mut output_frame =
            VideoFrame::new(self.target_width, self.target_height, video_frame.format);

        output_frame.data = vec![
            Buffer::from_vec(y_scaled.into_raw()),
            Buffer::from_vec(u_scaled.into_raw()),
            Buffer::from_vec(v_scaled.into_raw()),
        ];

        output_frame.linesize = vec![
            self.target_width as usize,
            (self.target_width / 2) as usize,
            (self.target_width / 2) as usize,
        ];

        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }
}

impl Filter for ScaleFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                // Only handle YUV420P for now
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Scale filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let scaled = self.scale_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(scaled)])
            }
            Frame::Audio(_) => Err(Error::filter("Scale filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Crop video filter
pub struct CropFilter {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl CropFilter {
    /// Create a new crop filter
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        CropFilter {
            x,
            y,
            width,
            height,
        }
    }
}

impl Filter for CropFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        // Placeholder implementation
        Ok(vec![input])
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}
