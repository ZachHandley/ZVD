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

    /// Crop a YUV420P frame
    fn crop_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let src_width = video_frame.width;
        let src_height = video_frame.height;

        // Validate crop region
        if self.x + self.width > src_width || self.y + self.height > src_height {
            return Err(Error::filter(format!(
                "Crop region ({}x{} at {},{}) exceeds frame dimensions ({}x{})",
                self.width, self.height, self.x, self.y, src_width, src_height
            )));
        }

        // Crop Y plane
        let y_plane = video_frame.data[0].as_slice();
        let mut y_cropped = Vec::with_capacity((self.width * self.height) as usize);

        for row in self.y..self.y + self.height {
            let start = (row * src_width + self.x) as usize;
            let end = start + self.width as usize;
            y_cropped.extend_from_slice(&y_plane[start..end]);
        }

        // Crop U plane (half resolution for 420)
        let u_plane = video_frame.data[1].as_slice();
        let u_x = self.x / 2;
        let u_y = self.y / 2;
        let u_width = self.width / 2;
        let u_height = self.height / 2;
        let u_src_width = src_width / 2;

        let mut u_cropped = Vec::with_capacity((u_width * u_height) as usize);
        for row in u_y..u_y + u_height {
            let start = (row * u_src_width + u_x) as usize;
            let end = start + u_width as usize;
            u_cropped.extend_from_slice(&u_plane[start..end]);
        }

        // Crop V plane (half resolution for 420)
        let v_plane = video_frame.data[2].as_slice();
        let mut v_cropped = Vec::with_capacity((u_width * u_height) as usize);
        for row in u_y..u_y + u_height {
            let start = (row * u_src_width + u_x) as usize;
            let end = start + u_width as usize;
            v_cropped.extend_from_slice(&v_plane[start..end]);
        }

        // Create output frame
        let mut output_frame = VideoFrame::new(self.width, self.height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_cropped),
            Buffer::from_vec(u_cropped),
            Buffer::from_vec(v_cropped),
        ];
        output_frame.linesize = vec![
            self.width as usize,
            (self.width / 2) as usize,
            (self.width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }
}

impl Filter for CropFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                // Only handle YUV420P for now
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Crop filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let cropped = self.crop_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(cropped)])
            }
            Frame::Audio(_) => Err(Error::filter("Crop filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Rotate video filter (90, 180, 270 degrees)
pub struct RotateFilter {
    angle: i32, // 90, 180, 270, or -90, -180, -270
}

impl RotateFilter {
    /// Create a new rotate filter
    /// angle: rotation angle in degrees (90, 180, 270, or negative values)
    pub fn new(angle: i32) -> Result<Self> {
        let normalized = ((angle % 360) + 360) % 360;
        if normalized != 0 && normalized != 90 && normalized != 180 && normalized != 270 {
            return Err(Error::filter(format!(
                "Rotate filter only supports 90, 180, and 270 degree rotations, got {}",
                angle
            )));
        }
        Ok(RotateFilter { angle: normalized })
    }

    /// Rotate a YUV420P frame
    fn rotate_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let src_width = video_frame.width;
        let src_height = video_frame.height;

        // Calculate output dimensions
        let (dst_width, dst_height) = if self.angle == 90 || self.angle == 270 {
            (src_height, src_width) // Swap dimensions
        } else {
            (src_width, src_height) // Keep dimensions
        };

        // Rotate Y plane
        let y_plane = video_frame.data[0].as_slice();
        let y_rotated = self.rotate_plane(y_plane, src_width, src_height, self.angle)?;

        // Rotate U plane (half resolution)
        let u_plane = video_frame.data[1].as_slice();
        let u_rotated = self.rotate_plane(u_plane, src_width / 2, src_height / 2, self.angle)?;

        // Rotate V plane (half resolution)
        let v_plane = video_frame.data[2].as_slice();
        let v_rotated = self.rotate_plane(v_plane, src_width / 2, src_height / 2, self.angle)?;

        // Create output frame
        let mut output_frame = VideoFrame::new(dst_width, dst_height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_rotated),
            Buffer::from_vec(u_rotated),
            Buffer::from_vec(v_rotated),
        ];
        output_frame.linesize = vec![
            dst_width as usize,
            (dst_width / 2) as usize,
            (dst_width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }

    /// Rotate a single plane
    fn rotate_plane(&self, src: &[u8], width: u32, height: u32, angle: i32) -> Result<Vec<u8>> {
        match angle {
            0 => Ok(src.to_vec()),
            90 => {
                // Rotate 90 degrees clockwise
                let mut dst = vec![0u8; (width * height) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let src_idx = (y * width + x) as usize;
                        let dst_x = height - 1 - y;
                        let dst_y = x;
                        let dst_idx = (dst_y * height + dst_x) as usize;
                        dst[dst_idx] = src[src_idx];
                    }
                }
                Ok(dst)
            }
            180 => {
                // Rotate 180 degrees
                let mut dst = vec![0u8; (width * height) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let src_idx = (y * width + x) as usize;
                        let dst_x = width - 1 - x;
                        let dst_y = height - 1 - y;
                        let dst_idx = (dst_y * width + dst_x) as usize;
                        dst[dst_idx] = src[src_idx];
                    }
                }
                Ok(dst)
            }
            270 => {
                // Rotate 270 degrees clockwise (90 counter-clockwise)
                let mut dst = vec![0u8; (width * height) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let src_idx = (y * width + x) as usize;
                        let dst_x = y;
                        let dst_y = width - 1 - x;
                        let dst_idx = (dst_y * height + dst_x) as usize;
                        dst[dst_idx] = src[src_idx];
                    }
                }
                Ok(dst)
            }
            _ => Err(Error::filter(format!("Unsupported rotation angle: {}", angle))),
        }
    }
}

impl Filter for RotateFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                // Only handle YUV420P for now
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Rotate filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let rotated = self.rotate_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(rotated)])
            }
            Frame::Audio(_) => Err(Error::filter("Rotate filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}
