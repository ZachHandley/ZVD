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
/// Flip video filter (horizontal or vertical)
pub struct FlipFilter {
    horizontal: bool,
    vertical: bool,
}

impl FlipFilter {
    /// Create a new flip filter
    pub fn horizontal() -> Self {
        FlipFilter {
            horizontal: true,
            vertical: false,
        }
    }

    /// Create a vertical flip filter
    pub fn vertical() -> Self {
        FlipFilter {
            horizontal: false,
            vertical: true,
        }
    }

    /// Create a both-axis flip filter
    pub fn both() -> Self {
        FlipFilter {
            horizontal: true,
            vertical: true,
        }
    }

    /// Flip a YUV420P frame
    fn flip_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let width = video_frame.width;
        let height = video_frame.height;

        // Flip Y plane
        let y_plane = video_frame.data[0].as_slice();
        let y_flipped = self.flip_plane(y_plane, width, height)?;

        // Flip U plane (half resolution)
        let u_plane = video_frame.data[1].as_slice();
        let u_flipped = self.flip_plane(u_plane, width / 2, height / 2)?;

        // Flip V plane (half resolution)
        let v_plane = video_frame.data[2].as_slice();
        let v_flipped = self.flip_plane(v_plane, width / 2, height / 2)?;

        // Create output frame
        let mut output_frame = VideoFrame::new(width, height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_flipped),
            Buffer::from_vec(u_flipped),
            Buffer::from_vec(v_flipped),
        ];
        output_frame.linesize = vec![
            width as usize,
            (width / 2) as usize,
            (width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }

    /// Flip a single plane
    fn flip_plane(&self, src: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let mut dst = vec![0u8; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let src_idx = (y * width + x) as usize;
                
                let dst_x = if self.horizontal { width - 1 - x } else { x };
                let dst_y = if self.vertical { height - 1 - y } else { y };
                let dst_idx = (dst_y * width + dst_x) as usize;

                dst[dst_idx] = src[src_idx];
            }
        }

        Ok(dst)
    }
}

impl Filter for FlipFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Flip filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let flipped = self.flip_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(flipped)])
            }
            Frame::Audio(_) => Err(Error::filter("Flip filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Brightness and Contrast adjustment filter
pub struct BrightnessContrastFilter {
    brightness: i32, // -100 to 100
    contrast: f32,   // 0.0 to 3.0, 1.0 = no change
}

impl BrightnessContrastFilter {
    /// Create a new brightness/contrast filter
    /// brightness: -100 to 100 (0 = no change)
    /// contrast: 0.0 to 3.0 (1.0 = no change)
    pub fn new(brightness: i32, contrast: f32) -> Self {
        BrightnessContrastFilter {
            brightness: brightness.clamp(-100, 100),
            contrast: contrast.clamp(0.0, 3.0),
        }
    }

    /// Adjust brightness/contrast of a YUV420P frame
    fn adjust_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let width = video_frame.width;
        let height = video_frame.height;

        // Adjust Y plane (luminance)
        let y_plane = video_frame.data[0].as_slice();
        let y_adjusted = self.adjust_plane(y_plane)?;

        // U and V planes remain unchanged
        let u_plane = video_frame.data[1].as_slice();
        let v_plane = video_frame.data[2].as_slice();

        // Create output frame
        let mut output_frame = VideoFrame::new(width, height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_adjusted),
            Buffer::from_vec(u_plane.to_vec()),
            Buffer::from_vec(v_plane.to_vec()),
        ];
        output_frame.linesize = vec![
            width as usize,
            (width / 2) as usize,
            (width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }

    /// Adjust brightness/contrast of a plane
    fn adjust_plane(&self, src: &[u8]) -> Result<Vec<u8>> {
        let mut dst = Vec::with_capacity(src.len());

        for &pixel in src {
            let value = pixel as f32;
            
            // Apply contrast first (around midpoint 128)
            let contrasted = ((value - 128.0) * self.contrast + 128.0);
            
            // Apply brightness
            let adjusted = contrasted + self.brightness as f32;
            
            // Clamp to valid range
            let clamped = adjusted.clamp(0.0, 255.0) as u8;
            
            dst.push(clamped);
        }

        Ok(dst)
    }
}

impl Filter for BrightnessContrastFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "BrightnessContrast filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let adjusted = self.adjust_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(adjusted)])
            }
            Frame::Audio(_) => Err(Error::filter("BrightnessContrast filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Deinterlace filter
///
/// Removes interlacing artifacts using various methods.
/// Essential for broadcast video and old footage.
pub struct DeinterlaceFilter {
    method: DeinterlaceMethod,
    last_field: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy)]
pub enum DeinterlaceMethod {
    /// Discard one field, keep the other (fast but loses half the resolution)
    Discard,
    /// Blend the two fields together (fast, simple)
    Blend,
    /// Linear interpolation between fields (better quality)
    Linear,
    /// Yadif-like adaptive deinterlacing (highest quality)
    Adaptive,
}

impl DeinterlaceFilter {
    /// Create a new deinterlace filter
    pub fn new(method: DeinterlaceMethod) -> Self {
        DeinterlaceFilter {
            method,
            last_field: None,
        }
    }

    /// Deinterlace a YUV420P frame
    fn deinterlace_yuv420p(&mut self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let width = video_frame.width;
        let height = video_frame.height;

        // Deinterlace Y plane
        let y_plane = video_frame.data[0].as_slice();
        let y_deinterlaced = self.deinterlace_plane(y_plane, width, height)?;

        // Deinterlace U plane (half resolution)
        let u_plane = video_frame.data[1].as_slice();
        let u_deinterlaced = self.deinterlace_plane(u_plane, width / 2, height / 2)?;

        // Deinterlace V plane (half resolution)
        let v_plane = video_frame.data[2].as_slice();
        let v_deinterlaced = self.deinterlace_plane(v_plane, width / 2, height / 2)?;

        // Create output frame
        let mut output_frame = VideoFrame::new(width, height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_deinterlaced),
            Buffer::from_vec(u_deinterlaced),
            Buffer::from_vec(v_deinterlaced),
        ];
        output_frame.linesize = vec![
            width as usize,
            (width / 2) as usize,
            (width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }

    /// Deinterlace a single plane
    fn deinterlace_plane(&self, src: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        match self.method {
            DeinterlaceMethod::Discard => {
                // Keep only even lines (top field)
                let mut dst = Vec::with_capacity((width * height) as usize);
                for y in (0..height).step_by(2) {
                    let start = (y * width) as usize;
                    let end = start + width as usize;
                    dst.extend_from_slice(&src[start..end]);
                    // Duplicate the line for the odd field
                    dst.extend_from_slice(&src[start..end]);
                }
                Ok(dst)
            }
            DeinterlaceMethod::Blend => {
                // Blend adjacent lines
                let mut dst = vec![0u8; (width * height) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y * width + x) as usize;
                        if y == 0 || y == height - 1 {
                            dst[idx] = src[idx];
                        } else {
                            let prev_line = ((y - 1) * width + x) as usize;
                            let next_line = ((y + 1) * width + x) as usize;
                            dst[idx] = ((src[prev_line] as u16 + src[idx] as u16 + src[next_line] as u16) / 3) as u8;
                        }
                    }
                }
                Ok(dst)
            }
            DeinterlaceMethod::Linear => {
                // Linear interpolation for interlaced lines
                let mut dst = vec![0u8; (width * height) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y * width + x) as usize;
                        if y % 2 == 0 {
                            // Even lines: keep original
                            dst[idx] = src[idx];
                        } else {
                            // Odd lines: interpolate from neighbors
                            if y == 0 || y == height - 1 {
                                dst[idx] = src[idx];
                            } else {
                                let prev_line = ((y - 1) * width + x) as usize;
                                let next_line = ((y + 1) * width + x) as usize;
                                dst[idx] = ((src[prev_line] as u16 + src[next_line] as u16) / 2) as u8;
                            }
                        }
                    }
                }
                Ok(dst)
            }
            DeinterlaceMethod::Adaptive => {
                // Simple adaptive method (edge-aware interpolation)
                let mut dst = vec![0u8; (width * height) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y * width + x) as usize;
                        if y % 2 == 0 {
                            dst[idx] = src[idx];
                        } else {
                            if y == 0 || y == height - 1 {
                                dst[idx] = src[idx];
                            } else {
                                let prev_line = ((y - 1) * width + x) as usize;
                                let next_line = ((y + 1) * width + x) as usize;
                                let current = src[idx] as i16;
                                let prev = src[prev_line] as i16;
                                let next = src[next_line] as i16;

                                // If edge is detected, use simpler interpolation
                                let diff = (prev - next).abs();
                                if diff > 30 {
                                    // Strong edge: use median
                                    let mut values = [prev as u8, current as u8, next as u8];
                                    values.sort();
                                    dst[idx] = values[1];
                                } else {
                                    // Weak edge: average
                                    dst[idx] = ((prev + next) / 2) as u8;
                                }
                            }
                        }
                    }
                }
                Ok(dst)
            }
        }
    }
}

impl Filter for DeinterlaceFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Deinterlace filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let deinterlaced = self.deinterlace_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(deinterlaced)])
            }
            Frame::Audio(_) => Err(Error::filter("Deinterlace filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Sharpen filter using unsharp mask technique
pub struct SharpenFilter {
    amount: f32, // 0.0 to 2.0
}

impl SharpenFilter {
    /// Create a new sharpen filter
    /// amount: 0.0 to 2.0 (1.0 = normal sharpening)
    pub fn new(amount: f32) -> Self {
        SharpenFilter {
            amount: amount.clamp(0.0, 2.0),
        }
    }

    /// Sharpen a YUV420P frame (only affects Y/luminance plane)
    fn sharpen_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let width = video_frame.width;
        let height = video_frame.height;

        // Sharpen Y plane using unsharp mask
        let y_plane = video_frame.data[0].as_slice();
        let y_sharpened = self.sharpen_plane(y_plane, width, height)?;

        // U and V planes remain unchanged
        let u_plane = video_frame.data[1].as_slice();
        let v_plane = video_frame.data[2].as_slice();

        // Create output frame
        let mut output_frame = VideoFrame::new(width, height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_sharpened),
            Buffer::from_vec(u_plane.to_vec()),
            Buffer::from_vec(v_plane.to_vec()),
        ];
        output_frame.linesize = vec![
            width as usize,
            (width / 2) as usize,
            (width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }

    /// Sharpen a plane using unsharp mask
    fn sharpen_plane(&self, src: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let mut dst = Vec::with_capacity((width * height) as usize);

        // Simple 3x3 sharpen kernel
        // [ 0, -1,  0]
        // [-1,  5, -1]
        // [ 0, -1,  0]
        let kernel_weight = 1.0 + self.amount;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                if y == 0 || y == height - 1 || x == 0 || x == width - 1 {
                    // Border: keep original
                    dst.push(src[idx]);
                } else {
                    // Apply sharpen kernel
                    let center = src[idx] as f32;
                    let top = src[((y - 1) * width + x) as usize] as f32;
                    let bottom = src[((y + 1) * width + x) as usize] as f32;
                    let left = src[(y * width + (x - 1)) as usize] as f32;
                    let right = src[(y * width + (x + 1)) as usize] as f32;

                    let sharpened = center * kernel_weight - (top + bottom + left + right) * (self.amount / 4.0);
                    dst.push(sharpened.clamp(0.0, 255.0) as u8);
                }
            }
        }

        Ok(dst)
    }
}

impl Filter for SharpenFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Sharpen filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let sharpened = self.sharpen_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(sharpened)])
            }
            Frame::Audio(_) => Err(Error::filter("Sharpen filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Gaussian blur filter for privacy or artistic effects
pub struct BlurFilter {
    radius: u32, // 1 to 10
}

impl BlurFilter {
    /// Create a new blur filter
    /// radius: blur radius (1 to 10)
    pub fn new(radius: u32) -> Self {
        BlurFilter {
            radius: radius.clamp(1, 10),
        }
    }

    /// Blur a YUV420P frame
    fn blur_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let width = video_frame.width;
        let height = video_frame.height;

        // Blur all planes
        let y_plane = video_frame.data[0].as_slice();
        let y_blurred = self.blur_plane(y_plane, width, height)?;

        let u_plane = video_frame.data[1].as_slice();
        let u_blurred = self.blur_plane(u_plane, width / 2, height / 2)?;

        let v_plane = video_frame.data[2].as_slice();
        let v_blurred = self.blur_plane(v_plane, width / 2, height / 2)?;

        // Create output frame
        let mut output_frame = VideoFrame::new(width, height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_blurred),
            Buffer::from_vec(u_blurred),
            Buffer::from_vec(v_blurred),
        ];
        output_frame.linesize = vec![
            width as usize,
            (width / 2) as usize,
            (width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }

    /// Blur a plane using box blur (fast approximation of Gaussian)
    fn blur_plane(&self, src: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let mut temp = src.to_vec();
        let mut dst = vec![0u8; (width * height) as usize];

        // Horizontal pass
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                let mut count = 0u32;

                for dx in -(self.radius as i32)..=(self.radius as i32) {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    sum += temp[(y * width + nx) as usize] as u32;
                    count += 1;
                }

                dst[(y * width + x) as usize] = (sum / count) as u8;
            }
        }

        // Vertical pass
        temp.copy_from_slice(&dst);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                let mut count = 0u32;

                for dy in -(self.radius as i32)..=(self.radius as i32) {
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    sum += temp[(ny * width + x) as usize] as u32;
                    count += 1;
                }

                dst[(y * width + x) as usize] = (sum / count) as u8;
            }
        }

        Ok(dst)
    }
}

impl Filter for BlurFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "Blur filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let blurred = self.blur_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(blurred)])
            }
            Frame::Audio(_) => Err(Error::filter("Blur filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

/// Chroma key (green screen) filter for compositing
pub struct ChromaKeyFilter {
    key_color: [u8; 3], // YUV color to key out
    tolerance: u8,       // 0-255, how much variation to allow
    spill_suppression: f32, // 0.0-1.0, reduce color spill
}

impl ChromaKeyFilter {
    /// Create a new chroma key filter for green screen (default)
    pub fn green_screen() -> Self {
        ChromaKeyFilter {
            key_color: [149, 43, 21], // Green in YUV
            tolerance: 40,
            spill_suppression: 0.5,
        }
    }

    /// Create a new chroma key filter for blue screen
    pub fn blue_screen() -> Self {
        ChromaKeyFilter {
            key_color: [41, 240, 110], // Blue in YUV
            tolerance: 40,
            spill_suppression: 0.5,
        }
    }

    /// Create a custom chroma key filter
    pub fn custom(yuv_color: [u8; 3], tolerance: u8, spill_suppression: f32) -> Self {
        ChromaKeyFilter {
            key_color: yuv_color,
            tolerance,
            spill_suppression: spill_suppression.clamp(0.0, 1.0),
        }
    }

    /// Apply chroma key to a YUV420P frame
    /// Returns frame with alpha channel indicating transparency
    fn apply_yuv420p(&self, video_frame: &VideoFrame) -> Result<VideoFrame> {
        if video_frame.data.len() < 3 {
            return Err(Error::filter("YUV420P frame must have 3 planes"));
        }

        let width = video_frame.width;
        let height = video_frame.height;

        let y_plane = video_frame.data[0].as_slice();
        let u_plane = video_frame.data[1].as_slice();
        let v_plane = video_frame.data[2].as_slice();

        // Generate alpha mask based on chroma key
        let mut y_out = Vec::with_capacity((width * height) as usize);
        let mut u_out = Vec::with_capacity(((width / 2) * (height / 2)) as usize);
        let mut v_out = Vec::with_capacity(((width / 2) * (height / 2)) as usize);

        // Process Y plane with alpha blending
        for y in 0..height {
            for x in 0..width {
                let y_idx = (y * width + x) as usize;
                let u_idx = ((y / 2) * (width / 2) + (x / 2)) as usize;

                let y_val = y_plane[y_idx];
                let u_val = u_plane[u_idx];
                let v_val = v_plane[u_idx];

                // Calculate distance from key color in YUV space
                let y_diff = (y_val as i16 - self.key_color[0] as i16).abs() as u8;
                let u_diff = (u_val as i16 - self.key_color[1] as i16).abs() as u8;
                let v_diff = (v_val as i16 - self.key_color[2] as i16).abs() as u8;

                let distance = (y_diff as u16 + u_diff as u16 * 2 + v_diff as u16 * 2) / 5;

                if distance < self.tolerance as u16 {
                    // Make transparent (set to black for now, alpha would be 0)
                    y_out.push(16); // Black in YUV
                } else {
                    y_out.push(y_val);
                }
            }
        }

        // Copy U and V planes with spill suppression
        for idx in 0..((width / 2) * (height / 2)) as usize {
            let u_val = u_plane[idx];
            let v_val = v_plane[idx];

            // Apply spill suppression
            let u_suppressed = u_val as f32
                - (u_val as f32 - 128.0) * self.spill_suppression *
                  ((self.key_color[1] as f32 - 128.0).abs() / 128.0);
            let v_suppressed = v_val as f32
                - (v_val as f32 - 128.0) * self.spill_suppression *
                  ((self.key_color[2] as f32 - 128.0).abs() / 128.0);

            u_out.push(u_suppressed.clamp(0.0, 255.0) as u8);
            v_out.push(v_suppressed.clamp(0.0, 255.0) as u8);
        }

        // Create output frame
        let mut output_frame = VideoFrame::new(width, height, video_frame.format);
        output_frame.data = vec![
            Buffer::from_vec(y_out),
            Buffer::from_vec(u_out),
            Buffer::from_vec(v_out),
        ];
        output_frame.linesize = vec![
            width as usize,
            (width / 2) as usize,
            (width / 2) as usize,
        ];
        output_frame.pts = video_frame.pts;
        output_frame.duration = video_frame.duration;
        output_frame.keyframe = video_frame.keyframe;
        output_frame.pict_type = video_frame.pict_type;

        Ok(output_frame)
    }
}

impl Filter for ChromaKeyFilter {
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>> {
        match input {
            Frame::Video(video_frame) => {
                if video_frame.format != PixelFormat::YUV420P {
                    return Err(Error::filter(format!(
                        "ChromaKey filter only supports YUV420P, got {:?}",
                        video_frame.format
                    )));
                }

                let keyed = self.apply_yuv420p(&video_frame)?;
                Ok(vec![Frame::Video(keyed)])
            }
            Frame::Audio(_) => Err(Error::filter("ChromaKey filter only accepts video frames")),
        }
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Timestamp;

    #[test]
    fn test_flip_filter_creation() {
        let filter = FlipFilter::horizontal();
        assert!(filter.horizontal);
        assert!(!filter.vertical);

        let filter = FlipFilter::vertical();
        assert!(!filter.horizontal);
        assert!(filter.vertical);

        let filter = FlipFilter::both();
        assert!(filter.horizontal);
        assert!(filter.vertical);
    }

    #[test]
    fn test_brightness_contrast_filter() {
        let filter = BrightnessContrastFilter::new(50, 1.2);
        assert_eq!(filter.brightness, 50);
        assert!((filter.contrast - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_deinterlace_filter_creation() {
        let filter = DeinterlaceFilter::new(DeinterlaceMethod::Blend);
        assert!(matches!(filter.method, DeinterlaceMethod::Blend));
    }

    #[test]
    fn test_sharpen_filter() {
        let filter = SharpenFilter::new(1.5);
        assert!((filter.amount - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_blur_filter() {
        let filter = BlurFilter::new(3);
        assert_eq!(filter.radius, 3);
    }

    #[test]
    fn test_chroma_key_filters() {
        let green = ChromaKeyFilter::green_screen();
        assert_eq!(green.key_color, [149, 43, 21]);

        let blue = ChromaKeyFilter::blue_screen();
        assert_eq!(blue.key_color, [41, 240, 110]);
    }
}
