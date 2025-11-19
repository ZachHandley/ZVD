//! Frame Interpolation and Rate Conversion
//!
//! Generate intermediate frames for slow motion, smooth playback, or frame rate
//! conversion using various interpolation techniques.
//!
//! ## Interpolation Methods
//!
//! - **Frame Blending**: Simple weighted average (fast, ghosting artifacts)
//! - **Motion Estimation**: Block-matching motion vectors
//! - **Optical Flow**: Dense motion field (high quality, slow)
//! - **Frame Repeat**: Duplicate frames (no interpolation)
//!
//! ## Use Cases
//!
//! - Slow motion from high frame rate footage
//! - Frame rate conversion (24fps -> 60fps, 30fps -> 24fps)
//! - Smooth motion for action sequences
//! - Time stretching/compression
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::interpolation::{FrameInterpolator, InterpolationMethod};
//!
//! let interpolator = FrameInterpolator::new(InterpolationMethod::Blend);
//! let intermediate = interpolator.interpolate(&frame1, &frame2, 0.5)?;
//! ```

use crate::error::{Error, Result};
use std::cmp;

/// Interpolation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Frame blending (weighted average)
    Blend,
    /// Motion-compensated interpolation (block matching)
    MotionCompensated,
    /// Optical flow (dense motion field)
    OpticalFlow,
    /// Nearest neighbor (duplicate frame)
    Nearest,
}

/// Motion vector (2D displacement)
#[derive(Debug, Clone, Copy)]
pub struct MotionVector {
    pub dx: i32,
    pub dy: i32,
    pub cost: u32, // SAD (Sum of Absolute Differences)
}

impl MotionVector {
    /// Create zero motion vector
    pub fn zero() -> Self {
        MotionVector {
            dx: 0,
            dy: 0,
            cost: 0,
        }
    }

    /// Get magnitude
    pub fn magnitude(&self) -> f32 {
        ((self.dx * self.dx + self.dy * self.dy) as f32).sqrt()
    }
}

/// Frame data wrapper for interpolation
#[derive(Debug, Clone)]
pub struct FrameData {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>, // RGB or YUV
    pub format: FrameFormat,
}

/// Frame format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    RGB8,
    YUV420P,
    YUV444P,
}

impl FrameData {
    /// Create new frame data
    pub fn new(width: usize, height: usize, format: FrameFormat) -> Self {
        let size = match format {
            FrameFormat::RGB8 => width * height * 3,
            FrameFormat::YUV420P => width * height * 3 / 2,
            FrameFormat::YUV444P => width * height * 3,
        };

        FrameData {
            width,
            height,
            data: vec![0u8; size],
            format,
        }
    }

    /// Get Y plane (for YUV formats)
    pub fn y_plane(&self) -> &[u8] {
        match self.format {
            FrameFormat::YUV420P | FrameFormat::YUV444P => {
                &self.data[0..self.width * self.height]
            }
            _ => &[],
        }
    }
}

/// Frame interpolator
pub struct FrameInterpolator {
    method: InterpolationMethod,
    block_size: usize, // For motion estimation
    search_range: i32, // Search range for motion vectors
}

impl FrameInterpolator {
    /// Create new interpolator
    pub fn new(method: InterpolationMethod) -> Self {
        FrameInterpolator {
            method,
            block_size: 16, // 16x16 blocks
            search_range: 16, // ±16 pixel search
        }
    }

    /// Configure block size for motion estimation
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Configure search range
    pub fn with_search_range(mut self, range: i32) -> Self {
        self.search_range = range;
        self
    }

    /// Interpolate between two frames
    ///
    /// # Arguments
    /// * `frame1` - First frame (t=0.0)
    /// * `frame2` - Second frame (t=1.0)
    /// * `t` - Interpolation factor (0.0 to 1.0)
    pub fn interpolate(
        &self,
        frame1: &FrameData,
        frame2: &FrameData,
        t: f64,
    ) -> Result<FrameData> {
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(Error::InvalidInput(
                "Frame dimensions must match".to_string(),
            ));
        }

        if frame1.format != frame2.format {
            return Err(Error::InvalidInput("Frame formats must match".to_string()));
        }

        if !(0.0..=1.0).contains(&t) {
            return Err(Error::InvalidInput(
                "Interpolation factor must be in [0.0, 1.0]".to_string(),
            ));
        }

        match self.method {
            InterpolationMethod::Blend => self.blend_frames(frame1, frame2, t),
            InterpolationMethod::MotionCompensated => {
                self.motion_compensated_interpolation(frame1, frame2, t)
            }
            InterpolationMethod::OpticalFlow => self.optical_flow_interpolation(frame1, frame2, t),
            InterpolationMethod::Nearest => {
                if t < 0.5 {
                    Ok(frame1.clone())
                } else {
                    Ok(frame2.clone())
                }
            }
        }
    }

    /// Simple frame blending (weighted average)
    fn blend_frames(&self, frame1: &FrameData, frame2: &FrameData, t: f64) -> Result<FrameData> {
        let mut result = FrameData::new(frame1.width, frame1.height, frame1.format);

        let weight2 = t;
        let weight1 = 1.0 - t;

        for i in 0..frame1.data.len() {
            let v1 = frame1.data[i] as f64;
            let v2 = frame2.data[i] as f64;
            result.data[i] = (v1 * weight1 + v2 * weight2) as u8;
        }

        Ok(result)
    }

    /// Motion-compensated interpolation using block matching
    fn motion_compensated_interpolation(
        &self,
        frame1: &FrameData,
        frame2: &FrameData,
        t: f64,
    ) -> Result<FrameData> {
        // For non-YUV formats, fall back to blending
        if frame1.format == FrameFormat::RGB8 {
            return self.blend_frames(frame1, frame2, t);
        }

        let mut result = FrameData::new(frame1.width, frame1.height, frame1.format);

        // Estimate motion vectors
        let motion_field = self.estimate_motion(frame1, frame2)?;

        // Interpolate using motion vectors
        let y_plane1 = frame1.y_plane();
        let y_plane2 = frame2.y_plane();
        let result_y = &mut result.data[0..frame1.width * frame1.height];

        let blocks_x = (frame1.width + self.block_size - 1) / self.block_size;
        let blocks_y = (frame1.height + self.block_size - 1) / self.block_size;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                let mv = motion_field[block_idx];

                // Interpolate block with motion compensation
                for py in 0..self.block_size {
                    for px in 0..self.block_size {
                        let x = bx * self.block_size + px;
                        let y = by * self.block_size + py;

                        if x >= frame1.width || y >= frame1.height {
                            continue;
                        }

                        let idx1 = y * frame1.width + x;

                        // Motion-compensated position in frame2
                        let x2 = (x as i32 + (mv.dx as f64 * t) as i32)
                            .max(0)
                            .min(frame1.width as i32 - 1) as usize;
                        let y2 = (y as i32 + (mv.dy as f64 * t) as i32)
                            .max(0)
                            .min(frame1.height as i32 - 1) as usize;
                        let idx2 = y2 * frame1.width + x2;

                        // Blend with motion compensation
                        let v1 = y_plane1[idx1] as f64;
                        let v2 = y_plane2[idx2] as f64;
                        result_y[idx1] = ((1.0 - t) * v1 + t * v2) as u8;
                    }
                }
            }
        }

        // For UV planes, use simple blending
        let y_size = frame1.width * frame1.height;
        for i in y_size..frame1.data.len() {
            let v1 = frame1.data[i] as f64;
            let v2 = frame2.data[i] as f64;
            result.data[i] = ((1.0 - t) * v1 + t * v2) as u8;
        }

        Ok(result)
    }

    /// Estimate motion vectors using block matching
    fn estimate_motion(&self, frame1: &FrameData, frame2: &FrameData) -> Result<Vec<MotionVector>> {
        let y_plane1 = frame1.y_plane();
        let y_plane2 = frame2.y_plane();

        let blocks_x = (frame1.width + self.block_size - 1) / self.block_size;
        let blocks_y = (frame1.height + self.block_size - 1) / self.block_size;

        let mut motion_field = vec![MotionVector::zero(); blocks_x * blocks_y];

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_x = bx * self.block_size;
                let block_y = by * self.block_size;

                let mv = self.find_best_match(
                    y_plane1,
                    y_plane2,
                    frame1.width,
                    frame1.height,
                    block_x,
                    block_y,
                );

                motion_field[by * blocks_x + bx] = mv;
            }
        }

        Ok(motion_field)
    }

    /// Find best matching block using Sum of Absolute Differences (SAD)
    fn find_best_match(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
        block_x: usize,
        block_y: usize,
    ) -> MotionVector {
        let mut best_mv = MotionVector::zero();
        let mut best_cost = u32::MAX;

        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let cost = self.calculate_sad(
                    frame1,
                    frame2,
                    width,
                    height,
                    block_x,
                    block_y,
                    dx,
                    dy,
                );

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = MotionVector { dx, dy, cost };
                }
            }
        }

        best_mv
    }

    /// Calculate Sum of Absolute Differences (SAD)
    fn calculate_sad(
        &self,
        frame1: &[u8],
        frame2: &[u8],
        width: usize,
        height: usize,
        block_x: usize,
        block_y: usize,
        dx: i32,
        dy: i32,
    ) -> u32 {
        let mut sad = 0u32;

        for py in 0..self.block_size {
            for px in 0..self.block_size {
                let x1 = block_x + px;
                let y1 = block_y + py;

                if x1 >= width || y1 >= height {
                    continue;
                }

                let x2 = (x1 as i32 + dx).max(0).min(width as i32 - 1) as usize;
                let y2 = (y1 as i32 + dy).max(0).min(height as i32 - 1) as usize;

                let idx1 = y1 * width + x1;
                let idx2 = y2 * width + x2;

                sad += (frame1[idx1] as i32 - frame2[idx2] as i32).unsigned_abs();
            }
        }

        sad
    }

    /// Optical flow interpolation (simplified)
    fn optical_flow_interpolation(
        &self,
        frame1: &FrameData,
        frame2: &FrameData,
        t: f64,
    ) -> Result<FrameData> {
        // Use motion compensation as a simplified optical flow
        // Real optical flow would use dense pyramid algorithms (e.g., Farneback)
        self.motion_compensated_interpolation(frame1, frame2, t)
    }
}

/// Frame rate converter
pub struct FrameRateConverter {
    interpolator: FrameInterpolator,
    source_fps: f64,
    target_fps: f64,
}

impl FrameRateConverter {
    /// Create new frame rate converter
    pub fn new(source_fps: f64, target_fps: f64, method: InterpolationMethod) -> Self {
        FrameRateConverter {
            interpolator: FrameInterpolator::new(method),
            source_fps,
            target_fps,
        }
    }

    /// Calculate output frame count
    pub fn output_frame_count(&self, input_frames: usize) -> usize {
        let duration = input_frames as f64 / self.source_fps;
        (duration * self.target_fps).round() as usize
    }

    /// Calculate which source frames to use for output frame
    ///
    /// Returns (frame1_idx, frame2_idx, t) where t is interpolation factor
    pub fn source_frames_for_output(&self, output_frame: usize) -> (usize, usize, f64) {
        let output_time = output_frame as f64 / self.target_fps;
        let source_frame_f = output_time * self.source_fps;

        let frame1_idx = source_frame_f.floor() as usize;
        let frame2_idx = (source_frame_f.ceil() as usize).max(frame1_idx + 1);
        let t = source_frame_f.fract();

        (frame1_idx, frame2_idx, t)
    }
}

/// Slow motion generator
pub struct SlowMotionGenerator {
    interpolator: FrameInterpolator,
    speed_factor: f64, // 0.5 = half speed (2x slow motion)
}

impl SlowMotionGenerator {
    /// Create new slow motion generator
    ///
    /// # Arguments
    /// * `speed_factor` - Playback speed (0.5 = 2x slow, 0.25 = 4x slow)
    /// * `method` - Interpolation method
    pub fn new(speed_factor: f64, method: InterpolationMethod) -> Result<Self> {
        if speed_factor <= 0.0 || speed_factor > 1.0 {
            return Err(Error::InvalidInput(
                "Speed factor must be in (0.0, 1.0]".to_string(),
            ));
        }

        Ok(SlowMotionGenerator {
            interpolator: FrameInterpolator::new(method),
            speed_factor,
        })
    }

    /// Calculate output frame count
    pub fn output_frame_count(&self, input_frames: usize) -> usize {
        (input_frames as f64 / self.speed_factor).round() as usize
    }

    /// Calculate which source frames to use for output frame
    pub fn source_frames_for_output(&self, output_frame: usize) -> (usize, usize, f64) {
        let source_frame_f = output_frame as f64 * self.speed_factor;

        let frame1_idx = source_frame_f.floor() as usize;
        let frame2_idx = frame1_idx + 1;
        let t = source_frame_f.fract();

        (frame1_idx, frame2_idx, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: usize, height: usize, value: u8) -> FrameData {
        FrameData {
            width,
            height,
            data: vec![value; width * height * 3 / 2], // YUV420P
            format: FrameFormat::YUV420P,
        }
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector {
            dx: 3,
            dy: 4,
            cost: 0,
        };
        assert_eq!(mv.magnitude(), 5.0);
    }

    #[test]
    fn test_blend_interpolation() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 100);

        let interpolator = FrameInterpolator::new(InterpolationMethod::Blend);
        let result = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();

        // Middle value should be ~50
        assert_eq!(result.data[0], 50);
    }

    #[test]
    fn test_blend_interpolation_weights() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 100);

        let interpolator = FrameInterpolator::new(InterpolationMethod::Blend);

        let result_25 = interpolator.interpolate(&frame1, &frame2, 0.25).unwrap();
        assert_eq!(result_25.data[0], 25);

        let result_75 = interpolator.interpolate(&frame1, &frame2, 0.75).unwrap();
        assert_eq!(result_75.data[0], 75);
    }

    #[test]
    fn test_nearest_interpolation() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 100);

        let interpolator = FrameInterpolator::new(InterpolationMethod::Nearest);

        let result_low = interpolator.interpolate(&frame1, &frame2, 0.3).unwrap();
        assert_eq!(result_low.data[0], 0); // Should be frame1

        let result_high = interpolator.interpolate(&frame1, &frame2, 0.6).unwrap();
        assert_eq!(result_high.data[0], 100); // Should be frame2
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
        assert_eq!(mv.magnitude(), 0.0);
    }

    #[test]
    fn test_frame_rate_converter() {
        let converter = FrameRateConverter::new(24.0, 60.0, InterpolationMethod::Blend);

        // 24fps -> 60fps
        let output_count = converter.output_frame_count(24);
        assert_eq!(output_count, 60);

        // First output frame should use first two source frames
        let (f1, f2, t) = converter.source_frames_for_output(0);
        assert_eq!(f1, 0);
        assert!(t >= 0.0 && t <= 1.0);
    }

    #[test]
    fn test_slow_motion_2x() {
        let slow_mo = SlowMotionGenerator::new(0.5, InterpolationMethod::Blend).unwrap();

        // 2x slow motion should double frame count
        let output_count = slow_mo.output_frame_count(100);
        assert_eq!(output_count, 200);

        let (f1, f2, t) = slow_mo.source_frames_for_output(1);
        assert_eq!(f1, 0);
        assert_eq!(f2, 1);
        assert!(t >= 0.0 && t <= 1.0);
    }

    #[test]
    fn test_slow_motion_4x() {
        let slow_mo = SlowMotionGenerator::new(0.25, InterpolationMethod::Blend).unwrap();

        // 4x slow motion should quadruple frame count
        let output_count = slow_mo.output_frame_count(50);
        assert_eq!(output_count, 200);
    }

    #[test]
    fn test_invalid_interpolation_factor() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 100);

        let interpolator = FrameInterpolator::new(InterpolationMethod::Blend);

        assert!(interpolator.interpolate(&frame1, &frame2, -0.1).is_err());
        assert!(interpolator.interpolate(&frame1, &frame2, 1.5).is_err());
    }

    #[test]
    fn test_mismatched_frame_dimensions() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(128, 128, 100);

        let interpolator = FrameInterpolator::new(InterpolationMethod::Blend);
        assert!(interpolator.interpolate(&frame1, &frame2, 0.5).is_err());
    }
}
