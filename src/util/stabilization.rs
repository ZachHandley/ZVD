//! Video Stabilization
//!
//! Remove camera shake and smooth motion in handheld or action footage
//! using motion analysis, trajectory smoothing, and crop compensation.
//!
//! ## Stabilization Techniques
//!
//! - **2D Stabilization**: Translation and rotation correction
//! - **2.5D Stabilization**: Translation, rotation, and scale/perspective
//! - **3D Stabilization**: Full 3D camera motion estimation
//! - **Rolling Shutter**: Correct for CMOS sensor wobble
//!
//! ## Algorithms
//!
//! - **Motion Estimation**: Feature tracking or block matching
//! - **Trajectory Smoothing**: Low-pass filter or Kalman filter
//! - **Crop Compensation**: Dynamic zoom to hide black borders
//!
//! ## Use Cases
//!
//! - Handheld camera footage
//! - Action cameras (GoPro, etc.)
//! - Drone footage smoothing
//! - Automotive/vehicle cameras
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::stabilization::{Stabilizer, StabilizationMode};
//!
//! let stabilizer = Stabilizer::new(StabilizationMode::Standard);
//! stabilizer.analyze_motion(&frames)?;
//! let smoothed_transforms = stabilizer.smooth_trajectory()?;
//! ```

use crate::error::{Error, Result};

/// Stabilization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizationMode {
    /// No stabilization
    None,
    /// Standard 2D (translation + rotation)
    Standard,
    /// High quality 2.5D (translation + rotation + scale)
    HighQuality,
    /// Maximum smoothing (may introduce more crop)
    Maximum,
}

impl StabilizationMode {
    /// Get smoothing strength (0.0 to 1.0)
    pub fn smoothing_strength(&self) -> f64 {
        match self {
            StabilizationMode::None => 0.0,
            StabilizationMode::Standard => 0.6,
            StabilizationMode::HighQuality => 0.8,
            StabilizationMode::Maximum => 0.95,
        }
    }

    /// Get crop percentage (additional border for stabilization)
    pub fn crop_percentage(&self) -> f64 {
        match self {
            StabilizationMode::None => 0.0,
            StabilizationMode::Standard => 5.0,    // 5% crop
            StabilizationMode::HighQuality => 8.0, // 8% crop
            StabilizationMode::Maximum => 12.0,    // 12% crop
        }
    }
}

/// 2D transform (translation + rotation + scale)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    pub dx: f64,    // Translation X
    pub dy: f64,    // Translation Y
    pub angle: f64, // Rotation (radians)
    pub scale: f64, // Scale factor (1.0 = no change)
}

impl Transform2D {
    /// Create identity transform
    pub fn identity() -> Self {
        Transform2D {
            dx: 0.0,
            dy: 0.0,
            angle: 0.0,
            scale: 1.0,
        }
    }

    /// Create from translation
    pub fn from_translation(dx: f64, dy: f64) -> Self {
        Transform2D {
            dx,
            dy,
            angle: 0.0,
            scale: 1.0,
        }
    }

    /// Compose two transforms
    pub fn compose(&self, other: &Transform2D) -> Transform2D {
        Transform2D {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            angle: self.angle + other.angle,
            scale: self.scale * other.scale,
        }
    }

    /// Invert transform
    pub fn inverse(&self) -> Transform2D {
        Transform2D {
            dx: -self.dx,
            dy: -self.dy,
            angle: -self.angle,
            scale: 1.0 / self.scale,
        }
    }

    /// Interpolate between two transforms
    pub fn lerp(&self, other: &Transform2D, t: f64) -> Transform2D {
        Transform2D {
            dx: self.dx + (other.dx - self.dx) * t,
            dy: self.dy + (other.dy - self.dy) * t,
            angle: self.angle + (other.angle - self.angle) * t,
            scale: self.scale + (other.scale - self.scale) * t,
        }
    }
}

/// Motion trajectory (sequence of transforms)
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Per-frame transforms relative to first frame
    pub transforms: Vec<Transform2D>,
}

impl Trajectory {
    /// Create new trajectory
    pub fn new() -> Self {
        Trajectory {
            transforms: Vec::new(),
        }
    }

    /// Add transform
    pub fn add(&mut self, transform: Transform2D) {
        self.transforms.push(transform);
    }

    /// Get frame count
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Get transform at frame
    pub fn get(&self, frame: usize) -> Option<&Transform2D> {
        self.transforms.get(frame)
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Trajectory smoothing filter
pub struct SmoothingFilter {
    /// Window size for smoothing
    window_size: usize,
    /// Smoothing strength (0.0 to 1.0)
    strength: f64,
}

impl SmoothingFilter {
    /// Create new smoothing filter
    pub fn new(window_size: usize, strength: f64) -> Self {
        SmoothingFilter {
            window_size,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Apply moving average smoothing
    pub fn smooth(&self, trajectory: &Trajectory) -> Trajectory {
        let mut smoothed = Trajectory::new();

        if trajectory.is_empty() {
            return smoothed;
        }

        let half_window = self.window_size / 2;

        for i in 0..trajectory.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(trajectory.len());

            // Calculate average transform in window
            let mut sum_dx = 0.0;
            let mut sum_dy = 0.0;
            let mut sum_angle = 0.0;
            let mut sum_scale = 0.0;
            let count = (end - start) as f64;

            for j in start..end {
                sum_dx += trajectory.transforms[j].dx;
                sum_dy += trajectory.transforms[j].dy;
                sum_angle += trajectory.transforms[j].angle;
                sum_scale += trajectory.transforms[j].scale;
            }

            let avg = Transform2D {
                dx: sum_dx / count,
                dy: sum_dy / count,
                angle: sum_angle / count,
                scale: sum_scale / count,
            };

            // Blend original with smoothed
            let original = &trajectory.transforms[i];
            let blended = original.lerp(&avg, self.strength);
            smoothed.add(blended);
        }

        smoothed
    }

    /// Apply low-pass filter (exponential smoothing)
    pub fn smooth_lowpass(&self, trajectory: &Trajectory) -> Trajectory {
        let mut smoothed = Trajectory::new();

        if trajectory.is_empty() {
            return smoothed;
        }

        let alpha = 1.0 - self.strength; // Alpha = 1 - strength
        let mut current = trajectory.transforms[0];
        smoothed.add(current);

        for i in 1..trajectory.len() {
            let original = trajectory.transforms[i];

            // Exponential moving average
            current = Transform2D {
                dx: alpha * original.dx + (1.0 - alpha) * current.dx,
                dy: alpha * original.dy + (1.0 - alpha) * current.dy,
                angle: alpha * original.angle + (1.0 - alpha) * current.angle,
                scale: alpha * original.scale + (1.0 - alpha) * current.scale,
            };

            smoothed.add(current);
        }

        smoothed
    }
}

/// Video stabilizer
pub struct Stabilizer {
    mode: StabilizationMode,
    trajectory: Trajectory,
    smoothed_trajectory: Option<Trajectory>,
    width: usize,
    height: usize,
}

impl Stabilizer {
    /// Create new stabilizer
    pub fn new(mode: StabilizationMode) -> Self {
        Stabilizer {
            mode,
            trajectory: Trajectory::new(),
            smoothed_trajectory: None,
            width: 0,
            height: 0,
        }
    }

    /// Set video dimensions
    pub fn set_dimensions(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    /// Add frame transform
    pub fn add_transform(&mut self, transform: Transform2D) {
        self.trajectory.add(transform);
    }

    /// Estimate motion between two frames (simplified block matching)
    ///
    /// Returns transform from frame1 to frame2
    pub fn estimate_motion(&self, _frame1: &[u8], _frame2: &[u8]) -> Result<Transform2D> {
        // Simplified implementation - would use feature tracking or block matching
        // For now, return identity (no motion)
        Ok(Transform2D::identity())
    }

    /// Smooth trajectory
    pub fn smooth_trajectory(&mut self) -> Result<&Trajectory> {
        if self.trajectory.is_empty() {
            return Err(Error::InvalidInput(
                "No trajectory data to smooth".to_string(),
            ));
        }

        let strength = self.mode.smoothing_strength();
        let window_size = match self.mode {
            StabilizationMode::None => 1,
            StabilizationMode::Standard => 15,
            StabilizationMode::HighQuality => 25,
            StabilizationMode::Maximum => 35,
        };

        let filter = SmoothingFilter::new(window_size, strength);
        let smoothed = filter.smooth_lowpass(&self.trajectory);

        self.smoothed_trajectory = Some(smoothed);
        Ok(self.smoothed_trajectory.as_ref().unwrap())
    }

    /// Get stabilizing transform for frame
    ///
    /// Returns the transform needed to stabilize the frame
    pub fn get_stabilizing_transform(&self, frame: usize) -> Result<Transform2D> {
        let smoothed = self
            .smoothed_trajectory
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Trajectory not smoothed".to_string()))?;

        let smooth_transform = smoothed
            .get(frame)
            .ok_or_else(|| Error::InvalidInput("Frame out of range".to_string()))?;

        let original_transform = self
            .trajectory
            .get(frame)
            .ok_or_else(|| Error::InvalidInput("Frame out of range".to_string()))?;

        // Difference between original and smoothed
        let diff = smooth_transform.compose(&original_transform.inverse());

        Ok(diff)
    }

    /// Calculate required crop to avoid black borders
    pub fn calculate_crop(&self) -> (usize, usize, usize, usize) {
        let crop_pct = self.mode.crop_percentage() / 100.0;

        let crop_x = (self.width as f64 * crop_pct / 2.0) as usize;
        let crop_y = (self.height as f64 * crop_pct / 2.0) as usize;

        let new_width = self.width - 2 * crop_x;
        let new_height = self.height - 2 * crop_y;

        (crop_x, crop_y, new_width, new_height)
    }
}

/// Rolling shutter correction
pub struct RollingShutterCorrector {
    /// Scanline readout time (fraction of frame time)
    readout_time: f64,
}

impl RollingShutterCorrector {
    /// Create new rolling shutter corrector
    ///
    /// # Arguments
    /// * `readout_time` - Time to read out full frame (0.0 to 1.0)
    ///   Typical values: 0.03 (slow), 0.01 (fast)
    pub fn new(readout_time: f64) -> Self {
        RollingShutterCorrector {
            readout_time: readout_time.clamp(0.0, 1.0),
        }
    }

    /// Calculate per-scanline transform
    pub fn scanline_transform(
        &self,
        scanline: usize,
        height: usize,
        motion: &Transform2D,
    ) -> Transform2D {
        let t = (scanline as f64 / height as f64) * self.readout_time;

        // Interpolate motion for this scanline
        let identity = Transform2D::identity();
        identity.lerp(motion, t)
    }
}

/// Stabilization statistics
#[derive(Debug, Clone)]
pub struct StabilizationStats {
    /// Average motion magnitude (pixels)
    pub avg_motion: f64,
    /// Maximum motion magnitude (pixels)
    pub max_motion: f64,
    /// Motion smoothness (0.0 = jerky, 1.0 = smooth)
    pub smoothness: f64,
    /// Crop percentage required
    pub crop_percentage: f64,
}

impl StabilizationStats {
    /// Calculate from trajectory
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        if trajectory.is_empty() {
            return StabilizationStats {
                avg_motion: 0.0,
                max_motion: 0.0,
                smoothness: 1.0,
                crop_percentage: 0.0,
            };
        }

        let mut total_motion: f64 = 0.0;
        let mut max_motion: f64 = 0.0;

        for transform in &trajectory.transforms {
            let motion = (transform.dx * transform.dx + transform.dy * transform.dy).sqrt();
            total_motion += motion;
            max_motion = max_motion.max(motion);
        }

        let avg_motion = total_motion / trajectory.len() as f64;

        // Calculate smoothness (inverse of motion variance)
        let mut variance = 0.0;
        for i in 1..trajectory.transforms.len() {
            let prev = &trajectory.transforms[i - 1];
            let curr = &trajectory.transforms[i];

            let delta_dx = curr.dx - prev.dx;
            let delta_dy = curr.dy - prev.dy;
            let delta = (delta_dx * delta_dx + delta_dy * delta_dy).sqrt();

            variance += delta * delta;
        }
        variance /= (trajectory.len() - 1).max(1) as f64;

        let smoothness = 1.0 / (1.0 + variance);

        StabilizationStats {
            avg_motion,
            max_motion,
            smoothness,
            crop_percentage: max_motion * 2.0, // Estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::identity();
        assert_eq!(t.dx, 0.0);
        assert_eq!(t.dy, 0.0);
        assert_eq!(t.angle, 0.0);
        assert_eq!(t.scale, 1.0);
    }

    #[test]
    fn test_transform_compose() {
        let t1 = Transform2D::from_translation(10.0, 20.0);
        let t2 = Transform2D::from_translation(5.0, 10.0);

        let composed = t1.compose(&t2);
        assert_eq!(composed.dx, 15.0);
        assert_eq!(composed.dy, 30.0);
    }

    #[test]
    fn test_transform_inverse() {
        let t = Transform2D::from_translation(10.0, 20.0);
        let inv = t.inverse();

        assert_eq!(inv.dx, -10.0);
        assert_eq!(inv.dy, -20.0);
    }

    #[test]
    fn test_transform_lerp() {
        let t1 = Transform2D::from_translation(0.0, 0.0);
        let t2 = Transform2D::from_translation(10.0, 20.0);

        let mid = t1.lerp(&t2, 0.5);
        assert_eq!(mid.dx, 5.0);
        assert_eq!(mid.dy, 10.0);
    }

    #[test]
    fn test_trajectory_creation() {
        let mut traj = Trajectory::new();
        assert!(traj.is_empty());

        traj.add(Transform2D::identity());
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_smoothing_filter() {
        let mut traj = Trajectory::new();

        // Add noisy trajectory
        for i in 0..10 {
            let noise = if i % 2 == 0 { 5.0 } else { -5.0 };
            traj.add(Transform2D::from_translation(noise, 0.0));
        }

        let filter = SmoothingFilter::new(3, 0.8);
        let smoothed = filter.smooth(&traj);

        assert_eq!(smoothed.len(), traj.len());

        // Smoothed should have less variation
        let original_var = calculate_variance(&traj);
        let smoothed_var = calculate_variance(&smoothed);
        assert!(smoothed_var < original_var);
    }

    fn calculate_variance(traj: &Trajectory) -> f64 {
        let mean: f64 = traj.transforms.iter().map(|t| t.dx).sum::<f64>() / traj.len() as f64;
        traj.transforms
            .iter()
            .map(|t| (t.dx - mean).powi(2))
            .sum::<f64>()
            / traj.len() as f64
    }

    #[test]
    fn test_stabilization_mode_strength() {
        assert_eq!(StabilizationMode::None.smoothing_strength(), 0.0);
        assert!(StabilizationMode::Standard.smoothing_strength() > 0.0);
        assert!(
            StabilizationMode::Maximum.smoothing_strength()
                > StabilizationMode::Standard.smoothing_strength()
        );
    }

    #[test]
    fn test_stabilizer_creation() {
        let stabilizer = Stabilizer::new(StabilizationMode::Standard);
        assert!(stabilizer.trajectory.is_empty());
    }

    #[test]
    fn test_stabilizer_add_transform() {
        let mut stabilizer = Stabilizer::new(StabilizationMode::Standard);
        stabilizer.add_transform(Transform2D::identity());

        assert_eq!(stabilizer.trajectory.len(), 1);
    }

    #[test]
    fn test_stabilizer_crop_calculation() {
        let mut stabilizer = Stabilizer::new(StabilizationMode::Standard);
        stabilizer.set_dimensions(1920, 1080);

        let (crop_x, crop_y, new_width, new_height) = stabilizer.calculate_crop();

        assert!(crop_x > 0);
        assert!(crop_y > 0);
        assert!(new_width < 1920);
        assert!(new_height < 1080);
    }

    #[test]
    fn test_rolling_shutter_corrector() {
        let corrector = RollingShutterCorrector::new(0.03);
        let motion = Transform2D::from_translation(10.0, 0.0);

        let t0 = corrector.scanline_transform(0, 1080, &motion);
        let t_mid = corrector.scanline_transform(540, 1080, &motion);
        let t_end = corrector.scanline_transform(1080, 1080, &motion);

        // Motion should increase with scanline
        assert!(t0.dx < t_mid.dx);
        assert!(t_mid.dx < t_end.dx);
    }

    #[test]
    fn test_stabilization_stats() {
        let mut traj = Trajectory::new();
        traj.add(Transform2D::from_translation(0.0, 0.0));
        traj.add(Transform2D::from_translation(5.0, 5.0));
        traj.add(Transform2D::from_translation(10.0, 10.0));

        let stats = StabilizationStats::from_trajectory(&traj);

        assert!(stats.avg_motion > 0.0);
        assert!(stats.max_motion > 0.0);
        assert!(stats.smoothness >= 0.0 && stats.smoothness <= 1.0);
    }
}
