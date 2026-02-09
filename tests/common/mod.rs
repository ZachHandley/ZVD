//! Common test utilities for ZVD integration tests
//!
//! This module provides helper functions for creating test frames, audio samples,
//! and verifying codec output across all test suites.

use zvd_lib::codec::frame::{AudioFrame, PictureType, VideoFrame};
use zvd_lib::format::{Packet, PacketFlags};
use zvd_lib::util::{Buffer, MediaType, PixelFormat, SampleFormat, Timestamp};

// ============================================================================
// Video Frame Generation
// ============================================================================

/// Create a test video frame with YUV420P format
///
/// The frame contains a test pattern that varies based on the fill value.
pub fn create_test_video_frame(width: u32, height: u32) -> VideoFrame {
    create_test_video_frame_with_pattern(width, height, PixelFormat::YUV420P, 0, 128)
}

/// Create a test video frame with specified dimensions, format, PTS, and fill pattern
pub fn create_test_video_frame_with_pattern(
    width: u32,
    height: u32,
    format: PixelFormat,
    pts: i64,
    fill_pattern: u8,
) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, format);
    frame.pts = Timestamp::new(pts);
    frame.keyframe = pts == 0;
    frame.pict_type = if pts == 0 {
        PictureType::I
    } else {
        PictureType::P
    };

    match format {
        PixelFormat::YUV420P => {
            // Y plane (full resolution) - gradient pattern
            let y_size = (width * height) as usize;
            let mut y_data = Vec::with_capacity(y_size);
            for row in 0..height {
                for col in 0..width {
                    // Create gradient based on position and fill pattern
                    let y_val =
                        ((fill_pattern as u32 + row + col) % 256) as u8;
                    y_data.push(y_val);
                }
            }
            frame.data.push(Buffer::from_vec(y_data));
            frame.linesize.push(width as usize);

            // U plane (half width, half height)
            let uv_width = (width / 2) as usize;
            let uv_height = (height / 2) as usize;
            let uv_size = uv_width * uv_height;
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(uv_width);

            // V plane (half width, half height)
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(uv_width);
        }
        PixelFormat::YUV422P => {
            // Y plane (full resolution)
            let y_size = (width * height) as usize;
            frame
                .data
                .push(Buffer::from_vec(vec![fill_pattern; y_size]));
            frame.linesize.push(width as usize);

            // U plane (half width, full height)
            let uv_width = (width / 2) as usize;
            let uv_size = uv_width * height as usize;
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(uv_width);

            // V plane (half width, full height)
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(uv_width);
        }
        PixelFormat::YUV444P => {
            // Y plane (full resolution)
            let size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![fill_pattern; size]));
            frame.linesize.push(width as usize);

            // U plane (full resolution)
            frame.data.push(Buffer::from_vec(vec![128u8; size]));
            frame.linesize.push(width as usize);

            // V plane (full resolution)
            frame.data.push(Buffer::from_vec(vec![128u8; size]));
            frame.linesize.push(width as usize);
        }
        PixelFormat::GRAY8 => {
            // Single plane (grayscale)
            let size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![fill_pattern; size]));
            frame.linesize.push(width as usize);
        }
        PixelFormat::RGB24 => {
            // Packed RGB (3 bytes per pixel)
            let size = (width * height * 3) as usize;
            let mut data = Vec::with_capacity(size);
            for row in 0..height {
                for col in 0..width {
                    // Create color bars pattern
                    let r = ((fill_pattern as u32 + col) % 256) as u8;
                    let g = ((fill_pattern as u32 + row) % 256) as u8;
                    let b = ((fill_pattern as u32 + col + row) % 256) as u8;
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
            }
            frame.data.push(Buffer::from_vec(data));
            frame.linesize.push((width * 3) as usize);
        }
        _ => {
            // For unsupported formats, create empty frame with proper structure
            let y_size = (width * height) as usize;
            frame
                .data
                .push(Buffer::from_vec(vec![fill_pattern; y_size]));
            frame.linesize.push(width as usize);
        }
    }

    frame
}

/// Generate SMPTE color bars pattern
///
/// Creates a frame with the standard SMPTE color bars test pattern
/// used for calibrating video equipment.
pub fn generate_color_bars(width: u32, height: u32) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;
    frame.pict_type = PictureType::I;

    // SMPTE color bars (Y, U, V values) - 8 bars
    // White, Yellow, Cyan, Green, Magenta, Red, Blue, Black
    let bars: [(u8, u8, u8); 8] = [
        (235, 128, 128), // White
        (210, 16, 146),  // Yellow
        (170, 166, 16),  // Cyan
        (145, 54, 34),   // Green
        (106, 202, 222), // Magenta
        (81, 90, 240),   // Red
        (41, 240, 110),  // Blue
        (16, 128, 128),  // Black
    ];

    let bar_width = width / 8;

    // Y plane
    let y_size = (width * height) as usize;
    let mut y_data = Vec::with_capacity(y_size);
    for _row in 0..height {
        for col in 0..width {
            let bar_index = (col / bar_width).min(7) as usize;
            y_data.push(bars[bar_index].0);
        }
    }
    frame.data.push(Buffer::from_vec(y_data));
    frame.linesize.push(width as usize);

    // U plane (half resolution)
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_size = (uv_width * uv_height) as usize;
    let mut u_data = Vec::with_capacity(uv_size);
    for _row in 0..uv_height {
        for col in 0..uv_width {
            let bar_index = ((col * 2) / bar_width).min(7) as usize;
            u_data.push(bars[bar_index].1);
        }
    }
    frame.data.push(Buffer::from_vec(u_data));
    frame.linesize.push(uv_width as usize);

    // V plane (half resolution)
    let mut v_data = Vec::with_capacity(uv_size);
    for _row in 0..uv_height {
        for col in 0..uv_width {
            let bar_index = ((col * 2) / bar_width).min(7) as usize;
            v_data.push(bars[bar_index].2);
        }
    }
    frame.data.push(Buffer::from_vec(v_data));
    frame.linesize.push(uv_width as usize);

    frame
}

/// Generate a gradient test pattern
pub fn generate_gradient(width: u32, height: u32) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;
    frame.pict_type = PictureType::I;

    // Y plane - horizontal gradient
    let y_size = (width * height) as usize;
    let mut y_data = Vec::with_capacity(y_size);
    for _row in 0..height {
        for col in 0..width {
            let y = (16 + (col * 219) / width) as u8;
            y_data.push(y);
        }
    }
    frame.data.push(Buffer::from_vec(y_data));
    frame.linesize.push(width as usize);

    // U and V planes - neutral
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_size = (uv_width * uv_height) as usize;
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push(uv_width as usize);
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push(uv_width as usize);

    frame
}

// ============================================================================
// Audio Frame Generation
// ============================================================================

/// Create a test audio frame with a sine wave
pub fn create_test_audio_frame(sample_rate: u32, channels: u16, samples: usize) -> AudioFrame {
    create_test_audio_frame_with_frequency(sample_rate, channels, samples, 440.0)
}

/// Create a test audio frame with a specific frequency sine wave
pub fn create_test_audio_frame_with_frequency(
    sample_rate: u32,
    channels: u16,
    samples: usize,
    frequency: f64,
) -> AudioFrame {
    let mut frame = AudioFrame::new(samples, sample_rate, channels, SampleFormat::F32);
    frame.pts = Timestamp::new(0);

    let total_samples = samples * channels as usize;
    let mut data = Vec::with_capacity(total_samples * 4); // f32 = 4 bytes

    for i in 0..samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * frequency * t).sin() * 0.5;

        for _ch in 0..channels {
            data.extend_from_slice(&(sample as f32).to_le_bytes());
        }
    }

    frame.data.push(Buffer::from_vec(data));
    frame
}

/// Create a test audio frame with i16 samples
pub fn create_test_audio_frame_i16(sample_rate: u32, channels: u16, samples: usize) -> AudioFrame {
    let mut frame = AudioFrame::new(samples, sample_rate, channels, SampleFormat::I16);
    frame.pts = Timestamp::new(0);

    let total_samples = samples * channels as usize;
    let mut data = Vec::with_capacity(total_samples * 2); // i16 = 2 bytes

    let frequency = 440.0;
    for i in 0..samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * frequency * t).sin() * 0.5;
        let sample_i16 = (sample * i16::MAX as f64) as i16;

        for _ch in 0..channels {
            data.extend_from_slice(&sample_i16.to_le_bytes());
        }
    }

    frame.data.push(Buffer::from_vec(data));
    frame
}

/// Create silence audio frame
pub fn create_silence_audio_frame(
    sample_rate: u32,
    channels: u16,
    samples: usize,
    format: SampleFormat,
) -> AudioFrame {
    let mut frame = AudioFrame::new(samples, sample_rate, channels, format);
    frame.pts = Timestamp::new(0);

    let sample_size = format.sample_size();
    let total_size = samples * channels as usize * sample_size;

    frame.data.push(Buffer::from_vec(vec![0u8; total_size]));
    frame
}

// ============================================================================
// Packet Creation
// ============================================================================

/// Create a test video packet
pub fn create_test_video_packet(data: Vec<u8>, pts: i64, keyframe: bool) -> Packet {
    let mut packet = Packet::new_video(0, Buffer::from_vec(data));
    packet.pts = Timestamp::new(pts);
    packet.dts = Timestamp::new(pts);
    packet.flags = PacketFlags {
        keyframe,
        corrupt: false,
        config: false,
    };
    packet
}

/// Create a test audio packet
pub fn create_test_audio_packet(data: Vec<u8>, pts: i64) -> Packet {
    let mut packet = Packet::new_audio(0, Buffer::from_vec(data));
    packet.pts = Timestamp::new(pts);
    packet.dts = Timestamp::new(pts);
    packet
}

/// Create garbage/malformed data for error handling tests
pub fn create_garbage_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Create random-looking data for fuzzing-style tests
pub fn create_random_data(size: usize, seed: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    let mut state = seed;
    for _ in 0..size {
        // Simple PRNG for reproducible test data
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push((state >> 56) as u8);
    }
    data
}

// ============================================================================
// Frame Verification
// ============================================================================

/// Verify that a decoded video frame has valid dimensions and format
pub fn verify_video_frame_properties(
    frame: &VideoFrame,
    expected_width: u32,
    expected_height: u32,
) -> Result<(), String> {
    if frame.width != expected_width {
        return Err(format!(
            "Width mismatch: expected {}, got {}",
            expected_width, frame.width
        ));
    }

    if frame.height != expected_height {
        return Err(format!(
            "Height mismatch: expected {}, got {}",
            expected_height, frame.height
        ));
    }

    // Verify frame has data
    if frame.data.is_empty() {
        return Err("Frame has no data planes".to_string());
    }

    // Verify each plane has data
    for (i, plane) in frame.data.iter().enumerate() {
        if plane.is_empty() {
            return Err(format!("Plane {} is empty", i));
        }
    }

    Ok(())
}

/// Verify that a decoded audio frame has valid properties
pub fn verify_audio_frame_properties(
    frame: &AudioFrame,
    expected_sample_rate: u32,
    expected_channels: u16,
) -> Result<(), String> {
    if frame.sample_rate != expected_sample_rate {
        return Err(format!(
            "Sample rate mismatch: expected {}, got {}",
            expected_sample_rate, frame.sample_rate
        ));
    }

    if frame.channels != expected_channels {
        return Err(format!(
            "Channel count mismatch: expected {}, got {}",
            expected_channels, frame.channels
        ));
    }

    if frame.data.is_empty() {
        return Err("Frame has no audio data".to_string());
    }

    Ok(())
}

/// Calculate PSNR between two video frames (for quality comparison)
pub fn calculate_psnr(original: &VideoFrame, decoded: &VideoFrame) -> f64 {
    if original.width != decoded.width || original.height != decoded.height {
        return 0.0;
    }

    if original.data.is_empty() || decoded.data.is_empty() {
        return 0.0;
    }

    // Compare Y plane only for simplicity
    let orig_y = original.data[0].as_slice();
    let dec_y = decoded.data[0].as_slice();

    if orig_y.len() != dec_y.len() {
        return 0.0;
    }

    let mse: f64 = orig_y
        .iter()
        .zip(dec_y.iter())
        .map(|(a, b)| {
            let diff = *a as f64 - *b as f64;
            diff * diff
        })
        .sum::<f64>()
        / orig_y.len() as f64;

    if mse == 0.0 {
        return f64::INFINITY; // Perfect match
    }

    let max_pixel_value = 255.0;
    10.0 * ((max_pixel_value * max_pixel_value) / mse).log10()
}

// ============================================================================
// Test Constants
// ============================================================================

/// Standard test resolutions
pub mod resolutions {
    pub const QCIF: (u32, u32) = (176, 144);
    pub const CIF: (u32, u32) = (352, 288);
    pub const VGA: (u32, u32) = (640, 480);
    pub const HD720: (u32, u32) = (1280, 720);
    pub const HD1080: (u32, u32) = (1920, 1080);
    pub const UHD4K: (u32, u32) = (3840, 2160);

    /// Small resolution suitable for fast tests
    pub const TEST_SMALL: (u32, u32) = (320, 240);
    /// Medium resolution for more thorough tests
    pub const TEST_MEDIUM: (u32, u32) = (640, 480);
}

/// Standard audio parameters
pub mod audio {
    pub const SAMPLE_RATE_8000: u32 = 8000;
    pub const SAMPLE_RATE_16000: u32 = 16000;
    pub const SAMPLE_RATE_44100: u32 = 44100;
    pub const SAMPLE_RATE_48000: u32 = 48000;

    pub const CHANNELS_MONO: u16 = 1;
    pub const CHANNELS_STEREO: u16 = 2;

    /// Standard frame size for many codecs
    pub const FRAME_SIZE: usize = 1024;
}

// ============================================================================
// Test Helpers
// ============================================================================

/// Helper to create a sequence of test video frames
pub fn create_video_sequence(
    width: u32,
    height: u32,
    num_frames: usize,
) -> Vec<VideoFrame> {
    (0..num_frames)
        .map(|i| {
            let fill = (i * 10) as u8;
            create_test_video_frame_with_pattern(
                width,
                height,
                PixelFormat::YUV420P,
                i as i64,
                fill,
            )
        })
        .collect()
}

/// Helper to create a sequence of test audio frames
pub fn create_audio_sequence(
    sample_rate: u32,
    channels: u16,
    samples_per_frame: usize,
    num_frames: usize,
) -> Vec<AudioFrame> {
    (0..num_frames)
        .map(|i| {
            let frequency = 440.0 + (i as f64 * 50.0); // Varying frequency
            let mut frame = create_test_audio_frame_with_frequency(
                sample_rate,
                channels,
                samples_per_frame,
                frequency,
            );
            frame.pts = Timestamp::new((i * samples_per_frame) as i64);
            frame
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_video_frame() {
        let frame = create_test_video_frame(320, 240);
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.format, PixelFormat::YUV420P);
        assert_eq!(frame.data.len(), 3); // Y, U, V planes
    }

    #[test]
    fn test_create_color_bars() {
        let frame = generate_color_bars(320, 240);
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert!(frame.keyframe);
    }

    #[test]
    fn test_create_audio_frame() {
        let frame = create_test_audio_frame(48000, 2, 1024);
        assert_eq!(frame.sample_rate, 48000);
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.nb_samples, 1024);
    }

    #[test]
    fn test_video_frame_verification() {
        let frame = create_test_video_frame(640, 480);
        assert!(verify_video_frame_properties(&frame, 640, 480).is_ok());
        assert!(verify_video_frame_properties(&frame, 320, 240).is_err());
    }

    #[test]
    fn test_audio_frame_verification() {
        let frame = create_test_audio_frame(48000, 2, 1024);
        assert!(verify_audio_frame_properties(&frame, 48000, 2).is_ok());
        assert!(verify_audio_frame_properties(&frame, 44100, 2).is_err());
    }

    #[test]
    fn test_psnr_identical_frames() {
        let frame = create_test_video_frame(320, 240);
        let psnr = calculate_psnr(&frame, &frame);
        assert!(psnr.is_infinite()); // Perfect match
    }

    #[test]
    fn test_garbage_data() {
        let data = create_garbage_data(100);
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_random_data_reproducibility() {
        let data1 = create_random_data(100, 12345);
        let data2 = create_random_data(100, 12345);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_video_sequence() {
        let frames = create_video_sequence(320, 240, 5);
        assert_eq!(frames.len(), 5);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.pts.value, i as i64);
        }
    }

    #[test]
    fn test_audio_sequence() {
        let frames = create_audio_sequence(48000, 2, 1024, 5);
        assert_eq!(frames.len(), 5);
    }
}
