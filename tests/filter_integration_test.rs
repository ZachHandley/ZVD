//! Integration tests for video and audio filters
//!
//! These tests verify filter functionality and filter chain operations.

use zvd_lib::codec::{AudioFrame, Frame, VideoFrame};
use zvd_lib::util::{Buffer, PixelFormat, SampleFormat, Timestamp};

/// Test ScaleFilter creation and basic operation
#[test]
fn test_scale_filter_creation() {
    use zvd_lib::filter::video::ScaleFilter;

    let filter = ScaleFilter::new(1280, 720);
    assert_eq!(filter.output_width(), 1280);
    assert_eq!(filter.output_height(), 720);
}

/// Test ScaleFilter with various resolutions
#[test]
fn test_scale_filter_resolutions() {
    use zvd_lib::filter::video::ScaleFilter;

    // Test common resolutions
    let filter = ScaleFilter::new(1920, 1080);
    assert_eq!(filter.output_width(), 1920);
    assert_eq!(filter.output_height(), 1080);

    let filter = ScaleFilter::new(3840, 2160);
    assert_eq!(filter.output_width(), 3840);
    assert_eq!(filter.output_height(), 2160);

    let filter = ScaleFilter::new(640, 480);
    assert_eq!(filter.output_width(), 640);
    assert_eq!(filter.output_height(), 480);
}

/// Test CropFilter creation
#[test]
fn test_crop_filter_creation() {
    use zvd_lib::filter::video::CropFilter;

    let filter = CropFilter::new(100, 100, 640, 480);
    // Crop 640x480 region starting at (100, 100)
}

/// Test CropFilter validation
#[test]
fn test_crop_filter_validation() {
    use zvd_lib::filter::video::CropFilter;

    // Valid crop
    let filter = CropFilter::new(0, 0, 1920, 1080);

    // Crop with offset
    let filter = CropFilter::new(100, 100, 1280, 720);
}

/// Test RotateFilter creation
#[test]
fn test_rotate_filter_creation() {
    use zvd_lib::filter::video::{RotateAngle, RotateFilter};

    let filter_90 = RotateFilter::new(RotateAngle::Rotate90);
    let filter_180 = RotateFilter::new(RotateAngle::Rotate180);
    let filter_270 = RotateFilter::new(RotateAngle::Rotate270);

    // Verify filters were created
}

/// Test FlipFilter creation
#[test]
fn test_flip_filter_creation() {
    use zvd_lib::filter::video::{FlipDirection, FlipFilter};

    let filter_h = FlipFilter::new(FlipDirection::Horizontal);
    let filter_v = FlipFilter::new(FlipDirection::Vertical);

    // Verify filters were created
}

/// Test BrightnessContrastFilter creation
#[test]
fn test_brightness_contrast_filter_creation() {
    use zvd_lib::filter::video::BrightnessContrastFilter;

    // Neutral settings (no change)
    let filter = BrightnessContrastFilter::new(0.0, 1.0);

    // Increase brightness
    let filter = BrightnessContrastFilter::new(20.0, 1.0);

    // Increase contrast
    let filter = BrightnessContrastFilter::new(0.0, 1.5);

    // Combined
    let filter = BrightnessContrastFilter::new(10.0, 1.2);
}

/// Test BrightnessContrastFilter validation
#[test]
fn test_brightness_contrast_filter_validation() {
    use zvd_lib::filter::video::BrightnessContrastFilter;

    // Valid ranges
    let filter = BrightnessContrastFilter::new(-100.0, 0.5);
    let filter = BrightnessContrastFilter::new(100.0, 2.0);

    // Extreme values (should still create filter)
    let filter = BrightnessContrastFilter::new(0.0, 0.1);
    let filter = BrightnessContrastFilter::new(0.0, 5.0);
}

/// Test VolumeFilter creation
#[test]
fn test_volume_filter_creation() {
    use zvd_lib::filter::audio::VolumeFilter;

    // Unity gain (no change)
    let filter = VolumeFilter::new(1.0);

    // Increase volume (2x)
    let filter = VolumeFilter::new(2.0);

    // Decrease volume (50%)
    let filter = VolumeFilter::new(0.5);

    // Mute
    let filter = VolumeFilter::new(0.0);
}

/// Test VolumeFilter validation
#[test]
fn test_volume_filter_validation() {
    use zvd_lib::filter::audio::VolumeFilter;

    // Valid gains
    let filter = VolumeFilter::new(0.0);
    let filter = VolumeFilter::new(1.0);
    let filter = VolumeFilter::new(10.0);

    // Very low gain
    let filter = VolumeFilter::new(0.001);

    // High gain
    let filter = VolumeFilter::new(100.0);
}

/// Test ResampleFilter creation
#[test]
fn test_resample_filter_creation() {
    use zvd_lib::filter::audio::ResampleFilter;

    // Common sample rate conversions
    let filter = ResampleFilter::new(48000, 44100);
    let filter = ResampleFilter::new(44100, 48000);
    let filter = ResampleFilter::new(96000, 48000);
    let filter = ResampleFilter::new(48000, 16000);
}

/// Test ResampleFilter with various sample rates
#[test]
fn test_resample_filter_sample_rates() {
    use zvd_lib::filter::audio::ResampleFilter;

    // Standard sample rates
    let rates = [
        8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000,
    ];

    for &from_rate in &rates {
        for &to_rate in &rates {
            let filter = ResampleFilter::new(from_rate, to_rate);
            // Filter should be created successfully
        }
    }
}

/// Test NormalizeFilter creation
#[test]
fn test_normalize_filter_creation() {
    use zvd_lib::filter::audio::NormalizeFilter;

    // Standard normalization levels
    let filter = NormalizeFilter::new(-20.0); // -20 dB RMS
    let filter = NormalizeFilter::new(-16.0); // -16 dB RMS
    let filter = NormalizeFilter::new(-12.0); // -12 dB RMS
    let filter = NormalizeFilter::new(-3.0); // -3 dB RMS
}

/// Test NormalizeFilter validation
#[test]
fn test_normalize_filter_validation() {
    use zvd_lib::filter::audio::NormalizeFilter;

    // Valid levels
    let filter = NormalizeFilter::new(-30.0);
    let filter = NormalizeFilter::new(-0.1);

    // Reasonable range
    let filter = NormalizeFilter::new(-60.0);
    let filter = NormalizeFilter::new(0.0);
}

/// Test VideoFilterChain creation
#[test]
fn test_video_filter_chain_creation() {
    use zvd_lib::filter::chain::VideoFilterChain;

    let mut chain = VideoFilterChain::new();
    assert_eq!(chain.len(), 0);
}

/// Test VideoFilterChain with multiple filters
#[test]
fn test_video_filter_chain_multiple() {
    use zvd_lib::filter::chain::VideoFilterChain;
    use zvd_lib::filter::video::{BrightnessContrastFilter, ScaleFilter};

    let mut chain = VideoFilterChain::new();

    // Add scale filter
    chain.add(Box::new(ScaleFilter::new(1280, 720)));

    // Add brightness/contrast
    chain.add(Box::new(BrightnessContrastFilter::new(10.0, 1.2)));

    assert_eq!(chain.len(), 2);
}

/// Test VideoFilterChain complex pipeline
#[test]
fn test_video_filter_chain_complex() {
    use zvd_lib::filter::chain::VideoFilterChain;
    use zvd_lib::filter::video::{
        BrightnessContrastFilter, CropFilter, FlipDirection, FlipFilter, RotateAngle, RotateFilter,
        ScaleFilter,
    };

    let mut chain = VideoFilterChain::new();

    // Build complex pipeline
    chain.add(Box::new(ScaleFilter::new(1920, 1080)));
    chain.add(Box::new(CropFilter::new(0, 0, 1920, 1000)));
    chain.add(Box::new(RotateFilter::new(RotateAngle::Rotate90)));
    chain.add(Box::new(FlipFilter::new(FlipDirection::Horizontal)));
    chain.add(Box::new(BrightnessContrastFilter::new(15.0, 1.3)));

    assert_eq!(chain.len(), 5);
}

/// Test AudioFilterChain creation
#[test]
fn test_audio_filter_chain_creation() {
    use zvd_lib::filter::chain::AudioFilterChain;

    let mut chain = AudioFilterChain::new();
    assert_eq!(chain.len(), 0);
}

/// Test AudioFilterChain with multiple filters
#[test]
fn test_audio_filter_chain_multiple() {
    use zvd_lib::filter::audio::{NormalizeFilter, VolumeFilter};
    use zvd_lib::filter::chain::AudioFilterChain;

    let mut chain = AudioFilterChain::new();

    // Add volume filter
    chain.add(Box::new(VolumeFilter::new(1.5)));

    // Add normalization
    chain.add(Box::new(NormalizeFilter::new(-16.0)));

    assert_eq!(chain.len(), 2);
}

/// Test AudioFilterChain complex pipeline
#[test]
fn test_audio_filter_chain_complex() {
    use zvd_lib::filter::audio::{NormalizeFilter, ResampleFilter, VolumeFilter};
    use zvd_lib::filter::chain::AudioFilterChain;

    let mut chain = AudioFilterChain::new();

    // Build complex pipeline
    chain.add(Box::new(ResampleFilter::new(48000, 44100)));
    chain.add(Box::new(VolumeFilter::new(1.2)));
    chain.add(Box::new(NormalizeFilter::new(-18.0)));

    assert_eq!(chain.len(), 3);
}

/// Test filter chain clearing
#[test]
fn test_filter_chain_clear() {
    use zvd_lib::filter::chain::VideoFilterChain;
    use zvd_lib::filter::video::ScaleFilter;

    let mut chain = VideoFilterChain::new();
    chain.add(Box::new(ScaleFilter::new(1280, 720)));
    chain.add(Box::new(ScaleFilter::new(640, 480)));
    assert_eq!(chain.len(), 2);

    chain.clear();
    assert_eq!(chain.len(), 0);
}

/// Test ScaleFilter with frame processing
#[test]
fn test_scale_filter_process() {
    use zvd_lib::filter::video::ScaleFilter;
    use zvd_lib::filter::VideoFilter;

    let mut filter = ScaleFilter::new(640, 480);

    // Create input frame (1920x1080)
    let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![128u8; 1920 * 1080]));
    frame.linesize.push(1920);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.pts = Timestamp::new(0);

    // Process frame
    let result = filter.process(&Frame::Video(frame));
    assert!(result.is_ok(), "Scale filter should process frame");

    if let Ok(Frame::Video(output)) = result {
        assert_eq!(output.width, 640);
        assert_eq!(output.height, 480);
    } else {
        panic!("Expected video frame output");
    }
}

/// Test BrightnessContrastFilter with frame processing
#[test]
fn test_brightness_contrast_filter_process() {
    use zvd_lib::filter::video::BrightnessContrastFilter;
    use zvd_lib::filter::VideoFilter;

    let mut filter = BrightnessContrastFilter::new(20.0, 1.2);

    // Create input frame
    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
    frame.linesize.push(640);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.pts = Timestamp::new(0);

    // Process frame
    let result = filter.process(&Frame::Video(frame));
    assert!(
        result.is_ok(),
        "Brightness/contrast filter should process frame"
    );

    if let Ok(Frame::Video(output)) = result {
        assert_eq!(output.width, 640);
        assert_eq!(output.height, 480);
        // Brightness should be modified
    } else {
        panic!("Expected video frame output");
    }
}

/// Test VolumeFilter with frame processing
#[test]
fn test_volume_filter_process() {
    use zvd_lib::filter::audio::VolumeFilter;
    use zvd_lib::filter::AudioFilter;

    let mut filter = VolumeFilter::new(1.5);

    // Create input frame
    let mut frame = AudioFrame::new(1024, 2, SampleFormat::F32);
    let samples = vec![0.5f32; 1024 * 2];
    frame.data.push(Buffer::from_vec(
        samples
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    ));
    frame.pts = Timestamp::new(0);

    // Process frame
    let result = filter.process(&Frame::Audio(frame));
    assert!(result.is_ok(), "Volume filter should process frame");

    if let Ok(Frame::Audio(output)) = result {
        assert_eq!(output.nb_samples, 1024);
        assert_eq!(output.channels, 2);
        // Volume should be amplified
    } else {
        panic!("Expected audio frame output");
    }
}

/// Test filter chain processing
#[test]
fn test_video_filter_chain_processing() {
    use zvd_lib::filter::chain::VideoFilterChain;
    use zvd_lib::filter::video::{BrightnessContrastFilter, ScaleFilter};
    use zvd_lib::filter::FilterChain;

    let mut chain = VideoFilterChain::new();
    chain.add(Box::new(ScaleFilter::new(640, 480)));
    chain.add(Box::new(BrightnessContrastFilter::new(10.0, 1.1)));

    // Create input frame
    let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 1920 * 1080]));
    frame.linesize.push(1920);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.pts = Timestamp::new(0);

    // Process through chain
    let result = chain.process(Frame::Video(frame));
    assert!(result.is_ok(), "Filter chain should process frame");

    if let Ok(Frame::Video(output)) = result {
        assert_eq!(output.width, 640);
        assert_eq!(output.height, 480);
    } else {
        panic!("Expected video frame output");
    }
}

/// Test audio filter chain processing
#[test]
fn test_audio_filter_chain_processing() {
    use zvd_lib::filter::audio::VolumeFilter;
    use zvd_lib::filter::chain::AudioFilterChain;
    use zvd_lib::filter::FilterChain;

    let mut chain = AudioFilterChain::new();
    chain.add(Box::new(VolumeFilter::new(1.2)));

    // Create input frame
    let mut frame = AudioFrame::new(1024, 2, SampleFormat::F32);
    let samples = vec![0.5f32; 1024 * 2];
    frame.data.push(Buffer::from_vec(
        samples
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    ));
    frame.pts = Timestamp::new(0);

    // Process through chain
    let result = chain.process(Frame::Audio(frame));
    assert!(result.is_ok(), "Audio filter chain should process frame");

    if let Ok(Frame::Audio(output)) = result {
        assert_eq!(output.nb_samples, 1024);
        assert_eq!(output.channels, 2);
    } else {
        panic!("Expected audio frame output");
    }
}

/// Test empty filter chain (passthrough)
#[test]
fn test_empty_filter_chain() {
    use zvd_lib::filter::chain::VideoFilterChain;
    use zvd_lib::filter::FilterChain;

    let chain = VideoFilterChain::new();

    // Create input frame
    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
    frame.linesize.push(640);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.pts = Timestamp::new(42);

    // Process through empty chain (should be passthrough)
    let result = chain.process(Frame::Video(frame));
    assert!(result.is_ok(), "Empty filter chain should passthrough");

    if let Ok(Frame::Video(output)) = result {
        assert_eq!(output.width, 640);
        assert_eq!(output.height, 480);
        assert_eq!(output.pts.value(), 42);
    } else {
        panic!("Expected video frame output");
    }
}
