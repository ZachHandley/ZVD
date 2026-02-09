//! Hardware acceleration integration tests
//!
//! Tests hardware-accelerated encoding and decoding functionality.
//! These tests are designed to work with or without actual hardware.

use zvd_lib::codec::{create_encoder_with_hw, list_hw_encoders, create_decoder_with_hw, list_hw_decoders};
use zvd_lib::codec::{HwEncoderPreference, HwDecoderPreference};
use zvd_lib::hwaccel::{
    detect_hw_devices, get_hw_device_info, select_best_hw_encoder, select_best_hw_decoder,
    HwAccelContext, HwAccelType, HwCodecType, HwEncoderConfig, HwDecoderConfig,
    HwPixelFormat, HwProfile, HwRateControlMode, HwSurface, HwSurfacePool,
    create_hw_encoder, create_hw_decoder,
};

#[test]
fn test_detect_hardware_devices() {
    let devices = detect_hw_devices();
    println!("Available hardware acceleration devices:");
    for device in &devices {
        println!("  - {:?}", device);
    }
    // Test should not panic, devices may be empty on systems without GPU
}

#[test]
fn test_get_device_info() {
    let infos = get_hw_device_info();
    println!("Hardware device information:");
    for info in &infos {
        println!(
            "  {} ({:?}):",
            info.name, info.device_type
        );
        println!("    Vendor: {}", info.vendor);
        println!("    Encode codecs: {:?}", info.encode_codecs);
        println!("    Decode codecs: {:?}", info.decode_codecs);
        println!(
            "    Max resolution: {}x{}",
            info.max_width, info.max_height
        );
    }
}

#[test]
fn test_hw_accel_type_properties() {
    // Test encoding support
    assert!(HwAccelType::NVENC.supports_encoding());
    assert!(HwAccelType::VAAPI.supports_encoding());
    assert!(HwAccelType::QSV.supports_encoding());
    assert!(HwAccelType::VideoToolbox.supports_encoding());
    assert!(!HwAccelType::NVDEC.supports_encoding());
    assert!(!HwAccelType::None.supports_encoding());

    // Test decoding support
    assert!(HwAccelType::NVDEC.supports_decoding());
    assert!(HwAccelType::VAAPI.supports_decoding());
    assert!(HwAccelType::QSV.supports_decoding());
    assert!(HwAccelType::VideoToolbox.supports_decoding());
    assert!(!HwAccelType::NVENC.supports_decoding());
    assert!(!HwAccelType::None.supports_decoding());

    // Test display names
    assert_eq!(HwAccelType::NVENC.name(), "NVIDIA NVENC");
    assert_eq!(HwAccelType::VAAPI.name(), "VA-API");
    assert_eq!(HwAccelType::QSV.name(), "Intel Quick Sync");
    assert_eq!(HwAccelType::VideoToolbox.name(), "Apple VideoToolbox");
}

#[test]
fn test_select_best_encoder() {
    let best_h264 = select_best_hw_encoder(HwCodecType::H264);
    let best_h265 = select_best_hw_encoder(HwCodecType::H265);
    let best_av1 = select_best_hw_encoder(HwCodecType::AV1);
    let best_vp9 = select_best_hw_encoder(HwCodecType::VP9);

    println!("Best encoders:");
    println!("  H.264: {:?}", best_h264);
    println!("  H.265: {:?}", best_h265);
    println!("  AV1:   {:?}", best_av1);
    println!("  VP9:   {:?}", best_vp9);
}

#[test]
fn test_select_best_decoder() {
    let best_h264 = select_best_hw_decoder(HwCodecType::H264);
    let best_h265 = select_best_hw_decoder(HwCodecType::H265);
    let best_av1 = select_best_hw_decoder(HwCodecType::AV1);
    let best_vp9 = select_best_hw_decoder(HwCodecType::VP9);

    println!("Best decoders:");
    println!("  H.264: {:?}", best_h264);
    println!("  H.265: {:?}", best_h265);
    println!("  AV1:   {:?}", best_av1);
    println!("  VP9:   {:?}", best_vp9);
}

#[test]
fn test_encoder_config_defaults() {
    let config = HwEncoderConfig::default();
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.framerate_num, 30);
    assert_eq!(config.framerate_den, 1);
    assert_eq!(config.bitrate, 5_000_000);
    assert_eq!(config.qp, 23);
    assert!(config.validate().is_ok());
}

#[test]
fn test_encoder_config_streaming() {
    let config = HwEncoderConfig::streaming(1280, 720, 2_500_000);
    assert_eq!(config.width, 1280);
    assert_eq!(config.height, 720);
    assert_eq!(config.bitrate, 2_500_000);
    assert_eq!(config.b_frames, 0); // No B-frames for low latency
    assert!(config.low_latency);
    assert!(config.validate().is_ok());
}

#[test]
fn test_encoder_config_archival() {
    let config = HwEncoderConfig::archival(3840, 2160);
    assert_eq!(config.width, 3840);
    assert_eq!(config.height, 2160);
    assert_eq!(config.preset, 7); // Highest quality
    assert!(config.temporal_aq);
    assert!(config.spatial_aq);
    assert!(config.validate().is_ok());
}

#[test]
fn test_encoder_config_validation() {
    // Test invalid width
    let mut config = HwEncoderConfig::default();
    config.width = 0;
    assert!(config.validate().is_err());

    // Test odd dimensions
    config.width = 1921;
    config.height = 1080;
    assert!(config.validate().is_err());

    // Test invalid QP
    config.width = 1920;
    config.qp = 60;
    assert!(config.validate().is_err());

    // Test invalid preset
    config.qp = 23;
    config.preset = 10;
    assert!(config.validate().is_err());
}

#[test]
fn test_decoder_config_defaults() {
    let config = HwDecoderConfig::default();
    assert_eq!(config.max_width, 4096);
    assert_eq!(config.max_height, 2160);
    assert_eq!(config.num_surfaces, 8);
}

#[test]
fn test_hw_surface_creation() {
    let surface = HwSurface::new(1920, 1080, HwPixelFormat::NV12);
    assert_eq!(surface.width, 1920);
    assert_eq!(surface.height, 1080);
    assert_eq!(surface.format, HwPixelFormat::NV12);
    assert_eq!(surface.strides.len(), 2); // NV12 has 2 planes
}

#[test]
fn test_hw_surface_pool() {
    let mut pool = HwSurfacePool::new(1920, 1080, HwPixelFormat::NV12, 8);
    pool.preallocate(4);
    assert_eq!(pool.available_count(), 4);

    let surface = pool.acquire();
    assert_eq!(surface.width, 1920);
    assert_eq!(pool.available_count(), 3);

    pool.release(surface);
    assert_eq!(pool.available_count(), 4);
}

#[test]
fn test_hw_pixel_format_properties() {
    // Test bytes per pixel
    assert_eq!(HwPixelFormat::NV12.bytes_per_pixel(), 1.5);
    assert_eq!(HwPixelFormat::P010.bytes_per_pixel(), 3.0);
    assert_eq!(HwPixelFormat::RGBA.bytes_per_pixel(), 4.0);

    // Test plane count
    assert_eq!(HwPixelFormat::NV12.num_planes(), 2);
    assert_eq!(HwPixelFormat::YUV420P.num_planes(), 3);
    assert_eq!(HwPixelFormat::BGRA.num_planes(), 1);

    // Test HDR capability
    assert!(HwPixelFormat::P010.is_hdr_capable());
    assert!(HwPixelFormat::Y410.is_hdr_capable());
    assert!(!HwPixelFormat::NV12.is_hdr_capable());
}

#[test]
fn test_hw_context_none() {
    let ctx = HwAccelContext::new(HwAccelType::None);
    assert_eq!(ctx.device_type(), HwAccelType::None);
    assert!(!ctx.is_initialized());
    assert!(!ctx.is_available());
}

#[test]
fn test_create_encoder_software_fallback() {
    // This should fall back to software encoding on systems without GPU
    let result = create_encoder_with_hw("av1", 1920, 1080, HwEncoderPreference::PreferHw);
    assert!(result.is_ok(), "Should succeed with software fallback");
}

#[test]
fn test_create_encoder_software_only() {
    let result = create_encoder_with_hw("av1", 1920, 1080, HwEncoderPreference::SoftwareOnly);
    assert!(result.is_ok(), "Should succeed with software encoder");
}

#[test]
fn test_list_hw_encoders_info() {
    let encoders = list_hw_encoders();
    println!("Available hardware encoders:");
    for (device, codec) in &encoders {
        println!("  {} -> {}", device, codec);
    }
}

#[test]
fn test_list_hw_decoders_info() {
    let decoders = list_hw_decoders();
    println!("Available hardware decoders:");
    for (device, codec) in &decoders {
        println!("  {} -> {}", device, codec);
    }
}

#[test]
fn test_hw_codec_type_conversion() {
    assert_eq!(HwCodecType::from_str("h264"), Some(HwCodecType::H264));
    assert_eq!(HwCodecType::from_str("avc"), Some(HwCodecType::H264));
    assert_eq!(HwCodecType::from_str("hevc"), Some(HwCodecType::H265));
    assert_eq!(HwCodecType::from_str("h.265"), Some(HwCodecType::H265));
    assert_eq!(HwCodecType::from_str("vp9"), Some(HwCodecType::VP9));
    assert_eq!(HwCodecType::from_str("av1"), Some(HwCodecType::AV1));
    assert_eq!(HwCodecType::from_str("unknown"), None);
}

#[test]
fn test_hw_codec_type_name() {
    assert_eq!(HwCodecType::H264.codec_name(), "h264");
    assert_eq!(HwCodecType::H265.codec_name(), "hevc");
    assert_eq!(HwCodecType::VP9.codec_name(), "vp9");
    assert_eq!(HwCodecType::AV1.codec_name(), "av1");
}

#[test]
fn test_surface_yuv420p_to_nv12_conversion() {
    let mut surface = HwSurface::new(4, 4, HwPixelFormat::YUV420P);
    surface.allocate_data();

    // Fill with test pattern
    if let Some(data) = &mut surface.data {
        // Y plane: 16 bytes
        for i in 0..16 {
            data[i] = i as u8;
        }
        // U plane: 4 bytes
        for i in 0..4 {
            data[16 + i] = 100 + i as u8;
        }
        // V plane: 4 bytes
        for i in 0..4 {
            data[20 + i] = 200 + i as u8;
        }
    }

    // Convert to NV12
    surface.convert_yuv420p_to_nv12().unwrap();

    assert_eq!(surface.format, HwPixelFormat::NV12);
    if let Some(data) = &surface.data {
        // Y plane should be unchanged
        for i in 0..16 {
            assert_eq!(data[i], i as u8);
        }
        // UV should be interleaved
        assert_eq!(data[16], 100); // U[0]
        assert_eq!(data[17], 200); // V[0]
        assert_eq!(data[18], 101); // U[1]
        assert_eq!(data[19], 201); // V[1]
    }
}

#[test]
fn test_surface_nv12_to_yuv420p_conversion() {
    let mut surface = HwSurface::new(4, 4, HwPixelFormat::NV12);
    surface.allocate_data();

    // Fill with test pattern
    if let Some(data) = &mut surface.data {
        // Y plane: 16 bytes
        for i in 0..16 {
            data[i] = i as u8;
        }
        // Interleaved UV: 8 bytes
        for i in 0..4 {
            data[16 + i * 2] = 100 + i as u8; // U
            data[16 + i * 2 + 1] = 200 + i as u8; // V
        }
    }

    // Convert to YUV420P
    surface.convert_nv12_to_yuv420p().unwrap();

    assert_eq!(surface.format, HwPixelFormat::YUV420P);
    if let Some(data) = &surface.data {
        // Y plane should be unchanged
        for i in 0..16 {
            assert_eq!(data[i], i as u8);
        }
        // U plane
        for i in 0..4 {
            assert_eq!(data[16 + i], 100 + i as u8);
        }
        // V plane
        for i in 0..4 {
            assert_eq!(data[20 + i], 200 + i as u8);
        }
    }
}

#[test]
fn test_hw_profile_default() {
    assert_eq!(HwProfile::default(), HwProfile::Auto);
}

#[test]
fn test_hw_rate_control_default() {
    assert_eq!(HwRateControlMode::default(), HwRateControlMode::VBR);
}

#[test]
fn test_hw_encoder_preference_default() {
    assert_eq!(HwEncoderPreference::default(), HwEncoderPreference::Auto);
}

#[test]
fn test_hw_decoder_preference_default() {
    assert_eq!(HwDecoderPreference::default(), HwDecoderPreference::Auto);
}

#[cfg(feature = "vaapi")]
mod vaapi_tests {
    use super::*;
    use zvd_lib::hwaccel::vaapi::{VaapiDevice, is_available, list_devices};

    #[test]
    fn test_vaapi_availability() {
        let available = is_available();
        println!("VAAPI available: {}", available);
    }

    #[test]
    fn test_vaapi_device_creation() {
        let device = VaapiDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_vaapi_list_devices() {
        let devices = list_devices();
        println!("VAAPI devices: {:?}", devices);
    }
}

#[cfg(feature = "qsv")]
mod qsv_tests {
    use super::*;
    use zvd_lib::hwaccel::qsv::{QsvDevice, is_available, list_devices};

    #[test]
    fn test_qsv_availability() {
        let available = is_available();
        println!("QSV available: {}", available);
    }

    #[test]
    fn test_qsv_device_creation() {
        let device = QsvDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_qsv_list_devices() {
        let devices = list_devices();
        println!("QSV devices: {:?}", devices);
    }
}

#[cfg(feature = "videotoolbox")]
mod videotoolbox_tests {
    use super::*;
    use zvd_lib::hwaccel::videotoolbox::{VideoToolboxDevice, is_available, get_version_info, list_supported_codecs};

    #[test]
    fn test_videotoolbox_availability() {
        let available = is_available();
        println!("VideoToolbox available: {}", available);

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert!(available, "VideoToolbox should be available on macOS/iOS");
    }

    #[test]
    fn test_videotoolbox_device_creation() {
        let device = VideoToolboxDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_videotoolbox_version() {
        let version = get_version_info();
        println!("VideoToolbox version: {}", version);
    }

    #[test]
    fn test_videotoolbox_supported_codecs() {
        let codecs = list_supported_codecs();
        println!("VideoToolbox supported codecs: {:?}", codecs);

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            assert!(!codecs.is_empty(), "Should support at least one codec");
            assert!(codecs.contains(&HwCodecType::H264), "Should support H.264");
        }
    }
}
