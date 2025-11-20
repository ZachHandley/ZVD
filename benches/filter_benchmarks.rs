//! Filter performance benchmarks
//!
//! Benchmarks for video and audio filter processing

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zvd_lib::codec::{AudioFrame, Frame, VideoFrame};
use zvd_lib::filter::audio::{NormalizeFilter, ResampleFilter, VolumeFilter};
use zvd_lib::filter::chain::{AudioFilterChain, VideoFilterChain};
use zvd_lib::filter::video::{
    BrightnessContrastFilter, CropFilter, FlipDirection, FlipFilter, RotateAngle, RotateFilter,
    ScaleFilter,
};
use zvd_lib::filter::{AudioFilter, FilterChain, VideoFilter};
use zvd_lib::util::{Buffer, PixelFormat, SampleFormat, Timestamp};

/// Create a test video frame
fn create_test_video_frame(width: u32, height: u32) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);

    // Y plane
    frame
        .data
        .push(Buffer::from_vec(vec![100u8; (width * height) as usize]));
    frame.linesize.push(width as usize);

    // U plane
    let uv_size = ((width / 2) * (height / 2)) as usize;
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push((width / 2) as usize);

    // V plane
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push((width / 2) as usize);

    frame.pts = Timestamp::new(0);
    frame
}

/// Create a test audio frame
fn create_test_audio_frame(samples: usize, channels: u16) -> AudioFrame {
    let mut frame = AudioFrame::new(samples, channels, SampleFormat::F32);

    let total_samples = samples * channels as usize;
    let sample_data: Vec<u8> = (0..total_samples)
        .flat_map(|i| {
            let sample = ((i as f32 * 0.01).sin() * 0.5) as f32;
            sample.to_le_bytes()
        })
        .collect();

    frame.data.push(Buffer::from_vec(sample_data));
    frame.pts = Timestamp::new(0);

    frame
}

/// Benchmark ScaleFilter at various resolutions
fn bench_scale_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_filter");

    // Test scaling from 1080p to various resolutions
    let source_width = 1920u32;
    let source_height = 1080u32;

    for &(target_width, target_height) in &[(1280, 720), (640, 480), (3840, 2160)] {
        let pixels = target_width * target_height;
        group.throughput(Throughput::Elements(pixels as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "{}x{}_to_{}x{}",
                source_width, source_height, target_width, target_height
            )),
            &(target_width, target_height),
            |b, &(tw, th)| {
                let mut filter = ScaleFilter::new(tw, th);
                let frame = create_test_video_frame(source_width, source_height);

                b.iter(|| {
                    let _ = filter.process(black_box(&Frame::Video(frame.clone())));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CropFilter
fn bench_crop_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("crop_filter");

    let width = 1920u32;
    let height = 1080u32;
    let pixels = (1280 * 720) as u64;

    group.throughput(Throughput::Elements(pixels));

    group.bench_function("1080p_to_720p", |b| {
        let mut filter = CropFilter::new(320, 180, 1280, 720);
        let frame = create_test_video_frame(width, height);

        b.iter(|| {
            let _ = filter.process(black_box(&Frame::Video(frame.clone())));
        });
    });

    group.finish();
}

/// Benchmark RotateFilter
fn bench_rotate_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotate_filter");

    let width = 1920u32;
    let height = 1080u32;
    let pixels = (width * height) as u64;

    group.throughput(Throughput::Elements(pixels));

    for angle in &[
        RotateAngle::Rotate90,
        RotateAngle::Rotate180,
        RotateAngle::Rotate270,
    ] {
        let angle_name = match angle {
            RotateAngle::Rotate90 => "90deg",
            RotateAngle::Rotate180 => "180deg",
            RotateAngle::Rotate270 => "270deg",
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(angle_name),
            angle,
            |b, &angle| {
                let mut filter = RotateFilter::new(angle);
                let frame = create_test_video_frame(width, height);

                b.iter(|| {
                    let _ = filter.process(black_box(&Frame::Video(frame.clone())));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FlipFilter
fn bench_flip_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip_filter");

    let width = 1920u32;
    let height = 1080u32;
    let pixels = (width * height) as u64;

    group.throughput(Throughput::Elements(pixels));

    for direction in &[FlipDirection::Horizontal, FlipDirection::Vertical] {
        let dir_name = match direction {
            FlipDirection::Horizontal => "horizontal",
            FlipDirection::Vertical => "vertical",
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(dir_name),
            direction,
            |b, &direction| {
                let mut filter = FlipFilter::new(direction);
                let frame = create_test_video_frame(width, height);

                b.iter(|| {
                    let _ = filter.process(black_box(&Frame::Video(frame.clone())));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark BrightnessContrastFilter
fn bench_brightness_contrast_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("brightness_contrast_filter");

    let width = 1920u32;
    let height = 1080u32;
    let pixels = (width * height) as u64;

    group.throughput(Throughput::Elements(pixels));

    group.bench_function("1080p", |b| {
        let mut filter = BrightnessContrastFilter::new(15.0, 1.2);
        let frame = create_test_video_frame(width, height);

        b.iter(|| {
            let _ = filter.process(black_box(&Frame::Video(frame.clone())));
        });
    });

    group.finish();
}

/// Benchmark VolumeFilter
fn bench_volume_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_filter");

    let samples = 1024;
    let channels = 2;

    group.throughput(Throughput::Elements((samples * channels) as u64));

    group.bench_function("stereo_1024_samples", |b| {
        let mut filter = VolumeFilter::new(1.5);
        let frame = create_test_audio_frame(samples, channels);

        b.iter(|| {
            let _ = filter.process(black_box(&Frame::Audio(frame.clone())));
        });
    });

    group.finish();
}

/// Benchmark ResampleFilter
fn bench_resample_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_filter");

    let samples = 1024;
    let channels = 2;

    group.throughput(Throughput::Elements((samples * channels) as u64));

    for &(from_rate, to_rate) in &[(48000, 44100), (44100, 48000), (96000, 48000)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}Hz_to_{}Hz", from_rate, to_rate)),
            &(from_rate, to_rate),
            |b, &(from, to)| {
                let mut filter = ResampleFilter::new(from, to);
                let frame = create_test_audio_frame(samples, channels);

                b.iter(|| {
                    let _ = filter.process(black_box(&Frame::Audio(frame.clone())));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark NormalizeFilter
fn bench_normalize_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_filter");

    let samples = 1024;
    let channels = 2;

    group.throughput(Throughput::Elements((samples * channels) as u64));

    group.bench_function("stereo_1024_samples", |b| {
        let mut filter = NormalizeFilter::new(-16.0);
        let frame = create_test_audio_frame(samples, channels);

        b.iter(|| {
            let _ = filter.process(black_box(&Frame::Audio(frame.clone())));
        });
    });

    group.finish();
}

/// Benchmark VideoFilterChain with multiple filters
fn bench_video_filter_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_filter_chain");

    let width = 1920u32;
    let height = 1080u32;

    group.throughput(Throughput::Elements((1280 * 720) as u64));

    group.bench_function("scale_brightness_3_filters", |b| {
        let mut chain = VideoFilterChain::new();
        chain.add(Box::new(ScaleFilter::new(1280, 720)));
        chain.add(Box::new(BrightnessContrastFilter::new(10.0, 1.1)));
        chain.add(Box::new(FlipFilter::new(FlipDirection::Horizontal)));

        let frame = create_test_video_frame(width, height);

        b.iter(|| {
            let _ = chain.process(black_box(Frame::Video(frame.clone())));
        });
    });

    group.finish();
}

/// Benchmark AudioFilterChain with multiple filters
fn bench_audio_filter_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_filter_chain");

    let samples = 1024;
    let channels = 2;

    group.throughput(Throughput::Elements((samples * channels) as u64));

    group.bench_function("resample_volume_normalize", |b| {
        let mut chain = AudioFilterChain::new();
        chain.add(Box::new(ResampleFilter::new(48000, 44100)));
        chain.add(Box::new(VolumeFilter::new(1.2)));
        chain.add(Box::new(NormalizeFilter::new(-18.0)));

        let frame = create_test_audio_frame(samples, channels);

        b.iter(|| {
            let _ = chain.process(black_box(Frame::Audio(frame.clone())));
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_scale_filter,
        bench_crop_filter,
        bench_rotate_filter,
        bench_flip_filter,
        bench_brightness_contrast_filter,
        bench_volume_filter,
        bench_resample_filter,
        bench_normalize_filter,
        bench_video_filter_chain,
        bench_audio_filter_chain,
}

criterion_main!(benches);
