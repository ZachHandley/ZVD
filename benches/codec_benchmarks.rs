//! Codec performance benchmarks
//!
//! Benchmarks for video and audio codec encode/decode performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use zvd_lib::codec::{create_encoder, create_decoder, Encoder, Decoder, Frame, VideoFrame, AudioFrame};
use zvd_lib::util::{Buffer, PixelFormat, SampleFormat, Timestamp};

/// Create a test video frame with the specified dimensions
fn create_test_video_frame(width: u32, height: u32) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);

    // Y plane
    frame.data.push(Buffer::from_vec(vec![100u8; (width * height) as usize]));
    frame.linesize.push(width as usize);

    // U plane (half size)
    let uv_size = ((width / 2) * (height / 2)) as usize;
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push((width / 2) as usize);

    // V plane (half size)
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push((width / 2) as usize);

    frame.pts = Timestamp::new(0);
    frame.keyframe = true;

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

/// Benchmark AV1 encoding at various resolutions
fn bench_av1_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_encode");

    for &(width, height) in &[(640, 480), (1280, 720), (1920, 1080)] {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    let mut encoder = create_encoder("av1", w, h)
                        .expect("Failed to create encoder");
                    let frame = create_test_video_frame(w, h);

                    encoder.send_frame(&Frame::Video(frame))
                        .expect("Failed to encode");
                    encoder.flush().expect("Failed to flush");

                    // Try to receive packet
                    let _ = encoder.receive_packet();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark AV1 decoding
fn bench_av1_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("av1_decode");

    let width = 1280u32;
    let height = 720u32;
    let pixels = width * height;

    // Pre-encode a frame
    let mut encoder = create_encoder("av1", width, height)
        .expect("Failed to create encoder");
    let frame = create_test_video_frame(width, height);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");
    let packet = encoder.receive_packet().expect("Failed to receive packet");

    group.throughput(Throughput::Elements(pixels as u64));

    group.bench_function("720p", |b| {
        b.iter(|| {
            let mut decoder = create_decoder("av1")
                .expect("Failed to create decoder");

            decoder.send_packet(black_box(&packet))
                .expect("Failed to decode");

            let _ = decoder.receive_frame();
        });
    });

    group.finish();
}

/// Benchmark H.264 encoding
#[cfg(feature = "h264")]
fn bench_h264_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_encode");

    for &(width, height) in &[(640, 480), (1280, 720), (1920, 1080)] {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    let mut encoder = create_encoder("h264", w, h)
                        .expect("Failed to create encoder");
                    let frame = create_test_video_frame(w, h);

                    encoder.send_frame(&Frame::Video(frame))
                        .expect("Failed to encode");
                    encoder.flush().expect("Failed to flush");

                    let _ = encoder.receive_packet();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark H.264 decoding
#[cfg(feature = "h264")]
fn bench_h264_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_decode");

    let width = 1280u32;
    let height = 720u32;
    let pixels = width * height;

    // Pre-encode a frame
    let mut encoder = create_encoder("h264", width, height)
        .expect("Failed to create encoder");
    let frame = create_test_video_frame(width, height);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");
    let packet = encoder.receive_packet().expect("Failed to receive packet");

    group.throughput(Throughput::Elements(pixels as u64));

    group.bench_function("720p", |b| {
        b.iter(|| {
            let mut decoder = create_decoder("h264")
                .expect("Failed to create decoder");

            decoder.send_packet(black_box(&packet))
                .expect("Failed to decode");

            let _ = decoder.receive_frame();
        });
    });

    group.finish();
}

/// Benchmark VP8 encoding
#[cfg(feature = "vp8-codec")]
fn bench_vp8_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("vp8_encode");

    for &(width, height) in &[(640, 480), (1280, 720)] {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    let mut encoder = create_encoder("vp8", w, h)
                        .expect("Failed to create encoder");
                    let frame = create_test_video_frame(w, h);

                    encoder.send_frame(&Frame::Video(frame))
                        .expect("Failed to encode");
                    encoder.flush().expect("Failed to flush");

                    let _ = encoder.receive_packet();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark VP9 encoding
#[cfg(feature = "vp9-codec")]
fn bench_vp9_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("vp9_encode");

    for &(width, height) in &[(640, 480), (1280, 720)] {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    let mut encoder = create_encoder("vp9", w, h)
                        .expect("Failed to create encoder");
                    let frame = create_test_video_frame(w, h);

                    encoder.send_frame(&Frame::Video(frame))
                        .expect("Failed to encode");
                    encoder.flush().expect("Failed to flush");

                    let _ = encoder.receive_packet();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Opus encoding
#[cfg(feature = "opus-codec")]
fn bench_opus_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode");

    let samples = 960; // 20ms at 48kHz
    let channels = 2;

    group.throughput(Throughput::Elements((samples * channels) as u64));

    group.bench_function("stereo_20ms", |b| {
        b.iter(|| {
            let mut encoder = zvd_lib::codec::opus::OpusEncoder::new(48000, channels)
                .expect("Failed to create encoder");
            let frame = create_test_audio_frame(samples, channels);

            let _ = encoder.send_frame(&Frame::Audio(frame));
            let _ = encoder.receive_packet();
        });
    });

    group.finish();
}

/// Benchmark Opus decoding
#[cfg(feature = "opus-codec")]
fn bench_opus_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_decode");

    let samples = 960;
    let channels = 2;

    // Pre-encode a frame
    let mut encoder = zvd_lib::codec::opus::OpusEncoder::new(48000, channels)
        .expect("Failed to create encoder");
    let frame = create_test_audio_frame(samples, channels);
    encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    let packet = encoder.receive_packet().expect("Failed to receive packet");

    group.throughput(Throughput::Elements((samples * channels) as u64));

    group.bench_function("stereo_20ms", |b| {
        b.iter(|| {
            let mut decoder = zvd_lib::codec::opus::OpusDecoder::new(48000, channels)
                .expect("Failed to create decoder");

            decoder.send_packet(black_box(&packet))
                .expect("Failed to decode");

            let _ = decoder.receive_frame();
        });
    });

    group.finish();
}

/// Benchmark video frame creation
fn bench_frame_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_creation");

    for &(width, height) in &[(640, 480), (1280, 720), (1920, 1080)] {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    black_box(create_test_video_frame(w, h));
                });
            },
        );
    }

    group.finish();
}

// Conditional compilation for benchmark groups
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_av1_encode,
        bench_av1_decode,
        bench_frame_creation,

        #[cfg(feature = "h264")]
        bench_h264_encode,
        #[cfg(feature = "h264")]
        bench_h264_decode,

        #[cfg(feature = "vp8-codec")]
        bench_vp8_encode,

        #[cfg(feature = "vp9-codec")]
        bench_vp9_encode,

        #[cfg(feature = "opus-codec")]
        bench_opus_encode,
        #[cfg(feature = "opus-codec")]
        bench_opus_decode,
}

criterion_main!(benches);
