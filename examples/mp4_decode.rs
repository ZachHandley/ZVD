//! MP4 Decoding Example
//!
//! This example demonstrates reading and decoding an MP4 file containing
//! H.264 video and AAC audio using the ZVD library.
//!
//! Run with: cargo run --example mp4_decode --features "h264,aac,mp4-support" -- <input.mp4>

#[cfg(not(all(feature = "h264", feature = "aac", feature = "mp4-support")))]
fn main() {
    eprintln!("This example requires the h264, aac, and mp4-support features.");
    eprintln!("Run with: cargo run --example mp4_decode --features \"h264,aac,mp4-support\" -- <input.mp4>");
}

#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use std::env;
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use std::io::Write;
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use std::path::Path;
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use zvd_lib::codec::h264::nal::{convert_avcc_to_annex_b, parse_avcc, prepend_parameter_sets};
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use zvd_lib::codec::{AacDecoder, Decoder, H264Decoder};
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use zvd_lib::format::demuxer::create_demuxer;
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use zvd_lib::format::Packet;
#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
use zvd_lib::util::{Buffer, MediaType};

#[cfg(all(feature = "h264", feature = "aac", feature = "mp4-support"))]
fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.mp4>", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    println!("ZVD MP4 Decoding Example");
    println!("========================");
    println!("Input: {}", input_path.display());
    println!();

    // Open the demuxer
    let mut demuxer = create_demuxer(input_path)?;

    // Get stream information
    let streams = demuxer.streams();
    println!("Found {} streams:", streams.len());

    // Find video and audio streams
    let mut video_stream_idx = None;
    let mut audio_stream_idx = None;
    let mut audio_extradata: Option<Vec<u8>> = None;
    let mut audio_sample_rate = 48000u32;
    let mut audio_channels = 2u16;

    // H.264 AVCC parameters for format conversion
    let mut avcc_length_size: u8 = 4;
    let mut sps_data: Option<Vec<u8>> = None;
    let mut pps_data: Option<Vec<u8>> = None;

    for stream in streams {
        println!(
            "  Stream #{}: {:?} - {}",
            stream.info.index, stream.info.media_type, stream.info.codec_id
        );

        match stream.info.media_type {
            MediaType::Video => {
                video_stream_idx = Some(stream.info.index);
                if let Some(ref video) = stream.info.video_info {
                    println!("    Resolution: {}x{}", video.width, video.height);
                    println!("    Frame rate: {}", video.frame_rate);
                }
                // Parse AVCC extradata to get SPS/PPS and length size
                if let Some(ref extradata) = stream.extradata {
                    if let Ok(avcc) = parse_avcc(extradata) {
                        avcc_length_size = avcc.length_size_minus_one + 1;
                        if !avcc.sps.is_empty() {
                            sps_data = Some(avcc.sps[0].clone());
                        }
                        if !avcc.pps.is_empty() {
                            pps_data = Some(avcc.pps[0].clone());
                        }
                        println!(
                            "    AVCC: {} SPS, {} PPS, {}-byte length",
                            avcc.sps.len(),
                            avcc.pps.len(),
                            avcc_length_size
                        );
                    }
                }
            }
            MediaType::Audio => {
                audio_stream_idx = Some(stream.info.index);
                audio_extradata = stream.extradata.clone();
                if let Some(ref audio) = stream.info.audio_info {
                    audio_sample_rate = audio.sample_rate;
                    audio_channels = audio.channels;
                    println!("    Sample rate: {} Hz", audio.sample_rate);
                    println!("    Channels: {}", audio.channels);
                }
            }
            _ => {}
        }
    }

    println!();

    // Create decoders
    let mut h264_decoder = H264Decoder::new()?;
    println!("Created H.264 decoder");

    // Try to create AAC decoder (may fail if system libs not available)
    let mut aac_decoder: Option<AacDecoder> = if let Some(ref extradata) = audio_extradata {
        match AacDecoder::with_extradata(audio_sample_rate, audio_channels, extradata) {
            Ok(dec) => {
                println!(
                    "Created AAC decoder ({}Hz, {} channels)",
                    audio_sample_rate, audio_channels
                );
                Some(dec)
            }
            Err(e) => {
                println!("Warning: AAC decoder unavailable ({}), skipping audio", e);
                println!("         Install libfdk-aac-dev to enable AAC decoding");
                None
            }
        }
    } else {
        match AacDecoder::new(audio_sample_rate, audio_channels) {
            Ok(dec) => {
                println!(
                    "Created AAC decoder ({}Hz, {} channels)",
                    audio_sample_rate, audio_channels
                );
                Some(dec)
            }
            Err(e) => {
                println!("Warning: AAC decoder unavailable ({}), skipping audio", e);
                println!("         Install libfdk-aac-dev to enable AAC decoding");
                None
            }
        }
    };

    // Decode packets
    let mut video_packets = 0;
    let mut video_frames = 0;
    let mut video_keyframes = 0;
    let mut audio_packets = 0;
    let mut audio_frames = 0;
    let mut total_audio_samples = 0usize;
    let max_packets = 500; // Limit for demo (about 4 seconds at 60fps)

    println!();
    println!("Decoding first {} packets...", max_packets);
    println!();

    loop {
        // Read next packet
        let packet = match demuxer.read_packet() {
            Ok(p) => p,
            Err(zvd_lib::Error::EndOfStream) => break,
            Err(e) => {
                eprintln!("Error reading packet: {}", e);
                break;
            }
        };

        // Process based on stream type
        if Some(packet.stream_index) == video_stream_idx {
            video_packets += 1;

            if packet.is_keyframe() {
                video_keyframes += 1;
            }

            // Convert AVCC format to Annex B format for OpenH264
            let annex_b_data = convert_avcc_to_annex_b(packet.data.as_slice(), avcc_length_size);

            // Prepend SPS/PPS for keyframes or first packet
            let final_data = if packet.is_keyframe() || video_packets == 1 {
                if let (Some(ref sps), Some(ref pps)) = (&sps_data, &pps_data) {
                    prepend_parameter_sets(&annex_b_data, sps, pps)
                } else {
                    annex_b_data
                }
            } else {
                annex_b_data
            };

            // Create a new packet with the converted data
            let converted_packet = Packet {
                stream_index: packet.stream_index,
                codec_type: packet.codec_type,
                data: Buffer::from_vec(final_data),
                pts: packet.pts,
                dts: packet.dts,
                duration: packet.duration,
                flags: packet.flags,
                position: packet.position,
            };

            // Decode the video packet
            if let Err(e) = h264_decoder.send_packet(&converted_packet) {
                // Only warn after first few packets (decoder needs time to initialize)
                if video_packets > 3 {
                    eprintln!("Warning: H.264 decode error: {}", e);
                }
                continue;
            }

            // Receive decoded frames
            while let Ok(frame) = h264_decoder.receive_frame() {
                video_frames += 1;
                if let zvd_lib::codec::Frame::Video(vf) = &frame {
                    if video_frames == 1 {
                        println!(
                            "First video frame: {}x{} {:?}",
                            vf.width, vf.height, vf.format
                        );
                    }
                }
            }
        } else if Some(packet.stream_index) == audio_stream_idx {
            audio_packets += 1;

            // Skip audio if decoder not available
            if let Some(ref mut decoder) = aac_decoder {
                // Decode the audio packet
                if let Err(e) = decoder.send_packet(&packet) {
                    // AAC decoder may fail on first few packets without proper setup
                    if audio_packets <= 3 {
                        continue;
                    }
                    eprintln!("Warning: AAC decode error: {}", e);
                    continue;
                }

                // Receive decoded frames
                while let Ok(frame) = decoder.receive_frame() {
                    audio_frames += 1;
                    if let zvd_lib::codec::Frame::Audio(af) = &frame {
                        total_audio_samples += af.nb_samples;
                        if audio_frames == 1 {
                            println!(
                                "First audio frame: {} samples at {} Hz, {} channels",
                                af.nb_samples, af.sample_rate, af.channels
                            );
                        }
                    }
                }
            }
        }

        // Progress indicator
        let total = video_packets + audio_packets;
        if total % 10 == 0 {
            print!(
                "\rProcessed {} packets (V:{} A:{})...",
                total, video_packets, audio_packets
            );
            std::io::stdout().flush().ok();
        }

        if total >= max_packets {
            break;
        }
    }

    println!("\r");
    println!("Decoding Results:");
    println!("=================");
    println!("Video:");
    println!("  Packets:   {}", video_packets);
    println!("  Keyframes: {}", video_keyframes);
    println!("  Frames:    {}", video_frames);
    println!();
    println!("Audio:");
    println!("  Packets:   {}", audio_packets);
    println!("  Frames:    {}", audio_frames);
    println!("  Samples:   {}", total_audio_samples);
    if audio_frames > 0 {
        let duration_sec = total_audio_samples as f64 / audio_sample_rate as f64;
        println!("  Duration:  {:.2} seconds", duration_sec);
    }

    println!();
    println!("MP4 decoding example completed successfully!");

    Ok(())
}
