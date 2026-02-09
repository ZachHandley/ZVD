//! ZVD CLI - FFMPEG in Rust
//!
//! A command-line tool for multimedia processing

use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use tracing::info;
use zvd_lib::{init, Config};

#[derive(Parser)]
#[command(name = "zvd")]
#[command(about = "ZVD - FFMPEG in Rust", long_about = None)]
#[command(version)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,

    /// Number of threads to use
    #[arg(short = 't', long)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show information about a media file
    Info {
        /// Input file path
        input: PathBuf,
    },

    /// Convert/transcode media files
    Convert {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Video codec (e.g., h264, h265, vp9)
        #[arg(long)]
        vcodec: Option<String>,

        /// Audio codec (e.g., aac, opus, mp3)
        #[arg(long)]
        acodec: Option<String>,

        /// Video bitrate (e.g., 2M, 500k)
        #[arg(long)]
        vbitrate: Option<String>,

        /// Audio bitrate (e.g., 128k, 192k)
        #[arg(long)]
        abitrate: Option<String>,

        /// Output format (override auto-detection)
        #[arg(short, long)]
        format: Option<String>,
    },

    /// Extract streams from a media file
    Extract {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Stream index to extract
        #[arg(short, long)]
        stream: usize,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Probe file format and streams
    Probe {
        /// Input file path
        input: PathBuf,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },

    /// List available codecs
    Codecs {
        /// Filter by type (video, audio)
        #[arg(short, long)]
        filter: Option<String>,
    },

    /// List available formats
    Formats {
        /// Show only muxers
        #[arg(long)]
        muxers: bool,

        /// Show only demuxers
        #[arg(long)]
        demuxers: bool,
    },

    /// Start a transcoding coordinator server
    #[cfg(feature = "server")]
    Coordinator {
        /// REST API port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// gRPC port for worker communication
        #[arg(long, default_value = "50051")]
        grpc_port: u16,

        /// Bind address
        #[arg(long, default_value = "0.0.0.0")]
        bind: String,

        /// Heartbeat timeout in seconds
        #[arg(long, default_value = "60")]
        heartbeat_timeout: u64,

        /// Maximum queue size
        #[arg(long, default_value = "10000")]
        max_queue: usize,
    },

    /// Start a transcoding worker daemon
    #[cfg(feature = "server")]
    Worker {
        /// Coordinator gRPC address
        #[arg(short, long, default_value = "http://localhost:50051")]
        coordinator: String,

        /// Maximum concurrent jobs
        #[arg(short, long, default_value = "1")]
        jobs: u32,

        /// Heartbeat interval in seconds
        #[arg(long, default_value = "30")]
        heartbeat_interval: u64,

        /// Job poll interval in seconds
        #[arg(long, default_value = "5")]
        poll_interval: u64,

        /// Worker hostname (auto-detected if not provided)
        #[arg(long)]
        hostname: Option<String>,
    },

    /// Submit a transcode job to the coordinator
    #[cfg(feature = "server")]
    Dispatch {
        /// Coordinator REST API URL
        #[arg(short, long, default_value = "http://localhost:8080")]
        coordinator: String,

        /// Input file path
        #[arg(short, long)]
        input: String,

        /// Output file path
        #[arg(short, long)]
        output: String,

        /// Video codec (e.g., h264, h265, vp9, av1)
        #[arg(long, value_name = "CODEC")]
        vcodec: Option<String>,

        /// Audio codec (e.g., aac, opus, mp3)
        #[arg(long, value_name = "CODEC")]
        acodec: Option<String>,

        /// Video bitrate (e.g., 5M, 2000k)
        #[arg(long, value_name = "BITRATE")]
        vbitrate: Option<String>,

        /// Audio bitrate (e.g., 128k, 192k)
        #[arg(long, value_name = "BITRATE")]
        abitrate: Option<String>,

        /// Output resolution (e.g., 1920x1080)
        #[arg(long, value_name = "WxH")]
        resolution: Option<String>,

        /// Quality/CRF value (0-63, lower is better)
        #[arg(long)]
        quality: Option<u32>,

        /// Encoder preset (e.g., fast, medium, slow)
        #[arg(long)]
        preset: Option<String>,

        /// Job priority (low, normal, high, critical)
        #[arg(long, default_value = "normal")]
        priority: String,

        /// Wait for job completion
        #[arg(long)]
        wait: bool,

        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },

    /// Parse FFmpeg-style arguments and submit as a job
    #[cfg(feature = "server")]
    Ffmpeg {
        /// Coordinator REST API URL
        #[arg(long, default_value = "http://localhost:8080")]
        coordinator: String,

        /// FFmpeg-style arguments (everything after --)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,

        /// Wait for job completion
        #[arg(long)]
        wait: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize the library
    let config = Config {
        max_threads: cli.threads,
        verbose: cli.verbose,
        debug: cli.debug,
    };

    init(config)?;

    info!("ZVD v{} - FFMPEG in Rust", zvd_lib::VERSION);

    // Execute command
    match cli.command {
        Commands::Info { input } => {
            info!("Getting info for: {}", input.display());
            cmd_info(&input)?;
        }
        Commands::Convert {
            input,
            output,
            vcodec,
            acodec,
            vbitrate,
            abitrate,
            format,
        } => {
            info!("Converting {} -> {}", input.display(), output.display());
            cmd_convert(&input, &output, vcodec, acodec, vbitrate, abitrate, format)?;
        }
        Commands::Extract {
            input,
            stream,
            output,
        } => {
            info!(
                "Extracting stream {} from {} to {}",
                stream,
                input.display(),
                output.display()
            );
            cmd_extract(&input, stream, &output)?;
        }
        Commands::Probe { input, json } => {
            cmd_probe(&input, json)?;
        }
        Commands::Codecs { filter } => {
            cmd_codecs(filter.as_deref())?;
        }
        Commands::Formats { muxers, demuxers } => {
            cmd_formats(muxers, demuxers)?;
        }

        #[cfg(feature = "server")]
        Commands::Coordinator {
            port,
            grpc_port,
            bind,
            heartbeat_timeout,
            max_queue,
        } => {
            cmd_coordinator(port, grpc_port, bind, heartbeat_timeout, max_queue)?;
        }

        #[cfg(feature = "server")]
        Commands::Worker {
            coordinator,
            jobs,
            heartbeat_interval,
            poll_interval,
            hostname,
        } => {
            cmd_worker(coordinator, jobs, heartbeat_interval, poll_interval, hostname)?;
        }

        #[cfg(feature = "server")]
        Commands::Dispatch {
            coordinator,
            input,
            output,
            vcodec,
            acodec,
            vbitrate,
            abitrate,
            resolution,
            quality,
            preset,
            priority,
            wait,
            json,
        } => {
            cmd_dispatch(
                coordinator, input, output, vcodec, acodec, vbitrate, abitrate,
                resolution, quality, preset, priority, wait, json,
            )?;
        }

        #[cfg(feature = "server")]
        Commands::Ffmpeg {
            coordinator,
            args,
            wait,
        } => {
            cmd_ffmpeg(coordinator, args, wait)?;
        }
    }

    Ok(())
}

fn cmd_info(input: &PathBuf) -> anyhow::Result<()> {
    use zvd_lib::format::demuxer::create_demuxer;

    println!("File: {}", input.display());
    println!();

    // Try to open the file
    let demuxer = match create_demuxer(input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            return Err(anyhow::anyhow!("Failed to open file"));
        }
    };

    // Get streams
    let streams = demuxer.streams();

    if streams.is_empty() {
        println!("No streams found in file");
        return Ok(());
    }

    // Display stream information
    println!("Streams: {}", streams.len());
    println!();

    for stream in streams {
        println!("Stream #{}:", stream.info.index);
        println!("  Type: {}", stream.info.media_type);
        println!("  Codec: {}", stream.info.codec_id);
        println!("  Time Base: {}", stream.info.time_base);

        if let Some(duration) = stream.info.nb_frames {
            println!("  Frames: {}", duration);
            println!("  Duration: {:.2}s", stream.info.duration_seconds());
        }

        // Audio-specific info
        if let Some(ref audio) = stream.info.audio_info {
            println!("  Sample Rate: {} Hz", audio.sample_rate);
            println!("  Channels: {}", audio.channels);
            println!("  Sample Format: {}", audio.sample_fmt);
            println!("  Bits Per Sample: {}", audio.bits_per_sample);
            if let Some(bitrate) = audio.bit_rate {
                println!("  Bit Rate: {} kbps", bitrate / 1000);
            }
        }

        // Video-specific info
        if let Some(ref video) = stream.info.video_info {
            println!("  Resolution: {}x{}", video.width, video.height);
            println!("  Frame Rate: {}", video.frame_rate);
            println!("  Pixel Format: {}", video.pix_fmt);
        }

        // Metadata
        if !stream.info.metadata.is_empty() {
            println!("  Metadata:");
            for (key, value) in &stream.info.metadata {
                println!("    {}: {}", key, value);
            }
        }

        println!();
    }

    Ok(())
}

fn cmd_convert(
    input: &PathBuf,
    output: &PathBuf,
    _vcodec: Option<String>,
    _acodec: Option<String>,
    _vbitrate: Option<String>,
    _abitrate: Option<String>,
    format: Option<String>,
) -> anyhow::Result<()> {
    use zvd_lib::codec::{PcmConfig, PcmDecoder, PcmEncoder};
    use zvd_lib::format::demuxer::create_demuxer;
    use zvd_lib::format::muxer::create_muxer;
    use zvd_lib::util::MediaType;

    println!("Converting {} -> {}", input.display(), output.display());

    // Open input demuxer
    let mut demuxer =
        create_demuxer(input).map_err(|e| anyhow::anyhow!("Failed to open input: {}", e))?;

    // Get input streams
    let streams = demuxer.streams();
    if streams.is_empty() {
        return Err(anyhow::anyhow!("No streams found in input file"));
    }

    // Find first audio stream
    let audio_stream = streams
        .iter()
        .find(|s| s.info.media_type == MediaType::Audio)
        .ok_or_else(|| anyhow::anyhow!("No audio stream found"))?;

    let audio_info = audio_stream
        .info
        .audio_info
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Audio stream missing info"))?;

    println!("Input:");
    println!("  Codec: {}", audio_stream.info.codec_id);
    println!("  Sample Rate: {} Hz", audio_info.sample_rate);
    println!("  Channels: {}", audio_info.channels);
    println!("  Format: {}", audio_info.sample_fmt);

    // Detect output format
    let output_format = if let Some(fmt) = format {
        fmt
    } else {
        // Detect from extension
        let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("wav");
        ext.to_string()
    };

    println!("\nOutput:");
    println!("  Format: {}", output_format);

    // Only support WAV for now
    if output_format != "wav" {
        return Err(anyhow::anyhow!("Only WAV output is currently supported"));
    }

    // Only support PCM codec for now
    if audio_stream.info.codec_id != "pcm" {
        return Err(anyhow::anyhow!("Only PCM codec is currently supported"));
    }

    // Parse sample format
    let sample_format = match audio_info.sample_fmt.as_str() {
        "u8" => zvd_lib::util::SampleFormat::U8,
        "s16" | "i16" => zvd_lib::util::SampleFormat::I16,
        "s32" | "i32" => zvd_lib::util::SampleFormat::I32,
        "f32" => zvd_lib::util::SampleFormat::F32,
        "f64" => zvd_lib::util::SampleFormat::F64,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format: {}",
                audio_info.sample_fmt
            ))
        }
    };

    // Create decoder
    let pcm_config = PcmConfig::new(sample_format, audio_info.channels, audio_info.sample_rate);
    let decoder = PcmDecoder::new(pcm_config.clone());

    // Create encoder (same config for now)
    let encoder = PcmEncoder::new(pcm_config);

    // Create output muxer
    let mut muxer = create_muxer(&output_format)
        .map_err(|e| anyhow::anyhow!("Failed to create muxer: {}", e))?;

    muxer
        .create(output)
        .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;

    // Clone the stream for output
    let output_stream = audio_stream.clone();
    muxer
        .add_stream(output_stream)
        .map_err(|e| anyhow::anyhow!("Failed to add stream: {}", e))?;

    muxer
        .write_header()
        .map_err(|e| anyhow::anyhow!("Failed to write header: {}", e))?;

    // Process packets
    let mut packets_processed = 0;
    let mut total_bytes = 0;

    println!("\nProcessing...");

    loop {
        // Read packet from input
        let packet = match demuxer.read_packet() {
            Ok(p) => p,
            Err(zvd_lib::Error::EndOfStream) => break,
            Err(e) => return Err(anyhow::anyhow!("Failed to read packet: {}", e)),
        };

        total_bytes += packet.data.len();

        // Decode packet to frame
        let frame = decoder
            .decode_packet(&packet)
            .map_err(|e| anyhow::anyhow!("Failed to decode packet: {}", e))?;

        // Encode frame to packet
        let mut output_packet = encoder
            .encode_frame(&frame)
            .map_err(|e| anyhow::anyhow!("Failed to encode frame: {}", e))?;

        // Write packet to output
        muxer
            .write_packet(&output_packet)
            .map_err(|e| anyhow::anyhow!("Failed to write packet: {}", e))?;

        packets_processed += 1;

        if packets_processed % 10 == 0 {
            print!(
                "\rProcessed {} packets ({} KB)...",
                packets_processed,
                total_bytes / 1024
            );
            std::io::stdout().flush().ok();
        }
    }

    // Finalize output
    muxer
        .write_trailer()
        .map_err(|e| anyhow::anyhow!("Failed to write trailer: {}", e))?;

    println!("\n\nConversion complete!");
    println!("  Packets processed: {}", packets_processed);
    println!("  Total data: {} KB", total_bytes / 1024);

    Ok(())
}

fn cmd_extract(input: &PathBuf, stream_index: usize, output: &PathBuf) -> anyhow::Result<()> {
    use zvd_lib::format::demuxer::create_demuxer;
    use zvd_lib::format::muxer::create_muxer;
    use zvd_lib::format::{detect_format_from_extension, Stream};

    println!(
        "Extracting stream {} from {} to {}",
        stream_index,
        input.display(),
        output.display()
    );

    // Open input demuxer
    let mut demuxer =
        create_demuxer(input).map_err(|e| anyhow::anyhow!("Failed to open input file: {}", e))?;

    // Get input streams and validate stream index
    let streams = demuxer.streams();
    if streams.is_empty() {
        return Err(anyhow::anyhow!("No streams found in input file"));
    }

    if stream_index >= streams.len() {
        return Err(anyhow::anyhow!(
            "Invalid stream index {}. File has {} streams (0-{})",
            stream_index,
            streams.len(),
            streams.len() - 1
        ));
    }

    // Clone the stream we want to extract
    let source_stream = streams[stream_index].clone();

    println!("\nSource stream:");
    println!("  Index: {}", source_stream.info.index);
    println!("  Type: {}", source_stream.info.media_type);
    println!("  Codec: {}", source_stream.info.codec_id);

    // Display stream-specific info
    if let Some(ref audio) = source_stream.info.audio_info {
        println!("  Sample Rate: {} Hz", audio.sample_rate);
        println!("  Channels: {}", audio.channels);
    }
    if let Some(ref video) = source_stream.info.video_info {
        println!("  Resolution: {}x{}", video.width, video.height);
        println!("  Frame Rate: {}", video.frame_rate);
    }

    // Detect output format from extension
    let output_path_str = output
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;

    let output_format = detect_format_from_extension(output_path_str).ok_or_else(|| {
        anyhow::anyhow!("Cannot detect output format from extension. Supported formats: wav, y4m")
    })?;

    println!("\nOutput:");
    println!("  Format: {}", output_format);
    println!("  File: {}", output.display());

    // Create output muxer
    let mut muxer = create_muxer(output_format).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create muxer for format '{}': {}",
            output_format,
            e
        )
    })?;

    muxer
        .create(output)
        .map_err(|e| anyhow::anyhow!("Failed to create output file: {}", e))?;

    // Create an output stream with index 0 (since we're only extracting one stream)
    let mut output_stream_info = source_stream.info.clone();
    output_stream_info.index = 0;
    let output_stream = Stream::new(output_stream_info);

    muxer
        .add_stream(output_stream)
        .map_err(|e| anyhow::anyhow!("Failed to add stream to output: {}", e))?;

    muxer
        .write_header()
        .map_err(|e| anyhow::anyhow!("Failed to write output header: {}", e))?;

    // Process packets - copy only packets from the specified stream
    let mut packets_extracted = 0;
    let mut total_bytes = 0;

    println!("\nExtracting...");

    loop {
        // Read packet from input
        let mut packet = match demuxer.read_packet() {
            Ok(p) => p,
            Err(zvd_lib::Error::EndOfStream) => break,
            Err(e) => return Err(anyhow::anyhow!("Failed to read packet: {}", e)),
        };

        // Only process packets from the stream we're extracting
        if packet.stream_index != stream_index {
            continue;
        }

        total_bytes += packet.data.len();

        // Remap stream index to 0 for output (since we only have one output stream)
        packet.stream_index = 0;

        // Write packet to output
        muxer
            .write_packet(&packet)
            .map_err(|e| anyhow::anyhow!("Failed to write packet: {}", e))?;

        packets_extracted += 1;

        if packets_extracted % 10 == 0 {
            print!(
                "\rExtracted {} packets ({} KB)...",
                packets_extracted,
                total_bytes / 1024
            );
            std::io::stdout().flush().ok();
        }
    }

    // Finalize output
    muxer
        .write_trailer()
        .map_err(|e| anyhow::anyhow!("Failed to write output trailer: {}", e))?;

    println!("\n\nExtraction complete!");
    println!("  Packets extracted: {}", packets_extracted);
    println!("  Total data: {} KB", total_bytes / 1024);

    if packets_extracted == 0 {
        println!("\nWarning: No packets were extracted. The stream may be empty or in an unsupported format.");
    }

    Ok(())
}

fn cmd_probe(input: &PathBuf, json: bool) -> anyhow::Result<()> {
    use serde::Serialize;
    use zvd_lib::format::demuxer::create_demuxer;
    use zvd_lib::format::{detect_format_from_extension, get_format_info};
    use zvd_lib::util::MediaType;

    // Serializable structures for JSON output
    #[derive(Serialize)]
    struct ProbeResult {
        file: String,
        format: FormatProbe,
        streams: Vec<StreamProbe>,
    }

    #[derive(Serialize)]
    struct FormatProbe {
        name: String,
        long_name: String,
        duration_seconds: Option<f64>,
        bit_rate: Option<u64>,
        seekable: bool,
        metadata: std::collections::HashMap<String, String>,
    }

    #[derive(Serialize)]
    struct StreamProbe {
        index: usize,
        #[serde(rename = "type")]
        stream_type: String,
        codec: String,
        time_base: String,
        duration_seconds: Option<f64>,
        nb_frames: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        video: Option<VideoProbe>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<AudioProbe>,
        metadata: std::collections::HashMap<String, String>,
    }

    #[derive(Serialize)]
    struct VideoProbe {
        width: u32,
        height: u32,
        frame_rate: String,
        frame_rate_num: i64,
        frame_rate_den: i64,
        pixel_format: String,
        bits_per_sample: u8,
        aspect_ratio: String,
    }

    #[derive(Serialize)]
    struct AudioProbe {
        sample_rate: u32,
        channels: u16,
        sample_format: String,
        bits_per_sample: u8,
        bit_rate: Option<u64>,
    }

    // Try to open the file with demuxer
    let demuxer = match create_demuxer(input) {
        Ok(d) => d,
        Err(e) => {
            if json {
                let error_json = serde_json::json!({
                    "error": format!("Failed to open file: {}", e),
                    "file": input.display().to_string()
                });
                println!("{}", serde_json::to_string_pretty(&error_json)?);
            } else {
                eprintln!("Error: Failed to open file: {}", e);
            }
            return Err(anyhow::anyhow!("Failed to open file"));
        }
    };

    // Detect format from extension
    let format_name =
        detect_format_from_extension(input.to_str().unwrap_or_default()).unwrap_or("unknown");

    let format_info = get_format_info(format_name);

    // Get streams from demuxer
    let streams = demuxer.streams();

    // Calculate total duration and bit rate from streams
    let mut max_duration_seconds: Option<f64> = None;
    for stream in streams {
        let dur = stream.info.duration_seconds();
        if dur > 0.0 {
            max_duration_seconds = Some(max_duration_seconds.map_or(dur, |d| d.max(dur)));
        }
    }

    // Estimate bit rate from file size and duration
    let bit_rate =
        if let (Some(duration), Ok(metadata)) = (max_duration_seconds, std::fs::metadata(input)) {
            if duration > 0.0 {
                Some(((metadata.len() as f64 * 8.0) / duration) as u64)
            } else {
                None
            }
        } else {
            None
        };

    // Build probe result
    let format_probe = FormatProbe {
        name: format_name.to_string(),
        long_name: format_info
            .as_ref()
            .map(|f| f.long_name.clone())
            .unwrap_or_else(|| format_name.to_string()),
        duration_seconds: max_duration_seconds,
        bit_rate,
        seekable: format_info
            .as_ref()
            .map(|f| f.capabilities.seekable)
            .unwrap_or(false),
        metadata: std::collections::HashMap::new(),
    };

    let mut stream_probes = Vec::new();
    for stream in streams {
        let stream_type = match stream.info.media_type {
            MediaType::Video => "video",
            MediaType::Audio => "audio",
            MediaType::Subtitle => "subtitle",
            MediaType::Data => "data",
            MediaType::Unknown => "unknown",
        };

        let video_probe = stream.info.video_info.as_ref().map(|v| VideoProbe {
            width: v.width,
            height: v.height,
            frame_rate: format!("{}", v.frame_rate),
            frame_rate_num: v.frame_rate.num,
            frame_rate_den: v.frame_rate.den,
            pixel_format: v.pix_fmt.clone(),
            bits_per_sample: v.bits_per_sample,
            aspect_ratio: format!("{}", v.aspect_ratio),
        });

        let audio_probe = stream.info.audio_info.as_ref().map(|a| AudioProbe {
            sample_rate: a.sample_rate,
            channels: a.channels,
            sample_format: a.sample_fmt.clone(),
            bits_per_sample: a.bits_per_sample,
            bit_rate: a.bit_rate,
        });

        let duration_secs = stream.info.duration_seconds();

        stream_probes.push(StreamProbe {
            index: stream.info.index,
            stream_type: stream_type.to_string(),
            codec: stream.info.codec_id.clone(),
            time_base: format!("{}", stream.info.time_base),
            duration_seconds: if duration_secs > 0.0 {
                Some(duration_secs)
            } else {
                None
            },
            nb_frames: stream.info.nb_frames,
            video: video_probe,
            audio: audio_probe,
            metadata: stream.info.metadata.clone(),
        });
    }

    let probe_result = ProbeResult {
        file: input.display().to_string(),
        format: format_probe,
        streams: stream_probes,
    };

    if json {
        // JSON output
        println!("{}", serde_json::to_string_pretty(&probe_result)?);
    } else {
        // Human-readable output
        println!("File: {}", probe_result.file);
        println!();
        println!("Format:");
        println!(
            "  Name: {} ({})",
            probe_result.format.name, probe_result.format.long_name
        );
        if let Some(duration) = probe_result.format.duration_seconds {
            let hours = (duration / 3600.0) as u32;
            let minutes = ((duration % 3600.0) / 60.0) as u32;
            let seconds = duration % 60.0;
            if hours > 0 {
                println!("  Duration: {:02}:{:02}:{:06.3}", hours, minutes, seconds);
            } else {
                println!("  Duration: {:02}:{:06.3}", minutes, seconds);
            }
        }
        if let Some(br) = probe_result.format.bit_rate {
            if br >= 1_000_000 {
                println!("  Bit Rate: {:.2} Mbps", br as f64 / 1_000_000.0);
            } else {
                println!("  Bit Rate: {} kbps", br / 1000);
            }
        }
        println!(
            "  Seekable: {}",
            if probe_result.format.seekable {
                "yes"
            } else {
                "no"
            }
        );
        println!();

        if probe_result.streams.is_empty() {
            println!("No streams found");
        } else {
            println!("Streams ({}):", probe_result.streams.len());
            println!();

            for stream in &probe_result.streams {
                println!("  Stream #{}:", stream.index);
                println!("    Type: {}", stream.stream_type);
                println!("    Codec: {}", stream.codec);
                println!("    Time Base: {}", stream.time_base);

                if let Some(duration) = stream.duration_seconds {
                    println!("    Duration: {:.3}s", duration);
                }

                if let Some(frames) = stream.nb_frames {
                    println!("    Frames: {}", frames);
                }

                if let Some(ref video) = stream.video {
                    println!("    Resolution: {}x{}", video.width, video.height);
                    println!(
                        "    Frame Rate: {} ({}/{})",
                        video.frame_rate, video.frame_rate_num, video.frame_rate_den
                    );
                    println!("    Pixel Format: {}", video.pixel_format);
                    println!("    Bits Per Sample: {}", video.bits_per_sample);
                    println!("    Aspect Ratio: {}", video.aspect_ratio);
                }

                if let Some(ref audio) = stream.audio {
                    println!("    Sample Rate: {} Hz", audio.sample_rate);
                    println!("    Channels: {}", audio.channels);
                    println!("    Sample Format: {}", audio.sample_format);
                    println!("    Bits Per Sample: {}", audio.bits_per_sample);
                    if let Some(br) = audio.bit_rate {
                        println!("    Bit Rate: {} kbps", br / 1000);
                    }
                }

                if !stream.metadata.is_empty() {
                    println!("    Metadata:");
                    for (key, value) in &stream.metadata {
                        println!("      {}: {}", key, value);
                    }
                }
                println!();
            }
        }
    }

    Ok(())
}

fn cmd_codecs(filter: Option<&str>) -> anyhow::Result<()> {
    use zvd_lib::codec::get_codec_info;
    use zvd_lib::util::MediaType;

    println!("Available Codecs:");
    println!("─────────────────────────────────────────────────────────");
    println!("{:<10} {:<8} {:<30}", "ID", "Type", "Description");
    println!("─────────────────────────────────────────────────────────");

    let codecs = vec!["h264", "h265", "vp8", "vp9", "aac", "mp3", "opus", "pcm"];

    for codec_id in codecs {
        if let Some(info) = get_codec_info(codec_id) {
            let should_show = match filter {
                Some("video") => info.media_type == MediaType::Video,
                Some("audio") => info.media_type == MediaType::Audio,
                _ => true,
            };

            if should_show {
                println!(
                    "{:<10} {:<8} {:<30}",
                    info.id,
                    format!("{}", info.media_type),
                    info.long_name
                );
            }
        }
    }

    println!("\nNote: This is a subset of supported codecs.");
    println!("Full codec implementations are in progress.");
    Ok(())
}

fn cmd_formats(muxers: bool, demuxers: bool) -> anyhow::Result<()> {
    use zvd_lib::format::get_format_info;

    let show_both = !muxers && !demuxers;

    if show_both || demuxers {
        println!("Demuxers (Input Formats):");
        println!("─────────────────────────────────────────────────────────");
    }

    if show_both || demuxers {
        let formats = vec!["mp4", "matroska", "wav"];
        for fmt in formats {
            if let Some(info) = get_format_info(fmt) {
                println!("{:<15} {}", info.name, info.long_name);
            }
        }
    }

    if show_both {
        println!();
    }

    if show_both || muxers {
        println!("Muxers (Output Formats):");
        println!("─────────────────────────────────────────────────────────");
        let formats = vec!["mp4", "matroska", "wav"];
        for fmt in formats {
            if let Some(info) = get_format_info(fmt) {
                println!("{:<15} {}", info.name, info.long_name);
            }
        }
    }

    println!("\nNote: This is a subset of supported formats.");
    println!("Full format implementations are in progress.");
    Ok(())
}

// ============================================================================
// Server Mode Commands (behind "server" feature)
// ============================================================================

#[cfg(feature = "server")]
fn cmd_coordinator(
    port: u16,
    grpc_port: u16,
    bind: String,
    heartbeat_timeout: u64,
    max_queue: usize,
) -> anyhow::Result<()> {
    use zvd_lib::server::{Coordinator, CoordinatorConfig};

    let config = CoordinatorConfig {
        rest_port: port,
        grpc_port,
        bind_addr: bind,
        heartbeat_timeout,
        max_queue_size: max_queue,
        ..Default::default()
    };

    println!("Starting ZVD Transcoding Coordinator");
    println!("  REST API: http://{}:{}", config.bind_addr, config.rest_port);
    println!("  gRPC:     {}:{}", config.bind_addr, config.grpc_port);
    println!();

    // Create async runtime and run coordinator
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let coordinator = Coordinator::new(config);
        coordinator.run().await
    })
    .map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}

#[cfg(feature = "server")]
fn cmd_worker(
    coordinator: String,
    jobs: u32,
    heartbeat_interval: u64,
    poll_interval: u64,
    hostname: Option<String>,
) -> anyhow::Result<()> {
    use zvd_lib::server::{Worker, WorkerConfig};

    let config = WorkerConfig {
        coordinator_addr: coordinator.clone(),
        max_concurrent_jobs: jobs,
        heartbeat_interval,
        poll_interval,
        hostname,
    };

    println!("Starting ZVD Transcoding Worker");
    println!("  Coordinator: {}", coordinator);
    println!("  Max concurrent jobs: {}", jobs);
    println!();

    // Create async runtime and run worker
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let worker = Worker::new(config);
        worker.run().await
    })
    .map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}

#[cfg(feature = "server")]
fn cmd_dispatch(
    coordinator: String,
    input: String,
    output: String,
    vcodec: Option<String>,
    acodec: Option<String>,
    vbitrate: Option<String>,
    abitrate: Option<String>,
    resolution: Option<String>,
    quality: Option<u32>,
    preset: Option<String>,
    priority: String,
    wait: bool,
    json: bool,
) -> anyhow::Result<()> {
    use std::time::Duration;
    use zvd_lib::server::{
        DispatchClient, DispatchConfig, TranscodeRequestBuilder,
        protocol::JobPriority,
    };

    // Parse priority
    let job_priority = match priority.to_lowercase().as_str() {
        "low" => JobPriority::Low,
        "normal" => JobPriority::Normal,
        "high" => JobPriority::High,
        "critical" => JobPriority::Critical,
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid priority: {}. Use low, normal, high, or critical",
                priority
            ));
        }
    };

    // Build request
    let mut builder = TranscodeRequestBuilder::new(&input, &output)
        .priority(job_priority);

    if let Some(codec) = vcodec {
        builder = builder.video_codec(codec);
    }
    if let Some(codec) = acodec {
        builder = builder.audio_codec(codec);
    }
    if let Some(bitrate) = vbitrate {
        let parsed = parse_bitrate_arg(&bitrate)?;
        builder = builder.video_bitrate(parsed);
    }
    if let Some(bitrate) = abitrate {
        let parsed = parse_bitrate_arg(&bitrate)?;
        builder = builder.audio_bitrate(parsed);
    }
    if let Some(res) = resolution {
        let (w, h) = parse_resolution_arg(&res)?;
        builder = builder.resolution(w, h);
    }
    if let Some(q) = quality {
        builder = builder.quality(q);
    }
    if let Some(p) = preset {
        builder = builder.preset(p);
    }

    let request = builder.build();

    // Create client
    let config = DispatchConfig {
        coordinator_url: coordinator.clone(),
        ..Default::default()
    };

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let client = DispatchClient::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create client: {}", e))?;

        // Health check first
        if !client.health_check().await.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Cannot connect to coordinator at {}",
                coordinator
            ));
        }

        // Submit job
        let response = client
            .submit_job(request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to submit job: {}", e))?;

        if json {
            println!("{}", serde_json::to_string_pretty(&response)?);
        } else {
            println!("Job submitted: {}", response.id);
            println!("Status: {:?}", response.status.state);
        }

        // Wait for completion if requested
        if wait {
            if !json {
                println!("\nWaiting for job completion...");
            }

            let final_status = client
                .wait_for_job(response.id, Duration::from_secs(2), None)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to wait for job: {}", e))?;

            if json {
                println!("{}", serde_json::to_string_pretty(&final_status)?);
            } else {
                println!("\nJob completed:");
                println!("  Status: {:?}", final_status.state);
            }
        }

        Ok(())
    })
}

#[cfg(feature = "server")]
fn cmd_ffmpeg(
    coordinator: String,
    args: Vec<String>,
    wait: bool,
) -> anyhow::Result<()> {
    use std::time::Duration;
    use zvd_lib::server::{
        DispatchClient, DispatchConfig,
        ffcompat::parse_ffmpeg_args,
    };

    if args.is_empty() {
        return Err(anyhow::anyhow!(
            "No FFmpeg arguments provided. Usage: zvd ffmpeg -- -i input.mp4 -c:v libx264 output.mp4"
        ));
    }

    // Parse FFmpeg-style arguments
    let parsed = parse_ffmpeg_args(&args)
        .map_err(|e| anyhow::anyhow!("Failed to parse arguments: {}", e))?;

    let request = parsed
        .into_request()
        .map_err(|e| anyhow::anyhow!("Failed to create request: {}", e))?;

    println!("Parsed transcode request:");
    println!("  Input: {}", request.input);
    println!("  Output: {}", request.output);
    if let Some(ref v) = request.params.video_codec {
        println!("  Video codec: {}", v);
    }
    if let Some(ref a) = request.params.audio_codec {
        println!("  Audio codec: {}", a);
    }

    // Create client
    let config = DispatchConfig {
        coordinator_url: coordinator.clone(),
        ..Default::default()
    };

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let client = DispatchClient::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create client: {}", e))?;

        // Health check
        if !client.health_check().await.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Cannot connect to coordinator at {}",
                coordinator
            ));
        }

        // Submit job
        let response = client
            .submit_job(request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to submit job: {}", e))?;

        println!("\nJob submitted: {}", response.id);

        // Wait if requested
        if wait {
            println!("Waiting for job completion...");

            let final_status = client
                .wait_for_job(response.id, Duration::from_secs(2), None)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to wait for job: {}", e))?;

            println!("Job completed: {:?}", final_status.state);
        }

        Ok(())
    })
}

#[cfg(feature = "server")]
fn parse_bitrate_arg(value: &str) -> anyhow::Result<u64> {
    let value = value.trim();

    let (num_str, multiplier) = if value.ends_with('k') || value.ends_with('K') {
        (&value[..value.len() - 1], 1_000u64)
    } else if value.ends_with('M') || value.ends_with('m') {
        (&value[..value.len() - 1], 1_000_000u64)
    } else if value.ends_with('G') || value.ends_with('g') {
        (&value[..value.len() - 1], 1_000_000_000u64)
    } else {
        (value, 1u64)
    };

    let num: f64 = num_str
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid bitrate: {}", value))?;

    Ok((num * multiplier as f64) as u64)
}

#[cfg(feature = "server")]
fn parse_resolution_arg(value: &str) -> anyhow::Result<(u32, u32)> {
    let parts: Vec<&str> = value.split(|c| c == 'x' || c == ':').collect();

    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "Invalid resolution: {}. Use WxH format (e.g., 1920x1080)",
            value
        ));
    }

    let width: u32 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid width in resolution: {}", value))?;
    let height: u32 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid height in resolution: {}", value))?;

    Ok((width, height))
}
