//! ZVD CLI - FFMPEG in Rust
//!
//! A command-line tool for multimedia processing

use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use tracing::info;
use zvd_lib::{Config, init};

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
            println!(
                "  Duration: {:.2}s",
                stream.info.duration_seconds()
            );
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
    let mut demuxer = create_demuxer(input)
        .map_err(|e| anyhow::anyhow!("Failed to open input: {}", e))?;

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
        let ext = output
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("wav");
        ext.to_string()
    };

    println!("\nOutput:");
    println!("  Format: {}", output_format);

    // Only support WAV for now
    if output_format != "wav" {
        return Err(anyhow::anyhow!(
            "Only WAV output is currently supported"
        ));
    }

    // Only support PCM codec for now
    if audio_stream.info.codec_id != "pcm" {
        return Err(anyhow::anyhow!(
            "Only PCM codec is currently supported"
        ));
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
    let pcm_config = PcmConfig::new(
        sample_format,
        audio_info.channels,
        audio_info.sample_rate,
    );
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
            print!("\rProcessed {} packets ({} KB)...", packets_processed, total_bytes / 1024);
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

fn cmd_extract(_input: &PathBuf, _stream: usize, _output: &PathBuf) -> anyhow::Result<()> {
    println!("Note: Stream extraction not yet implemented.");
    Ok(())
}

fn cmd_probe(input: &PathBuf, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{{");
        println!("  \"file\": \"{}\",", input.display());
        println!("  \"note\": \"Probe functionality not yet implemented\"");
        println!("}}");
    } else {
        println!("Probing: {}", input.display());
        println!("\nNote: Probe functionality not yet implemented.");
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
