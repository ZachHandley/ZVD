//! ZVD CLI - FFMPEG in Rust
//!
//! A command-line tool for multimedia processing

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};
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
    println!("File: {}", input.display());
    println!("\nNote: Full media info functionality not yet implemented.");
    println!("This is a placeholder that will be expanded with actual demuxer integration.");
    Ok(())
}

fn cmd_convert(
    _input: &PathBuf,
    _output: &PathBuf,
    _vcodec: Option<String>,
    _acodec: Option<String>,
    _vbitrate: Option<String>,
    _abitrate: Option<String>,
    _format: Option<String>,
) -> anyhow::Result<()> {
    println!("Note: Conversion functionality not yet implemented.");
    println!("This will be implemented with:");
    println!("  1. Demuxer to read input");
    println!("  2. Decoder to decompress streams");
    println!("  3. Filters for processing");
    println!("  4. Encoder to compress streams");
    println!("  5. Muxer to write output");
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

    let codecs = vec!["h264", "h265", "vp8", "vp9", "aac", "mp3", "opus"];

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
        let formats = vec!["mp4", "matroska"];
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
        let formats = vec!["mp4", "matroska"];
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
