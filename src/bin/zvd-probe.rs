//! zvd-probe - FFprobe equivalent for ZVD
//!
//! A command-line tool for inspecting media files and extracting metadata.
//!
//! # Usage
//!
//! ```bash
//! # Show human-readable output
//! zvd-probe video.mp4
//!
//! # Show JSON output
//! zvd-probe --format json video.mp4
//!
//! # Show compact JSON output
//! zvd-probe --format json --compact video.mp4
//! ```

use clap::{Parser, ValueEnum};
use zvd_lib::probe::MediaProbe;
use std::process;

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    /// Human-readable text output (default)
    Text,
    /// Pretty-printed JSON
    Json,
}

#[derive(Parser, Debug)]
#[command(name = "zvd-probe")]
#[command(about = "Probe media files and extract metadata (FFprobe equivalent)", long_about = None)]
struct Args {
    /// Media file to probe
    #[arg(value_name = "FILE")]
    file: String,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Compact JSON output (only with --format json)
    #[arg(short, long)]
    compact: bool,

    /// Show only specific stream type (video, audio, subtitle)
    #[arg(short, long)]
    stream_type: Option<String>,
}

fn main() {
    let args = Args::parse();

    // Create probe
    let probe = match MediaProbe::new(&args.file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: Failed to open file '{}': {}", args.file, e);
            process::exit(1);
        }
    };

    // Analyze file
    let metadata = match probe.analyze() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: Failed to analyze file '{}': {}", args.file, e);
            process::exit(1);
        }
    };

    // Filter streams if requested
    let filtered_metadata = if let Some(ref stream_type) = args.stream_type {
        let mut filtered = metadata.clone();
        filtered.streams.retain(|s| {
            stream_type.to_lowercase() == format!("{}", s.stream_type).to_lowercase()
        });
        filtered
    } else {
        metadata
    };

    // Output results
    match args.format {
        OutputFormat::Text => {
            println!("{}", filtered_metadata);
        }
        OutputFormat::Json => {
            let json = if args.compact {
                filtered_metadata.to_json_compact()
            } else {
                filtered_metadata.to_json()
            };

            match json {
                Ok(j) => println!("{}", j),
                Err(e) => {
                    eprintln!("Error: Failed to serialize JSON: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}
