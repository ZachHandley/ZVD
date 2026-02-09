//! FFmpeg Compatibility Layer
//!
//! This module provides parsing of FFmpeg-style command-line arguments,
//! allowing users to submit transcode jobs using familiar syntax.
//!
//! # Supported Arguments
//!
//! ## Input/Output
//! - `-i <file>` - Input file
//! - `-o <file>` or positional - Output file
//! - `-y` - Overwrite output
//!
//! ## Video Options
//! - `-c:v <codec>` / `-vcodec <codec>` - Video codec
//! - `-b:v <bitrate>` / `-vb <bitrate>` - Video bitrate
//! - `-s <WxH>` - Output resolution
//! - `-r <fps>` - Frame rate
//! - `-pix_fmt <format>` - Pixel format
//! - `-crf <value>` - Constant Rate Factor
//! - `-preset <value>` - Encoder preset
//! - `-pass <1|2>` - Two-pass encoding pass
//!
//! ## Audio Options
//! - `-c:a <codec>` / `-acodec <codec>` - Audio codec
//! - `-b:a <bitrate>` / `-ab <bitrate>` - Audio bitrate
//! - `-ar <rate>` - Audio sample rate
//! - `-ac <channels>` - Audio channels
//!
//! ## Seeking/Trimming
//! - `-ss <time>` - Start time
//! - `-t <duration>` - Duration
//! - `-to <time>` - End time
//!
//! ## Hardware Acceleration
//! - `-hwaccel <type>` - Hardware acceleration type
//!
//! # Example
//!
//! ```rust
//! use zvd_lib::server::ffcompat::parse_ffmpeg_args;
//!
//! let args = vec![
//!     "-i", "input.mp4",
//!     "-c:v", "libx264",
//!     "-crf", "23",
//!     "-c:a", "aac",
//!     "-b:a", "128k",
//!     "output.mp4"
//! ];
//!
//! let request = parse_ffmpeg_args(&args).unwrap();
//! ```

use std::collections::HashMap;

use super::protocol::{CreateJobRequest, JobPriority, TranscodeParams};
use crate::hwaccel::HwAccelType;

/// Error type for FFmpeg argument parsing
#[derive(Debug)]
pub enum ParseError {
    /// Missing required argument
    MissingRequired(String),
    /// Invalid argument value
    InvalidValue { arg: String, value: String, reason: String },
    /// Unknown argument
    Unknown(String),
    /// Missing value for argument
    MissingValue(String),
    /// Conflicting arguments
    Conflict(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingRequired(arg) => write!(f, "Missing required argument: {}", arg),
            ParseError::InvalidValue { arg, value, reason } => {
                write!(f, "Invalid value '{}' for {}: {}", value, arg, reason)
            }
            ParseError::Unknown(arg) => write!(f, "Unknown argument: {}", arg),
            ParseError::MissingValue(arg) => write!(f, "Missing value for argument: {}", arg),
            ParseError::Conflict(msg) => write!(f, "Conflicting arguments: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

/// Result type for FFmpeg argument parsing
pub type Result<T> = std::result::Result<T, ParseError>;

/// Parsed FFmpeg arguments
#[derive(Debug, Clone, Default)]
pub struct ParsedArgs {
    /// Input files
    pub inputs: Vec<String>,

    /// Output file
    pub output: Option<String>,

    /// Video codec
    pub video_codec: Option<String>,

    /// Audio codec
    pub audio_codec: Option<String>,

    /// Video bitrate
    pub video_bitrate: Option<u64>,

    /// Audio bitrate
    pub audio_bitrate: Option<u64>,

    /// Output width
    pub width: Option<u32>,

    /// Output height
    pub height: Option<u32>,

    /// Frame rate
    pub frame_rate: Option<f64>,

    /// Pixel format
    pub pixel_format: Option<String>,

    /// CRF value
    pub crf: Option<u32>,

    /// Encoder preset
    pub preset: Option<String>,

    /// Two-pass encoding pass number
    pub pass: Option<u32>,

    /// Audio sample rate
    pub audio_sample_rate: Option<u32>,

    /// Audio channels
    pub audio_channels: Option<u32>,

    /// Start time in seconds
    pub start_time: Option<f64>,

    /// Duration in seconds
    pub duration: Option<f64>,

    /// End time in seconds
    pub end_time: Option<f64>,

    /// Hardware acceleration type
    pub hwaccel: Option<String>,

    /// Overwrite output
    pub overwrite: bool,

    /// Extra parameters
    pub extra: HashMap<String, String>,
}

impl ParsedArgs {
    /// Convert to a CreateJobRequest
    pub fn into_request(self) -> Result<CreateJobRequest> {
        let input = self
            .inputs
            .first()
            .ok_or_else(|| ParseError::MissingRequired("input (-i)".to_string()))?
            .clone();

        let output = self
            .output
            .ok_or_else(|| ParseError::MissingRequired("output".to_string()))?;

        // Calculate actual duration if end_time is specified
        let duration = match (self.duration, self.end_time, self.start_time) {
            (Some(d), _, _) => Some(d),
            (None, Some(end), Some(start)) => Some(end - start),
            (None, Some(end), None) => Some(end),
            _ => None,
        };

        // Convert hwaccel string to enum
        let hw_accel = self.hwaccel.as_ref().and_then(|h| parse_hwaccel(h));

        let params = TranscodeParams {
            video_codec: self.video_codec,
            audio_codec: self.audio_codec,
            video_bitrate: self.video_bitrate,
            audio_bitrate: self.audio_bitrate,
            width: self.width,
            height: self.height,
            frame_rate: self.frame_rate,
            pixel_format: self.pixel_format,
            preset: self.preset,
            quality: self.crf,
            two_pass: self.pass.map_or(false, |p| p == 2),
            hw_accel,
            start_time: self.start_time,
            duration,
            audio_channels: self.audio_channels,
            audio_sample_rate: self.audio_sample_rate,
            extra_params: self.extra,
        };

        Ok(CreateJobRequest {
            input,
            output,
            params,
            priority: JobPriority::Normal,
            timeout: None,
            callback_url: None,
            metadata: HashMap::new(),
        })
    }
}

/// Parse FFmpeg-style command-line arguments
pub fn parse_ffmpeg_args<S: AsRef<str>>(args: &[S]) -> Result<ParsedArgs> {
    let mut parsed = ParsedArgs::default();
    let mut iter = args.iter().peekable();

    while let Some(arg) = iter.next() {
        let arg = arg.as_ref();

        match arg {
            // Input file
            "-i" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue("-i".to_string()))?;
                parsed.inputs.push(value.as_ref().to_string());
            }

            // Output file (explicit)
            "-o" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue("-o".to_string()))?;
                parsed.output = Some(value.as_ref().to_string());
            }

            // Video codec
            "-c:v" | "-vcodec" | "-codec:v" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.video_codec = Some(normalize_codec(value.as_ref()));
            }

            // Audio codec
            "-c:a" | "-acodec" | "-codec:a" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.audio_codec = Some(normalize_codec(value.as_ref()));
            }

            // Copy all streams
            "-c" | "-codec" if iter.peek().map(|v| v.as_ref()) == Some("copy") => {
                iter.next();
                parsed.video_codec = Some("copy".to_string());
                parsed.audio_codec = Some("copy".to_string());
            }

            // Video bitrate
            "-b:v" | "-vb" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.video_bitrate = Some(parse_bitrate(value.as_ref(), arg)?);
            }

            // Audio bitrate
            "-b:a" | "-ab" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.audio_bitrate = Some(parse_bitrate(value.as_ref(), arg)?);
            }

            // Resolution
            "-s" | "-video_size" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                let (w, h) = parse_resolution(value.as_ref(), arg)?;
                parsed.width = Some(w);
                parsed.height = Some(h);
            }

            // Frame rate
            "-r" | "-framerate" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.frame_rate = Some(parse_frame_rate(value.as_ref(), arg)?);
            }

            // Pixel format
            "-pix_fmt" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.pixel_format = Some(value.as_ref().to_string());
            }

            // CRF
            "-crf" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.crf = Some(parse_u32(value.as_ref(), arg)?);
            }

            // Preset
            "-preset" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.preset = Some(value.as_ref().to_string());
            }

            // Two-pass
            "-pass" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.pass = Some(parse_u32(value.as_ref(), arg)?);
            }

            // Audio sample rate
            "-ar" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.audio_sample_rate = Some(parse_u32(value.as_ref(), arg)?);
            }

            // Audio channels
            "-ac" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.audio_channels = Some(parse_u32(value.as_ref(), arg)?);
            }

            // Start time
            "-ss" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.start_time = Some(parse_time(value.as_ref(), arg)?);
            }

            // Duration
            "-t" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.duration = Some(parse_time(value.as_ref(), arg)?);
            }

            // End time
            "-to" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.end_time = Some(parse_time(value.as_ref(), arg)?);
            }

            // Hardware acceleration
            "-hwaccel" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.hwaccel = Some(value.as_ref().to_string());
            }

            // Overwrite output
            "-y" => {
                parsed.overwrite = true;
            }

            // Quality (libvpx-vp9)
            "-quality" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("quality".to_string(), value.as_ref().to_string());
            }

            // Profile
            "-profile:v" | "-profile" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("profile".to_string(), value.as_ref().to_string());
            }

            // Level
            "-level" | "-level:v" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("level".to_string(), value.as_ref().to_string());
            }

            // Tune
            "-tune" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("tune".to_string(), value.as_ref().to_string());
            }

            // GOP size
            "-g" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("gop".to_string(), value.as_ref().to_string());
            }

            // B-frames
            "-bf" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("bframes".to_string(), value.as_ref().to_string());
            }

            // Max rate (for VBV)
            "-maxrate" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("maxrate".to_string(), value.as_ref().to_string());
            }

            // Buffer size (for VBV)
            "-bufsize" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("bufsize".to_string(), value.as_ref().to_string());
            }

            // Threads
            "-threads" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("threads".to_string(), value.as_ref().to_string());
            }

            // No audio
            "-an" => {
                parsed.audio_codec = None;
                parsed.extra.insert("no_audio".to_string(), "true".to_string());
            }

            // No video
            "-vn" => {
                parsed.video_codec = None;
                parsed.extra.insert("no_video".to_string(), "true".to_string());
            }

            // Filter complex (just store for now)
            "-filter_complex" | "-lavfi" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("filter_complex".to_string(), value.as_ref().to_string());
            }

            // Video filter
            "-vf" | "-filter:v" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("vf".to_string(), value.as_ref().to_string());
            }

            // Audio filter
            "-af" | "-filter:a" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("af".to_string(), value.as_ref().to_string());
            }

            // Format
            "-f" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                parsed.extra.insert("format".to_string(), value.as_ref().to_string());
            }

            // Map streams
            "-map" => {
                let value = iter.next().ok_or_else(|| ParseError::MissingValue(arg.to_string()))?;
                // Store map arguments, might have multiple
                let key = format!("map_{}", parsed.extra.iter().filter(|(k, _)| k.starts_with("map_")).count());
                parsed.extra.insert(key, value.as_ref().to_string());
            }

            // Unknown argument starting with -
            other if other.starts_with('-') => {
                // Try to consume the next value if it exists and doesn't start with -
                if let Some(next) = iter.peek() {
                    if !next.as_ref().starts_with('-') {
                        let value = iter.next().unwrap();
                        parsed.extra.insert(other[1..].to_string(), value.as_ref().to_string());
                        continue;
                    }
                }
                // Flag without value
                parsed.extra.insert(other[1..].to_string(), "true".to_string());
            }

            // Positional argument (output file)
            other => {
                if parsed.output.is_none() {
                    parsed.output = Some(other.to_string());
                }
            }
        }
    }

    Ok(parsed)
}

/// Normalize codec name from FFmpeg to ZVD
fn normalize_codec(codec: &str) -> String {
    match codec.to_lowercase().as_str() {
        // Video codecs
        "libx264" | "h264" | "avc" => "h264".to_string(),
        "libx265" | "h265" | "hevc" => "h265".to_string(),
        "libvpx" | "vp8" => "vp8".to_string(),
        "libvpx-vp9" | "vp9" => "vp9".to_string(),
        "libaom-av1" | "av1" | "libsvtav1" | "librav1e" => "av1".to_string(),
        "prores" | "prores_ks" | "prores_aw" => "prores".to_string(),
        "dnxhd" | "dnxhr" => "dnxhd".to_string(),

        // Audio codecs
        "aac" | "libfdk_aac" | "aac_at" => "aac".to_string(),
        "libopus" | "opus" => "opus".to_string(),
        "libmp3lame" | "mp3" => "mp3".to_string(),
        "flac" => "flac".to_string(),
        "pcm_s16le" | "pcm_s24le" | "pcm_s32le" | "pcm_f32le" => "pcm".to_string(),

        // Copy
        "copy" => "copy".to_string(),

        // Unknown - pass through
        other => other.to_string(),
    }
}

/// Parse bitrate string (e.g., "5M", "128k", "2000000")
fn parse_bitrate(value: &str, arg: &str) -> Result<u64> {
    let value = value.trim();

    // Check for suffix
    let (num_str, multiplier) = if value.ends_with('k') || value.ends_with('K') {
        (&value[..value.len() - 1], 1_000u64)
    } else if value.ends_with('M') || value.ends_with('m') {
        (&value[..value.len() - 1], 1_000_000u64)
    } else if value.ends_with('G') || value.ends_with('g') {
        (&value[..value.len() - 1], 1_000_000_000u64)
    } else {
        (value, 1u64)
    };

    // Parse number (can be float for things like "1.5M")
    let num: f64 = num_str.parse().map_err(|_| ParseError::InvalidValue {
        arg: arg.to_string(),
        value: value.to_string(),
        reason: "Invalid bitrate value".to_string(),
    })?;

    Ok((num * multiplier as f64) as u64)
}

/// Parse resolution string (e.g., "1920x1080", "1280:720")
fn parse_resolution(value: &str, arg: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = value.split(|c| c == 'x' || c == ':').collect();

    if parts.len() != 2 {
        return Err(ParseError::InvalidValue {
            arg: arg.to_string(),
            value: value.to_string(),
            reason: "Resolution must be WxH or W:H".to_string(),
        });
    }

    let width: u32 = parts[0].parse().map_err(|_| ParseError::InvalidValue {
        arg: arg.to_string(),
        value: value.to_string(),
        reason: "Invalid width".to_string(),
    })?;

    let height: u32 = parts[1].parse().map_err(|_| ParseError::InvalidValue {
        arg: arg.to_string(),
        value: value.to_string(),
        reason: "Invalid height".to_string(),
    })?;

    Ok((width, height))
}

/// Parse frame rate (e.g., "30", "29.97", "30000/1001")
fn parse_frame_rate(value: &str, arg: &str) -> Result<f64> {
    // Check for fraction
    if value.contains('/') {
        let parts: Vec<&str> = value.split('/').collect();
        if parts.len() != 2 {
            return Err(ParseError::InvalidValue {
                arg: arg.to_string(),
                value: value.to_string(),
                reason: "Frame rate fraction must be num/den".to_string(),
            });
        }

        let num: f64 = parts[0].parse().map_err(|_| ParseError::InvalidValue {
            arg: arg.to_string(),
            value: value.to_string(),
            reason: "Invalid numerator".to_string(),
        })?;

        let den: f64 = parts[1].parse().map_err(|_| ParseError::InvalidValue {
            arg: arg.to_string(),
            value: value.to_string(),
            reason: "Invalid denominator".to_string(),
        })?;

        if den == 0.0 {
            return Err(ParseError::InvalidValue {
                arg: arg.to_string(),
                value: value.to_string(),
                reason: "Denominator cannot be zero".to_string(),
            });
        }

        return Ok(num / den);
    }

    // Parse as float
    value.parse().map_err(|_| ParseError::InvalidValue {
        arg: arg.to_string(),
        value: value.to_string(),
        reason: "Invalid frame rate".to_string(),
    })
}

/// Parse unsigned 32-bit integer
fn parse_u32(value: &str, arg: &str) -> Result<u32> {
    value.parse().map_err(|_| ParseError::InvalidValue {
        arg: arg.to_string(),
        value: value.to_string(),
        reason: "Invalid integer value".to_string(),
    })
}

/// Parse time string (e.g., "10", "1:30", "01:30:00", "90.5")
fn parse_time(value: &str, arg: &str) -> Result<f64> {
    // Check for HH:MM:SS format
    if value.contains(':') {
        let parts: Vec<&str> = value.split(':').collect();
        let mut seconds = 0.0;

        match parts.len() {
            2 => {
                // MM:SS
                let mins: f64 = parts[0].parse().map_err(|_| ParseError::InvalidValue {
                    arg: arg.to_string(),
                    value: value.to_string(),
                    reason: "Invalid minutes".to_string(),
                })?;
                let secs: f64 = parts[1].parse().map_err(|_| ParseError::InvalidValue {
                    arg: arg.to_string(),
                    value: value.to_string(),
                    reason: "Invalid seconds".to_string(),
                })?;
                seconds = mins * 60.0 + secs;
            }
            3 => {
                // HH:MM:SS
                let hours: f64 = parts[0].parse().map_err(|_| ParseError::InvalidValue {
                    arg: arg.to_string(),
                    value: value.to_string(),
                    reason: "Invalid hours".to_string(),
                })?;
                let mins: f64 = parts[1].parse().map_err(|_| ParseError::InvalidValue {
                    arg: arg.to_string(),
                    value: value.to_string(),
                    reason: "Invalid minutes".to_string(),
                })?;
                let secs: f64 = parts[2].parse().map_err(|_| ParseError::InvalidValue {
                    arg: arg.to_string(),
                    value: value.to_string(),
                    reason: "Invalid seconds".to_string(),
                })?;
                seconds = hours * 3600.0 + mins * 60.0 + secs;
            }
            _ => {
                return Err(ParseError::InvalidValue {
                    arg: arg.to_string(),
                    value: value.to_string(),
                    reason: "Time must be SS, MM:SS, or HH:MM:SS".to_string(),
                });
            }
        }

        return Ok(seconds);
    }

    // Parse as float (seconds)
    value.parse().map_err(|_| ParseError::InvalidValue {
        arg: arg.to_string(),
        value: value.to_string(),
        reason: "Invalid time value".to_string(),
    })
}

/// Parse hardware acceleration type
fn parse_hwaccel(value: &str) -> Option<HwAccelType> {
    match value.to_lowercase().as_str() {
        "none" | "auto" => Some(HwAccelType::None),
        "vaapi" => Some(HwAccelType::VAAPI),
        "nvdec" | "cuda" | "cuvid" => Some(HwAccelType::NVDEC),
        "nvenc" => Some(HwAccelType::NVENC),
        "qsv" => Some(HwAccelType::QSV),
        "videotoolbox" | "vt" => Some(HwAccelType::VideoToolbox),
        "amf" => Some(HwAccelType::AMF),
        "dxva2" => Some(HwAccelType::DXVA2),
        "d3d11va" => Some(HwAccelType::D3D11VA),
        "vulkan" => Some(HwAccelType::Vulkan),
        _ => None,
    }
}

/// Parse a Jellyfin-style transcode command
///
/// Jellyfin typically generates commands like:
/// ```text
/// -i /path/to/media.mkv -map 0:0 -map 0:1 -c:v:0 libx264 -preset veryfast
/// -crf 23 -maxrate 10000000 -bufsize 20000000 -profile:v high -level 41
/// -c:a:0 aac -b:a 192000 -f mp4 /path/to/output.mp4
/// ```
pub fn parse_jellyfin_command(cmd: &str) -> Result<ParsedArgs> {
    // Split command into args, handling quoted strings
    let args = shell_words::split(cmd).map_err(|e| {
        ParseError::InvalidValue {
            arg: "command".to_string(),
            value: cmd.to_string(),
            reason: format!("Failed to parse command: {}", e),
        }
    })?;

    parse_ffmpeg_args(&args)
}

/// Parse a Plex-style transcode command
///
/// Plex generates similar commands but may include additional metadata arguments
pub fn parse_plex_command(cmd: &str) -> Result<ParsedArgs> {
    // Plex commands are similar to FFmpeg, just parse them normally
    parse_jellyfin_command(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_args() {
        let args = vec!["-i", "input.mp4", "output.webm"];
        let parsed = parse_ffmpeg_args(&args).unwrap();

        assert_eq!(parsed.inputs, vec!["input.mp4"]);
        assert_eq!(parsed.output, Some("output.webm".to_string()));
    }

    #[test]
    fn test_parse_video_codec() {
        let args = vec!["-i", "in.mp4", "-c:v", "libx264", "out.mp4"];
        let parsed = parse_ffmpeg_args(&args).unwrap();

        assert_eq!(parsed.video_codec, Some("h264".to_string()));
    }

    #[test]
    fn test_parse_audio_codec() {
        let args = vec!["-i", "in.mp4", "-c:a", "aac", "out.mp4"];
        let parsed = parse_ffmpeg_args(&args).unwrap();

        assert_eq!(parsed.audio_codec, Some("aac".to_string()));
    }

    #[test]
    fn test_parse_bitrate() {
        assert_eq!(parse_bitrate("5M", "test").unwrap(), 5_000_000);
        assert_eq!(parse_bitrate("128k", "test").unwrap(), 128_000);
        assert_eq!(parse_bitrate("1.5M", "test").unwrap(), 1_500_000);
        assert_eq!(parse_bitrate("2000000", "test").unwrap(), 2_000_000);
    }

    #[test]
    fn test_parse_resolution() {
        assert_eq!(parse_resolution("1920x1080", "test").unwrap(), (1920, 1080));
        assert_eq!(parse_resolution("1280:720", "test").unwrap(), (1280, 720));
    }

    #[test]
    fn test_parse_frame_rate() {
        assert!((parse_frame_rate("30", "test").unwrap() - 30.0).abs() < 0.001);
        assert!((parse_frame_rate("29.97", "test").unwrap() - 29.97).abs() < 0.001);
        assert!((parse_frame_rate("30000/1001", "test").unwrap() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_parse_time() {
        assert!((parse_time("90", "test").unwrap() - 90.0).abs() < 0.001);
        assert!((parse_time("1:30", "test").unwrap() - 90.0).abs() < 0.001);
        assert!((parse_time("01:30:00", "test").unwrap() - 5400.0).abs() < 0.001);
        assert!((parse_time("1.5", "test").unwrap() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_full_command() {
        let args = vec![
            "-i", "input.mp4",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            "-s", "1920x1080",
            "-r", "30",
            "-ss", "10",
            "-t", "60",
            "output.mp4",
        ];
        let parsed = parse_ffmpeg_args(&args).unwrap();

        assert_eq!(parsed.inputs, vec!["input.mp4"]);
        assert_eq!(parsed.output, Some("output.mp4".to_string()));
        assert_eq!(parsed.video_codec, Some("h264".to_string()));
        assert_eq!(parsed.audio_codec, Some("aac".to_string()));
        assert_eq!(parsed.crf, Some(23));
        assert_eq!(parsed.preset, Some("medium".to_string()));
        assert_eq!(parsed.audio_bitrate, Some(128_000));
        assert_eq!(parsed.width, Some(1920));
        assert_eq!(parsed.height, Some(1080));
        assert_eq!(parsed.frame_rate, Some(30.0));
        assert_eq!(parsed.start_time, Some(10.0));
        assert_eq!(parsed.duration, Some(60.0));
    }

    #[test]
    fn test_convert_to_request() {
        let args = vec![
            "-i", "input.mp4",
            "-c:v", "libx264",
            "-crf", "23",
            "output.mp4",
        ];
        let parsed = parse_ffmpeg_args(&args).unwrap();
        let request = parsed.into_request().unwrap();

        assert_eq!(request.input, "input.mp4");
        assert_eq!(request.output, "output.mp4");
        assert_eq!(request.params.video_codec, Some("h264".to_string()));
        assert_eq!(request.params.quality, Some(23));
    }

    #[test]
    fn test_codec_normalization() {
        assert_eq!(normalize_codec("libx264"), "h264");
        assert_eq!(normalize_codec("libx265"), "h265");
        assert_eq!(normalize_codec("libvpx-vp9"), "vp9");
        assert_eq!(normalize_codec("libaom-av1"), "av1");
        assert_eq!(normalize_codec("libopus"), "opus");
        assert_eq!(normalize_codec("copy"), "copy");
    }
}
