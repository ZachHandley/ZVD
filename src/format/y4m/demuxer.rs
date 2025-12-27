//! Y4M demuxer implementation

use crate::error::{Error, Result};
use crate::format::{Demuxer, DemuxerContext, Packet, Stream, StreamInfo, VideoInfo};
use crate::util::{Buffer, MediaType, Rational};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use y4m::Decoder;

/// Y4M demuxer for reading YUV4MPEG2 files
pub struct Y4mDemuxer {
    decoder: Option<Decoder<BufReader<File>>>,
    context: DemuxerContext,
    frame_number: u64,
    /// Path to the file (needed for seeking/re-opening)
    file_path: Option<PathBuf>,
    /// Byte offset where frame data starts (after Y4M header)
    header_size: u64,
    /// Size of each frame in bytes (based on colorspace)
    frame_data_size: usize,
    /// Size of frame header ("FRAME\n" = 6 bytes typically)
    frame_header_size: usize,
}

impl Y4mDemuxer {
    /// Create a new Y4M demuxer
    pub fn new() -> Self {
        Y4mDemuxer {
            decoder: None,
            context: DemuxerContext::new("y4m".to_string()),
            frame_number: 0,
            file_path: None,
            header_size: 0,
            frame_data_size: 0,
            frame_header_size: 6, // "FRAME\n" is 6 bytes
        }
    }

    /// Convert y4m colorspace to our pixel format string
    fn colorspace_to_pixel_format(colorspace: y4m::Colorspace) -> String {
        match colorspace {
            y4m::Colorspace::C420 | y4m::Colorspace::C420jpeg | y4m::Colorspace::C420paldv => {
                "yuv420p".to_string()
            }
            y4m::Colorspace::C422 => "yuv422p".to_string(),
            y4m::Colorspace::C444 => "yuv444p".to_string(),
            y4m::Colorspace::Cmono => "gray".to_string(),
            _ => format!("{:?}", colorspace).to_lowercase(),
        }
    }

    /// Calculate frame data size in bytes based on colorspace
    fn calculate_frame_size(width: usize, height: usize, colorspace: y4m::Colorspace) -> usize {
        let luma_size = width * height;
        match colorspace {
            // 4:2:0 - chroma is 1/4 size each (half width, half height)
            y4m::Colorspace::C420
            | y4m::Colorspace::C420jpeg
            | y4m::Colorspace::C420paldv
            | y4m::Colorspace::C420mpeg2
            | y4m::Colorspace::C420p10
            | y4m::Colorspace::C420p12 => {
                let bytes_per_sample = match colorspace {
                    y4m::Colorspace::C420p10 | y4m::Colorspace::C420p12 => 2,
                    _ => 1,
                };
                (luma_size + luma_size / 2) * bytes_per_sample
            }
            // 4:2:2 - chroma is 1/2 width, full height
            y4m::Colorspace::C422 | y4m::Colorspace::C422p10 | y4m::Colorspace::C422p12 => {
                let bytes_per_sample = match colorspace {
                    y4m::Colorspace::C422p10 | y4m::Colorspace::C422p12 => 2,
                    _ => 1,
                };
                luma_size * 2 * bytes_per_sample
            }
            // 4:4:4 - full resolution chroma
            y4m::Colorspace::C444 | y4m::Colorspace::C444p10 | y4m::Colorspace::C444p12 => {
                let bytes_per_sample = match colorspace {
                    y4m::Colorspace::C444p10 | y4m::Colorspace::C444p12 => 2,
                    _ => 1,
                };
                luma_size * 3 * bytes_per_sample
            }
            // Monochrome - luma only
            y4m::Colorspace::Cmono | y4m::Colorspace::Cmono12 => {
                let bytes_per_sample = match colorspace {
                    y4m::Colorspace::Cmono12 => 2,
                    _ => 1,
                };
                luma_size * bytes_per_sample
            }
            // Unknown or future colorspaces - estimate as 4:2:0
            _ => luma_size + luma_size / 2,
        }
    }

    /// Calculate the byte offset for a given frame number
    fn frame_byte_offset(&self, frame_number: u64) -> u64 {
        let frame_total_size = (self.frame_header_size + self.frame_data_size) as u64;
        self.header_size + frame_number * frame_total_size
    }

    /// Re-open the file and seek to a specific frame
    fn reopen_and_seek(&mut self, target_frame: u64) -> Result<()> {
        let path = self
            .file_path
            .as_ref()
            .ok_or_else(|| Error::invalid_state("File path not stored"))?
            .clone();

        // Open the file for direct seeking
        let mut file = File::open(&path)
            .map_err(|e| Error::format(format!("Failed to reopen file: {}", e)))?;

        // Calculate target byte offset
        let target_offset = self.frame_byte_offset(target_frame);

        // Seek to the target position
        file.seek(SeekFrom::Start(target_offset))
            .map_err(|e| Error::format(format!("Failed to seek: {}", e)))?;

        // Now we need to re-create the decoder starting from this position
        // The y4m crate expects to read the header first, so we need to re-open
        // and manually position after reading header info

        // Re-open fresh for the y4m decoder
        let file = File::open(&path)
            .map_err(|e| Error::format(format!("Failed to reopen file: {}", e)))?;
        let reader = BufReader::new(file);

        let decoder = y4m::decode(reader)
            .map_err(|e| Error::format(format!("Failed to decode Y4M header: {}", e)))?;

        self.decoder = Some(decoder);

        // Now skip frames until we reach the target
        for _ in 0..target_frame {
            let decoder = self.decoder.as_mut().unwrap();
            decoder.read_frame().map_err(|e| match e {
                y4m::Error::EOF => Error::invalid_input(format!(
                    "Seek target {} is past end of file",
                    target_frame
                )),
                _ => Error::format(format!("Failed to skip frame during seek: {}", e)),
            })?;
        }

        self.frame_number = target_frame;
        Ok(())
    }
}

impl Default for Y4mDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for Y4mDemuxer {
    fn open(&mut self, path: &Path) -> Result<()> {
        // Store the file path for seeking
        self.file_path = Some(path.to_path_buf());

        // First, we need to determine the header size by reading the file manually
        // Y4M header starts with "YUV4MPEG2 " and ends with newline
        let mut file =
            File::open(path).map_err(|e| Error::format(format!("Failed to open file: {}", e)))?;

        // Read the header to find its size
        let mut header_buf = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            file.read_exact(&mut byte)
                .map_err(|e| Error::format(format!("Failed to read Y4M header: {}", e)))?;
            header_buf.push(byte[0]);
            if byte[0] == b'\n' {
                break;
            }
            // Sanity check - header shouldn't be too long
            if header_buf.len() > 1024 {
                return Err(Error::format("Y4M header too long"));
            }
        }
        self.header_size = header_buf.len() as u64;

        // Now re-open with BufReader for the y4m crate
        let file =
            File::open(path).map_err(|e| Error::format(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);

        // Create Y4M decoder
        let decoder = y4m::decode(reader)
            .map_err(|e| Error::format(format!("Failed to decode Y4M header: {}", e)))?;

        // Get video parameters
        let width = decoder.get_width();
        let height = decoder.get_height();
        let framerate = decoder.get_framerate();
        let colorspace = decoder.get_colorspace();

        // Calculate frame data size for seeking
        self.frame_data_size = Self::calculate_frame_size(width, height, colorspace);

        // Determine bits per sample based on colorspace
        let bits_per_sample = match colorspace {
            y4m::Colorspace::C420p10
            | y4m::Colorspace::C422p10
            | y4m::Colorspace::C444p10 => 10,
            y4m::Colorspace::C420p12
            | y4m::Colorspace::C422p12
            | y4m::Colorspace::C444p12
            | y4m::Colorspace::Cmono12 => 12,
            _ => 8,
        };

        // Create stream info for the video stream
        let mut stream_info = StreamInfo::new(0, MediaType::Video, "rawvideo".to_string());

        // Set time base from framerate
        stream_info.time_base = Rational::new(framerate.den as i64, framerate.num as i64);

        // Set video info
        stream_info.video_info = Some(VideoInfo {
            width: width as u32,
            height: height as u32,
            pix_fmt: Self::colorspace_to_pixel_format(colorspace),
            frame_rate: Rational::new(framerate.num as i64, framerate.den as i64),
            aspect_ratio: Rational::new(1, 1), // Default to square pixels
            bits_per_sample,
        });

        // Add stream to context
        let stream = Stream::new(stream_info);
        self.context.add_stream(stream);

        self.decoder = Some(decoder);

        Ok(())
    }

    fn streams(&self) -> &[Stream] {
        self.context.streams()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        // Read next frame
        let frame = decoder.read_frame().map_err(|e| match e {
            y4m::Error::EOF => Error::EndOfStream,
            _ => Error::format(format!("Failed to read frame: {}", e)),
        })?;

        // Convert frame planes to a single buffer
        // Y4M frames are stored as separate planes (Y, U, V)
        let y_plane = frame.get_y_plane();
        let u_plane = frame.get_u_plane();
        let v_plane = frame.get_v_plane();

        // Concatenate planes into a single buffer
        let mut data = Vec::with_capacity(y_plane.len() + u_plane.len() + v_plane.len());
        data.extend_from_slice(y_plane);
        data.extend_from_slice(u_plane);
        data.extend_from_slice(v_plane);

        // Create packet
        let mut packet = Packet::new(0, Buffer::from_vec(data));

        // Set PTS based on frame number and time base
        packet.pts = crate::util::Timestamp::new(self.frame_number as i64);
        packet.dts = crate::util::Timestamp::new(self.frame_number as i64);
        packet.duration = 1; // Each frame has duration of 1 in time_base units

        self.frame_number += 1;

        Ok(packet)
    }

    fn seek(&mut self, _stream_index: usize, timestamp: i64) -> Result<()> {
        // Validate that the demuxer is open
        if self.decoder.is_none() {
            return Err(Error::invalid_state("Demuxer not opened"));
        }

        // Timestamp in Y4M corresponds directly to frame number
        // (since time_base is set to 1/framerate)
        let target_frame = if timestamp < 0 { 0u64 } else { timestamp as u64 };

        // Handle seek to frame 0 specially - just reopen and reset
        if target_frame == 0 {
            return self.reopen_and_seek(0);
        }

        // For forward seeks from current position, we can just skip frames
        if target_frame > self.frame_number {
            let frames_to_skip = target_frame - self.frame_number;

            // If we need to skip many frames, it might be more efficient to reopen
            // But for small seeks, just read and discard
            if frames_to_skip <= 100 {
                let decoder = self.decoder.as_mut().unwrap();
                for _ in 0..frames_to_skip {
                    decoder.read_frame().map_err(|e| match e {
                        y4m::Error::EOF => Error::invalid_input(format!(
                            "Seek target frame {} is past end of file",
                            target_frame
                        )),
                        _ => Error::format(format!("Failed to skip frame during seek: {}", e)),
                    })?;
                }
                self.frame_number = target_frame;
                return Ok(());
            }
        }

        // For backward seeks or large forward seeks, reopen and seek
        self.reopen_and_seek(target_frame)
    }

    fn close(&mut self) -> Result<()> {
        self.decoder = None;
        self.frame_number = 0;
        self.file_path = None;
        self.header_size = 0;
        self.frame_data_size = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_y4m_demuxer_creation() {
        let demuxer = Y4mDemuxer::new();
        assert_eq!(demuxer.context.format_name(), "y4m");
        assert_eq!(demuxer.frame_number, 0);
    }

    #[test]
    fn test_frame_size_calculation() {
        // 4:2:0 - luma + chroma/2 = 1.5 * luma
        assert_eq!(
            Y4mDemuxer::calculate_frame_size(320, 240, y4m::Colorspace::C420),
            320 * 240 * 3 / 2
        );
        assert_eq!(
            Y4mDemuxer::calculate_frame_size(320, 240, y4m::Colorspace::C420jpeg),
            320 * 240 * 3 / 2
        );

        // 4:2:2 - luma + chroma = 2 * luma
        assert_eq!(
            Y4mDemuxer::calculate_frame_size(320, 240, y4m::Colorspace::C422),
            320 * 240 * 2
        );

        // 4:4:4 - full resolution for all planes
        assert_eq!(
            Y4mDemuxer::calculate_frame_size(320, 240, y4m::Colorspace::C444),
            320 * 240 * 3
        );

        // Monochrome - luma only
        assert_eq!(
            Y4mDemuxer::calculate_frame_size(320, 240, y4m::Colorspace::Cmono),
            320 * 240
        );
    }

    /// Create a minimal Y4M file for testing with the specified number of frames
    fn create_test_y4m_file(width: usize, height: usize, num_frames: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");

        // Write Y4M header
        writeln!(file, "YUV4MPEG2 W{} H{} F30:1 Ip C420", width, height)
            .expect("Failed to write header");

        // Calculate frame size for 4:2:0
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let frame_data_size = y_size + 2 * uv_size;

        // Write frames
        for frame_num in 0..num_frames {
            // Write FRAME header
            writeln!(file, "FRAME").expect("Failed to write FRAME header");

            // Write frame data - fill with a pattern based on frame number
            let y_value = ((frame_num * 10) % 256) as u8;
            let u_value = ((frame_num * 20) % 256) as u8;
            let v_value = ((frame_num * 30) % 256) as u8;

            let y_plane: Vec<u8> = vec![y_value; y_size];
            let u_plane: Vec<u8> = vec![u_value; uv_size];
            let v_plane: Vec<u8> = vec![v_value; uv_size];

            file.write_all(&y_plane).expect("Failed to write Y plane");
            file.write_all(&u_plane).expect("Failed to write U plane");
            file.write_all(&v_plane).expect("Failed to write V plane");
        }

        file.flush().expect("Failed to flush file");
        file
    }

    #[test]
    fn test_y4m_seek_to_zero() {
        let temp_file = create_test_y4m_file(8, 8, 5);
        let mut demuxer = Y4mDemuxer::new();

        // Open the file
        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Read a few frames to advance position
        for _ in 0..3 {
            demuxer.read_packet().expect("Failed to read packet");
        }
        assert_eq!(demuxer.frame_number, 3);

        // Seek to frame 0
        demuxer.seek(0, 0).expect("Failed to seek to frame 0");
        assert_eq!(demuxer.frame_number, 0);

        // Read and verify we're at the beginning
        let packet = demuxer.read_packet().expect("Failed to read packet after seek");
        assert_eq!(packet.pts.value, 0);
    }

    #[test]
    fn test_y4m_seek_forward() {
        let temp_file = create_test_y4m_file(8, 8, 10);
        let mut demuxer = Y4mDemuxer::new();

        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Seek forward to frame 5
        demuxer.seek(0, 5).expect("Failed to seek to frame 5");
        assert_eq!(demuxer.frame_number, 5);

        // Read and verify PTS
        let packet = demuxer.read_packet().expect("Failed to read packet after seek");
        assert_eq!(packet.pts.value, 5);
        assert_eq!(demuxer.frame_number, 6);
    }

    #[test]
    fn test_y4m_seek_backward() {
        let temp_file = create_test_y4m_file(8, 8, 10);
        let mut demuxer = Y4mDemuxer::new();

        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Read to frame 7
        for _ in 0..7 {
            demuxer.read_packet().expect("Failed to read packet");
        }
        assert_eq!(demuxer.frame_number, 7);

        // Seek backward to frame 2
        demuxer.seek(0, 2).expect("Failed to seek to frame 2");
        assert_eq!(demuxer.frame_number, 2);

        // Read and verify PTS
        let packet = demuxer.read_packet().expect("Failed to read packet after seek");
        assert_eq!(packet.pts.value, 2);
    }

    #[test]
    fn test_y4m_seek_negative_clamps_to_zero() {
        let temp_file = create_test_y4m_file(8, 8, 5);
        let mut demuxer = Y4mDemuxer::new();

        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Read a few frames
        for _ in 0..3 {
            demuxer.read_packet().expect("Failed to read packet");
        }

        // Seek to negative timestamp - should clamp to 0
        demuxer.seek(0, -10).expect("Failed to seek to negative timestamp");
        assert_eq!(demuxer.frame_number, 0);
    }

    #[test]
    fn test_y4m_seek_past_end_fails() {
        let temp_file = create_test_y4m_file(8, 8, 5);
        let mut demuxer = Y4mDemuxer::new();

        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Seek past the last frame - should fail
        let result = demuxer.seek(0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_y4m_seek_without_open_fails() {
        let mut demuxer = Y4mDemuxer::new();

        // Seek without opening - should fail
        let result = demuxer.seek(0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_y4m_multiple_seeks() {
        let temp_file = create_test_y4m_file(8, 8, 20);
        let mut demuxer = Y4mDemuxer::new();

        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Perform multiple seeks
        let seek_positions = [5, 10, 3, 15, 0, 8];

        for &pos in &seek_positions {
            demuxer
                .seek(0, pos)
                .expect(&format!("Failed to seek to frame {}", pos));
            assert_eq!(demuxer.frame_number, pos as u64);

            let packet = demuxer
                .read_packet()
                .expect(&format!("Failed to read packet at frame {}", pos));
            assert_eq!(packet.pts.value, pos);
        }
    }

    #[test]
    fn test_y4m_frame_data_integrity_after_seek() {
        let temp_file = create_test_y4m_file(8, 8, 10);
        let mut demuxer = Y4mDemuxer::new();

        demuxer
            .open(temp_file.path())
            .expect("Failed to open Y4M file");

        // Read frame 3 directly
        demuxer.seek(0, 3).expect("Failed to seek to frame 3");
        let packet_a = demuxer.read_packet().expect("Failed to read packet");
        let data_a = packet_a.data.as_slice().to_vec();

        // Seek back to frame 0, then read to frame 3
        demuxer.seek(0, 0).expect("Failed to seek to frame 0");
        for _ in 0..3 {
            demuxer.read_packet().expect("Failed to read packet");
        }
        let packet_b = demuxer.read_packet().expect("Failed to read packet");
        let data_b = packet_b.data.as_slice().to_vec();

        // The data should be identical
        assert_eq!(data_a, data_b, "Frame data mismatch after seek");
    }
}
