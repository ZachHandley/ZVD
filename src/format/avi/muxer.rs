//! AVI muxer implementation
//!
//! This module provides complete AVI muxing with RIFF/LIST structure and index generation.

use super::{AviMainHeader, AviStreamHeader, RiffChunk};
use crate::error::{Error, Result};
use crate::format::{Muxer, Packet, Stream};
use crate::util::MediaType;
use std::io::{Seek, SeekFrom, Write};

/// Index entry for idx1 chunk
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Chunk ID (e.g., "00dc", "01wb")
    chunk_id: [u8; 4],
    /// Flags (AVIIF_KEYFRAME = 0x10)
    flags: u32,
    /// Offset from movi start (after LIST/movi header)
    offset: u32,
    /// Size of chunk data (without header)
    size: u32,
}

/// Stream context for muxing
#[derive(Debug, Clone)]
struct MuxerStream {
    /// Stream index
    index: usize,
    /// Two-character stream number
    stream_num: [u8; 2],
    /// Media type
    media_type: MediaType,
    /// Frame count
    frame_count: u32,
    /// Codec FourCC
    fourcc: [u8; 4],
    /// Sample rate (audio) or fps numerator (video)
    rate: u32,
    /// Scale (audio=1, video=fps denominator)
    scale: u32,
    /// Total bytes written for this stream
    total_bytes: u64,
}

/// AVI muxer with seekable writer
pub struct AviMuxer<W: Write + Seek> {
    writer: W,
    main_header: AviMainHeader,
    stream_headers: Vec<AviStreamHeader>,
    streams: Vec<Stream>,
    muxer_streams: Vec<MuxerStream>,
    frame_count: u32,
    started: bool,
    /// Index entries for idx1 chunk
    index: Vec<IndexEntry>,
    /// Position of RIFF size field
    riff_size_pos: u64,
    /// Position of movi LIST size field
    movi_size_pos: u64,
    /// Start of movi data (for index offsets)
    movi_data_start: u64,
    /// Current write position in movi
    movi_current_offset: u32,
    /// Position of avih total_frames field
    avih_frames_pos: u64,
    /// Positions of strh length fields
    strh_length_positions: Vec<u64>,
}

impl<W: Write + Seek> AviMuxer<W> {
    /// Create a new AVI muxer
    pub fn new(writer: W, width: u32, height: u32, fps: u32) -> Self {
        AviMuxer {
            writer,
            main_header: AviMainHeader::new(width, height, fps),
            stream_headers: Vec::new(),
            streams: Vec::new(),
            muxer_streams: Vec::new(),
            frame_count: 0,
            started: false,
            index: Vec::new(),
            riff_size_pos: 0,
            movi_size_pos: 0,
            movi_data_start: 0,
            movi_current_offset: 0,
            avih_frames_pos: 0,
            strh_length_positions: Vec::new(),
        }
    }

    /// Write 4-byte FourCC
    fn write_fourcc(&mut self, fourcc: &[u8; 4]) -> Result<()> {
        self.writer.write_all(fourcc).map_err(Error::Io)?;
        Ok(())
    }

    /// Write 32-bit little-endian value
    fn write_u32(&mut self, value: u32) -> Result<()> {
        self.writer
            .write_all(&value.to_le_bytes())
            .map_err(Error::Io)?;
        Ok(())
    }

    /// Write 16-bit little-endian value
    fn write_u16(&mut self, value: u16) -> Result<()> {
        self.writer
            .write_all(&value.to_le_bytes())
            .map_err(Error::Io)?;
        Ok(())
    }

    /// Get current position
    fn position(&mut self) -> Result<u64> {
        self.writer.stream_position().map_err(Error::Io)
    }

    /// Seek to position
    fn seek_to(&mut self, pos: u64) -> Result<()> {
        self.writer.seek(SeekFrom::Start(pos)).map_err(Error::Io)?;
        Ok(())
    }

    /// Write chunk header
    fn write_chunk_header(&mut self, fourcc: &[u8; 4], size: u32) -> Result<()> {
        self.write_fourcc(fourcc)?;
        self.write_u32(size)?;
        Ok(())
    }

    /// Get codec FourCC from stream info
    fn get_fourcc(codec_id: &str) -> [u8; 4] {
        let codec_lower = codec_id.to_lowercase();
        match codec_lower.as_str() {
            "h264" | "avc" | "avc1" => *b"H264",
            "hevc" | "h265" | "hvc1" => *b"HEVC",
            "mjpeg" | "mjpg" => *b"MJPG",
            "mpeg4" | "mp4v" => *b"XVID",
            "mpeg2video" | "mpg2" => *b"MPG2",
            "prores" => *b"apcn",
            _ => {
                let mut fourcc = [0u8; 4];
                let bytes = codec_id.as_bytes();
                for (i, byte) in bytes.iter().take(4).enumerate() {
                    fourcc[i] = *byte;
                }
                fourcc
            }
        }
    }

    /// Get audio format tag from codec ID
    fn get_audio_format_tag(codec_id: &str) -> u16 {
        let codec_lower = codec_id.to_lowercase();
        match codec_lower.as_str() {
            "pcm_s16le" | "pcm" => 0x0001, // WAVE_FORMAT_PCM
            "pcm_f32le" => 0x0003,         // WAVE_FORMAT_IEEE_FLOAT
            "mp3" | "mp2" => 0x0055,       // WAVE_FORMAT_MPEGLAYER3
            "aac" => 0x00FF,               // WAVE_FORMAT_AAC
            "ac3" => 0x2000,               // WAVE_FORMAT_DVM
            _ => 0x0001,                   // Default to PCM
        }
    }

    /// Write main header (avih)
    fn write_main_header(&mut self) -> Result<()> {
        self.write_chunk_header(b"avih", 56)?;

        self.write_u32(self.main_header.microsec_per_frame)?;
        self.write_u32(self.main_header.max_bytes_per_sec)?;
        self.write_u32(0)?; // padding granularity
        self.write_u32(self.main_header.flags)?;

        // Save position for total_frames update
        self.avih_frames_pos = self.position()?;
        self.write_u32(0)?; // total_frames - will update

        self.write_u32(self.main_header.initial_frames)?;
        self.write_u32(self.muxer_streams.len() as u32)?; // streams
        self.write_u32(self.main_header.suggested_buffer_size)?;
        self.write_u32(self.main_header.width)?;
        self.write_u32(self.main_header.height)?;

        // Reserved
        self.write_u32(0)?;
        self.write_u32(0)?;
        self.write_u32(0)?;
        self.write_u32(0)?;

        Ok(())
    }

    /// Write stream header (strh) and format (strf) in strl LIST
    fn write_stream_list(&mut self, stream_idx: usize) -> Result<()> {
        // Extract all needed data upfront to avoid borrow conflicts
        let media_type = self.muxer_streams[stream_idx].media_type;
        let fourcc = self.muxer_streams[stream_idx].fourcc;
        let scale = self.muxer_streams[stream_idx].scale;
        let rate = self.muxer_streams[stream_idx].rate;
        let main_width = self.main_header.width;
        let main_height = self.main_header.height;

        // Extract video/audio info
        let (video_width, video_height) = self.streams[stream_idx]
            .info
            .video_info
            .as_ref()
            .map(|v| (v.width, v.height))
            .unwrap_or((main_width, main_height));

        let (channels, sample_rate_audio, bits_per_sample) = self.streams[stream_idx]
            .info
            .audio_info
            .as_ref()
            .map(|a| (a.channels, a.sample_rate, a.bits_per_sample as u16))
            .unwrap_or((2, 48000, 16));

        let codec_id = self.streams[stream_idx].info.codec_id.clone();

        // Calculate strl LIST size
        let strh_size = 56u32;
        let strf_size = match media_type {
            MediaType::Video => 40u32, // BITMAPINFOHEADER
            MediaType::Audio => 18u32, // WAVEFORMATEX
            _ => 0,
        };
        let strl_size = 4 + 8 + strh_size + 8 + strf_size;

        // LIST strl header
        self.write_chunk_header(b"LIST", strl_size)?;
        self.write_fourcc(b"strl")?;

        // strh chunk
        self.write_chunk_header(b"strh", strh_size)?;

        match media_type {
            MediaType::Video => {
                self.write_fourcc(b"vids")?;
                self.write_fourcc(&fourcc)?;
            }
            MediaType::Audio => {
                self.write_fourcc(b"auds")?;
                self.write_u32(0)?; // No codec for audio header
            }
            _ => {
                self.write_fourcc(b"data")?;
                self.write_u32(0)?;
            }
        }

        self.write_u32(0)?; // flags
        self.write_u16(0)?; // priority
        self.write_u16(0)?; // language
        self.write_u32(0)?; // initial_frames
        self.write_u32(scale)?; // scale
        self.write_u32(rate)?; // rate
        self.write_u32(0)?; // start

        // Save position for length update
        let pos = self.position()?;
        self.strh_length_positions.push(pos);
        self.write_u32(0)?; // length - will update

        self.write_u32(1024 * 1024)?; // suggested_buffer_size
        self.write_u32(10000)?; // quality
        self.write_u32(0)?; // sample_size

        // rcFrame
        self.write_u16(0)?;
        self.write_u16(0)?;
        self.write_u16(main_width as u16)?;
        self.write_u16(main_height as u16)?;

        // strf chunk
        self.write_chunk_header(b"strf", strf_size)?;

        match media_type {
            MediaType::Video => {
                // BITMAPINFOHEADER
                self.write_u32(40)?; // biSize
                self.write_u32(video_width)?; // biWidth
                self.write_u32(video_height)?; // biHeight (positive = bottom-up)
                self.write_u16(1)?; // biPlanes
                self.write_u16(24)?; // biBitCount
                self.write_fourcc(&fourcc)?; // biCompression
                self.write_u32(video_width * video_height * 3)?; // biSizeImage
                self.write_u32(0)?; // biXPelsPerMeter
                self.write_u32(0)?; // biYPelsPerMeter
                self.write_u32(0)?; // biClrUsed
                self.write_u32(0)?; // biClrImportant
            }
            MediaType::Audio => {
                // WAVEFORMATEX
                let block_align = channels * (bits_per_sample / 8);
                let avg_bytes_per_sec = sample_rate_audio * block_align as u32;

                let format_tag = Self::get_audio_format_tag(&codec_id);

                self.write_u16(format_tag)?; // wFormatTag
                self.write_u16(channels)?; // nChannels
                self.write_u32(sample_rate_audio)?; // nSamplesPerSec
                self.write_u32(avg_bytes_per_sec)?; // nAvgBytesPerSec
                self.write_u16(block_align)?; // nBlockAlign
                self.write_u16(bits_per_sample)?; // wBitsPerSample
                self.write_u16(0)?; // cbSize
            }
            _ => {}
        }

        Ok(())
    }

    /// Write AVI headers
    fn write_headers(&mut self) -> Result<()> {
        // RIFF header
        self.write_fourcc(b"RIFF")?;
        self.riff_size_pos = self.position()?;
        self.write_u32(0)?; // Size placeholder
        self.write_fourcc(b"AVI ")?;

        // Calculate hdrl LIST size
        let avih_chunk_size = 8 + 56;
        let mut strl_total_size = 0u32;
        for stream in &self.muxer_streams {
            let strf_size = match stream.media_type {
                MediaType::Video => 40,
                MediaType::Audio => 18,
                _ => 0,
            };
            strl_total_size += 12 + 8 + 56 + 8 + strf_size; // LIST + strl + strh + strf
        }
        let hdrl_size = 4 + avih_chunk_size + strl_total_size;

        // LIST hdrl
        self.write_chunk_header(b"LIST", hdrl_size)?;
        self.write_fourcc(b"hdrl")?;

        // avih
        self.write_main_header()?;

        // strl for each stream
        for i in 0..self.muxer_streams.len() {
            self.write_stream_list(i)?;
        }

        // LIST movi
        self.write_chunk_header(b"LIST", 0)?; // Size placeholder
        self.movi_size_pos = self.position()? - 4;
        self.write_fourcc(b"movi")?;
        self.movi_data_start = self.position()?;

        Ok(())
    }

    /// Write index chunk (idx1)
    fn write_index(&mut self) -> Result<()> {
        if self.index.is_empty() {
            return Ok(());
        }

        let idx_size = (self.index.len() * 16) as u32;
        self.write_chunk_header(b"idx1", idx_size)?;

        // Copy index entries to avoid borrow conflict
        let entries: Vec<_> = self.index.iter().map(|e| {
            (e.chunk_id, e.flags, e.offset, e.size)
        }).collect();

        for (chunk_id, flags, offset, size) in entries {
            self.write_fourcc(&chunk_id)?;
            self.write_u32(flags)?;
            self.write_u32(offset)?;
            self.write_u32(size)?;
        }

        Ok(())
    }

    /// Update sizes in header
    fn update_sizes(&mut self) -> Result<()> {
        let end_pos = self.position()?;

        // Update RIFF size
        let riff_size = (end_pos - 8) as u32;
        self.seek_to(self.riff_size_pos)?;
        self.write_u32(riff_size)?;

        // Update movi LIST size
        let movi_size = (self.movi_data_start - self.movi_size_pos - 4 + self.movi_current_offset as u64) as u32;
        self.seek_to(self.movi_size_pos)?;
        self.write_u32(movi_size)?;

        // Update total frames in avih
        self.seek_to(self.avih_frames_pos)?;
        self.write_u32(self.frame_count)?;

        // Collect stream length update info to avoid borrow conflicts
        let updates: Vec<_> = self.strh_length_positions.iter()
            .enumerate()
            .filter(|(i, _)| *i < self.muxer_streams.len())
            .map(|(i, &pos)| (pos, self.muxer_streams[i].frame_count))
            .collect();

        for (pos, frame_count) in updates {
            self.seek_to(pos)?;
            self.write_u32(frame_count)?;
        }

        // Seek back to end
        self.seek_to(end_pos)?;

        Ok(())
    }

    /// Get chunk ID for stream
    fn get_chunk_id(stream_idx: usize, media_type: MediaType) -> [u8; 4] {
        let num1 = b'0' + (stream_idx / 10) as u8;
        let num2 = b'0' + (stream_idx % 10) as u8;

        match media_type {
            MediaType::Video => [num1, num2, b'd', b'c'], // compressed video
            MediaType::Audio => [num1, num2, b'w', b'b'], // audio
            _ => [num1, num2, b'd', b'd'],                // data
        }
    }

    /// Add video stream
    pub fn add_video_stream(&mut self, codec_fourcc: [u8; 4]) {
        let header = AviStreamHeader::video(
            codec_fourcc,
            self.main_header.width,
            self.main_header.height,
            1_000_000 / self.main_header.microsec_per_frame,
        );
        self.stream_headers.push(header);
    }

    /// Add audio stream
    pub fn add_audio_stream(&mut self, sample_rate: u32, channels: u16) {
        let header = AviStreamHeader::audio(1, sample_rate, channels);
        self.stream_headers.push(header);
    }
}

impl<W: Write + Seek> Muxer for AviMuxer<W> {
    fn create(&mut self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        if self.started {
            return Err(Error::invalid_state("Cannot add stream after header written"));
        }

        let index = self.streams.len();
        let stream_num = [b'0' + (index / 10) as u8, b'0' + (index % 10) as u8];

        let (fourcc, rate, scale) = match stream.info.media_type {
            MediaType::Video => {
                let fourcc = Self::get_fourcc(&stream.info.codec_id);
                let (rate, scale) = if let Some(ref vi) = stream.info.video_info {
                    (vi.frame_rate.num.abs() as u32, vi.frame_rate.den.abs() as u32)
                } else {
                    (30, 1)
                };
                // Update main header with video dimensions
                if let Some(ref vi) = stream.info.video_info {
                    self.main_header.width = vi.width;
                    self.main_header.height = vi.height;
                    if scale > 0 {
                        self.main_header.microsec_per_frame = (1_000_000 * scale) / rate;
                    }
                }
                (fourcc, rate, scale)
            }
            MediaType::Audio => {
                let rate = stream.info.audio_info.as_ref().map(|a| a.sample_rate).unwrap_or(48000);
                ([0u8; 4], rate, 1)
            }
            _ => ([0u8; 4], 1, 1),
        };

        let muxer_stream = MuxerStream {
            index,
            stream_num,
            media_type: stream.info.media_type,
            frame_count: 0,
            fourcc,
            rate,
            scale,
            total_bytes: 0,
        };

        self.muxer_streams.push(muxer_stream);
        self.streams.push(stream);

        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        if self.started {
            return Err(Error::invalid_state("Header already written"));
        }

        if self.muxer_streams.is_empty() {
            return Err(Error::invalid_state("No streams added"));
        }

        self.write_headers()?;
        self.started = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.started {
            return Err(Error::invalid_state("Header not written"));
        }

        if packet.stream_index >= self.muxer_streams.len() {
            return Err(Error::invalid_input("Invalid stream index"));
        }

        let media_type = self.muxer_streams[packet.stream_index].media_type;
        let chunk_id = Self::get_chunk_id(packet.stream_index, media_type);
        let data = packet.data.as_slice();
        let data_size = data.len() as u32;

        // Create index entry
        let flags = if packet.flags.keyframe { 0x10 } else { 0 };
        let index_entry = IndexEntry {
            chunk_id,
            flags,
            offset: self.movi_current_offset,
            size: data_size,
        };
        self.index.push(index_entry);

        // Write chunk
        self.write_chunk_header(&chunk_id, data_size)?;
        self.writer.write_all(data).map_err(Error::Io)?;

        // Pad to word boundary
        if data_size % 2 != 0 {
            self.writer.write_all(&[0]).map_err(Error::Io)?;
            self.movi_current_offset += 1;
        }

        self.movi_current_offset += 8 + data_size; // header + data

        // Update stream stats
        self.muxer_streams[packet.stream_index].frame_count += 1;
        self.muxer_streams[packet.stream_index].total_bytes += data_size as u64;

        // Update video frame count
        if media_type == MediaType::Video {
            self.frame_count += 1;
        }

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }

        // Write index
        self.write_index()?;

        // Update all size fields
        self.update_sizes()?;

        // Flush
        self.writer.flush().map_err(Error::Io)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_avi_muxer_creation() {
        let buffer = Cursor::new(Vec::new());
        let muxer = AviMuxer::new(buffer, 1920, 1080, 30);
        assert!(!muxer.started);
        assert_eq!(muxer.frame_count, 0);
    }

    #[test]
    fn test_get_fourcc() {
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_fourcc("h264"), *b"H264");
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_fourcc("mjpeg"), *b"MJPG");
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_fourcc("hevc"), *b"HEVC");
    }

    #[test]
    fn test_get_audio_format_tag() {
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_audio_format_tag("pcm_s16le"), 0x0001);
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_audio_format_tag("mp3"), 0x0055);
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_audio_format_tag("aac"), 0x00FF);
    }

    #[test]
    fn test_get_chunk_id() {
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_chunk_id(0, MediaType::Video), *b"00dc");
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_chunk_id(0, MediaType::Audio), *b"00wb");
        assert_eq!(AviMuxer::<Cursor<Vec<u8>>>::get_chunk_id(1, MediaType::Video), *b"01dc");
    }

    #[test]
    fn test_avi_muxer_write_header() {
        use crate::format::{StreamInfo, VideoInfo};

        let buffer = Cursor::new(Vec::new());
        let mut muxer = AviMuxer::new(buffer, 1920, 1080, 30);

        // Add a video stream
        let mut info = StreamInfo::new(0, MediaType::Video, "h264".to_string());
        info.video_info = Some(VideoInfo::new(1920, 1080));
        let stream = Stream::new(info);
        muxer.add_stream(stream).unwrap();

        let result = muxer.write_header();
        assert!(result.is_ok());
        assert!(muxer.started);
    }
}
