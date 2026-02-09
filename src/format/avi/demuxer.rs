//! AVI demuxer implementation
//!
//! This module provides complete AVI demuxing with RIFF/LIST chunk parsing.

use super::{AviMainHeader, AviStreamHeader, RiffChunk};
use crate::error::{Error, Result};
use crate::format::{AudioInfo, Demuxer, Packet, PacketFlags, Stream, StreamInfo, VideoInfo};
use crate::util::{Buffer, MediaType, Rational, Timestamp};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

/// BITMAPINFOHEADER structure
#[derive(Debug, Clone)]
struct BitmapInfoHeader {
    size: u32,
    width: i32,
    height: i32,
    planes: u16,
    bit_count: u16,
    compression: [u8; 4],
    size_image: u32,
    x_pels_per_meter: i32,
    y_pels_per_meter: i32,
    clr_used: u32,
    clr_important: u32,
}

impl BitmapInfoHeader {
    fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 40 {
            return Err(Error::invalid_input("BITMAPINFOHEADER too small"));
        }

        Ok(BitmapInfoHeader {
            size: u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
            width: i32::from_le_bytes([data[4], data[5], data[6], data[7]]),
            height: i32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            planes: u16::from_le_bytes([data[12], data[13]]),
            bit_count: u16::from_le_bytes([data[14], data[15]]),
            compression: [data[16], data[17], data[18], data[19]],
            size_image: u32::from_le_bytes([data[20], data[21], data[22], data[23]]),
            x_pels_per_meter: i32::from_le_bytes([data[24], data[25], data[26], data[27]]),
            y_pels_per_meter: i32::from_le_bytes([data[28], data[29], data[30], data[31]]),
            clr_used: u32::from_le_bytes([data[32], data[33], data[34], data[35]]),
            clr_important: u32::from_le_bytes([data[36], data[37], data[38], data[39]]),
        })
    }
}

/// WAVEFORMATEX structure
#[derive(Debug, Clone)]
struct WaveFormatEx {
    format_tag: u16,
    channels: u16,
    samples_per_sec: u32,
    avg_bytes_per_sec: u32,
    block_align: u16,
    bits_per_sample: u16,
}

impl WaveFormatEx {
    fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(Error::invalid_input("WAVEFORMATEX too small"));
        }

        Ok(WaveFormatEx {
            format_tag: u16::from_le_bytes([data[0], data[1]]),
            channels: u16::from_le_bytes([data[2], data[3]]),
            samples_per_sec: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
            avg_bytes_per_sec: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            block_align: u16::from_le_bytes([data[12], data[13]]),
            bits_per_sample: u16::from_le_bytes([data[14], data[15]]),
        })
    }
}

/// Index entry from idx1 chunk
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Chunk ID (e.g., "00dc", "01wb")
    chunk_id: [u8; 4],
    /// Flags (AVIIF_KEYFRAME = 0x10)
    flags: u32,
    /// Offset from movi start
    offset: u32,
    /// Size of chunk data
    size: u32,
}

/// Stream context for demuxing
#[derive(Debug)]
struct StreamContext {
    /// Stream index
    index: usize,
    /// Stream type (video/audio)
    media_type: MediaType,
    /// Two-character stream number ("00", "01", etc.)
    stream_num: [u8; 2],
    /// Current frame/sample number
    current_frame: u64,
    /// Frames per second (for video)
    fps: Rational,
    /// Samples per second (for audio)
    sample_rate: u32,
    /// Block align for audio
    block_align: u16,
}

/// AVI demuxer
pub struct AviDemuxer<R: Read + Seek> {
    reader: R,
    main_header: Option<AviMainHeader>,
    stream_headers: Vec<AviStreamHeader>,
    streams: Vec<Stream>,
    stream_contexts: Vec<StreamContext>,
    current_frame: u64,
    /// Start of movi data
    movi_offset: u64,
    /// End of movi data
    movi_end: u64,
    /// Current read position in movi
    current_offset: u64,
    /// Index entries
    index: Vec<IndexEntry>,
    /// Current index position
    index_pos: usize,
    /// Whether we're using index for reading
    use_index: bool,
}

impl<R: Read + Seek> AviDemuxer<R> {
    /// Create a new AVI demuxer
    pub fn new(reader: R) -> Self {
        AviDemuxer {
            reader,
            main_header: None,
            stream_headers: Vec::new(),
            streams: Vec::new(),
            stream_contexts: Vec::new(),
            current_frame: 0,
            movi_offset: 0,
            movi_end: 0,
            current_offset: 0,
            index: Vec::new(),
            index_pos: 0,
            use_index: false,
        }
    }

    /// Read RIFF chunk header
    fn read_chunk_header(&mut self) -> Result<RiffChunk> {
        let mut fourcc = [0u8; 4];
        let mut size_buf = [0u8; 4];

        self.reader
            .read_exact(&mut fourcc)
            .map_err(|e| Error::Io(e))?;
        self.reader
            .read_exact(&mut size_buf)
            .map_err(|e| Error::Io(e))?;

        let size = u32::from_le_bytes(size_buf);

        Ok(RiffChunk { fourcc, size })
    }

    /// Read exact bytes
    fn read_bytes(&mut self, size: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0u8; size];
        self.reader
            .read_exact(&mut buffer)
            .map_err(|e| Error::Io(e))?;
        Ok(buffer)
    }

    /// Skip bytes
    fn skip_bytes(&mut self, size: u64) -> Result<()> {
        self.reader
            .seek(SeekFrom::Current(size as i64))
            .map_err(|e| Error::Io(e))?;
        Ok(())
    }

    /// Parse AVI main header (avih)
    fn parse_main_header(&mut self, size: u32) -> Result<AviMainHeader> {
        if size < 56 {
            return Err(Error::invalid_input("avih chunk too small"));
        }

        let data = self.read_bytes(size as usize)?;

        let header = AviMainHeader {
            microsec_per_frame: u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
            max_bytes_per_sec: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
            flags: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            total_frames: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
            initial_frames: u32::from_le_bytes([data[20], data[21], data[22], data[23]]),
            streams: u32::from_le_bytes([data[24], data[25], data[26], data[27]]),
            suggested_buffer_size: u32::from_le_bytes([data[28], data[29], data[30], data[31]]),
            width: u32::from_le_bytes([data[32], data[33], data[34], data[35]]),
            height: u32::from_le_bytes([data[36], data[37], data[38], data[39]]),
        };

        Ok(header)
    }

    /// Parse stream header (strh)
    fn parse_stream_header(&mut self, size: u32) -> Result<AviStreamHeader> {
        if size < 56 {
            return Err(Error::invalid_input("strh chunk too small"));
        }

        let data = self.read_bytes(size as usize)?;

        let mut fcc_type = [0u8; 4];
        let mut fcc_handler = [0u8; 4];
        fcc_type.copy_from_slice(&data[0..4]);
        fcc_handler.copy_from_slice(&data[4..8]);

        let header = AviStreamHeader {
            fcc_type,
            fcc_handler,
            flags: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            priority: u16::from_le_bytes([data[12], data[13]]),
            language: u16::from_le_bytes([data[14], data[15]]),
            initial_frames: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
            scale: u32::from_le_bytes([data[20], data[21], data[22], data[23]]),
            rate: u32::from_le_bytes([data[24], data[25], data[26], data[27]]),
            start: u32::from_le_bytes([data[28], data[29], data[30], data[31]]),
            length: u32::from_le_bytes([data[32], data[33], data[34], data[35]]),
            suggested_buffer_size: u32::from_le_bytes([data[36], data[37], data[38], data[39]]),
            quality: u32::from_le_bytes([data[40], data[41], data[42], data[43]]),
            sample_size: u32::from_le_bytes([data[44], data[45], data[46], data[47]]),
        };

        Ok(header)
    }

    /// Get codec ID from FourCC
    fn fourcc_to_codec_id(fourcc: &[u8; 4]) -> String {
        let fourcc_str = String::from_utf8_lossy(fourcc).to_lowercase();

        match fourcc_str.as_str() {
            "h264" | "x264" | "avc1" => "h264".to_string(),
            "hevc" | "hvc1" | "h265" => "hevc".to_string(),
            "mjpg" | "mjpa" | "mjpb" => "mjpeg".to_string(),
            "xvid" | "divx" | "dx50" | "fmp4" => "mpeg4".to_string(),
            "mp4v" | "mp4s" => "mpeg4".to_string(),
            "mpeg" | "mpg1" | "mpg2" => "mpeg2video".to_string(),
            "apcn" | "apcs" | "apch" | "ap4h" | "ap4x" => "prores".to_string(),
            _ => fourcc_str,
        }
    }

    /// Get audio codec ID from format tag
    fn format_tag_to_codec_id(tag: u16) -> String {
        match tag {
            0x0001 => "pcm_s16le".to_string(), // WAVE_FORMAT_PCM
            0x0003 => "pcm_f32le".to_string(), // WAVE_FORMAT_IEEE_FLOAT
            0x0055 => "mp3".to_string(),       // WAVE_FORMAT_MPEGLAYER3
            0x00FF => "aac".to_string(),       // WAVE_FORMAT_AAC
            0x2000 => "ac3".to_string(),       // WAVE_FORMAT_DVM (AC-3)
            0xFFFE => "pcm".to_string(),       // WAVE_FORMAT_EXTENSIBLE
            _ => format!("audio_0x{:04X}", tag),
        }
    }

    /// Parse stream list (strl)
    fn parse_stream_list(&mut self, list_size: u32) -> Result<()> {
        let end_pos = self
            .reader
            .stream_position()
            .map_err(|e| Error::Io(e))?
            + list_size as u64;

        let mut strh: Option<AviStreamHeader> = None;
        let mut video_format: Option<BitmapInfoHeader> = None;
        let mut audio_format: Option<WaveFormatEx> = None;
        let mut extradata: Option<Vec<u8>> = None;

        while self
            .reader
            .stream_position()
            .map_err(|e| Error::Io(e))?
            < end_pos
        {
            let chunk = self.read_chunk_header()?;
            let chunk_end = self
                .reader
                .stream_position()
                .map_err(|e| Error::Io(e))?
                + chunk.size as u64;

            match &chunk.fourcc {
                b"strh" => {
                    strh = Some(self.parse_stream_header(chunk.size)?);
                }
                b"strf" => {
                    if let Some(ref header) = strh {
                        if &header.fcc_type == b"vids" {
                            video_format = Some(BitmapInfoHeader::from_bytes(
                                &self.read_bytes(chunk.size as usize)?,
                            )?);
                            // Check for extradata after BITMAPINFOHEADER (40 bytes)
                            if chunk.size > 40 {
                                let extra_size = chunk.size - 40;
                                // Already read, so seek back and read just extradata
                                self.reader
                                    .seek(SeekFrom::Current(-(chunk.size as i64 - 40)))
                                    .map_err(|e| Error::Io(e))?;
                                extradata = Some(self.read_bytes(extra_size as usize)?);
                            }
                        } else if &header.fcc_type == b"auds" {
                            audio_format =
                                Some(WaveFormatEx::from_bytes(&self.read_bytes(chunk.size as usize)?)?);
                        } else {
                            self.skip_bytes(chunk.size as u64)?;
                        }
                    } else {
                        self.skip_bytes(chunk.size as u64)?;
                    }
                }
                _ => {
                    // Skip unknown chunks
                    self.skip_bytes(chunk.size as u64)?;
                }
            }

            // Align to word boundary
            let current = self
                .reader
                .stream_position()
                .map_err(|e| Error::Io(e))?;
            if current < chunk_end {
                self.skip_bytes(chunk_end - current)?;
            }
            if chunk.size % 2 != 0 {
                self.skip_bytes(1)?;
            }
        }

        // Create stream from parsed data
        if let Some(header) = strh {
            let stream_index = self.streams.len();
            let stream_num = [
                b'0' + (stream_index / 10) as u8,
                b'0' + (stream_index % 10) as u8,
            ];

            if &header.fcc_type == b"vids" {
                let codec_id = Self::fourcc_to_codec_id(&header.fcc_handler);
                let mut info = StreamInfo::new(stream_index, MediaType::Video, codec_id);

                let fps = if header.scale > 0 {
                    Rational::new(header.rate as i64, header.scale as i64)
                } else {
                    Rational::new(25, 1)
                };

                info.time_base = Rational::new(header.scale as i64, header.rate as i64);
                info.duration = header.length as i64;

                if let Some(vf) = video_format {
                    let mut video_info = VideoInfo::new(vf.width.abs() as u32, vf.height.abs() as u32);
                    video_info.frame_rate = fps;
                    info.video_info = Some(video_info);
                }

                let mut stream = Stream::new(info);
                stream.extradata = extradata;

                self.streams.push(stream);
                self.stream_headers.push(header.clone());
                self.stream_contexts.push(StreamContext {
                    index: stream_index,
                    media_type: MediaType::Video,
                    stream_num,
                    current_frame: 0,
                    fps,
                    sample_rate: 0,
                    block_align: 0,
                });
            } else if &header.fcc_type == b"auds" {
                if let Some(af) = audio_format {
                    let codec_id = Self::format_tag_to_codec_id(af.format_tag);
                    let mut info = StreamInfo::new(stream_index, MediaType::Audio, codec_id);

                    info.time_base = Rational::new(1, af.samples_per_sec as i64);

                    let mut audio_info = AudioInfo::new(af.samples_per_sec, af.channels);
                    audio_info.bits_per_sample = af.bits_per_sample as u8;
                    audio_info.bit_rate = Some((af.avg_bytes_per_sec * 8) as u64);
                    info.audio_info = Some(audio_info);

                    let stream = Stream::new(info);
                    self.streams.push(stream);
                    self.stream_headers.push(header);
                    self.stream_contexts.push(StreamContext {
                        index: stream_index,
                        media_type: MediaType::Audio,
                        stream_num,
                        current_frame: 0,
                        fps: Rational::new(1, 1),
                        sample_rate: af.samples_per_sec,
                        block_align: af.block_align,
                    });
                }
            }
        }

        Ok(())
    }

    /// Parse index chunk (idx1)
    fn parse_index(&mut self, size: u32) -> Result<()> {
        let num_entries = size / 16;
        self.index.clear();
        self.index.reserve(num_entries as usize);

        for _ in 0..num_entries {
            let data = self.read_bytes(16)?;
            let mut chunk_id = [0u8; 4];
            chunk_id.copy_from_slice(&data[0..4]);

            let entry = IndexEntry {
                chunk_id,
                flags: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                offset: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
                size: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            };
            self.index.push(entry);
        }

        self.use_index = !self.index.is_empty();
        Ok(())
    }

    /// Parse AVI headers
    fn parse_headers(&mut self) -> Result<()> {
        // Read RIFF header
        let riff = self.read_chunk_header()?;
        if &riff.fourcc != b"RIFF" {
            return Err(Error::invalid_input("Not a valid RIFF file"));
        }

        // Read AVI type
        let mut avi_type = [0u8; 4];
        self.reader
            .read_exact(&mut avi_type)
            .map_err(|e| Error::Io(e))?;
        if &avi_type != b"AVI " {
            return Err(Error::invalid_input("Not a valid AVI file"));
        }

        let file_end = 8 + riff.size as u64;

        // Parse chunks
        while self
            .reader
            .stream_position()
            .map_err(|e| Error::Io(e))?
            < file_end
        {
            let chunk = self.read_chunk_header()?;
            let chunk_start = self
                .reader
                .stream_position()
                .map_err(|e| Error::Io(e))?;

            match &chunk.fourcc {
                b"LIST" => {
                    let mut list_type = [0u8; 4];
                    self.reader
                        .read_exact(&mut list_type)
                        .map_err(|e| Error::Io(e))?;

                    match &list_type {
                        b"hdrl" => {
                            // Header list - parse contained chunks
                            let list_end = chunk_start + chunk.size as u64;
                            while self
                                .reader
                                .stream_position()
                                .map_err(|e| Error::Io(e))?
                                < list_end
                            {
                                let inner_chunk = self.read_chunk_header()?;

                                match &inner_chunk.fourcc {
                                    b"avih" => {
                                        self.main_header =
                                            Some(self.parse_main_header(inner_chunk.size)?);
                                    }
                                    b"LIST" => {
                                        let mut inner_list_type = [0u8; 4];
                                        self.reader
                                            .read_exact(&mut inner_list_type)
                                            .map_err(|e| Error::Io(e))?;
                                        if &inner_list_type == b"strl" {
                                            self.parse_stream_list(inner_chunk.size - 4)?;
                                        } else {
                                            self.skip_bytes((inner_chunk.size - 4) as u64)?;
                                        }
                                    }
                                    _ => {
                                        self.skip_bytes(inner_chunk.size as u64)?;
                                    }
                                }

                                // Word align
                                if inner_chunk.size % 2 != 0 {
                                    self.skip_bytes(1)?;
                                }
                            }
                        }
                        b"movi" => {
                            // Movie data - record position
                            self.movi_offset = self
                                .reader
                                .stream_position()
                                .map_err(|e| Error::Io(e))?;
                            self.movi_end = chunk_start + chunk.size as u64;
                            self.current_offset = self.movi_offset;
                            // Skip movi content for now
                            self.skip_bytes((chunk.size - 4) as u64)?;
                        }
                        _ => {
                            self.skip_bytes((chunk.size - 4) as u64)?;
                        }
                    }
                }
                b"idx1" => {
                    self.parse_index(chunk.size)?;
                }
                _ => {
                    self.skip_bytes(chunk.size as u64)?;
                }
            }

            // Word align
            if chunk.size % 2 != 0 {
                self.skip_bytes(1)?;
            }
        }

        // Seek back to movi start for reading
        if self.movi_offset > 0 {
            self.reader
                .seek(SeekFrom::Start(self.movi_offset))
                .map_err(|e| Error::Io(e))?;
        }

        Ok(())
    }

    /// Find stream context by chunk ID prefix
    fn find_stream_by_chunk_id(&self, chunk_id: &[u8; 4]) -> Option<usize> {
        let stream_num = [chunk_id[0], chunk_id[1]];
        self.stream_contexts
            .iter()
            .position(|s| s.stream_num == stream_num)
    }

    /// Check if chunk is a keyframe (static version for use without self borrow)
    fn is_keyframe_chunk_static(chunk_id: &[u8; 4], data: &[u8]) -> bool {
        Self::is_keyframe_chunk_impl(chunk_id, data)
    }

    /// Check if chunk is a keyframe
    fn is_keyframe_chunk(&self, chunk_id: &[u8; 4], data: &[u8]) -> bool {
        Self::is_keyframe_chunk_impl(chunk_id, data)
    }

    /// Implementation for keyframe check
    fn is_keyframe_chunk_impl(chunk_id: &[u8; 4], data: &[u8]) -> bool {
        // Check chunk type suffix
        let suffix = &chunk_id[2..4];

        // dc = compressed video, db = uncompressed video (always keyframe), wb = audio
        if suffix == b"db" {
            return true;
        }

        // For video, check for keyframe markers
        if suffix == b"dc" {
            // Check for H.264 IDR
            if data.len() >= 5 {
                // Look for start code + NAL type 5 (IDR)
                if data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1 {
                    let nal_type = data[4] & 0x1F;
                    if nal_type == 5 {
                        return true;
                    }
                }
                // 3-byte start code
                if data[0] == 0 && data[1] == 0 && data[2] == 1 {
                    let nal_type = data[3] & 0x1F;
                    if nal_type == 5 {
                        return true;
                    }
                }
            }

            // Check for MJPEG SOI marker
            if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
                return true;
            }
        }

        // Audio is always a keyframe
        if suffix == b"wb" {
            return true;
        }

        false
    }

    /// Read next packet from movi data
    fn read_movi_packet(&mut self) -> Result<Packet> {
        loop {
            let current = self
                .reader
                .stream_position()
                .map_err(|e| Error::Io(e))?;

            if current >= self.movi_end {
                return Err(Error::EndOfStream);
            }

            let chunk = self.read_chunk_header()?;

            // Skip LIST chunks inside movi
            if &chunk.fourcc == b"LIST" {
                let mut list_type = [0u8; 4];
                self.reader
                    .read_exact(&mut list_type)
                    .map_err(|e| Error::Io(e))?;
                // Continue scanning inside the LIST
                continue;
            }

            // Skip JUNK chunks
            if &chunk.fourcc == b"JUNK" || &chunk.fourcc == b"PAD " {
                self.skip_bytes(chunk.size as u64)?;
                if chunk.size % 2 != 0 {
                    self.skip_bytes(1)?;
                }
                continue;
            }

            // Check if this is a valid stream chunk
            if let Some(stream_idx) = self.find_stream_by_chunk_id(&chunk.fourcc) {
                let data = self.read_bytes(chunk.size as usize)?;

                // Word align
                if chunk.size % 2 != 0 {
                    self.skip_bytes(1)?;
                }

                // Capture the fourcc for keyframe check before borrowing stream_contexts
                let chunk_fourcc = chunk.fourcc;

                // Check keyframe before mutably borrowing stream_contexts
                let is_keyframe = Self::is_keyframe_chunk_static(&chunk_fourcc, &data);

                let ctx = &mut self.stream_contexts[stream_idx];

                let pts = ctx.current_frame as i64;
                ctx.current_frame += 1;

                let media_type = ctx.media_type;

                let mut packet = Packet::new(stream_idx, Buffer::from_vec(data));
                packet.codec_type = media_type;
                packet.pts = Timestamp::new(pts);
                packet.dts = Timestamp::new(pts);
                packet.duration = 1;
                packet.flags.keyframe = is_keyframe;

                return Ok(packet);
            } else {
                // Unknown chunk, skip it
                self.skip_bytes(chunk.size as u64)?;
                if chunk.size % 2 != 0 {
                    self.skip_bytes(1)?;
                }
            }
        }
    }

    /// Read packet using index
    fn read_index_packet(&mut self) -> Result<Packet> {
        while self.index_pos < self.index.len() {
            // Copy entry data before borrowing to avoid borrow checker issues
            let entry_chunk_id = self.index[self.index_pos].chunk_id;
            let entry_flags = self.index[self.index_pos].flags;
            let entry_offset = self.index[self.index_pos].offset;
            self.index_pos += 1;

            // Find stream for this chunk
            if let Some(stream_idx) = self.find_stream_by_chunk_id(&entry_chunk_id) {
                // Seek to chunk data (offset is from movi start, after 'movi' fourcc)
                let offset = self.movi_offset + entry_offset as u64;
                self.reader
                    .seek(SeekFrom::Start(offset))
                    .map_err(|e| Error::Io(e))?;

                // Read chunk header
                let chunk = self.read_chunk_header()?;

                // Read data
                let data = self.read_bytes(chunk.size as usize)?;

                // Check keyframe before mutably borrowing stream_contexts
                let is_keyframe = (entry_flags & 0x10) != 0
                    || Self::is_keyframe_chunk_static(&entry_chunk_id, &data);

                let ctx = &mut self.stream_contexts[stream_idx];

                let pts = ctx.current_frame as i64;
                ctx.current_frame += 1;

                let media_type = ctx.media_type;

                let mut packet = Packet::new(stream_idx, Buffer::from_vec(data));
                packet.codec_type = media_type;
                packet.pts = Timestamp::new(pts);
                packet.dts = Timestamp::new(pts);
                packet.duration = 1;
                packet.flags.keyframe = is_keyframe;

                return Ok(packet);
            }
        }

        Err(Error::EndOfStream)
    }
}

impl<R: Read + Seek> Demuxer for AviDemuxer<R> {
    fn open(&mut self, _path: &std::path::Path) -> Result<()> {
        self.parse_headers()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        if self.use_index {
            self.read_index_packet()
        } else {
            self.read_movi_packet()
        }
    }

    fn seek(&mut self, stream_index: usize, timestamp: i64) -> Result<()> {
        if !self.use_index {
            return Err(Error::unsupported("Seeking requires index"));
        }

        // Find nearest keyframe before timestamp
        let mut target_pos = 0;
        let stream_prefix = if stream_index < self.stream_contexts.len() {
            self.stream_contexts[stream_index].stream_num
        } else {
            return Err(Error::invalid_input("Invalid stream index"));
        };

        let mut frame_count = 0i64;
        for (i, entry) in self.index.iter().enumerate() {
            if entry.chunk_id[0] == stream_prefix[0] && entry.chunk_id[1] == stream_prefix[1] {
                if frame_count <= timestamp {
                    if (entry.flags & 0x10) != 0 {
                        // Keyframe
                        target_pos = i;
                    }
                }
                frame_count += 1;
                if frame_count > timestamp {
                    break;
                }
            }
        }

        self.index_pos = target_pos;

        // Reset frame counters
        for ctx in &mut self.stream_contexts {
            ctx.current_frame = 0;
        }

        // Count frames up to target position
        for i in 0..target_pos {
            if let Some(idx) = self.find_stream_by_chunk_id(&self.index[i].chunk_id) {
                self.stream_contexts[idx].current_frame += 1;
            }
        }

        Ok(())
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_avi_demuxer_creation() {
        let data = Vec::new();
        let cursor = Cursor::new(data);
        let demuxer = AviDemuxer::new(cursor);
        assert_eq!(demuxer.current_frame, 0);
        assert!(demuxer.streams.is_empty());
    }

    #[test]
    fn test_fourcc_to_codec() {
        assert_eq!(AviDemuxer::<Cursor<Vec<u8>>>::fourcc_to_codec_id(b"H264"), "h264");
        assert_eq!(AviDemuxer::<Cursor<Vec<u8>>>::fourcc_to_codec_id(b"MJPG"), "mjpeg");
        assert_eq!(AviDemuxer::<Cursor<Vec<u8>>>::fourcc_to_codec_id(b"XVID"), "mpeg4");
    }

    #[test]
    fn test_format_tag_to_codec() {
        assert_eq!(AviDemuxer::<Cursor<Vec<u8>>>::format_tag_to_codec_id(0x0001), "pcm_s16le");
        assert_eq!(AviDemuxer::<Cursor<Vec<u8>>>::format_tag_to_codec_id(0x0055), "mp3");
        assert_eq!(AviDemuxer::<Cursor<Vec<u8>>>::format_tag_to_codec_id(0x00FF), "aac");
    }
}
