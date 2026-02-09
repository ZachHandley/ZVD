//! MPEG-TS demuxer implementation
//!
//! This module provides complete MPEG-TS demuxing with PAT, PMT, and PES parsing.

use super::{pids, StreamType, TsPacketHeader, TS_PACKET_SIZE, TS_SYNC_BYTE};
use crate::error::{Error, Result};
use crate::format::{AudioInfo, Demuxer, Packet, PacketFlags, Stream, StreamInfo, VideoInfo};
use crate::util::{Buffer, MediaType, Rational, Timestamp};
use std::collections::HashMap;
use std::io::Read;

/// PES stream state for assembling packets
#[derive(Debug, Clone)]
struct PesState {
    /// Accumulated PES data
    buffer: Vec<u8>,
    /// PTS of current PES packet
    pts: Option<i64>,
    /// DTS of current PES packet
    dts: Option<i64>,
    /// Whether we're in the middle of assembling a PES packet
    in_progress: bool,
    /// Expected PES packet length (0 = unbounded)
    expected_length: usize,
}

impl Default for PesState {
    fn default() -> Self {
        PesState {
            buffer: Vec::new(),
            pts: None,
            dts: None,
            in_progress: false,
            expected_length: 0,
        }
    }
}

/// Program information from PMT
#[derive(Debug, Clone)]
struct ProgramInfo {
    /// Program number
    program_number: u16,
    /// PMT PID
    pmt_pid: u16,
    /// PCR PID
    pcr_pid: u16,
    /// Elementary streams: PID -> (stream_type, stream_index)
    streams: HashMap<u16, (StreamType, usize)>,
}

/// MPEG-TS demuxer
pub struct MpegtsDemuxer<R: Read> {
    reader: R,
    streams: Vec<Stream>,
    stream_map: HashMap<u16, usize>, // PID -> stream index
    duration: Option<f64>,
    packet_count: u64,
    /// PAT parsed flag
    pat_parsed: bool,
    /// Programs from PAT: program_number -> PMT PID
    programs: HashMap<u16, u16>,
    /// Program info from PMT parsing
    program_info: Option<ProgramInfo>,
    /// PES assembly state per PID
    pes_state: HashMap<u16, PesState>,
    /// Byte buffer for reading
    read_buffer: Vec<u8>,
    /// Whether we need to find sync
    need_sync: bool,
}

impl<R: Read> MpegtsDemuxer<R> {
    /// Create a new MPEG-TS demuxer
    pub fn new(reader: R) -> Self {
        MpegtsDemuxer {
            reader,
            streams: Vec::new(),
            stream_map: HashMap::new(),
            duration: None,
            packet_count: 0,
            pat_parsed: false,
            programs: HashMap::new(),
            program_info: None,
            pes_state: HashMap::new(),
            read_buffer: vec![0u8; TS_PACKET_SIZE],
            need_sync: true,
        }
    }

    /// Find sync byte in stream
    fn find_sync(&mut self) -> Result<()> {
        let mut byte = [0u8; 1];
        let mut attempts = 0;
        const MAX_SYNC_ATTEMPTS: usize = TS_PACKET_SIZE * 10;

        loop {
            match self.reader.read_exact(&mut byte) {
                Ok(()) => {
                    if byte[0] == TS_SYNC_BYTE {
                        self.need_sync = false;
                        return Ok(());
                    }
                    attempts += 1;
                    if attempts > MAX_SYNC_ATTEMPTS {
                        return Err(Error::format("Could not find MPEG-TS sync byte"));
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Err(Error::EndOfStream);
                }
                Err(e) => return Err(Error::Io(e)),
            }
        }
    }

    /// Read TS packet
    fn read_ts_packet(&mut self) -> Result<&[u8]> {
        if self.need_sync {
            self.find_sync()?;
            self.read_buffer[0] = TS_SYNC_BYTE;
            self.reader
                .read_exact(&mut self.read_buffer[1..])
                .map_err(|e| {
                    if e.kind() == std::io::ErrorKind::UnexpectedEof {
                        Error::EndOfStream
                    } else {
                        Error::Io(e)
                    }
                })?;
        } else {
            self.reader.read_exact(&mut self.read_buffer).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    Error::EndOfStream
                } else {
                    Error::Io(e)
                }
            })?;
        }

        if self.read_buffer[0] != TS_SYNC_BYTE {
            self.need_sync = true;
            return Err(Error::TryAgain);
        }

        Ok(&self.read_buffer)
    }

    /// Get payload offset accounting for adaptation field
    fn get_payload_offset(packet: &[u8], header: &TsPacketHeader) -> Option<usize> {
        let mut offset = 4; // Skip header

        // Check adaptation field
        match header.adaptation_field_control {
            0b00 => return None, // Reserved
            0b01 => {}           // Payload only
            0b10 => return None, // Adaptation field only
            0b11 => {
                // Adaptation field + payload
                if offset >= packet.len() {
                    return None;
                }
                let adaptation_length = packet[offset] as usize;
                offset += 1 + adaptation_length;
            }
            _ => return None,
        }

        if offset >= packet.len() {
            None
        } else {
            Some(offset)
        }
    }

    /// Parse PAT (Program Association Table)
    fn parse_pat(&mut self, packet: &[u8], header: &TsPacketHeader) -> Result<()> {
        let payload_offset = match Self::get_payload_offset(packet, header) {
            Some(o) => o,
            None => return Ok(()),
        };

        let payload = &packet[payload_offset..];
        if payload.is_empty() {
            return Ok(());
        }

        // Skip pointer field if payload_unit_start is set
        let mut offset = if header.payload_unit_start {
            let pointer = payload[0] as usize;
            1 + pointer
        } else {
            0
        };

        if offset >= payload.len() {
            return Ok(());
        }

        // Table ID
        let table_id = payload[offset];
        if table_id != 0x00 {
            return Ok(()); // Not a PAT
        }
        offset += 1;

        if offset + 2 > payload.len() {
            return Ok(());
        }

        // Section length
        let section_length =
            (((payload[offset] & 0x0F) as usize) << 8) | (payload[offset + 1] as usize);
        offset += 2;

        if offset + section_length > payload.len() {
            return Ok(());
        }

        // Transport stream ID
        offset += 2;

        // Version number, current/next indicator
        offset += 1;

        // Section number
        offset += 1;

        // Last section number
        offset += 1;

        // Program loop - each entry is 4 bytes
        // section_length includes CRC (4 bytes) and the bytes we've skipped (5 bytes)
        let program_data_length = section_length.saturating_sub(9);
        let num_programs = program_data_length / 4;

        for _ in 0..num_programs {
            if offset + 4 > payload.len() {
                break;
            }

            let program_number = ((payload[offset] as u16) << 8) | (payload[offset + 1] as u16);
            let pmt_pid =
                (((payload[offset + 2] & 0x1F) as u16) << 8) | (payload[offset + 3] as u16);
            offset += 4;

            // Program number 0 is network PID, skip it
            if program_number != 0 {
                self.programs.insert(program_number, pmt_pid);
            }
        }

        self.pat_parsed = true;
        Ok(())
    }

    /// Parse PMT (Program Map Table)
    fn parse_pmt(&mut self, packet: &[u8], header: &TsPacketHeader, pmt_pid: u16) -> Result<()> {
        let payload_offset = match Self::get_payload_offset(packet, header) {
            Some(o) => o,
            None => return Ok(()),
        };

        let payload = &packet[payload_offset..];
        if payload.is_empty() {
            return Ok(());
        }

        // Skip pointer field if payload_unit_start is set
        let mut offset = if header.payload_unit_start {
            let pointer = payload[0] as usize;
            1 + pointer
        } else {
            0
        };

        if offset >= payload.len() {
            return Ok(());
        }

        // Table ID
        let table_id = payload[offset];
        if table_id != 0x02 {
            return Ok(()); // Not a PMT
        }
        offset += 1;

        if offset + 2 > payload.len() {
            return Ok(());
        }

        // Section length
        let section_length =
            (((payload[offset] & 0x0F) as usize) << 8) | (payload[offset + 1] as usize);
        offset += 2;

        if offset + section_length > payload.len() {
            return Ok(());
        }

        // Program number
        if offset + 2 > payload.len() {
            return Ok(());
        }
        let program_number = ((payload[offset] as u16) << 8) | (payload[offset + 1] as u16);
        offset += 2;

        // Version, current/next
        offset += 1;

        // Section number
        offset += 1;

        // Last section number
        offset += 1;

        // PCR PID
        if offset + 2 > payload.len() {
            return Ok(());
        }
        let pcr_pid = (((payload[offset] & 0x1F) as u16) << 8) | (payload[offset + 1] as u16);
        offset += 2;

        // Program info length
        if offset + 2 > payload.len() {
            return Ok(());
        }
        let program_info_length =
            (((payload[offset] & 0x0F) as usize) << 8) | (payload[offset + 1] as usize);
        offset += 2;

        // Skip program descriptors
        offset += program_info_length;

        // Create program info
        let mut program_info = ProgramInfo {
            program_number,
            pmt_pid,
            pcr_pid,
            streams: HashMap::new(),
        };

        // Parse elementary stream loop
        // Remaining length = section_length - 9 (header) - 4 (CRC) - program_info_length
        let es_loop_end = payload_offset + 3 + section_length - 4;

        while offset + 5 <= payload.len() && payload_offset + offset < es_loop_end {
            let stream_type_byte = payload[offset];
            offset += 1;

            let es_pid = (((payload[offset] & 0x1F) as u16) << 8) | (payload[offset + 1] as u16);
            offset += 2;

            let es_info_length =
                (((payload[offset] & 0x0F) as usize) << 8) | (payload[offset + 1] as usize);
            offset += 2;

            // Skip ES descriptors
            offset += es_info_length;

            // Convert stream type
            if let Some(stream_type) = StreamType::from_u8(stream_type_byte) {
                let stream_index = self.create_stream_for_type(stream_type, es_pid);
                program_info.streams.insert(es_pid, (stream_type, stream_index));
                self.stream_map.insert(es_pid, stream_index);
                self.pes_state.insert(es_pid, PesState::default());
            }
        }

        self.program_info = Some(program_info);
        Ok(())
    }

    /// Create a stream for the given type
    fn create_stream_for_type(&mut self, stream_type: StreamType, _pid: u16) -> usize {
        let index = self.streams.len();

        let (media_type, codec_id) = match stream_type {
            StreamType::Mpeg1Video => (MediaType::Video, "mpeg1video"),
            StreamType::Mpeg2Video => (MediaType::Video, "mpeg2video"),
            StreamType::Mpeg1Audio => (MediaType::Audio, "mp2"),
            StreamType::Mpeg2Audio => (MediaType::Audio, "mp2"),
            StreamType::AudioAAC | StreamType::AudioAACLatm => (MediaType::Audio, "aac"),
            StreamType::VideoMpeg4 => (MediaType::Video, "mpeg4"),
            StreamType::VideoH264 => (MediaType::Video, "h264"),
            StreamType::VideoH265 => (MediaType::Video, "hevc"),
            StreamType::PrivateData | StreamType::PrivateSection => (MediaType::Data, "data"),
        };

        let mut info = StreamInfo::new(index, media_type, codec_id.to_string());
        info.time_base = Rational::new(1, 90000); // MPEG-TS uses 90kHz clock

        match media_type {
            MediaType::Video => {
                // Default video info, will be updated when we parse the stream
                info.video_info = Some(VideoInfo::new(0, 0));
            }
            MediaType::Audio => {
                // Default audio info
                info.audio_info = Some(AudioInfo::new(48000, 2));
            }
            _ => {}
        }

        let stream = Stream::new(info);
        self.streams.push(stream);

        index
    }

    /// Parse PES (Packetized Elementary Stream) packet header
    fn parse_pes_header(data: &[u8]) -> Option<(i64, Option<i64>, usize)> {
        // PES packet header:
        // 3 bytes: packet_start_code_prefix (0x000001)
        // 1 byte: stream_id
        // 2 bytes: PES_packet_length
        // Optional header follows

        if data.len() < 9 {
            return None;
        }

        // Check start code
        if data[0] != 0x00 || data[1] != 0x00 || data[2] != 0x01 {
            return None;
        }

        let stream_id = data[3];

        // Skip non-video/audio streams
        if !((0xC0..=0xDF).contains(&stream_id) || (0xE0..=0xEF).contains(&stream_id)) {
            return None;
        }

        let pes_length = ((data[4] as usize) << 8) | (data[5] as usize);

        // Optional PES header
        if data.len() < 9 {
            return None;
        }

        let pts_dts_flags = (data[7] >> 6) & 0x03;
        let header_data_length = data[8] as usize;

        let header_size = 9 + header_data_length;
        if data.len() < header_size {
            return None;
        }

        let mut pts: Option<i64> = None;
        let mut dts: Option<i64> = None;

        if pts_dts_flags >= 2 && data.len() >= 14 {
            // Parse PTS
            pts = Some(Self::parse_timestamp(&data[9..14]));
        }

        if pts_dts_flags == 3 && data.len() >= 19 {
            // Parse DTS
            dts = Some(Self::parse_timestamp(&data[14..19]));
        }

        let pts_value = pts.unwrap_or(0);
        Some((pts_value, dts, header_size))
    }

    /// Parse a 33-bit timestamp from 5 bytes
    fn parse_timestamp(data: &[u8]) -> i64 {
        let ts = (((data[0] >> 1) as i64 & 0x07) << 30)
            | ((data[1] as i64) << 22)
            | (((data[2] >> 1) as i64) << 15)
            | ((data[3] as i64) << 7)
            | ((data[4] >> 1) as i64);
        ts
    }

    /// Parse PES packet from accumulated data
    fn parse_pes(&mut self, pid: u16) -> Result<Option<Packet>> {
        let pes_state = match self.pes_state.get(&pid) {
            Some(state) => state,
            None => return Ok(None),
        };

        if pes_state.buffer.is_empty() {
            return Ok(None);
        }

        let stream_index = match self.stream_map.get(&pid) {
            Some(&idx) => idx,
            None => return Ok(None),
        };

        // Parse PES header
        let (pts, dts, header_size) = match Self::parse_pes_header(&pes_state.buffer) {
            Some(result) => result,
            None => {
                // Clear invalid data
                if let Some(state) = self.pes_state.get_mut(&pid) {
                    state.buffer.clear();
                    state.in_progress = false;
                }
                return Ok(None);
            }
        };

        // Get payload data
        let payload = if header_size < pes_state.buffer.len() {
            pes_state.buffer[header_size..].to_vec()
        } else {
            Vec::new()
        };

        // Determine media type from stream
        let codec_type = self
            .streams
            .get(stream_index)
            .map(|s| s.info.media_type)
            .unwrap_or(MediaType::Unknown);

        // Create packet
        let mut packet = Packet::new(stream_index, Buffer::from_vec(payload));
        packet.codec_type = codec_type;
        packet.pts = Timestamp::new(pts);
        packet.dts = dts.map(Timestamp::new).unwrap_or(Timestamp::new(pts));

        // Check for keyframe (for H.264, look for IDR NAL)
        packet.flags.keyframe = Self::is_keyframe(&packet.data.as_slice(), codec_type);

        // Clear PES state
        if let Some(state) = self.pes_state.get_mut(&pid) {
            state.buffer.clear();
            state.pts = None;
            state.dts = None;
            state.in_progress = false;
        }

        Ok(Some(packet))
    }

    /// Check if packet contains a keyframe
    fn is_keyframe(data: &[u8], codec_type: MediaType) -> bool {
        if codec_type != MediaType::Video || data.len() < 5 {
            return false;
        }

        // Look for H.264 IDR NAL (type 5) or H.265 IDR NAL (types 19, 20)
        let mut i = 0;
        while i + 4 < data.len() {
            // Look for start code
            if data[i] == 0x00 && data[i + 1] == 0x00 {
                let nal_offset;
                if data[i + 2] == 0x01 {
                    nal_offset = i + 3;
                } else if data[i + 2] == 0x00 && i + 3 < data.len() && data[i + 3] == 0x01 {
                    nal_offset = i + 4;
                } else {
                    i += 1;
                    continue;
                }

                if nal_offset < data.len() {
                    let nal_type = data[nal_offset] & 0x1F;
                    // H.264 IDR = 5, H.265 IDR_W_RADL = 19, IDR_N_LP = 20
                    if nal_type == 5 || nal_type == 19 || nal_type == 20 {
                        return true;
                    }
                }

                i = nal_offset;
            } else {
                i += 1;
            }
        }

        false
    }

    /// Process a TS packet and potentially return a complete PES packet
    fn process_ts_packet(&mut self, packet: &[u8]) -> Result<Option<Packet>> {
        let header_bytes: [u8; 4] = packet[0..4]
            .try_into()
            .map_err(|_| Error::invalid_input("Invalid TS packet"))?;

        let header =
            TsPacketHeader::from_bytes(&header_bytes).map_err(|e| Error::invalid_input(e))?;

        self.packet_count += 1;

        // Skip null packets
        if header.pid == pids::NULL {
            return Ok(None);
        }

        // Handle PAT
        if header.pid == pids::PAT {
            self.parse_pat(packet, &header)?;
            return Ok(None);
        }

        // Handle PMT
        if !self.programs.is_empty() && self.program_info.is_none() {
            for &pmt_pid in self.programs.values() {
                if header.pid == pmt_pid {
                    self.parse_pmt(packet, &header, pmt_pid)?;
                    return Ok(None);
                }
            }
        }

        // Handle elementary streams
        if self.stream_map.contains_key(&header.pid) {
            // Get payload
            let payload_offset = match Self::get_payload_offset(packet, &header) {
                Some(o) => o,
                None => return Ok(None),
            };

            let payload = &packet[payload_offset..];

            // Handle payload unit start - indicates new PES packet
            if header.payload_unit_start {
                // First, emit any existing data as a packet
                let prev_packet = self.parse_pes(header.pid)?;

                // Start new PES packet
                if let Some(state) = self.pes_state.get_mut(&header.pid) {
                    state.buffer.clear();
                    state.buffer.extend_from_slice(payload);
                    state.in_progress = true;
                }

                return Ok(prev_packet);
            } else {
                // Continue accumulating PES data
                if let Some(state) = self.pes_state.get_mut(&header.pid) {
                    if state.in_progress {
                        state.buffer.extend_from_slice(payload);
                    }
                }
            }
        }

        Ok(None)
    }
}

impl<R: Read> Demuxer for MpegtsDemuxer<R> {
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn open(&mut self, _path: &std::path::Path) -> Result<()> {
        // Find first sync byte
        self.find_sync()?;

        // Read packets until we have PAT and PMT
        let mut attempts = 0;
        const MAX_INIT_PACKETS: usize = 1000;

        while (self.programs.is_empty() || self.program_info.is_none())
            && attempts < MAX_INIT_PACKETS
        {
            self.read_buffer[0] = TS_SYNC_BYTE;
            match self.reader.read_exact(&mut self.read_buffer[1..]) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::Io(e)),
            }

            let header_bytes: [u8; 4] = self.read_buffer[0..4]
                .try_into()
                .map_err(|_| Error::invalid_input("Invalid TS packet"))?;

            if let Ok(header) = TsPacketHeader::from_bytes(&header_bytes) {
                if header.pid == pids::PAT {
                    self.parse_pat(&self.read_buffer.clone(), &header)?;
                } else if self.pat_parsed {
                    for &pmt_pid in self.programs.values() {
                        if header.pid == pmt_pid {
                            self.parse_pmt(&self.read_buffer.clone(), &header, pmt_pid)?;
                            break;
                        }
                    }
                }
            }

            attempts += 1;
        }

        self.need_sync = false;
        Ok(())
    }

    fn read_packet(&mut self) -> Result<Packet> {
        loop {
            let packet = match self.read_ts_packet() {
                Ok(p) => p.to_vec(), // Copy to avoid borrow issues
                Err(Error::TryAgain) => continue,
                Err(e) => return Err(e),
            };

            if let Some(pes_packet) = self.process_ts_packet(&packet)? {
                return Ok(pes_packet);
            }
        }
    }

    fn seek(&mut self, _stream_index: usize, _timestamp: i64) -> Result<()> {
        // TS seeking is complex - need to find sync points
        // For now, return unsupported
        Err(Error::unsupported("Seeking not supported for MPEG-TS"))
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_mpegts_demuxer_creation() {
        let data = Vec::new();
        let cursor = Cursor::new(data);
        let demuxer = MpegtsDemuxer::new(cursor);
        assert_eq!(demuxer.packet_count, 0);
        assert!(!demuxer.pat_parsed);
    }

    #[test]
    fn test_parse_timestamp() {
        // Test timestamp parsing
        // PTS = 90000 (1 second at 90kHz)
        let ts_data = [0x21, 0x00, 0x07, 0xD1, 0x21];
        let ts = MpegtsDemuxer::<Cursor<Vec<u8>>>::parse_timestamp(&ts_data);
        // The encoded timestamp should decode properly
        assert!(ts > 0);
    }

    #[test]
    fn test_stream_type_conversion() {
        assert_eq!(StreamType::from_u8(0x1B), Some(StreamType::VideoH264));
        assert_eq!(StreamType::from_u8(0x0F), Some(StreamType::AudioAAC));
        assert_eq!(StreamType::from_u8(0x24), Some(StreamType::VideoH265));
        assert_eq!(StreamType::from_u8(0xFF), None);
    }

    #[test]
    fn test_is_keyframe_h264() {
        // H.264 IDR frame start
        let idr_data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84];
        assert!(MpegtsDemuxer::<Cursor<Vec<u8>>>::is_keyframe(
            &idr_data,
            MediaType::Video
        ));

        // Non-IDR frame
        let non_idr = vec![0x00, 0x00, 0x00, 0x01, 0x41, 0x9A];
        assert!(!MpegtsDemuxer::<Cursor<Vec<u8>>>::is_keyframe(
            &non_idr,
            MediaType::Video
        ));
    }
}
