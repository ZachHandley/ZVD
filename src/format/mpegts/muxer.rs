//! MPEG-TS muxer implementation
//!
//! This module provides complete MPEG-TS muxing with PAT, PMT, and PES packet writing.

use super::{pids, StreamType, TsPacketHeader, TS_PACKET_SIZE, TS_SYNC_BYTE};
use crate::error::{Error, Result};
use crate::format::{Muxer, Packet, Stream};
use crate::util::MediaType;
use std::collections::HashMap;
use std::io::Write;

/// CRC32 table for MPEG-TS
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            if crc & 0x80000000 != 0 {
                crc = (crc << 1) ^ 0x04C11DB7;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Calculate CRC32 for MPEG-TS tables
fn calculate_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF;
    for &byte in data {
        let idx = ((crc >> 24) ^ (byte as u32)) & 0xFF;
        crc = (crc << 8) ^ CRC32_TABLE[idx as usize];
    }
    crc
}

/// Stream info for muxing
#[derive(Debug, Clone)]
struct MuxerStream {
    /// Stream index
    index: usize,
    /// PID for this stream
    pid: u16,
    /// Stream type for PMT
    stream_type: StreamType,
    /// Media type
    media_type: MediaType,
    /// Continuity counter
    continuity_counter: u8,
}

/// MPEG-TS muxer
pub struct MpegtsMuxer<W: Write> {
    writer: W,
    streams: Vec<Stream>,
    muxer_streams: Vec<MuxerStream>,
    stream_pids: HashMap<usize, u16>,      // stream index -> PID
    continuity_counters: HashMap<u16, u8>, // PID -> counter
    next_pid: u16,
    pmt_pid: u16,
    pcr_pid: u16,
    started: bool,
    packet_count: u64,
    /// PCR base value (90kHz clock)
    pcr_base: u64,
    /// Last DTS for PCR calculation
    last_dts: i64,
    /// PAT continuity counter
    pat_cc: u8,
    /// PMT continuity counter
    pmt_cc: u8,
    /// Packets since last PAT/PMT
    packets_since_tables: u32,
}

impl<W: Write> MpegtsMuxer<W> {
    /// Create a new MPEG-TS muxer
    pub fn new(writer: W) -> Self {
        MpegtsMuxer {
            writer,
            streams: Vec::new(),
            muxer_streams: Vec::new(),
            stream_pids: HashMap::new(),
            continuity_counters: HashMap::new(),
            next_pid: 256, // Start PIDs after reserved range
            pmt_pid: 4096, // Standard PMT PID
            pcr_pid: 256,  // Will be set to first video stream PID
            started: false,
            packet_count: 0,
            pcr_base: 0,
            last_dts: 0,
            pat_cc: 0,
            pmt_cc: 0,
            packets_since_tables: 0,
        }
    }

    /// Get and increment continuity counter for PID
    fn get_counter(&mut self, pid: u16) -> u8 {
        let counter = self.continuity_counters.entry(pid).or_insert(0);
        let current = *counter;
        *counter = (*counter + 1) & 0x0F;
        current
    }

    /// Write a complete TS packet with optional adaptation field
    fn write_ts_packet(
        &mut self,
        pid: u16,
        payload: &[u8],
        payload_start: bool,
        adaptation: Option<&[u8]>,
    ) -> Result<()> {
        let counter = self.get_counter(pid);

        let mut packet = vec![0u8; TS_PACKET_SIZE];

        // Build header
        packet[0] = TS_SYNC_BYTE;
        packet[1] = if payload_start { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
        packet[2] = (pid & 0xFF) as u8;

        let mut offset = 4;
        let adaptation_field_control;

        if let Some(adapt) = adaptation {
            // Adaptation field + payload
            adaptation_field_control = 0x30;
            packet[4] = adapt.len() as u8;
            offset = 5;
            packet[offset..offset + adapt.len()].copy_from_slice(adapt);
            offset += adapt.len();
        } else if payload.len() < TS_PACKET_SIZE - 4 {
            // Need stuffing - use adaptation field for padding
            adaptation_field_control = 0x30;
            let stuffing_len = TS_PACKET_SIZE - 4 - payload.len() - 1;
            packet[4] = stuffing_len as u8;
            offset = 5;
            if stuffing_len > 0 {
                // First byte of adaptation field content is flags
                packet[5] = 0x00; // No flags set
                offset = 6;
                // Fill rest with stuffing bytes (0xFF)
                for i in 0..stuffing_len.saturating_sub(1) {
                    packet[offset + i] = 0xFF;
                }
                offset = 5 + stuffing_len;
            }
        } else {
            // Payload only
            adaptation_field_control = 0x10;
        }

        packet[3] = adaptation_field_control | counter;

        // Copy payload
        let payload_len = payload.len().min(TS_PACKET_SIZE - offset);
        packet[offset..offset + payload_len].copy_from_slice(&payload[..payload_len]);

        // Pad remaining with stuffing if needed
        for i in (offset + payload_len)..TS_PACKET_SIZE {
            packet[i] = 0xFF;
        }

        self.writer.write_all(&packet).map_err(Error::Io)?;
        self.packet_count += 1;
        self.packets_since_tables += 1;

        Ok(())
    }

    /// Write PAT (Program Association Table)
    fn write_pat(&mut self) -> Result<()> {
        let mut pat_data = Vec::with_capacity(17);

        // Pointer field
        pat_data.push(0x00);

        // Table ID for PAT
        pat_data.push(0x00);

        // Section syntax indicator (1), private bit (0), reserved (11), section length (12 bits)
        // section_length = 5 (header after length) + 4 (program entry) + 4 (CRC) = 13 = 0x00D
        pat_data.push(0xB0);
        pat_data.push(0x0D);

        // Transport stream ID
        pat_data.push(0x00);
        pat_data.push(0x01);

        // Reserved (2), version number (5), current/next indicator (1)
        pat_data.push(0xC1);

        // Section number
        pat_data.push(0x00);

        // Last section number
        pat_data.push(0x00);

        // Program number 1
        pat_data.push(0x00);
        pat_data.push(0x01);

        // Reserved (3), PMT PID (13)
        pat_data.push(0xE0 | ((self.pmt_pid >> 8) & 0x1F) as u8);
        pat_data.push((self.pmt_pid & 0xFF) as u8);

        // Calculate CRC32 over data from table_id to last byte before CRC
        let crc = calculate_crc32(&pat_data[1..]);
        pat_data.extend_from_slice(&crc.to_be_bytes());

        // Write PAT packet
        let counter = self.pat_cc;
        self.pat_cc = (self.pat_cc + 1) & 0x0F;

        let mut packet = vec![0u8; TS_PACKET_SIZE];
        packet[0] = TS_SYNC_BYTE;
        packet[1] = 0x40; // Payload start
        packet[2] = 0x00; // PAT PID = 0
        packet[3] = 0x10 | counter; // Payload only

        packet[4..4 + pat_data.len()].copy_from_slice(&pat_data);

        // Fill rest with stuffing
        for i in (4 + pat_data.len())..TS_PACKET_SIZE {
            packet[i] = 0xFF;
        }

        self.writer.write_all(&packet).map_err(Error::Io)?;
        self.packet_count += 1;

        Ok(())
    }

    /// Write PMT (Program Map Table)
    fn write_pmt(&mut self) -> Result<()> {
        let mut pmt_data = Vec::with_capacity(64);

        // Pointer field
        pmt_data.push(0x00);

        // Table ID for PMT
        pmt_data.push(0x02);

        // Placeholder for section length - will update
        let section_length_pos = pmt_data.len();
        pmt_data.push(0xB0);
        pmt_data.push(0x00); // Will be updated

        // Program number
        pmt_data.push(0x00);
        pmt_data.push(0x01);

        // Reserved (2), version (5), current/next (1)
        pmt_data.push(0xC1);

        // Section number
        pmt_data.push(0x00);

        // Last section number
        pmt_data.push(0x00);

        // Reserved (3), PCR PID (13)
        pmt_data.push(0xE0 | ((self.pcr_pid >> 8) & 0x1F) as u8);
        pmt_data.push((self.pcr_pid & 0xFF) as u8);

        // Reserved (4), program info length (12) - no descriptors
        pmt_data.push(0xF0);
        pmt_data.push(0x00);

        // Elementary stream info
        for stream in &self.muxer_streams {
            // Stream type
            pmt_data.push(stream.stream_type as u8);

            // Reserved (3), elementary PID (13)
            pmt_data.push(0xE0 | ((stream.pid >> 8) & 0x1F) as u8);
            pmt_data.push((stream.pid & 0xFF) as u8);

            // Reserved (4), ES info length (12) - no descriptors
            pmt_data.push(0xF0);
            pmt_data.push(0x00);
        }

        // Calculate section length: data from program_number to end (before CRC) + CRC
        // = total length from byte after section_length to end including CRC
        let section_length = pmt_data.len() - 4 + 4; // -4 for bytes before section_length, +4 for CRC
        pmt_data[section_length_pos] = 0xB0 | ((section_length >> 8) & 0x0F) as u8;
        pmt_data[section_length_pos + 1] = (section_length & 0xFF) as u8;

        // Calculate CRC32
        let crc = calculate_crc32(&pmt_data[1..]);
        pmt_data.extend_from_slice(&crc.to_be_bytes());

        // Write PMT packet
        let counter = self.pmt_cc;
        self.pmt_cc = (self.pmt_cc + 1) & 0x0F;

        let mut packet = vec![0u8; TS_PACKET_SIZE];
        packet[0] = TS_SYNC_BYTE;
        packet[1] = 0x40 | ((self.pmt_pid >> 8) & 0x1F) as u8;
        packet[2] = (self.pmt_pid & 0xFF) as u8;
        packet[3] = 0x10 | counter;

        if pmt_data.len() > TS_PACKET_SIZE - 4 {
            return Err(Error::format("PMT too large for single packet"));
        }

        packet[4..4 + pmt_data.len()].copy_from_slice(&pmt_data);

        // Fill rest with stuffing
        for i in (4 + pmt_data.len())..TS_PACKET_SIZE {
            packet[i] = 0xFF;
        }

        self.writer.write_all(&packet).map_err(Error::Io)?;
        self.packet_count += 1;

        Ok(())
    }

    /// Write PAT and PMT tables
    fn write_tables(&mut self) -> Result<()> {
        self.write_pat()?;
        self.write_pmt()?;
        self.packets_since_tables = 0;
        Ok(())
    }

    /// Create PES header for packet
    fn create_pes_header(
        stream_id: u8,
        pts: Option<i64>,
        dts: Option<i64>,
        payload_len: usize,
    ) -> Vec<u8> {
        let mut header = Vec::with_capacity(19);

        // PES start code prefix
        header.push(0x00);
        header.push(0x00);
        header.push(0x01);

        // Stream ID
        header.push(stream_id);

        // Calculate header data length
        let pts_dts_flags = if pts.is_some() && dts.is_some() && pts != dts {
            0x03 // PTS and DTS present
        } else if pts.is_some() {
            0x02 // PTS only
        } else {
            0x00 // Neither
        };

        let header_data_length = match pts_dts_flags {
            0x03 => 10, // PTS (5) + DTS (5)
            0x02 => 5,  // PTS only
            _ => 0,
        };

        // PES packet length (0 for video = unbounded)
        let pes_packet_length = if stream_id >= 0xE0 {
            0 // Unbounded for video
        } else {
            // 3 (header extension) + header_data_length + payload
            3 + header_data_length + payload_len
        };
        header.push((pes_packet_length >> 8) as u8);
        header.push((pes_packet_length & 0xFF) as u8);

        // PES header extension
        // '10' marker bits, scrambling (00), priority (0), alignment (0), copyright (0), original (0)
        header.push(0x80);

        // PTS_DTS_flags, ESCR, ES_rate, DSM_trick, additional_copy, CRC, extension
        header.push(pts_dts_flags << 6);

        // Header data length
        header.push(header_data_length as u8);

        // PTS
        if let Some(pts_val) = pts {
            let pts_val = pts_val as u64;
            if pts_dts_flags == 0x03 {
                // PTS with DTS following
                header.push(0x31 | (((pts_val >> 30) & 0x07) << 1) as u8);
            } else {
                // PTS only
                header.push(0x21 | (((pts_val >> 30) & 0x07) << 1) as u8);
            }
            header.push(((pts_val >> 22) & 0xFF) as u8);
            header.push(0x01 | (((pts_val >> 15) & 0x7F) << 1) as u8);
            header.push(((pts_val >> 7) & 0xFF) as u8);
            header.push(0x01 | ((pts_val & 0x7F) << 1) as u8);
        }

        // DTS
        if pts_dts_flags == 0x03 {
            if let Some(dts_val) = dts {
                let dts_val = dts_val as u64;
                header.push(0x11 | (((dts_val >> 30) & 0x07) << 1) as u8);
                header.push(((dts_val >> 22) & 0xFF) as u8);
                header.push(0x01 | (((dts_val >> 15) & 0x7F) << 1) as u8);
                header.push(((dts_val >> 7) & 0xFF) as u8);
                header.push(0x01 | ((dts_val & 0x7F) << 1) as u8);
            }
        }

        header
    }

    /// Create adaptation field with PCR
    fn create_pcr_adaptation(pcr: u64) -> Vec<u8> {
        let mut adapt = Vec::with_capacity(8);

        // Adaptation field length (7 bytes of content)
        // Flags + PCR (6 bytes)

        // Flags: PCR flag set
        adapt.push(0x10);

        // PCR: 33 bits base + 6 reserved + 9 bits extension
        // For simplicity, we use the same value scaled
        let pcr_base = pcr;
        let pcr_ext = 0u16;

        adapt.push((pcr_base >> 25) as u8);
        adapt.push((pcr_base >> 17) as u8);
        adapt.push((pcr_base >> 9) as u8);
        adapt.push((pcr_base >> 1) as u8);
        adapt.push((((pcr_base & 0x01) << 7) | 0x7E | ((pcr_ext as u64 >> 8) & 0x01)) as u8);
        adapt.push((pcr_ext & 0xFF) as u8);

        adapt
    }

    /// Write a PES packet split into TS packets
    fn write_pes_packets(&mut self, pid: u16, pes_data: &[u8], include_pcr: bool) -> Result<()> {
        let mut offset = 0;
        let mut first = true;

        while offset < pes_data.len() {
            let remaining = pes_data.len() - offset;

            // Determine available space in this TS packet
            let adaptation = if first && include_pcr && pid == self.pcr_pid {
                Some(Self::create_pcr_adaptation(self.pcr_base))
            } else {
                None
            };

            let adapt_len = adaptation.as_ref().map(|a| a.len() + 1).unwrap_or(0);
            let available = TS_PACKET_SIZE - 4 - adapt_len;

            let payload_len = remaining.min(available);
            let payload = &pes_data[offset..offset + payload_len];

            self.write_ts_packet(pid, payload, first, adaptation.as_deref())?;

            offset += payload_len;
            first = false;
        }

        Ok(())
    }

    /// Get stream type from codec ID
    fn get_stream_type(codec_id: &str, media_type: MediaType) -> StreamType {
        let codec_lower = codec_id.to_lowercase();

        match media_type {
            MediaType::Video => {
                if codec_lower.contains("h264") || codec_lower.contains("avc") {
                    StreamType::VideoH264
                } else if codec_lower.contains("h265")
                    || codec_lower.contains("hevc")
                    || codec_lower.contains("h.265")
                {
                    StreamType::VideoH265
                } else if codec_lower.contains("mpeg2") {
                    StreamType::Mpeg2Video
                } else if codec_lower.contains("mpeg4") || codec_lower.contains("mp4v") {
                    StreamType::VideoMpeg4
                } else {
                    StreamType::VideoH264 // Default to H.264
                }
            }
            MediaType::Audio => {
                if codec_lower.contains("aac") {
                    StreamType::AudioAAC
                } else if codec_lower.contains("mp3") || codec_lower.contains("mp2") {
                    StreamType::Mpeg2Audio
                } else if codec_lower.contains("ac3") || codec_lower.contains("eac3") {
                    StreamType::PrivateData // AC-3 uses private PES
                } else {
                    StreamType::AudioAAC // Default to AAC
                }
            }
            _ => StreamType::PrivateData,
        }
    }

    /// Get stream ID for PES
    fn get_stream_id(media_type: MediaType, stream_index: usize) -> u8 {
        match media_type {
            MediaType::Video => 0xE0 + (stream_index as u8 % 16), // Video stream IDs: 0xE0-0xEF
            MediaType::Audio => 0xC0 + (stream_index as u8 % 32), // Audio stream IDs: 0xC0-0xDF
            _ => 0xBD,                                            // Private stream 1
        }
    }
}

impl<W: Write> Muxer for MpegtsMuxer<W> {
    fn create(&mut self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        if self.started {
            return Err(Error::invalid_state(
                "Cannot add stream after header written",
            ));
        }

        let index = self.streams.len();
        let pid = self.next_pid;
        self.next_pid += 1;

        // Set PCR PID to first video stream
        if stream.info.media_type == MediaType::Video && self.pcr_pid == 256 {
            self.pcr_pid = pid;
        }

        let stream_type = Self::get_stream_type(&stream.info.codec_id, stream.info.media_type);

        let muxer_stream = MuxerStream {
            index,
            pid,
            stream_type,
            media_type: stream.info.media_type,
            continuity_counter: 0,
        };

        self.stream_pids.insert(index, pid);
        self.muxer_streams.push(muxer_stream);
        self.streams.push(stream);

        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        if self.started {
            return Err(Error::invalid_state("Header already written"));
        }

        // If no streams, nothing to write
        if self.muxer_streams.is_empty() {
            return Err(Error::invalid_state("No streams added"));
        }

        // Write initial PAT and PMT
        self.write_tables()?;

        self.started = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.started {
            return Err(Error::invalid_state("Header not written"));
        }

        // Get PID for this stream
        let pid = *self
            .stream_pids
            .get(&packet.stream_index)
            .ok_or_else(|| Error::invalid_input("Invalid stream index"))?;

        // Periodically write PAT/PMT (every ~100 packets or 40ms worth)
        // Must be done before borrowing muxer_streams
        if self.packets_since_tables > 100 {
            self.write_tables()?;
        }

        // Get stream info - extract media_type before releasing borrow
        let media_type = {
            let stream = self
                .muxer_streams
                .iter()
                .find(|s| s.index == packet.stream_index)
                .ok_or_else(|| Error::invalid_input("Stream not found"))?;
            stream.media_type
        };

        // Get timestamps
        let pts = if packet.pts.is_valid() {
            Some(packet.pts.value)
        } else {
            None
        };
        let dts = if packet.dts.is_valid() {
            Some(packet.dts.value)
        } else {
            pts
        };

        // Update PCR base from DTS
        if let Some(d) = dts {
            if d > self.last_dts {
                self.pcr_base = d as u64;
                self.last_dts = d;
            }
        }

        // Get stream ID for PES header
        let stream_id = Self::get_stream_id(media_type, packet.stream_index);

        // Create PES header
        let pes_header = Self::create_pes_header(stream_id, pts, dts, packet.data.len());

        // Combine PES header and payload
        let mut pes_data = pes_header;
        pes_data.extend_from_slice(packet.data.as_slice());

        // Write PES packet as TS packets
        let include_pcr = packet.flags.keyframe && pid == self.pcr_pid;
        self.write_pes_packets(pid, &pes_data, include_pcr)?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        // Write final PAT/PMT
        if self.started {
            self.write_tables()?;
        }

        // Flush writer
        self.writer.flush().map_err(Error::Io)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpegts_muxer_creation() {
        let buffer = Vec::new();
        let muxer = MpegtsMuxer::new(buffer);
        assert!(!muxer.started);
        assert_eq!(muxer.packet_count, 0);
    }

    #[test]
    fn test_crc32_calculation() {
        // Test with known PAT data
        let test_data = [0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xF0, 0x00];
        let crc = calculate_crc32(&test_data);
        // CRC should be non-zero for valid data
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_pes_header_creation() {
        // Video stream, PTS only
        let header = MpegtsMuxer::<Vec<u8>>::create_pes_header(0xE0, Some(90000), None, 1000);
        assert!(header.len() >= 14); // Start code (3) + stream_id (1) + length (2) + extension (3) + PTS (5)

        // Verify start code
        assert_eq!(&header[0..3], &[0x00, 0x00, 0x01]);
        assert_eq!(header[3], 0xE0); // Video stream ID
    }

    #[test]
    fn test_pcr_adaptation_creation() {
        let adapt = MpegtsMuxer::<Vec<u8>>::create_pcr_adaptation(90000);
        assert_eq!(adapt.len(), 7); // Flags (1) + PCR (6)
        assert_eq!(adapt[0] & 0x10, 0x10); // PCR flag set
    }

    #[test]
    fn test_stream_type_detection() {
        assert_eq!(
            MpegtsMuxer::<Vec<u8>>::get_stream_type("h264", MediaType::Video),
            StreamType::VideoH264
        );
        assert_eq!(
            MpegtsMuxer::<Vec<u8>>::get_stream_type("hevc", MediaType::Video),
            StreamType::VideoH265
        );
        assert_eq!(
            MpegtsMuxer::<Vec<u8>>::get_stream_type("aac", MediaType::Audio),
            StreamType::AudioAAC
        );
    }

    #[test]
    fn test_mpegts_write_header() {
        use crate::format::{StreamInfo, VideoInfo};

        let buffer = Vec::new();
        let mut muxer = MpegtsMuxer::new(buffer);

        // Add a video stream
        let mut info = StreamInfo::new(0, MediaType::Video, "h264".to_string());
        info.video_info = Some(VideoInfo::new(1920, 1080));
        let stream = Stream::new(info);
        muxer.add_stream(stream).unwrap();

        let result = muxer.write_header();
        assert!(result.is_ok());
        assert!(muxer.started);
        assert!(muxer.packet_count >= 2); // At least PAT and PMT
    }
}
