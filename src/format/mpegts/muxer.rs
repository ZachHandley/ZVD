//! MPEG-TS muxer implementation

use super::{pids, StreamType, TsPacketHeader, TS_PACKET_SIZE};
use crate::error::{Error, Result};
use crate::format::{Muxer, Packet, Stream};
use std::collections::HashMap;
use std::io::Write;

/// MPEG-TS muxer
pub struct MpegtsMuxer<W: Write> {
    writer: W,
    streams: Vec<Stream>,
    stream_pids: HashMap<usize, u16>,      // stream index -> PID
    continuity_counters: HashMap<u16, u8>, // PID -> counter
    next_pid: u16,
    pmt_pid: u16,
    pcr_pid: u16,
    started: bool,
    packet_count: u64,
}

impl<W: Write> MpegtsMuxer<W> {
    /// Create a new MPEG-TS muxer
    pub fn new(writer: W) -> Self {
        MpegtsMuxer {
            writer,
            streams: Vec::new(),
            stream_pids: HashMap::new(),
            continuity_counters: HashMap::new(),
            next_pid: 256, // Start PIDs after reserved range
            pmt_pid: 256,
            pcr_pid: 256,
            started: false,
            packet_count: 0,
        }
    }

    /// Get continuity counter for PID
    fn get_counter(&mut self, pid: u16) -> u8 {
        let counter = self.continuity_counters.entry(pid).or_insert(0);
        let current = *counter;
        *counter = (*counter + 1) & 0x0F;
        current
    }

    /// Write TS packet
    fn write_ts_packet(&mut self, pid: u16, payload: &[u8], start: bool) -> Result<()> {
        let counter = self.get_counter(pid);
        let header = TsPacketHeader::new(pid, start, counter);

        // Write header
        self.writer
            .write_all(&header.to_bytes())
            .map_err(|e| Error::Io(e))?;

        // Write payload, padding with 0xFF if needed
        let payload_size = payload.len().min(TS_PACKET_SIZE - 4);
        self.writer
            .write_all(&payload[..payload_size])
            .map_err(|e| Error::Io(e))?;

        // Pad to packet size
        let padding_size = TS_PACKET_SIZE - 4 - payload_size;
        if padding_size > 0 {
            let padding = vec![0xFF; padding_size];
            self.writer.write_all(&padding).map_err(|e| Error::Io(e))?;
        }

        self.packet_count += 1;
        Ok(())
    }

    /// Write PAT (Program Association Table)
    fn write_pat(&mut self) -> Result<()> {
        let mut pat_data = Vec::new();

        // Table ID for PAT
        pat_data.push(0x00);

        // Section syntax indicator, reserved, section length
        pat_data.push(0xB0);
        pat_data.push(0x0D); // Length

        // Transport stream ID
        pat_data.extend_from_slice(&0x0001u16.to_be_bytes());

        // Version number, current/next indicator
        pat_data.push(0xC1);

        // Section number, last section number
        pat_data.push(0x00);
        pat_data.push(0x00);

        // Program number 1
        pat_data.extend_from_slice(&0x0001u16.to_be_bytes());

        // PMT PID
        pat_data.push(0xE0 | ((self.pmt_pid >> 8) & 0x1F) as u8);
        pat_data.push((self.pmt_pid & 0xFF) as u8);

        // CRC32 placeholder
        pat_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        self.write_ts_packet(pids::PAT, &pat_data, true)
    }

    /// Write PMT (Program Map Table)
    fn write_pmt(&mut self) -> Result<()> {
        let mut pmt_data = Vec::new();

        // Table ID for PMT
        pmt_data.push(0x02);

        // Section syntax indicator, reserved, section length (will be updated)
        pmt_data.push(0xB0);
        pmt_data.push(0x12); // Placeholder length

        // Program number
        pmt_data.extend_from_slice(&0x0001u16.to_be_bytes());

        // Version, current/next
        pmt_data.push(0xC1);

        // Section number, last section number
        pmt_data.push(0x00);
        pmt_data.push(0x00);

        // PCR PID
        pmt_data.push(0xE0 | ((self.pcr_pid >> 8) & 0x1F) as u8);
        pmt_data.push((self.pcr_pid & 0xFF) as u8);

        // Program info length
        pmt_data.push(0xF0);
        pmt_data.push(0x00);

        // Elementary streams (placeholder - would add actual stream info)

        // CRC32 placeholder
        pmt_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        self.write_ts_packet(self.pmt_pid, &pmt_data, true)
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

        self.stream_pids.insert(index, pid);
        self.streams.push(stream);

        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        if self.started {
            return Err(Error::invalid_state("Header already written"));
        }

        // Write PAT
        self.write_pat()?;

        // Write PMT
        self.write_pmt()?;

        self.started = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.started {
            return Err(Error::invalid_state("Header not written"));
        }

        // Get PID for this stream
        let pid = self
            .stream_pids
            .get(&packet.stream_index)
            .ok_or_else(|| Error::invalid_input("Invalid stream index"))?;

        // Placeholder - would create PES packet and write TS packets
        // For now, just write the data
        self.write_ts_packet(*pid, packet.data.as_slice(), true)?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        // TS has no trailer - can just stop writing packets
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
    fn test_mpegts_write_header() {
        let buffer = Vec::new();
        let mut muxer = MpegtsMuxer::new(buffer);
        let result = muxer.write_header();
        assert!(result.is_ok());
        assert!(muxer.started);
        assert!(muxer.packet_count >= 2); // At least PAT and PMT
    }
}
