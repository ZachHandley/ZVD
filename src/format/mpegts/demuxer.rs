//! MPEG-TS demuxer implementation

use super::{pids, StreamType, TsPacketHeader, TS_PACKET_SIZE, TS_SYNC_BYTE};
use crate::error::{Error, Result};
use crate::format::{Demuxer, Packet, Stream};
use std::collections::HashMap;
use std::io::Read;

/// MPEG-TS demuxer
pub struct MpegtsDemuxer<R: Read> {
    reader: R,
    streams: Vec<Stream>,
    stream_map: HashMap<u16, usize>, // PID -> stream index
    duration: Option<f64>,
    packet_count: u64,
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
        }
    }

    /// Find sync byte in stream
    fn find_sync(&mut self) -> Result<()> {
        let mut byte = [0u8; 1];
        loop {
            self.reader
                .read_exact(&mut byte)
                .map_err(|e| Error::Io(e))?;
            if byte[0] == TS_SYNC_BYTE {
                return Ok(());
            }
        }
    }

    /// Read TS packet
    fn read_ts_packet(&mut self) -> Result<Vec<u8>> {
        let mut packet = vec![0u8; TS_PACKET_SIZE];
        self.reader
            .read_exact(&mut packet)
            .map_err(|e| Error::Io(e))?;

        if packet[0] != TS_SYNC_BYTE {
            return Err(Error::invalid_input("Lost sync"));
        }

        Ok(packet)
    }

    /// Parse PAT (Program Association Table)
    fn parse_pat(&mut self, payload: &[u8]) -> Result<()> {
        // Placeholder - would parse PAT to find PMT PIDs
        Ok(())
    }

    /// Parse PMT (Program Map Table)
    fn parse_pmt(&mut self, payload: &[u8]) -> Result<()> {
        // Placeholder - would parse PMT to find elementary stream PIDs and types
        Ok(())
    }

    /// Parse PES (Packetized Elementary Stream) packet
    fn parse_pes(&mut self, pid: u16, payload: &[u8]) -> Result<Packet> {
        // Placeholder - would parse PES header and extract data
        Err(Error::TryAgain)
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
        for _ in 0..100 {
            let packet = self.read_ts_packet()?;
            let header_bytes: [u8; 4] = packet[0..4]
                .try_into()
                .map_err(|_| Error::invalid_input("Invalid TS packet"))?;

            let header =
                TsPacketHeader::from_bytes(&header_bytes).map_err(|e| Error::invalid_input(e))?;

            if header.pid == pids::PAT {
                // Parse PAT
                self.parse_pat(&packet[4..])?;
            }
        }

        Ok(())
    }

    fn read_packet(&mut self) -> Result<Packet> {
        loop {
            let packet = self.read_ts_packet()?;
            let header_bytes: [u8; 4] = packet[0..4]
                .try_into()
                .map_err(|_| Error::invalid_input("Invalid TS packet"))?;

            let header =
                TsPacketHeader::from_bytes(&header_bytes).map_err(|e| Error::invalid_input(e))?;

            self.packet_count += 1;

            // Skip null packets
            if header.pid == pids::NULL {
                continue;
            }

            // Check if this is a stream we're tracking
            if self.stream_map.contains_key(&header.pid) {
                // Parse PES packet
                return self.parse_pes(header.pid, &packet[4..]);
            }
        }
    }

    fn seek(&mut self, _stream_index: usize, _timestamp: i64) -> Result<()> {
        // TS seeking is complex - need to find sync points
        Err(Error::unsupported("Seeking not supported for MPEG-TS"))
    }

    fn streams(&self) -> &[crate::format::Stream] {
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
    }
}
