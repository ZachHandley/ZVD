//! MP4 muxer implementation
//!
//! This module provides MP4 container writing using the mp4 crate.
//! Note: H.264 codec is patent-encumbered. See CODEC_LICENSES.md for details.

#[cfg(feature = "mp4-support")]
use crate::error::{Error, Result};
#[cfg(feature = "mp4-support")]
use crate::format::{Muxer, MuxerContext, Packet, Stream, StreamInfo};
#[cfg(feature = "mp4-support")]
use crate::util::MediaType;
#[cfg(feature = "mp4-support")]
use mp4::{
    AacConfig, AudioObjectType, AvcConfig, ChannelConfig, MediaConfig, Mp4Config, Mp4Sample,
    Mp4Writer, SampleFreqIndex, TrackConfig, TrackType,
};
#[cfg(feature = "mp4-support")]
use std::collections::HashMap;
#[cfg(feature = "mp4-support")]
use std::fs::File;
#[cfg(feature = "mp4-support")]
use std::io::BufWriter;
#[cfg(feature = "mp4-support")]
use std::path::Path;

#[cfg(feature = "mp4-support")]
/// NAL unit types for H.264
mod nal_types {
    pub const NAL_SLICE: u8 = 1;
    pub const NAL_DPA: u8 = 2;
    pub const NAL_DPB: u8 = 3;
    pub const NAL_DPC: u8 = 4;
    pub const NAL_IDR_SLICE: u8 = 5;
    pub const NAL_SEI: u8 = 6;
    pub const NAL_SPS: u8 = 7;
    pub const NAL_PPS: u8 = 8;
    pub const NAL_AUD: u8 = 9;
    pub const NAL_END_SEQUENCE: u8 = 10;
    pub const NAL_END_STREAM: u8 = 11;
    pub const NAL_FILLER_DATA: u8 = 12;
}

#[cfg(feature = "mp4-support")]
/// H.264 NAL unit parser for extracting SPS/PPS
struct H264NalParser;

#[cfg(feature = "mp4-support")]
impl H264NalParser {
    /// Find all NAL unit start code positions in the data
    /// Supports both 3-byte (0x000001) and 4-byte (0x00000001) start codes
    fn find_nal_units(data: &[u8]) -> Vec<(usize, usize)> {
        let mut units = Vec::new();
        let mut i = 0;
        let len = data.len();

        while i < len {
            // Look for start code: 0x000001 or 0x00000001
            if i + 3 <= len && data[i] == 0x00 && data[i + 1] == 0x00 {
                let nal_start;
                let _start_code_len;

                if data[i + 2] == 0x01 {
                    // 3-byte start code
                    _start_code_len = 3;
                    nal_start = i + 3;
                } else if i + 4 <= len && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                    // 4-byte start code
                    _start_code_len = 4;
                    nal_start = i + 4;
                } else {
                    i += 1;
                    continue;
                }

                // Find the end of this NAL unit (start of next or end of data)
                let mut nal_end = len;
                let mut j = nal_start;
                while j + 3 <= len {
                    if data[j] == 0x00 && data[j + 1] == 0x00 {
                        if data[j + 2] == 0x01 {
                            nal_end = j;
                            break;
                        } else if j + 4 <= len && data[j + 2] == 0x00 && data[j + 3] == 0x01 {
                            nal_end = j;
                            break;
                        }
                    }
                    j += 1;
                }

                if nal_start < nal_end {
                    units.push((nal_start, nal_end));
                }

                i = nal_end;
            } else {
                i += 1;
            }
        }

        units
    }

    /// Extract SPS and PPS NAL units from Annex B format data
    fn extract_sps_pps(data: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut sps_list = Vec::new();
        let mut pps_list = Vec::new();

        let nal_units = Self::find_nal_units(data);

        for (start, end) in nal_units {
            if start >= end || start >= data.len() {
                continue;
            }

            let nal_type = data[start] & 0x1F;

            match nal_type {
                nal_types::NAL_SPS => {
                    let sps = data[start..end].to_vec();
                    // Avoid duplicates
                    if !sps_list.contains(&sps) {
                        sps_list.push(sps);
                    }
                }
                nal_types::NAL_PPS => {
                    let pps = data[start..end].to_vec();
                    // Avoid duplicates
                    if !pps_list.contains(&pps) {
                        pps_list.push(pps);
                    }
                }
                _ => {}
            }
        }

        (sps_list, pps_list)
    }

    /// Parse SPS to extract video dimensions
    fn parse_sps_dimensions(sps: &[u8]) -> Option<(u16, u16)> {
        if sps.is_empty() {
            return None;
        }

        // Simple SPS parsing - skip NAL header, profile, constraints, level
        // Then parse pic_width_in_mbs_minus1 and pic_height_in_map_units_minus1
        // This is a simplified parser - full SPS parsing requires exponential golomb decoding

        // For now, return None and rely on stream info for dimensions
        // A full implementation would decode exp-golomb coded values
        None
    }

    /// Convert Annex B format to AVC format (length-prefixed NAL units)
    fn annex_b_to_avc(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let nal_units = Self::find_nal_units(data);

        for (start, end) in nal_units {
            if start >= end || start >= data.len() {
                continue;
            }

            let nal_data = &data[start..end];
            let nal_type = nal_data[0] & 0x1F;

            // Skip SPS, PPS, AUD, SEI for sample data (they go in the decoder config)
            match nal_type {
                nal_types::NAL_SPS | nal_types::NAL_PPS | nal_types::NAL_AUD => continue,
                _ => {}
            }

            // Write 4-byte length prefix (big-endian)
            let length = nal_data.len() as u32;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(nal_data);
        }

        output
    }

    /// Check if data is in Annex B format (has start codes)
    fn is_annex_b(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        // Check for 3-byte or 4-byte start code at beginning
        (data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01)
            || (data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x01)
    }
}

#[cfg(feature = "mp4-support")]
/// MP4 muxer
pub struct Mp4Muxer {
    writer: Option<Mp4Writer<BufWriter<File>>>,
    context: MuxerContext,
    track_ids: HashMap<usize, u32>,
    path: Option<std::path::PathBuf>,
    /// Cached SPS data from first video packet
    cached_sps: Vec<Vec<u8>>,
    /// Cached PPS data from first video packet
    cached_pps: Vec<Vec<u8>>,
    /// Whether we've extracted SPS/PPS yet
    sps_pps_extracted: bool,
    /// Pending packets waiting for SPS/PPS extraction
    pending_packets: Vec<Packet>,
    /// Track stream indices that need AVC conversion
    avc_streams: std::collections::HashSet<usize>,
}

#[cfg(feature = "mp4-support")]
impl Mp4Muxer {
    /// Create a new MP4 muxer
    pub fn new() -> Self {
        Mp4Muxer {
            writer: None,
            context: MuxerContext::new("mp4".to_string()),
            track_ids: HashMap::new(),
            path: None,
            cached_sps: Vec::new(),
            cached_pps: Vec::new(),
            sps_pps_extracted: false,
            pending_packets: Vec::new(),
            avc_streams: std::collections::HashSet::new(),
        }
    }

    /// Map sample rate to MP4 sample frequency index
    fn sample_rate_to_freq_index(sample_rate: u32) -> SampleFreqIndex {
        match sample_rate {
            96000 => SampleFreqIndex::Freq96000,
            88200 => SampleFreqIndex::Freq88200,
            64000 => SampleFreqIndex::Freq64000,
            48000 => SampleFreqIndex::Freq48000,
            44100 => SampleFreqIndex::Freq44100,
            32000 => SampleFreqIndex::Freq32000,
            24000 => SampleFreqIndex::Freq24000,
            22050 => SampleFreqIndex::Freq22050,
            16000 => SampleFreqIndex::Freq16000,
            12000 => SampleFreqIndex::Freq12000,
            11025 => SampleFreqIndex::Freq11025,
            8000 => SampleFreqIndex::Freq8000,
            7350 => SampleFreqIndex::Freq7350,
            _ => SampleFreqIndex::Freq48000, // Default to 48kHz
        }
    }

    /// Map channel count to MP4 channel config
    fn channels_to_config(channels: u16) -> ChannelConfig {
        match channels {
            1 => ChannelConfig::Mono,
            2 => ChannelConfig::Stereo,
            3 => ChannelConfig::Three,
            4 => ChannelConfig::Four,
            5 => ChannelConfig::Five,
            6 => ChannelConfig::FiveOne,
            7 => ChannelConfig::SevenOne,
            8 => ChannelConfig::SevenOne,
            _ => ChannelConfig::Stereo, // Default to stereo
        }
    }

    /// Extract SPS/PPS from extradata or packet data
    fn extract_sps_pps_from_data(&mut self, data: &[u8], extradata: Option<&[u8]>) {
        // First try extradata
        if let Some(extra) = extradata {
            if extra.len() > 6 {
                // Check if it's AVC decoder configuration record format
                if extra[0] == 1 {
                    // Parse AVCDecoderConfigurationRecord
                    self.parse_avc_decoder_config(extra);
                } else if H264NalParser::is_annex_b(extra) {
                    // Annex B format in extradata
                    let (sps, pps) = H264NalParser::extract_sps_pps(extra);
                    if !sps.is_empty() {
                        self.cached_sps = sps;
                    }
                    if !pps.is_empty() {
                        self.cached_pps = pps;
                    }
                }
            }
        }

        // If still no SPS/PPS, try from packet data
        if self.cached_sps.is_empty() || self.cached_pps.is_empty() {
            if H264NalParser::is_annex_b(data) {
                let (sps, pps) = H264NalParser::extract_sps_pps(data);
                if self.cached_sps.is_empty() && !sps.is_empty() {
                    self.cached_sps = sps;
                }
                if self.cached_pps.is_empty() && !pps.is_empty() {
                    self.cached_pps = pps;
                }
            }
        }

        if !self.cached_sps.is_empty() && !self.cached_pps.is_empty() {
            self.sps_pps_extracted = true;
        }
    }

    /// Parse AVC decoder configuration record format
    fn parse_avc_decoder_config(&mut self, data: &[u8]) {
        if data.len() < 7 {
            return;
        }

        // AVCDecoderConfigurationRecord format:
        // configurationVersion = 1 byte
        // AVCProfileIndication = 1 byte
        // profile_compatibility = 1 byte
        // AVCLevelIndication = 1 byte
        // lengthSizeMinusOne = 1 byte (lower 2 bits)
        // numOfSequenceParameterSets = 1 byte (lower 5 bits)
        // Then SPS data
        // numOfPictureParameterSets = 1 byte
        // Then PPS data

        let num_sps = (data[5] & 0x1F) as usize;
        let mut offset = 6;

        // Read SPS
        for _ in 0..num_sps {
            if offset + 2 > data.len() {
                break;
            }
            let sps_len = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
            offset += 2;
            if offset + sps_len > data.len() {
                break;
            }
            let sps = data[offset..offset + sps_len].to_vec();
            if !self.cached_sps.contains(&sps) {
                self.cached_sps.push(sps);
            }
            offset += sps_len;
        }

        // Read PPS count
        if offset >= data.len() {
            return;
        }
        let num_pps = data[offset] as usize;
        offset += 1;

        // Read PPS
        for _ in 0..num_pps {
            if offset + 2 > data.len() {
                break;
            }
            let pps_len = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
            offset += 2;
            if offset + pps_len > data.len() {
                break;
            }
            let pps = data[offset..offset + pps_len].to_vec();
            if !self.cached_pps.contains(&pps) {
                self.cached_pps.push(pps);
            }
            offset += pps_len;
        }
    }

    /// Create a track config from stream info with cached SPS/PPS
    fn create_track_config(&self, stream_info: &StreamInfo, stream: &Stream) -> Result<TrackConfig> {
        let timescale = stream_info.time_base.den.abs() as u32;

        match stream_info.media_type {
            MediaType::Video => {
                let video_info = stream_info
                    .video_info
                    .as_ref()
                    .ok_or_else(|| Error::format("Video track missing video info"))?;

                // Use cached SPS/PPS or from extradata
                let mut sps_list = self.cached_sps.clone();
                let mut pps_list = self.cached_pps.clone();

                // Try to get from extradata if not cached
                if sps_list.is_empty() || pps_list.is_empty() {
                    if let Some(ref extradata) = stream.extradata {
                        if extradata.len() > 6 && extradata[0] == 1 {
                            // Parse AVCDecoderConfigurationRecord
                            let (sps, pps) = Self::parse_avc_config_static(extradata);
                            if sps_list.is_empty() {
                                sps_list = sps;
                            }
                            if pps_list.is_empty() {
                                pps_list = pps;
                            }
                        } else if H264NalParser::is_annex_b(extradata) {
                            let (sps, pps) = H264NalParser::extract_sps_pps(extradata);
                            if sps_list.is_empty() {
                                sps_list = sps;
                            }
                            if pps_list.is_empty() {
                                pps_list = pps;
                            }
                        }
                    }
                }

                // Flatten SPS/PPS lists for mp4 crate
                let seq_param_set = sps_list.into_iter().flatten().collect();
                let pic_param_set = pps_list.into_iter().flatten().collect();

                let avc_config = AvcConfig {
                    width: video_info.width as u16,
                    height: video_info.height as u16,
                    seq_param_set,
                    pic_param_set,
                };

                Ok(TrackConfig {
                    track_type: TrackType::Video,
                    timescale,
                    language: "und".to_string(),
                    media_conf: MediaConfig::AvcConfig(avc_config),
                })
            }
            MediaType::Audio => {
                let audio_info = stream_info
                    .audio_info
                    .as_ref()
                    .ok_or_else(|| Error::format("Audio track missing audio info"))?;

                // Create AAC config
                let aac_config = AacConfig {
                    bitrate: audio_info.bit_rate.unwrap_or(128000) as u32,
                    profile: AudioObjectType::AacLowComplexity,
                    freq_index: Self::sample_rate_to_freq_index(audio_info.sample_rate),
                    chan_conf: Self::channels_to_config(audio_info.channels),
                };

                Ok(TrackConfig {
                    track_type: TrackType::Audio,
                    timescale,
                    language: "und".to_string(),
                    media_conf: MediaConfig::AacConfig(aac_config),
                })
            }
            _ => Err(Error::format(format!(
                "Unsupported media type for MP4: {:?}",
                stream_info.media_type
            ))),
        }
    }

    /// Static method to parse AVC decoder config
    fn parse_avc_config_static(data: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut sps_list = Vec::new();
        let mut pps_list = Vec::new();

        if data.len() < 7 {
            return (sps_list, pps_list);
        }

        let num_sps = (data[5] & 0x1F) as usize;
        let mut offset = 6;

        for _ in 0..num_sps {
            if offset + 2 > data.len() {
                break;
            }
            let sps_len = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
            offset += 2;
            if offset + sps_len > data.len() {
                break;
            }
            sps_list.push(data[offset..offset + sps_len].to_vec());
            offset += sps_len;
        }

        if offset >= data.len() {
            return (sps_list, pps_list);
        }
        let num_pps = data[offset] as usize;
        offset += 1;

        for _ in 0..num_pps {
            if offset + 2 > data.len() {
                break;
            }
            let pps_len = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
            offset += 2;
            if offset + pps_len > data.len() {
                break;
            }
            pps_list.push(data[offset..offset + pps_len].to_vec());
            offset += pps_len;
        }

        (sps_list, pps_list)
    }

    /// Process a video packet, converting from Annex B to AVC format if needed
    fn process_video_packet(&self, data: &[u8], stream_index: usize) -> Vec<u8> {
        if self.avc_streams.contains(&stream_index) && H264NalParser::is_annex_b(data) {
            H264NalParser::annex_b_to_avc(data)
        } else {
            data.to_vec()
        }
    }
}

#[cfg(feature = "mp4-support")]
impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "mp4-support")]
impl Muxer for Mp4Muxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        // Store path for later when we write header
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        // Mark H.264 video streams for AVC conversion
        if stream.info.media_type == MediaType::Video {
            let codec_id = stream.info.codec_id.to_lowercase();
            if codec_id.contains("h264") || codec_id.contains("avc") {
                self.avc_streams.insert(self.context.streams().len());
            }
        }

        // Try to extract SPS/PPS from extradata
        if stream.info.media_type == MediaType::Video {
            if let Some(ref extradata) = stream.extradata {
                self.extract_sps_pps_from_data(extradata, Some(extradata));
            }
        }

        let index = self.context.add_stream(stream);
        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| Error::format("Path not set"))?;

        let file = File::create(path)
            .map_err(|e| Error::format(format!("Failed to create MP4 file: {}", e)))?;

        let buf_writer = BufWriter::new(file);

        // Create MP4 config
        let config = Mp4Config {
            major_brand: str::parse("isom").unwrap(),
            minor_version: 512,
            compatible_brands: vec![
                str::parse("isom").unwrap(),
                str::parse("iso2").unwrap(),
                str::parse("avc1").unwrap(),
                str::parse("mp41").unwrap(),
            ],
            timescale: 1000,
        };

        // Initialize writer
        let mut writer = Mp4Writer::write_start(buf_writer, &config)
            .map_err(|e| Error::format(format!("Failed to initialize MP4 writer: {}", e)))?;

        // Add tracks for each stream in context
        let mut track_id: u32 = 1;
        let streams: Vec<Stream> = self.context.streams().to_vec();
        for stream in &streams {
            let track_config = self.create_track_config(&stream.info, stream)?;

            writer
                .add_track(&track_config)
                .map_err(|e| Error::format(format!("Failed to add track: {}", e)))?;

            self.track_ids.insert(stream.info.index, track_id);
            track_id += 1;
        }

        self.writer = Some(writer);
        self.context.set_header_written();
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        // If we haven't extracted SPS/PPS yet and this is a video packet, try to extract
        if !self.sps_pps_extracted && packet.codec_type == MediaType::Video {
            // Clone extradata to avoid borrow conflict
            let extradata = self
                .context
                .streams()
                .get(packet.stream_index)
                .and_then(|s| s.extradata.clone());
            let extradata_slice = extradata.as_deref();
            self.extract_sps_pps_from_data(packet.data.as_slice(), extradata_slice);
        }

        // Get track_id first (copy the value to avoid borrow)
        let track_id = *self.track_ids.get(&packet.stream_index).ok_or_else(|| {
            Error::format(format!(
                "Track not found for stream {}",
                packet.stream_index
            ))
        })?;

        // Process video packets to convert Annex B to AVC format
        // Must be done before borrowing writer
        let data = if packet.codec_type == MediaType::Video {
            self.process_video_packet(packet.data.as_slice(), packet.stream_index)
        } else {
            packet.data.as_slice().to_vec()
        };

        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("MP4 writer not initialized"))?;

        // Convert our packet to MP4 sample
        let sample = Mp4Sample {
            start_time: if packet.pts.is_valid() {
                packet.pts.value as u64
            } else if packet.dts.is_valid() {
                packet.dts.value as u64
            } else {
                0
            },
            duration: packet.duration as u32,
            rendering_offset: if packet.pts.is_valid() && packet.dts.is_valid() {
                (packet.pts.value - packet.dts.value) as i32
            } else {
                0
            },
            is_sync: packet.flags.keyframe,
            bytes: bytes::Bytes::from(data),
        };

        writer
            .write_sample(track_id, &sample)
            .map_err(|e| Error::format(format!("Failed to write sample: {}", e)))?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer
                .write_end()
                .map_err(|e| Error::format(format!("Failed to finalize MP4 file: {}", e)))?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "mp4-support"))]
/// MP4 muxer (disabled - enable with 'mp4-support' feature)
pub struct Mp4Muxer;

#[cfg(not(feature = "mp4-support"))]
impl Mp4Muxer {
    pub fn new() -> Self {
        panic!("MP4 support not enabled. Build with --features mp4-support");
    }
}

#[cfg(not(feature = "mp4-support"))]
impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "mp4-support"))]
mod tests {
    use super::*;

    #[test]
    fn test_nal_unit_finding() {
        // Test data with two NAL units
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // Start code
            0x67, 0x42, 0x00, 0x1E, // SPS NAL (type 7)
            0x00, 0x00, 0x01, // Start code
            0x68, 0xCE, 0x38, 0x80, // PPS NAL (type 8)
        ];

        let units = H264NalParser::find_nal_units(&data);
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn test_sps_pps_extraction() {
        // Minimal SPS/PPS data
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // Start code
            0x67, 0x42, 0x00, 0x1E, // SPS
            0x00, 0x00, 0x01, // Start code
            0x68, 0xCE, 0x38, 0x80, // PPS
        ];

        let (sps, pps) = H264NalParser::extract_sps_pps(&data);
        assert_eq!(sps.len(), 1);
        assert_eq!(pps.len(), 1);
        assert_eq!(sps[0][0] & 0x1F, nal_types::NAL_SPS);
        assert_eq!(pps[0][0] & 0x1F, nal_types::NAL_PPS);
    }

    #[test]
    fn test_is_annex_b() {
        let annex_b = vec![0x00, 0x00, 0x00, 0x01, 0x67];
        let not_annex_b = vec![0x00, 0x00, 0x00, 0x04, 0x67];

        assert!(H264NalParser::is_annex_b(&annex_b));
        assert!(!H264NalParser::is_annex_b(&not_annex_b));
    }

    #[test]
    fn test_mp4_muxer_creation() {
        let muxer = Mp4Muxer::new();
        assert!(muxer.writer.is_none());
        assert!(muxer.cached_sps.is_empty());
        assert!(muxer.cached_pps.is_empty());
    }
}
