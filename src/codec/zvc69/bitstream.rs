//! ZVC69 Bitstream Format Implementation
//!
//! This module implements the ZVC69 neural video codec bitstream format,
//! providing serialization and deserialization of file headers, frame headers,
//! and index tables for random access seeking.
//!
//! ## Bitstream Structure
//!
//! ```text
//! +-----------------------+
//! |  FILE HEADER (64B)    |
//! +-----------------------+
//! |  INDEX TABLE (opt)    |
//! +-----------------------+
//! |  FRAME 0 (I-Frame)    |
//! +-----------------------+
//! |  FRAME 1 ...          |
//! +-----------------------+
//! |  TRAILER (opt)        |
//! +-----------------------+
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::bitstream::{BitstreamWriter, BitstreamReader, FileHeader};
//!
//! // Writing
//! let mut writer = BitstreamWriter::new(file);
//! writer.write_file_header(&header)?;
//! writer.write_frame_header(&frame_header)?;
//! writer.write_frame_data(&data)?;
//! writer.finalize()?;
//!
//! // Reading
//! let mut reader = BitstreamReader::new(file);
//! let header = reader.read_file_header()?;
//! let frame_header = reader.read_frame_header()?;
//! let data = reader.read_frame_data(frame_header.frame_size as usize)?;
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};

use super::error::ZVC69Error;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes for ZVC69 file format
pub const MAGIC: &[u8; 6] = b"ZVC69\0";

/// Size of the file header in bytes
pub const FILE_HEADER_SIZE: usize = 64;

/// Size of a frame header in bytes
pub const FRAME_HEADER_SIZE: usize = 32;

/// Size of an index entry in bytes
pub const INDEX_ENTRY_SIZE: usize = 16;

/// Size of the index table header in bytes
pub const INDEX_TABLE_HEADER_SIZE: usize = 8;

/// Current bitstream major version
pub const VERSION_MAJOR: u8 = 1;

/// Current bitstream minor version
pub const VERSION_MINOR: u8 = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Flag Definitions
// ─────────────────────────────────────────────────────────────────────────────

/// File header flags
pub mod flags {
    /// Index table present for random access
    pub const HAS_INDEX: u32 = 1 << 0;
    /// B-frames present in stream
    pub const HAS_B_FRAMES: u32 = 1 << 1;
    /// Trailer with checksums present
    pub const HAS_TRAILER: u32 = 1 << 2;
    /// Model uses integer arithmetic (deterministic)
    pub const INTEGERIZED: u32 = 1 << 3;
    /// Uses Gaussian Mixture Model entropy
    pub const GMM_ENTROPY: u32 = 1 << 4;
    /// Checkerboard autoregressive context
    pub const CHECKERBOARD: u32 = 1 << 5;
}

/// Index entry flags
pub mod index_flags {
    /// This is an I-frame (random access point)
    pub const IS_KEYFRAME: u32 = 1 << 0;
    /// Frame is used as reference
    pub const IS_REFERENCE: u32 = 1 << 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Color Space Enum
// ─────────────────────────────────────────────────────────────────────────────

/// Color space identifier for the bitstream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorSpace {
    /// YUV 4:2:0 subsampling
    #[default]
    Yuv420 = 0,
    /// YUV 4:4:4 (no subsampling)
    Yuv444 = 1,
    /// RGB color space
    Rgb = 2,
}

impl TryFrom<u8> for ColorSpace {
    type Error = ZVC69Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ColorSpace::Yuv420),
            1 => Ok(ColorSpace::Yuv444),
            2 => Ok(ColorSpace::Rgb),
            _ => Err(ZVC69Error::invalid_header(format!(
                "Invalid color space value: {}",
                value
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame Type Enum
// ─────────────────────────────────────────────────────────────────────────────

/// Frame type in the bitstream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    /// Intra frame (keyframe, random access point)
    #[default]
    I = 0,
    /// Predicted frame (forward reference)
    P = 1,
    /// Bidirectional frame (forward and backward reference)
    B = 2,
    /// Bidirectional frame with bidirectional prediction
    BBi = 3,
}

impl FrameType {
    /// Check if this frame type is a keyframe
    pub fn is_keyframe(&self) -> bool {
        matches!(self, FrameType::I)
    }

    /// Check if this frame type requires reference frames
    pub fn needs_reference(&self) -> bool {
        !matches!(self, FrameType::I)
    }
}

impl TryFrom<u8> for FrameType {
    type Error = ZVC69Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(FrameType::I),
            1 => Ok(FrameType::P),
            2 => Ok(FrameType::B),
            3 => Ok(FrameType::BBi),
            _ => Err(ZVC69Error::invalid_header(format!(
                "Invalid frame type value: {}",
                value
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// File Header (64 bytes)
// ─────────────────────────────────────────────────────────────────────────────

/// ZVC69 file header (64 bytes)
///
/// Contains global metadata about the video stream including dimensions,
/// framerate, color space, and index table location for seeking.
///
/// ## Layout (Little Endian)
///
/// | Offset | Size | Field              | Description                           |
/// |--------|------|--------------------|---------------------------------------|
/// | 0x00   | 6    | magic              | "ZVC69\0"                             |
/// | 0x06   | 1    | version_major      | Bitstream major version               |
/// | 0x07   | 1    | version_minor      | Bitstream minor version               |
/// | 0x08   | 4    | flags              | Feature flags                         |
/// | 0x0C   | 2    | width              | Video width in pixels                 |
/// | 0x0E   | 2    | height             | Video height in pixels                |
/// | 0x10   | 2    | framerate_num      | Framerate numerator                   |
/// | 0x12   | 2    | framerate_den      | Framerate denominator                 |
/// | 0x14   | 4    | total_frames       | Total frame count (0 = unknown)       |
/// | 0x18   | 1    | gop_size           | I-frame interval                      |
/// | 0x19   | 1    | quality_level      | Quality setting (0-100)               |
/// | 0x1A   | 1    | color_space        | 0=YUV420, 1=YUV444, 2=RGB             |
/// | 0x1B   | 1    | bit_depth          | 8, 10, or 12 bits per component       |
/// | 0x1C   | 4    | model_hash         | First 4 bytes of model SHA-256        |
/// | 0x20   | 8    | index_offset       | Byte offset to index table (0 = none) |
/// | 0x28   | 4    | latent_channels    | Number of latent channels             |
/// | 0x2C   | 4    | hyperprior_channels| Number of hyperprior channels         |
/// | 0x30   | 16   | reserved           | Reserved (must be zero)               |
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic bytes ("ZVC69\0")
    pub magic: [u8; 6],
    /// Bitstream major version
    pub version_major: u8,
    /// Bitstream minor version
    pub version_minor: u8,
    /// Feature flags (see `flags` module)
    pub flags: u32,
    /// Video width in pixels
    pub width: u16,
    /// Video height in pixels
    pub height: u16,
    /// Framerate numerator
    pub framerate_num: u16,
    /// Framerate denominator
    pub framerate_den: u16,
    /// Total frame count (0 = unknown/streaming)
    pub total_frames: u32,
    /// I-frame interval (GOP size)
    pub gop_size: u8,
    /// Quality level (0-100)
    pub quality_level: u8,
    /// Color space
    pub color_space: u8,
    /// Bit depth (8, 10, or 12)
    pub bit_depth: u8,
    /// First 4 bytes of model SHA-256 hash
    pub model_hash: [u8; 4],
    /// Byte offset to index table (0 = no index)
    pub index_offset: u64,
    /// Number of latent channels
    pub latent_channels: u32,
    /// Number of hyperprior channels
    pub hyperprior_channels: u32,
}

impl Default for FileHeader {
    fn default() -> Self {
        FileHeader {
            magic: *MAGIC,
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            flags: 0,
            width: 1920,
            height: 1080,
            framerate_num: 30,
            framerate_den: 1,
            total_frames: 0,
            gop_size: 10,
            quality_level: 50,
            color_space: ColorSpace::Yuv420 as u8,
            bit_depth: 8,
            model_hash: [0; 4],
            index_offset: 0,
            latent_channels: 192,
            hyperprior_channels: 128,
        }
    }
}

impl FileHeader {
    /// Create a new file header with specified dimensions
    pub fn new(width: u16, height: u16) -> Self {
        FileHeader {
            width,
            height,
            ..Default::default()
        }
    }

    /// Create a builder for constructing a file header
    pub fn builder() -> FileHeaderBuilder {
        FileHeaderBuilder::default()
    }

    /// Check if this header has valid magic bytes
    pub fn has_valid_magic(&self) -> bool {
        self.magic == *MAGIC
    }

    /// Check if this header's version is supported
    pub fn is_version_supported(&self) -> bool {
        self.version_major == VERSION_MAJOR
    }

    /// Check if index table is present
    pub fn has_index(&self) -> bool {
        (self.flags & flags::HAS_INDEX) != 0 && self.index_offset > 0
    }

    /// Check if B-frames are present
    pub fn has_b_frames(&self) -> bool {
        (self.flags & flags::HAS_B_FRAMES) != 0
    }

    /// Check if trailer is present
    pub fn has_trailer(&self) -> bool {
        (self.flags & flags::HAS_TRAILER) != 0
    }

    /// Get the color space enum
    pub fn color_space_enum(&self) -> Result<ColorSpace, ZVC69Error> {
        ColorSpace::try_from(self.color_space)
    }

    /// Get framerate as f64
    pub fn framerate(&self) -> f64 {
        if self.framerate_den == 0 {
            0.0
        } else {
            f64::from(self.framerate_num) / f64::from(self.framerate_den)
        }
    }

    /// Validate the header contents
    pub fn validate(&self) -> Result<(), ZVC69Error> {
        // Check magic
        if !self.has_valid_magic() {
            return Err(ZVC69Error::invalid_header("Invalid magic bytes"));
        }

        // Check version
        if !self.is_version_supported() {
            return Err(ZVC69Error::UnsupportedBitstreamVersion {
                version: u32::from(self.version_major),
                min_supported: u32::from(VERSION_MAJOR),
                max_supported: u32::from(VERSION_MAJOR),
            });
        }

        // Check dimensions
        if self.width == 0 || self.height == 0 {
            return Err(ZVC69Error::invalid_header("Invalid dimensions (zero)"));
        }

        // Check framerate
        if self.framerate_den == 0 {
            return Err(ZVC69Error::invalid_header(
                "Invalid framerate denominator (zero)",
            ));
        }

        // Check bit depth
        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err(ZVC69Error::invalid_header(format!(
                "Invalid bit depth: {}",
                self.bit_depth
            )));
        }

        // Check color space
        ColorSpace::try_from(self.color_space)?;

        Ok(())
    }
}

/// Builder for FileHeader
#[derive(Debug, Default)]
pub struct FileHeaderBuilder {
    header: FileHeader,
}

impl FileHeaderBuilder {
    /// Set video dimensions
    pub fn dimensions(mut self, width: u16, height: u16) -> Self {
        self.header.width = width;
        self.header.height = height;
        self
    }

    /// Set framerate
    pub fn framerate(mut self, num: u16, den: u16) -> Self {
        self.header.framerate_num = num;
        self.header.framerate_den = den;
        self
    }

    /// Set total frame count
    pub fn total_frames(mut self, count: u32) -> Self {
        self.header.total_frames = count;
        self
    }

    /// Set GOP size
    pub fn gop_size(mut self, size: u8) -> Self {
        self.header.gop_size = size;
        self
    }

    /// Set quality level
    pub fn quality_level(mut self, level: u8) -> Self {
        self.header.quality_level = level;
        self
    }

    /// Set color space
    pub fn color_space(mut self, cs: ColorSpace) -> Self {
        self.header.color_space = cs as u8;
        self
    }

    /// Set bit depth
    pub fn bit_depth(mut self, depth: u8) -> Self {
        self.header.bit_depth = depth;
        self
    }

    /// Set model hash
    pub fn model_hash(mut self, hash: [u8; 4]) -> Self {
        self.header.model_hash = hash;
        self
    }

    /// Set index offset
    pub fn index_offset(mut self, offset: u64) -> Self {
        self.header.index_offset = offset;
        if offset > 0 {
            self.header.flags |= flags::HAS_INDEX;
        }
        self
    }

    /// Set latent channels
    pub fn latent_channels(mut self, channels: u32) -> Self {
        self.header.latent_channels = channels;
        self
    }

    /// Set hyperprior channels
    pub fn hyperprior_channels(mut self, channels: u32) -> Self {
        self.header.hyperprior_channels = channels;
        self
    }

    /// Set flags
    pub fn flags(mut self, flags: u32) -> Self {
        self.header.flags = flags;
        self
    }

    /// Enable B-frames flag
    pub fn with_b_frames(mut self) -> Self {
        self.header.flags |= flags::HAS_B_FRAMES;
        self
    }

    /// Enable trailer flag
    pub fn with_trailer(mut self) -> Self {
        self.header.flags |= flags::HAS_TRAILER;
        self
    }

    /// Build and validate the header
    pub fn build(self) -> Result<FileHeader, ZVC69Error> {
        self.header.validate()?;
        Ok(self.header)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> FileHeader {
        self.header
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame Header (32 bytes)
// ─────────────────────────────────────────────────────────────────────────────

/// ZVC69 frame header (32 bytes)
///
/// Contains per-frame metadata including type, size, timestamps,
/// and reference information.
///
/// ## Layout (Little Endian)
///
/// | Offset | Size | Field           | Description                    |
/// |--------|------|-----------------|--------------------------------|
/// | 0x00   | 1    | frame_type      | 0=I, 1=P, 2=B, 3=B-bi          |
/// | 0x01   | 1    | temporal_layer  | Hierarchical layer (0-7)       |
/// | 0x02   | 2    | reference_flags | Which references are used      |
/// | 0x04   | 4    | frame_size      | Compressed frame size in bytes |
/// | 0x08   | 8    | pts             | Presentation timestamp (90kHz) |
/// | 0x10   | 8    | dts             | Decode timestamp (90kHz)       |
/// | 0x18   | 2    | qp_offset       | QP offset from base (signed)   |
/// | 0x1A   | 2    | checksum        | CRC-16 of frame data           |
/// | 0x1C   | 4    | reserved        | Reserved (must be zero)        |
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    /// Frame type (I, P, B, B-bi)
    pub frame_type: FrameType,
    /// Temporal layer for hierarchical B-frames (0-7)
    pub temporal_layer: u8,
    /// Reference flags indicating which references are used
    pub reference_flags: u16,
    /// Total compressed frame size in bytes
    pub frame_size: u32,
    /// Presentation timestamp (90kHz timebase)
    pub pts: u64,
    /// Decode timestamp (90kHz timebase)
    pub dts: u64,
    /// Quality parameter offset from base
    pub qp_offset: i16,
    /// CRC-16 checksum of frame data
    pub checksum: u16,
}

impl Default for FrameHeader {
    fn default() -> Self {
        FrameHeader {
            frame_type: FrameType::I,
            temporal_layer: 0,
            reference_flags: 0,
            frame_size: 0,
            pts: 0,
            dts: 0,
            qp_offset: 0,
            checksum: 0,
        }
    }
}

impl FrameHeader {
    /// Create a new frame header
    pub fn new(frame_type: FrameType, frame_size: u32) -> Self {
        FrameHeader {
            frame_type,
            frame_size,
            ..Default::default()
        }
    }

    /// Create an I-frame header
    pub fn i_frame(frame_size: u32, pts: u64) -> Self {
        FrameHeader {
            frame_type: FrameType::I,
            frame_size,
            pts,
            dts: pts,
            ..Default::default()
        }
    }

    /// Create a P-frame header
    pub fn p_frame(frame_size: u32, pts: u64, reference_flags: u16) -> Self {
        FrameHeader {
            frame_type: FrameType::P,
            frame_size,
            pts,
            dts: pts,
            reference_flags,
            ..Default::default()
        }
    }

    /// Create a B-frame header
    pub fn b_frame(
        frame_size: u32,
        pts: u64,
        dts: u64,
        temporal_layer: u8,
        reference_flags: u16,
    ) -> Self {
        FrameHeader {
            frame_type: FrameType::B,
            frame_size,
            pts,
            dts,
            temporal_layer,
            reference_flags,
            ..Default::default()
        }
    }

    /// Check if this is a keyframe
    pub fn is_keyframe(&self) -> bool {
        self.frame_type.is_keyframe()
    }

    /// Set the checksum
    pub fn with_checksum(mut self, checksum: u16) -> Self {
        self.checksum = checksum;
        self
    }

    /// Set the QP offset
    pub fn with_qp_offset(mut self, offset: i16) -> Self {
        self.qp_offset = offset;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Index Table
// ─────────────────────────────────────────────────────────────────────────────

/// Index entry for seeking (16 bytes)
///
/// Stores the byte offset and metadata for a single frame,
/// enabling efficient random access within the bitstream.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexEntry {
    /// Frame number (0-indexed)
    pub frame_number: u32,
    /// Byte offset from file start
    pub byte_offset: u64,
    /// Flags (IS_KEYFRAME, IS_REFERENCE)
    pub flags: u32,
}

impl IndexEntry {
    /// Create a new index entry
    pub fn new(frame_number: u32, byte_offset: u64, flags: u32) -> Self {
        IndexEntry {
            frame_number,
            byte_offset,
            flags,
        }
    }

    /// Create an index entry for a keyframe
    pub fn keyframe(frame_number: u32, byte_offset: u64) -> Self {
        IndexEntry {
            frame_number,
            byte_offset,
            flags: index_flags::IS_KEYFRAME | index_flags::IS_REFERENCE,
        }
    }

    /// Create an index entry for a reference frame
    pub fn reference_frame(frame_number: u32, byte_offset: u64) -> Self {
        IndexEntry {
            frame_number,
            byte_offset,
            flags: index_flags::IS_REFERENCE,
        }
    }

    /// Check if this is a keyframe
    pub fn is_keyframe(&self) -> bool {
        (self.flags & index_flags::IS_KEYFRAME) != 0
    }

    /// Check if this is a reference frame
    pub fn is_reference(&self) -> bool {
        (self.flags & index_flags::IS_REFERENCE) != 0
    }
}

/// Index table for random access seeking
///
/// Contains a list of frame entries with their byte offsets,
/// enabling efficient seeking to any frame in the stream.
#[derive(Debug, Clone, Default)]
pub struct IndexTable {
    /// List of index entries
    pub entries: Vec<IndexEntry>,
}

impl IndexTable {
    /// Create a new empty index table
    pub fn new() -> Self {
        IndexTable {
            entries: Vec::new(),
        }
    }

    /// Create an index table with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        IndexTable {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Add an entry to the index
    pub fn push(&mut self, entry: IndexEntry) {
        self.entries.push(entry);
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the byte offset for the index table in the file
    pub fn calculate_size(&self) -> usize {
        INDEX_TABLE_HEADER_SIZE + (self.entries.len() * INDEX_ENTRY_SIZE)
    }

    /// Find the nearest keyframe at or before the target frame
    pub fn find_keyframe_before(&self, target_frame: u32) -> Option<&IndexEntry> {
        self.entries
            .iter()
            .filter(|e| e.frame_number <= target_frame && e.is_keyframe())
            .max_by_key(|e| e.frame_number)
    }

    /// Find the entry for a specific frame
    pub fn find_frame(&self, frame_number: u32) -> Option<&IndexEntry> {
        self.entries.iter().find(|e| e.frame_number == frame_number)
    }

    /// Get all keyframe entries
    pub fn keyframes(&self) -> impl Iterator<Item = &IndexEntry> {
        self.entries.iter().filter(|e| e.is_keyframe())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Seek Result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a seek operation
#[derive(Debug, Clone)]
pub struct SeekResult {
    /// Byte offset to seek to
    pub seek_offset: u64,
    /// Number of frames to decode after seeking
    pub frames_to_decode: u32,
    /// Frame number of the keyframe
    pub keyframe_number: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitstream Writer
// ─────────────────────────────────────────────────────────────────────────────

/// Writer for ZVC69 bitstream format
///
/// Provides methods for writing file headers, frame headers,
/// frame data, and index tables to an output stream.
pub struct BitstreamWriter<W: Write> {
    writer: W,
    /// Current byte position in the stream
    position: u64,
    /// Index entries collected during writing
    index_entries: Vec<IndexEntry>,
    /// Current frame number
    frame_count: u32,
    /// File header (stored for finalization)
    file_header: Option<FileHeader>,
}

impl<W: Write> BitstreamWriter<W> {
    /// Create a new bitstream writer
    pub fn new(writer: W) -> Self {
        BitstreamWriter {
            writer,
            position: 0,
            index_entries: Vec::new(),
            frame_count: 0,
            file_header: None,
        }
    }

    /// Get the current byte position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get the number of frames written
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// Write the file header
    pub fn write_file_header(&mut self, header: &FileHeader) -> Result<(), ZVC69Error> {
        // Validate header
        header.validate()?;

        // Store for finalization
        self.file_header = Some(header.clone());

        // Write magic (6 bytes)
        self.writer
            .write_all(&header.magic)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write version (2 bytes)
        self.writer.write_u8(header.version_major).map_err(|e| {
            ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            }
        })?;
        self.writer.write_u8(header.version_minor).map_err(|e| {
            ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            }
        })?;

        // Write flags (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(header.flags)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write dimensions (4 bytes)
        self.writer
            .write_u16::<LittleEndian>(header.width)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;
        self.writer
            .write_u16::<LittleEndian>(header.height)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write framerate (4 bytes)
        self.writer
            .write_u16::<LittleEndian>(header.framerate_num)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;
        self.writer
            .write_u16::<LittleEndian>(header.framerate_den)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write total frames (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(header.total_frames)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write GOP size, quality, color space, bit depth (4 bytes)
        self.writer
            .write_u8(header.gop_size)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;
        self.writer.write_u8(header.quality_level).map_err(|e| {
            ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            }
        })?;
        self.writer
            .write_u8(header.color_space)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;
        self.writer
            .write_u8(header.bit_depth)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write model hash (4 bytes)
        self.writer.write_all(&header.model_hash).map_err(|e| {
            ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            }
        })?;

        // Write index offset (8 bytes)
        self.writer
            .write_u64::<LittleEndian>(header.index_offset)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write latent channels (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(header.latent_channels)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write hyperprior channels (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(header.hyperprior_channels)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write reserved padding (16 bytes)
        let padding = [0u8; 16];
        self.writer
            .write_all(&padding)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        self.position = FILE_HEADER_SIZE as u64;
        Ok(())
    }

    /// Write a frame header
    pub fn write_frame_header(&mut self, header: &FrameHeader) -> Result<(), ZVC69Error> {
        // Record frame position for index
        let frame_offset = self.position;

        // Determine flags for index entry
        let mut entry_flags = 0u32;
        if header.is_keyframe() {
            entry_flags |= index_flags::IS_KEYFRAME;
        }
        if header.frame_type != FrameType::B && header.frame_type != FrameType::BBi {
            entry_flags |= index_flags::IS_REFERENCE;
        }

        // Add to index
        self.index_entries.push(IndexEntry {
            frame_number: self.frame_count,
            byte_offset: frame_offset,
            flags: entry_flags,
        });

        // Write frame type (1 byte)
        self.writer.write_u8(header.frame_type as u8).map_err(|e| {
            ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            }
        })?;

        // Write temporal layer (1 byte)
        self.writer.write_u8(header.temporal_layer).map_err(|e| {
            ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            }
        })?;

        // Write reference flags (2 bytes)
        self.writer
            .write_u16::<LittleEndian>(header.reference_flags)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write frame size (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(header.frame_size)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write PTS (8 bytes)
        self.writer
            .write_u64::<LittleEndian>(header.pts)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write DTS (8 bytes)
        self.writer
            .write_u64::<LittleEndian>(header.dts)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write QP offset (2 bytes)
        self.writer
            .write_i16::<LittleEndian>(header.qp_offset)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write checksum (2 bytes)
        self.writer
            .write_u16::<LittleEndian>(header.checksum)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write reserved (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(0)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        self.position += FRAME_HEADER_SIZE as u64;
        Ok(())
    }

    /// Write frame data
    pub fn write_frame_data(&mut self, data: &[u8]) -> Result<(), ZVC69Error> {
        self.writer
            .write_all(data)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;
        self.position += data.len() as u64;
        self.frame_count += 1;
        Ok(())
    }

    /// Write the index table
    pub fn write_index_table(&mut self, table: &IndexTable) -> Result<(), ZVC69Error> {
        // Write entry count (4 bytes)
        self.writer
            .write_u32::<LittleEndian>(table.entries.len() as u32)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write entry size (2 bytes)
        self.writer
            .write_u16::<LittleEndian>(INDEX_ENTRY_SIZE as u16)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write reserved (2 bytes)
        self.writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })?;

        // Write entries
        for entry in &table.entries {
            // Write frame number (4 bytes)
            self.writer
                .write_u32::<LittleEndian>(entry.frame_number)
                .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                    reason: e.to_string(),
                })?;

            // Write byte offset (8 bytes)
            self.writer
                .write_u64::<LittleEndian>(entry.byte_offset)
                .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                    reason: e.to_string(),
                })?;

            // Write flags (4 bytes)
            self.writer
                .write_u32::<LittleEndian>(entry.flags)
                .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                    reason: e.to_string(),
                })?;
        }

        self.position += table.calculate_size() as u64;
        Ok(())
    }

    /// Get the collected index entries
    pub fn index_entries(&self) -> &[IndexEntry] {
        &self.index_entries
    }

    /// Build an index table from collected entries
    pub fn build_index_table(&self) -> IndexTable {
        IndexTable {
            entries: self.index_entries.clone(),
        }
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<(), ZVC69Error> {
        self.writer
            .flush()
            .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                reason: e.to_string(),
            })
    }

    /// Finalize the bitstream (flush and return any final data)
    pub fn finalize(mut self) -> Result<W, ZVC69Error> {
        self.flush()?;
        Ok(self.writer)
    }
}

impl<W: Write + Seek> BitstreamWriter<W> {
    /// Finalize with index table written at the end
    ///
    /// This method:
    /// 1. Records the current position as the index offset
    /// 2. Writes the index table
    /// 3. Seeks back to update the file header with the index offset
    pub fn finalize_with_index(mut self) -> Result<W, ZVC69Error> {
        // Get current position for index
        let index_offset = self.position;

        // Write index table
        let index_table = self.build_index_table();
        self.write_index_table(&index_table)?;

        // Update file header with index offset and frame count
        if let Some(mut header) = self.file_header.take() {
            header.index_offset = index_offset;
            header.flags |= flags::HAS_INDEX;
            header.total_frames = self.frame_count;

            // Seek back to beginning
            self.writer
                .seek(SeekFrom::Start(0))
                .map_err(|e| ZVC69Error::BitstreamWriteFailed {
                    reason: e.to_string(),
                })?;

            // Rewrite header
            self.position = 0;
            self.write_file_header(&header)?;
        }

        self.flush()?;
        Ok(self.writer)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitstream Reader
// ─────────────────────────────────────────────────────────────────────────────

/// Reader for ZVC69 bitstream format
///
/// Provides methods for reading file headers, frame headers,
/// frame data, and index tables from an input stream.
pub struct BitstreamReader<R: Read> {
    reader: R,
    /// Current byte position in the stream
    position: u64,
    /// Cached file header
    file_header: Option<FileHeader>,
    /// Cached index table
    index_table: Option<IndexTable>,
}

impl<R: Read> BitstreamReader<R> {
    /// Create a new bitstream reader
    pub fn new(reader: R) -> Self {
        BitstreamReader {
            reader,
            position: 0,
            file_header: None,
            index_table: None,
        }
    }

    /// Get the current byte position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get the cached file header if available
    pub fn file_header(&self) -> Option<&FileHeader> {
        self.file_header.as_ref()
    }

    /// Get the cached index table if available
    pub fn index_table(&self) -> Option<&IndexTable> {
        self.index_table.as_ref()
    }

    /// Read the file header
    pub fn read_file_header(&mut self) -> Result<FileHeader, ZVC69Error> {
        // Read magic (6 bytes)
        let mut magic = [0u8; 6];
        self.reader
            .read_exact(&mut magic)
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;

        // Validate magic
        if magic != *MAGIC {
            return Err(ZVC69Error::invalid_header(format!(
                "Invalid magic bytes: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        // Read version (2 bytes)
        let version_major = self
            .reader
            .read_u8()
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;
        let version_minor = self
            .reader
            .read_u8()
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;

        // Check version
        if version_major != VERSION_MAJOR {
            return Err(ZVC69Error::UnsupportedBitstreamVersion {
                version: u32::from(version_major),
                min_supported: u32::from(VERSION_MAJOR),
                max_supported: u32::from(VERSION_MAJOR),
            });
        }

        // Read flags (4 bytes)
        let flags = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read dimensions (4 bytes)
        let width = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;
        let height = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read framerate (4 bytes)
        let framerate_num = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;
        let framerate_den = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read total frames (4 bytes)
        let total_frames = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read GOP size, quality, color space, bit depth (4 bytes)
        let gop_size = self
            .reader
            .read_u8()
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;
        let quality_level = self
            .reader
            .read_u8()
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;
        let color_space = self
            .reader
            .read_u8()
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;
        let bit_depth = self
            .reader
            .read_u8()
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;

        // Read model hash (4 bytes)
        let mut model_hash = [0u8; 4];
        self.reader
            .read_exact(&mut model_hash)
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;

        // Read index offset (8 bytes)
        let index_offset = self.reader.read_u64::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read latent channels (4 bytes)
        let latent_channels = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read hyperprior channels (4 bytes)
        let hyperprior_channels = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read reserved padding (16 bytes)
        let mut reserved = [0u8; 16];
        self.reader
            .read_exact(&mut reserved)
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;

        let header = FileHeader {
            magic,
            version_major,
            version_minor,
            flags,
            width,
            height,
            framerate_num,
            framerate_den,
            total_frames,
            gop_size,
            quality_level,
            color_space,
            bit_depth,
            model_hash,
            index_offset,
            latent_channels,
            hyperprior_channels,
        };

        // Validate (skip magic and version since already checked)
        if header.framerate_den == 0 {
            return Err(ZVC69Error::invalid_header(
                "Invalid framerate denominator (zero)",
            ));
        }

        // Cache the header
        self.file_header = Some(header.clone());
        self.position = FILE_HEADER_SIZE as u64;

        Ok(header)
    }

    /// Read a frame header
    pub fn read_frame_header(&mut self) -> Result<FrameHeader, ZVC69Error> {
        // Read frame type (1 byte)
        let frame_type_byte =
            self.reader
                .read_u8()
                .map_err(|e| ZVC69Error::BitstreamReadFailed {
                    reason: e.to_string(),
                })?;
        let frame_type = FrameType::try_from(frame_type_byte)?;

        // Read temporal layer (1 byte)
        let temporal_layer =
            self.reader
                .read_u8()
                .map_err(|e| ZVC69Error::BitstreamReadFailed {
                    reason: e.to_string(),
                })?;

        // Read reference flags (2 bytes)
        let reference_flags = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read frame size (4 bytes)
        let frame_size = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read PTS (8 bytes)
        let pts = self.reader.read_u64::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read DTS (8 bytes)
        let dts = self.reader.read_u64::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read QP offset (2 bytes)
        let qp_offset = self.reader.read_i16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read checksum (2 bytes)
        let checksum = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read reserved (4 bytes)
        let _ = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        self.position += FRAME_HEADER_SIZE as u64;

        Ok(FrameHeader {
            frame_type,
            temporal_layer,
            reference_flags,
            frame_size,
            pts,
            dts,
            qp_offset,
            checksum,
        })
    }

    /// Read frame data of specified size
    pub fn read_frame_data(&mut self, size: usize) -> Result<Vec<u8>, ZVC69Error> {
        let mut data = vec![0u8; size];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;
        self.position += size as u64;
        Ok(data)
    }

    /// Read the index table
    pub fn read_index_table(&mut self) -> Result<IndexTable, ZVC69Error> {
        // Read entry count (4 bytes)
        let entry_count = self.reader.read_u32::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read entry size (2 bytes)
        let entry_size = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Validate entry size
        if entry_size != INDEX_ENTRY_SIZE as u16 {
            return Err(ZVC69Error::invalid_header(format!(
                "Unexpected index entry size: expected {}, got {}",
                INDEX_ENTRY_SIZE, entry_size
            )));
        }

        // Read reserved (2 bytes)
        let _ = self.reader.read_u16::<LittleEndian>().map_err(|e| {
            ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            }
        })?;

        // Read entries
        let mut entries = Vec::with_capacity(entry_count as usize);
        for _ in 0..entry_count {
            // Read frame number (4 bytes)
            let frame_number = self.reader.read_u32::<LittleEndian>().map_err(|e| {
                ZVC69Error::BitstreamReadFailed {
                    reason: e.to_string(),
                }
            })?;

            // Read byte offset (8 bytes)
            let byte_offset = self.reader.read_u64::<LittleEndian>().map_err(|e| {
                ZVC69Error::BitstreamReadFailed {
                    reason: e.to_string(),
                }
            })?;

            // Read flags (4 bytes)
            let flags = self.reader.read_u32::<LittleEndian>().map_err(|e| {
                ZVC69Error::BitstreamReadFailed {
                    reason: e.to_string(),
                }
            })?;

            entries.push(IndexEntry {
                frame_number,
                byte_offset,
                flags,
            });
        }

        let table = IndexTable { entries };

        // Cache the table
        self.index_table = Some(table.clone());
        self.position +=
            INDEX_TABLE_HEADER_SIZE as u64 + (entry_count as u64 * INDEX_ENTRY_SIZE as u64);

        Ok(table)
    }
}

impl<R: Read + Seek> BitstreamReader<R> {
    /// Seek to a specific byte position
    pub fn seek(&mut self, pos: u64) -> Result<(), ZVC69Error> {
        self.reader
            .seek(SeekFrom::Start(pos))
            .map_err(|e| ZVC69Error::BitstreamReadFailed {
                reason: e.to_string(),
            })?;
        self.position = pos;
        Ok(())
    }

    /// Seek to frame by number using the index table
    ///
    /// Returns a SeekResult containing the actual seek position and
    /// number of frames to decode to reach the target.
    pub fn seek_to_frame(&mut self, frame_number: u32) -> Result<SeekResult, ZVC69Error> {
        // Ensure we have the index table and find the keyframe
        // Copy the data we need to avoid borrow checker issues
        let (keyframe_offset, keyframe_number) = {
            let table =
                self.index_table
                    .as_ref()
                    .ok_or_else(|| ZVC69Error::BitstreamReadFailed {
                        reason: "Index table not loaded".to_string(),
                    })?;

            // Find nearest keyframe
            let keyframe = table.find_keyframe_before(frame_number).ok_or_else(|| {
                ZVC69Error::BitstreamReadFailed {
                    reason: format!("No keyframe found before frame {}", frame_number),
                }
            })?;

            (keyframe.byte_offset, keyframe.frame_number)
        };

        // Seek to keyframe position
        self.seek(keyframe_offset)?;

        Ok(SeekResult {
            seek_offset: keyframe_offset,
            frames_to_decode: frame_number - keyframe_number,
            keyframe_number,
        })
    }

    /// Load the index table from its location in the file
    pub fn load_index_table(&mut self) -> Result<IndexTable, ZVC69Error> {
        // Ensure we have the file header
        let index_offset = {
            let header =
                self.file_header
                    .as_ref()
                    .ok_or_else(|| ZVC69Error::BitstreamReadFailed {
                        reason: "File header not read".to_string(),
                    })?;

            if !header.has_index() {
                return Err(ZVC69Error::BitstreamReadFailed {
                    reason: "No index table in this file".to_string(),
                });
            }

            header.index_offset
        };

        // Seek to index
        self.seek(index_offset)?;

        // Read index table
        self.read_index_table()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checksum Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate CRC-16 checksum for frame data
pub fn calculate_crc16(data: &[u8]) -> u16 {
    // CRC-16-CCITT polynomial
    const POLY: u16 = 0x1021;

    let mut crc: u16 = 0xFFFF;

    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
        }
    }

    crc
}

/// Verify frame data checksum
pub fn verify_checksum(data: &[u8], expected: u16) -> bool {
    calculate_crc16(data) == expected
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_file_header_roundtrip() {
        let header = FileHeader::builder()
            .dimensions(1920, 1080)
            .framerate(30, 1)
            .total_frames(300)
            .gop_size(10)
            .quality_level(75)
            .color_space(ColorSpace::Yuv420)
            .bit_depth(8)
            .model_hash([0xAB, 0xCD, 0xEF, 0x12])
            .latent_channels(192)
            .hyperprior_channels(128)
            .build()
            .unwrap();

        let mut buffer = Vec::new();
        {
            let mut writer = BitstreamWriter::new(&mut buffer);
            writer.write_file_header(&header).unwrap();
        }

        assert_eq!(buffer.len(), FILE_HEADER_SIZE);

        let mut reader = BitstreamReader::new(Cursor::new(&buffer));
        let read_header = reader.read_file_header().unwrap();

        assert_eq!(read_header.magic, header.magic);
        assert_eq!(read_header.version_major, header.version_major);
        assert_eq!(read_header.version_minor, header.version_minor);
        assert_eq!(read_header.flags, header.flags);
        assert_eq!(read_header.width, header.width);
        assert_eq!(read_header.height, header.height);
        assert_eq!(read_header.framerate_num, header.framerate_num);
        assert_eq!(read_header.framerate_den, header.framerate_den);
        assert_eq!(read_header.total_frames, header.total_frames);
        assert_eq!(read_header.gop_size, header.gop_size);
        assert_eq!(read_header.quality_level, header.quality_level);
        assert_eq!(read_header.color_space, header.color_space);
        assert_eq!(read_header.bit_depth, header.bit_depth);
        assert_eq!(read_header.model_hash, header.model_hash);
        assert_eq!(read_header.index_offset, header.index_offset);
        assert_eq!(read_header.latent_channels, header.latent_channels);
        assert_eq!(read_header.hyperprior_channels, header.hyperprior_channels);
    }

    #[test]
    fn test_frame_header_roundtrip() {
        let headers = vec![
            FrameHeader::i_frame(1000, 0),
            FrameHeader::p_frame(500, 3000, 0x0001),
            FrameHeader::b_frame(300, 1500, 1000, 2, 0x0003)
                .with_checksum(0x1234)
                .with_qp_offset(-5),
        ];

        for original in headers {
            let mut buffer = Vec::new();
            {
                let mut writer = BitstreamWriter::new(&mut buffer);
                // Write a dummy file header first
                writer.write_file_header(&FileHeader::default()).unwrap();
                writer.write_frame_header(&original).unwrap();
            }

            let mut reader = BitstreamReader::new(Cursor::new(&buffer));
            reader.read_file_header().unwrap();
            let read_header = reader.read_frame_header().unwrap();

            assert_eq!(read_header.frame_type, original.frame_type);
            assert_eq!(read_header.temporal_layer, original.temporal_layer);
            assert_eq!(read_header.reference_flags, original.reference_flags);
            assert_eq!(read_header.frame_size, original.frame_size);
            assert_eq!(read_header.pts, original.pts);
            assert_eq!(read_header.dts, original.dts);
            assert_eq!(read_header.qp_offset, original.qp_offset);
            assert_eq!(read_header.checksum, original.checksum);
        }
    }

    #[test]
    fn test_index_table_roundtrip() {
        let mut table = IndexTable::new();
        table.push(IndexEntry::keyframe(0, 64));
        table.push(IndexEntry::reference_frame(1, 1064));
        table.push(IndexEntry::new(2, 1564, 0));
        table.push(IndexEntry::keyframe(10, 5000));

        let mut buffer = Vec::new();
        {
            let mut writer = BitstreamWriter::new(&mut buffer);
            writer.write_index_table(&table).unwrap();
        }

        let expected_size = INDEX_TABLE_HEADER_SIZE + (4 * INDEX_ENTRY_SIZE);
        assert_eq!(buffer.len(), expected_size);

        let mut reader = BitstreamReader::new(Cursor::new(&buffer));
        let read_table = reader.read_index_table().unwrap();

        assert_eq!(read_table.len(), table.len());
        for (original, read) in table.entries.iter().zip(read_table.entries.iter()) {
            assert_eq!(read.frame_number, original.frame_number);
            assert_eq!(read.byte_offset, original.byte_offset);
            assert_eq!(read.flags, original.flags);
        }
    }

    #[test]
    fn test_index_seeking() {
        let mut table = IndexTable::new();
        table.push(IndexEntry::keyframe(0, 64));
        table.push(IndexEntry::reference_frame(1, 1000));
        table.push(IndexEntry::reference_frame(2, 2000));
        table.push(IndexEntry::new(3, 3000, 0));
        table.push(IndexEntry::new(4, 4000, 0));
        table.push(IndexEntry::keyframe(5, 5000));
        table.push(IndexEntry::reference_frame(6, 6000));

        // Find keyframe for frame 3 (should be frame 0)
        let keyframe = table.find_keyframe_before(3).unwrap();
        assert_eq!(keyframe.frame_number, 0);
        assert_eq!(keyframe.byte_offset, 64);

        // Find keyframe for frame 6 (should be frame 5)
        let keyframe = table.find_keyframe_before(6).unwrap();
        assert_eq!(keyframe.frame_number, 5);
        assert_eq!(keyframe.byte_offset, 5000);

        // Find specific frame
        let entry = table.find_frame(3).unwrap();
        assert_eq!(entry.frame_number, 3);
        assert_eq!(entry.byte_offset, 3000);

        // Count keyframes
        let keyframe_count = table.keyframes().count();
        assert_eq!(keyframe_count, 2);
    }

    #[test]
    fn test_seek_to_frame() {
        let mut buffer = Vec::new();

        // Create a file with header, frames, and index
        {
            let mut writer = BitstreamWriter::new(Cursor::new(&mut buffer));

            let header = FileHeader::builder()
                .dimensions(1920, 1080)
                .framerate(30, 1)
                .gop_size(5)
                .build()
                .unwrap();

            writer.write_file_header(&header).unwrap();

            // Write some frames
            for i in 0..10 {
                let frame_type = if i % 5 == 0 {
                    FrameType::I
                } else {
                    FrameType::P
                };
                let frame_header = FrameHeader {
                    frame_type,
                    frame_size: 100,
                    pts: (i as u64) * 3000,
                    dts: (i as u64) * 3000,
                    ..Default::default()
                };
                writer.write_frame_header(&frame_header).unwrap();
                writer.write_frame_data(&[0u8; 100]).unwrap();
            }

            // Finalize with index
            writer.finalize_with_index().unwrap();
        }

        // Read back and test seeking
        let mut reader = BitstreamReader::new(Cursor::new(&buffer));
        reader.read_file_header().unwrap();
        reader.load_index_table().unwrap();

        // Seek to frame 7 (should go to keyframe at 5)
        let seek_result = reader.seek_to_frame(7).unwrap();
        assert_eq!(seek_result.keyframe_number, 5);
        assert_eq!(seek_result.frames_to_decode, 2);

        // Seek to frame 3 (should go to keyframe at 0)
        let seek_result = reader.seek_to_frame(3).unwrap();
        assert_eq!(seek_result.keyframe_number, 0);
        assert_eq!(seek_result.frames_to_decode, 3);
    }

    #[test]
    fn test_invalid_magic() {
        let buffer = b"WRONG\0xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        let mut reader = BitstreamReader::new(Cursor::new(&buffer[..]));
        let result = reader.read_file_header();
        assert!(result.is_err());
        if let Err(ZVC69Error::InvalidHeader { reason }) = result {
            assert!(reason.contains("Invalid magic"));
        } else {
            panic!("Expected InvalidHeader error");
        }
    }

    #[test]
    fn test_invalid_version() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(MAGIC);
        buffer.push(99); // Invalid major version
        buffer.push(0);
        buffer.extend_from_slice(&[0u8; 56]); // Rest of header

        let mut reader = BitstreamReader::new(Cursor::new(&buffer));
        let result = reader.read_file_header();
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ZVC69Error::UnsupportedBitstreamVersion { .. })
        ));
    }

    #[test]
    fn test_frame_types() {
        assert!(FrameType::I.is_keyframe());
        assert!(!FrameType::P.is_keyframe());
        assert!(!FrameType::B.is_keyframe());

        assert!(!FrameType::I.needs_reference());
        assert!(FrameType::P.needs_reference());
        assert!(FrameType::B.needs_reference());
    }

    #[test]
    fn test_color_space_conversion() {
        assert_eq!(ColorSpace::try_from(0).unwrap(), ColorSpace::Yuv420);
        assert_eq!(ColorSpace::try_from(1).unwrap(), ColorSpace::Yuv444);
        assert_eq!(ColorSpace::try_from(2).unwrap(), ColorSpace::Rgb);
        assert!(ColorSpace::try_from(3).is_err());
    }

    #[test]
    fn test_checksum() {
        let data = b"Hello, ZVC69!";
        let checksum = calculate_crc16(data);
        assert!(verify_checksum(data, checksum));
        assert!(!verify_checksum(data, checksum.wrapping_add(1)));
    }

    #[test]
    fn test_file_header_builder() {
        let header = FileHeader::builder()
            .dimensions(3840, 2160)
            .framerate(60, 1)
            .total_frames(1800)
            .gop_size(15)
            .quality_level(90)
            .color_space(ColorSpace::Yuv444)
            .bit_depth(10)
            .with_b_frames()
            .build()
            .unwrap();

        assert_eq!(header.width, 3840);
        assert_eq!(header.height, 2160);
        assert_eq!(header.framerate_num, 60);
        assert_eq!(header.gop_size, 15);
        assert_eq!(header.bit_depth, 10);
        assert!(header.has_b_frames());
    }

    #[test]
    fn test_index_entry_flags() {
        let kf = IndexEntry::keyframe(0, 64);
        assert!(kf.is_keyframe());
        assert!(kf.is_reference());

        let ref_frame = IndexEntry::reference_frame(1, 128);
        assert!(!ref_frame.is_keyframe());
        assert!(ref_frame.is_reference());

        let non_ref = IndexEntry::new(2, 256, 0);
        assert!(!non_ref.is_keyframe());
        assert!(!non_ref.is_reference());
    }

    #[test]
    fn test_full_write_read_cycle() {
        let mut buffer = Vec::new();

        // Write complete file
        {
            let mut writer = BitstreamWriter::new(Cursor::new(&mut buffer));

            let header = FileHeader::builder()
                .dimensions(640, 480)
                .framerate(24, 1)
                .total_frames(3)
                .gop_size(3)
                .quality_level(50)
                .build()
                .unwrap();

            writer.write_file_header(&header).unwrap();

            // I-frame
            let frame_data_1 = vec![0xAAu8; 500];
            let checksum_1 = calculate_crc16(&frame_data_1);
            let frame_header_1 = FrameHeader::i_frame(500, 0).with_checksum(checksum_1);
            writer.write_frame_header(&frame_header_1).unwrap();
            writer.write_frame_data(&frame_data_1).unwrap();

            // P-frame
            let frame_data_2 = vec![0xBBu8; 200];
            let checksum_2 = calculate_crc16(&frame_data_2);
            let frame_header_2 = FrameHeader::p_frame(200, 3000, 0x0001).with_checksum(checksum_2);
            writer.write_frame_header(&frame_header_2).unwrap();
            writer.write_frame_data(&frame_data_2).unwrap();

            // P-frame
            let frame_data_3 = vec![0xCCu8; 150];
            let checksum_3 = calculate_crc16(&frame_data_3);
            let frame_header_3 = FrameHeader::p_frame(150, 6000, 0x0001).with_checksum(checksum_3);
            writer.write_frame_header(&frame_header_3).unwrap();
            writer.write_frame_data(&frame_data_3).unwrap();

            writer.finalize_with_index().unwrap();
        }

        // Read back
        let mut reader = BitstreamReader::new(Cursor::new(&buffer));

        let file_header = reader.read_file_header().unwrap();
        assert_eq!(file_header.width, 640);
        assert_eq!(file_header.height, 480);
        assert_eq!(file_header.total_frames, 3);
        assert!(file_header.has_index());

        // Read frame 1
        let fh1 = reader.read_frame_header().unwrap();
        assert!(fh1.is_keyframe());
        assert_eq!(fh1.frame_size, 500);
        let data1 = reader.read_frame_data(fh1.frame_size as usize).unwrap();
        assert!(verify_checksum(&data1, fh1.checksum));

        // Read frame 2
        let fh2 = reader.read_frame_header().unwrap();
        assert_eq!(fh2.frame_type, FrameType::P);
        assert_eq!(fh2.frame_size, 200);
        let data2 = reader.read_frame_data(fh2.frame_size as usize).unwrap();
        assert!(verify_checksum(&data2, fh2.checksum));

        // Read frame 3
        let fh3 = reader.read_frame_header().unwrap();
        assert_eq!(fh3.frame_type, FrameType::P);
        assert_eq!(fh3.frame_size, 150);
        let data3 = reader.read_frame_data(fh3.frame_size as usize).unwrap();
        assert!(verify_checksum(&data3, fh3.checksum));

        // Load and verify index
        reader.load_index_table().unwrap();
        let index = reader.index_table().unwrap();
        assert_eq!(index.len(), 3);
        assert!(index.entries[0].is_keyframe());
    }
}
