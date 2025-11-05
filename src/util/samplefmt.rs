//! Audio sample format definitions

use std::fmt;

/// Audio sample format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit
    U8,
    /// Signed 16-bit
    I16,
    /// Signed 32-bit
    I32,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// Unsigned 8-bit planar
    U8P,
    /// Signed 16-bit planar
    I16P,
    /// Signed 32-bit planar
    I32P,
    /// 32-bit float planar
    F32P,
    /// 64-bit float planar
    F64P,
    /// Unknown format
    Unknown,
}

impl SampleFormat {
    /// Get the size in bytes of one sample
    pub fn sample_size(&self) -> usize {
        match self {
            SampleFormat::U8 | SampleFormat::U8P => 1,
            SampleFormat::I16 | SampleFormat::I16P => 2,
            SampleFormat::I32 | SampleFormat::I32P => 4,
            SampleFormat::F32 | SampleFormat::F32P => 4,
            SampleFormat::F64 | SampleFormat::F64P => 8,
            SampleFormat::Unknown => 0,
        }
    }

    /// Check if this is a planar format
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            SampleFormat::U8P
                | SampleFormat::I16P
                | SampleFormat::I32P
                | SampleFormat::F32P
                | SampleFormat::F64P
        )
    }

    /// Check if this is a packed format
    pub fn is_packed(&self) -> bool {
        matches!(
            self,
            SampleFormat::U8
                | SampleFormat::I16
                | SampleFormat::I32
                | SampleFormat::F32
                | SampleFormat::F64
        )
    }

    /// Check if this is a floating point format
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            SampleFormat::F32 | SampleFormat::F64 | SampleFormat::F32P | SampleFormat::F64P
        )
    }

    /// Get the packed equivalent of this format
    pub fn to_packed(&self) -> Self {
        match self {
            SampleFormat::U8P => SampleFormat::U8,
            SampleFormat::I16P => SampleFormat::I16,
            SampleFormat::I32P => SampleFormat::I32,
            SampleFormat::F32P => SampleFormat::F32,
            SampleFormat::F64P => SampleFormat::F64,
            _ => *self,
        }
    }

    /// Get the planar equivalent of this format
    pub fn to_planar(&self) -> Self {
        match self {
            SampleFormat::U8 => SampleFormat::U8P,
            SampleFormat::I16 => SampleFormat::I16P,
            SampleFormat::I32 => SampleFormat::I32P,
            SampleFormat::F32 => SampleFormat::F32P,
            SampleFormat::F64 => SampleFormat::F64P,
            _ => *self,
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SampleFormat::U8 => "u8",
            SampleFormat::I16 => "s16",
            SampleFormat::I32 => "s32",
            SampleFormat::F32 => "f32",
            SampleFormat::F64 => "f64",
            SampleFormat::U8P => "u8p",
            SampleFormat::I16P => "s16p",
            SampleFormat::I32P => "s32p",
            SampleFormat::F32P => "f32p",
            SampleFormat::F64P => "f64p",
            SampleFormat::Unknown => "unknown",
        };
        write!(f, "{}", name)
    }
}

impl Default for SampleFormat {
    fn default() -> Self {
        SampleFormat::Unknown
    }
}
