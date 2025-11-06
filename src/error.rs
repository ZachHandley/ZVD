//! Error types for ZVD

use std::fmt;
use thiserror::Error;

/// Result type alias for ZVD operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for ZVD
#[derive(Error, Debug)]
pub enum Error {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Format error
    #[error("Format error: {0}")]
    Format(String),

    /// Codec error
    #[error("Codec error: {0}")]
    Codec(String),

    /// Filter error
    #[error("Filter error: {0}")]
    Filter(String),

    /// Initialization error
    #[error("Initialization error: {0}")]
    Init(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Unsupported feature
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// End of stream
    #[error("End of stream")]
    EndOfStream,

    /// Try again later
    #[error("Try again")]
    TryAgain,

    /// Buffer too small
    #[error("Buffer too small: need {need}, have {have}")]
    BufferTooSmall { need: usize, have: usize },

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Unknown error
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl Error {
    /// Create a format error
    pub fn format<S: Into<String>>(msg: S) -> Self {
        Error::Format(msg.into())
    }

    /// Create a codec error
    pub fn codec<S: Into<String>>(msg: S) -> Self {
        Error::Codec(msg.into())
    }

    /// Create a filter error
    pub fn filter<S: Into<String>>(msg: S) -> Self {
        Error::Filter(msg.into())
    }

    /// Create an unsupported error
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        Error::Unsupported(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Error::InvalidInput(msg.into())
    }

    /// Create an invalid state error
    pub fn invalid_state<S: Into<String>>(msg: S) -> Self {
        Error::InvalidState(msg.into())
    }
}
