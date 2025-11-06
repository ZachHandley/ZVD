//! Timestamp handling for media streams

use super::Rational;
use std::fmt;

/// Time base for timestamps (1/time_base seconds per tick)
pub type TimeBase = Rational;

/// A timestamp in a media stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp {
    /// Timestamp value in time_base units
    pub value: i64,
}

impl Timestamp {
    /// Create a new timestamp
    pub fn new(value: i64) -> Self {
        Timestamp { value }
    }

    /// No timestamp / unknown timestamp
    pub fn none() -> Self {
        Timestamp { value: i64::MIN }
    }

    /// Check if timestamp is valid
    pub fn is_valid(&self) -> bool {
        self.value != i64::MIN
    }

    /// Convert timestamp to seconds
    pub fn to_seconds(&self, time_base: TimeBase) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }
        self.value as f64 * time_base.to_f64()
    }

    /// Convert seconds to timestamp
    pub fn from_seconds(seconds: f64, time_base: TimeBase) -> Self {
        let value = (seconds / time_base.to_f64()).round() as i64;
        Timestamp { value }
    }

    /// Rescale timestamp from one time base to another
    pub fn rescale(&self, from: TimeBase, to: TimeBase) -> Self {
        if !self.is_valid() {
            return *self;
        }

        // value * from / to
        let rescaled = (self.value as i128 * from.num as i128 * to.den as i128)
            / (from.den as i128 * to.num as i128);

        Timestamp {
            value: rescaled as i64,
        }
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Timestamp::none()
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "{}", self.value)
        } else {
            write!(f, "NOPTS")
        }
    }
}

impl From<i64> for Timestamp {
    fn from(value: i64) -> Self {
        Timestamp::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::new(100);
        assert!(ts.is_valid());
        assert_eq!(ts.value, 100);
    }

    #[test]
    fn test_timestamp_none() {
        let ts = Timestamp::none();
        assert!(!ts.is_valid());
    }

    #[test]
    fn test_timestamp_to_seconds() {
        let ts = Timestamp::new(1000);
        let time_base = Rational::new(1, 1000); // milliseconds
        assert_eq!(ts.to_seconds(time_base), 1.0);
    }

    #[test]
    fn test_timestamp_rescale() {
        let ts = Timestamp::new(1000);
        let from = Rational::new(1, 1000); // milliseconds
        let to = Rational::new(1, 90000); // 90kHz
        let rescaled = ts.rescale(from, to);
        assert_eq!(rescaled.value, 90000);
    }
}
