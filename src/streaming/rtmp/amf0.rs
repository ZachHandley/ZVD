//! AMF0 (Action Message Format 0) encoding and decoding
//!
//! AMF0 is used by RTMP for encoding commands and data

use crate::error::{Error, Result};
use std::collections::HashMap;

/// AMF0 data types
#[derive(Debug, Clone, PartialEq)]
pub enum Amf0Value {
    Number(f64),
    Boolean(bool),
    String(String),
    Object(HashMap<String, Amf0Value>),
    Null,
    Undefined,
    Array(Vec<Amf0Value>),
}

/// AMF0 type markers
mod markers {
    pub const NUMBER: u8 = 0x00;
    pub const BOOLEAN: u8 = 0x01;
    pub const STRING: u8 = 0x02;
    pub const OBJECT: u8 = 0x03;
    pub const NULL: u8 = 0x05;
    pub const UNDEFINED: u8 = 0x06;
    pub const ARRAY: u8 = 0x08;
    pub const OBJECT_END: u8 = 0x09;
}

impl Amf0Value {
    /// Encode AMF0 value to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        match self {
            Amf0Value::Number(n) => {
                bytes.push(markers::NUMBER);
                bytes.extend_from_slice(&n.to_bits().to_be_bytes());
            }
            Amf0Value::Boolean(b) => {
                bytes.push(markers::BOOLEAN);
                bytes.push(if *b { 1 } else { 0 });
            }
            Amf0Value::String(s) => {
                bytes.push(markers::STRING);
                let len = s.len() as u16;
                bytes.extend_from_slice(&len.to_be_bytes());
                bytes.extend_from_slice(s.as_bytes());
            }
            Amf0Value::Object(obj) => {
                bytes.push(markers::OBJECT);
                for (key, value) in obj {
                    // Key is encoded as string without type marker
                    let len = key.len() as u16;
                    bytes.extend_from_slice(&len.to_be_bytes());
                    bytes.extend_from_slice(key.as_bytes());
                    // Value is encoded normally
                    bytes.extend_from_slice(&value.encode());
                }
                // Object end marker
                bytes.extend_from_slice(&[0x00, 0x00, markers::OBJECT_END]);
            }
            Amf0Value::Null => {
                bytes.push(markers::NULL);
            }
            Amf0Value::Undefined => {
                bytes.push(markers::UNDEFINED);
            }
            Amf0Value::Array(arr) => {
                bytes.push(markers::ARRAY);
                let len = arr.len() as u32;
                bytes.extend_from_slice(&len.to_be_bytes());
                for value in arr {
                    bytes.extend_from_slice(&value.encode());
                }
            }
        }

        bytes
    }

    /// Helper: Create a command object
    pub fn command_object(properties: Vec<(&str, Amf0Value)>) -> Self {
        let mut map = HashMap::new();
        for (key, value) in properties {
            map.insert(key.to_string(), value);
        }
        Amf0Value::Object(map)
    }
}

/// AMF0 encoder for RTMP commands
pub struct Amf0Encoder;

impl Amf0Encoder {
    /// Encode connect command
    pub fn encode_connect(
        transaction_id: f64,
        app: &str,
        flash_ver: &str,
        tc_url: &str,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Command name: "connect"
        bytes.extend_from_slice(&Amf0Value::String("connect".to_string()).encode());

        // Transaction ID
        bytes.extend_from_slice(&Amf0Value::Number(transaction_id).encode());

        // Command object
        let command_obj = Amf0Value::command_object(vec![
            ("app", Amf0Value::String(app.to_string())),
            ("flashVer", Amf0Value::String(flash_ver.to_string())),
            ("tcUrl", Amf0Value::String(tc_url.to_string())),
            ("fpad", Amf0Value::Boolean(false)),
            ("capabilities", Amf0Value::Number(15.0)),
            ("audioCodecs", Amf0Value::Number(3575.0)),
            ("videoCodecs", Amf0Value::Number(252.0)),
            ("videoFunction", Amf0Value::Number(1.0)),
        ]);
        bytes.extend_from_slice(&command_obj.encode());

        bytes
    }

    /// Encode createStream command
    pub fn encode_create_stream(transaction_id: f64) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Command name: "createStream"
        bytes.extend_from_slice(&Amf0Value::String("createStream".to_string()).encode());

        // Transaction ID
        bytes.extend_from_slice(&Amf0Value::Number(transaction_id).encode());

        // Command object: null
        bytes.extend_from_slice(&Amf0Value::Null.encode());

        bytes
    }

    /// Encode publish command
    pub fn encode_publish(transaction_id: f64, stream_name: &str, publish_type: &str) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Command name: "publish"
        bytes.extend_from_slice(&Amf0Value::String("publish".to_string()).encode());

        // Transaction ID
        bytes.extend_from_slice(&Amf0Value::Number(transaction_id).encode());

        // Command object: null
        bytes.extend_from_slice(&Amf0Value::Null.encode());

        // Publishing name
        bytes.extend_from_slice(&Amf0Value::String(stream_name.to_string()).encode());

        // Publishing type ("live", "record", or "append")
        bytes.extend_from_slice(&Amf0Value::String(publish_type.to_string()).encode());

        bytes
    }

    /// Encode releaseStream command
    pub fn encode_release_stream(transaction_id: f64, stream_name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&Amf0Value::String("releaseStream".to_string()).encode());
        bytes.extend_from_slice(&Amf0Value::Number(transaction_id).encode());
        bytes.extend_from_slice(&Amf0Value::Null.encode());
        bytes.extend_from_slice(&Amf0Value::String(stream_name.to_string()).encode());

        bytes
    }

    /// Encode FCPublish command (used by some servers)
    pub fn encode_fc_publish(transaction_id: f64, stream_name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&Amf0Value::String("FCPublish".to_string()).encode());
        bytes.extend_from_slice(&Amf0Value::Number(transaction_id).encode());
        bytes.extend_from_slice(&Amf0Value::Null.encode());
        bytes.extend_from_slice(&Amf0Value::String(stream_name.to_string()).encode());

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amf0_number_encoding() {
        let value = Amf0Value::Number(42.0);
        let bytes = value.encode();
        assert_eq!(bytes[0], markers::NUMBER);
        assert_eq!(bytes.len(), 9); // 1 byte marker + 8 bytes f64
    }

    #[test]
    fn test_amf0_string_encoding() {
        let value = Amf0Value::String("test".to_string());
        let bytes = value.encode();
        assert_eq!(bytes[0], markers::STRING);
        assert_eq!(bytes.len(), 7); // 1 marker + 2 length + 4 chars
    }

    #[test]
    fn test_amf0_boolean_encoding() {
        let value = Amf0Value::Boolean(true);
        let bytes = value.encode();
        assert_eq!(bytes[0], markers::BOOLEAN);
        assert_eq!(bytes[1], 1);
    }

    #[test]
    fn test_amf0_null_encoding() {
        let value = Amf0Value::Null;
        let bytes = value.encode();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], markers::NULL);
    }

    #[test]
    fn test_connect_command_encoding() {
        let bytes = Amf0Encoder::encode_connect(1.0, "live", "FMLE/3.0", "rtmp://localhost/live");
        assert!(!bytes.is_empty());
        // Should start with "connect" string
        assert_eq!(bytes[0], markers::STRING);
    }

    #[test]
    fn test_create_stream_encoding() {
        let bytes = Amf0Encoder::encode_create_stream(2.0);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_publish_command_encoding() {
        let bytes = Amf0Encoder::encode_publish(0.0, "stream_key", "live");
        assert!(!bytes.is_empty());
    }
}
