//! WASM bindings for ZVD multimedia processing
//!
//! This module provides JavaScript-friendly bindings for ZVD's multimedia
//! processing capabilities, enabling video and audio processing in the browser.

use wasm_bindgen::prelude::*;
use web_sys::{console, ImageData};
use zvd_lib::codec::{Frame, VideoFrame, AudioFrame};
use zvd_lib::error::Result as ZvdResult;
use zvd_lib::util::{PixelFormat, SampleFormat, Buffer, Timestamp};

/// Initialize the WASM module
/// Call this before using any other functions
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages in browser console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    console::log_1(&"ZVD WASM module initialized".into());
}

/// Video encoder wrapper for WASM
#[wasm_bindgen]
pub struct WasmVideoEncoder {
    codec: String,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl WasmVideoEncoder {
    /// Create a new video encoder
    #[wasm_bindgen(constructor)]
    pub fn new(codec: String, width: u32, height: u32) -> Result<WasmVideoEncoder, JsValue> {
        console::log_3(
            &"Creating encoder:".into(),
            &codec.clone().into(),
            &format!("{}x{}", width, height).into(),
        );

        Ok(WasmVideoEncoder {
            codec,
            width,
            height,
        })
    }

    /// Encode a video frame
    /// Returns encoded data as Uint8Array
    pub fn encode_frame(&mut self, frame_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        // Placeholder - would actually encode the frame
        console::log_1(&"Encoding frame...".into());
        Ok(vec![])
    }

    /// Flush the encoder
    pub fn flush(&mut self) -> Result<Vec<u8>, JsValue> {
        console::log_1(&"Flushing encoder...".into());
        Ok(vec![])
    }
}

/// Video decoder wrapper for WASM
#[wasm_bindgen]
pub struct WasmVideoDecoder {
    codec: String,
}

#[wasm_bindgen]
impl WasmVideoDecoder {
    /// Create a new video decoder
    #[wasm_bindgen(constructor)]
    pub fn new(codec: String) -> Result<WasmVideoDecoder, JsValue> {
        console::log_2(&"Creating decoder:".into(), &codec.clone().into());

        Ok(WasmVideoDecoder { codec })
    }

    /// Decode a video packet
    /// Returns decoded frame data
    pub fn decode_packet(&mut self, packet_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        console::log_1(&"Decoding packet...".into());
        // Placeholder - would actually decode the packet
        Ok(vec![])
    }
}

/// Audio encoder wrapper for WASM
#[wasm_bindgen]
pub struct WasmAudioEncoder {
    codec: String,
    sample_rate: u32,
    channels: u16,
}

#[wasm_bindgen]
impl WasmAudioEncoder {
    /// Create a new audio encoder
    #[wasm_bindgen(constructor)]
    pub fn new(codec: String, sample_rate: u32, channels: u16) -> Result<WasmAudioEncoder, JsValue> {
        console::log_3(
            &"Creating audio encoder:".into(),
            &codec.clone().into(),
            &format!("{}Hz, {} ch", sample_rate, channels).into(),
        );

        Ok(WasmAudioEncoder {
            codec,
            sample_rate,
            channels,
        })
    }

    /// Encode audio samples
    pub fn encode_samples(&mut self, samples: &[f32]) -> Result<Vec<u8>, JsValue> {
        console::log_2(
            &"Encoding audio samples:".into(),
            &samples.len().into(),
        );
        Ok(vec![])
    }
}

/// Audio decoder wrapper for WASM
#[wasm_bindgen]
pub struct WasmAudioDecoder {
    codec: String,
}

#[wasm_bindgen]
impl WasmAudioDecoder {
    /// Create a new audio decoder
    #[wasm_bindgen(constructor)]
    pub fn new(codec: String) -> Result<WasmAudioDecoder, JsValue> {
        console::log_2(&"Creating audio decoder:".into(), &codec.clone().into());

        Ok(WasmAudioDecoder { codec })
    }

    /// Decode an audio packet
    pub fn decode_packet(&mut self, packet_data: &[u8]) -> Result<Vec<f32>, JsValue> {
        console::log_1(&"Decoding audio packet...".into());
        Ok(vec![])
    }
}

/// Video filter chain for WASM
#[wasm_bindgen]
pub struct WasmFilterChain {
    filters: Vec<String>,
}

#[wasm_bindgen]
impl WasmFilterChain {
    /// Create a new filter chain
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmFilterChain {
        WasmFilterChain {
            filters: Vec::new(),
        }
    }

    /// Add a scale filter
    pub fn add_scale(&mut self, width: u32, height: u32) {
        self.filters.push(format!("scale:{}x{}", width, height));
    }

    /// Add a crop filter
    pub fn add_crop(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.filters.push(format!("crop:{}:{}:{}:{}", x, y, width, height));
    }

    /// Add a rotate filter
    pub fn add_rotate(&mut self, angle: i32) {
        self.filters.push(format!("rotate:{}", angle));
    }

    /// Add a flip filter
    pub fn add_flip(&mut self, horizontal: bool, vertical: bool) {
        let mode = match (horizontal, vertical) {
            (true, true) => "both",
            (true, false) => "horizontal",
            (false, true) => "vertical",
            _ => "none",
        };
        self.filters.push(format!("flip:{}", mode));
    }

    /// Apply filters to a frame
    pub fn apply(&self, frame_data: &[u8]) -> Result<Vec<u8>, JsValue> {
        console::log_2(
            &"Applying filters:".into(),
            &self.filters.join(", ").into(),
        );
        // Placeholder
        Ok(frame_data.to_vec())
    }
}

/// Get ZVD version
#[wasm_bindgen]
pub fn get_version() -> String {
    "0.1.0".to_string()
}

/// List available codecs
#[wasm_bindgen]
pub fn list_codecs() -> Vec<JsValue> {
    vec![
        "av1".into(),
        "vp8".into(),
        "vp9".into(),
        "opus".into(),
        "vorbis".into(),
        "flac".into(),
        "mp3".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_version() {
        assert_eq!(get_version(), "0.1.0");
    }

    #[wasm_bindgen_test]
    fn test_list_codecs() {
        let codecs = list_codecs();
        assert!(!codecs.is_empty());
    }
}
