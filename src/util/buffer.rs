//! Buffer management for media data

use bytes::{Bytes, BytesMut};
use std::sync::Arc;

/// A reference-counted buffer for media data
#[derive(Debug, Clone)]
pub struct Buffer {
    data: Bytes,
}

impl Buffer {
    /// Create a new buffer from bytes
    pub fn new(data: Bytes) -> Self {
        Buffer { data }
    }

    /// Create a buffer from a vector
    pub fn from_vec(vec: Vec<u8>) -> Self {
        Buffer {
            data: Bytes::from(vec),
        }
    }

    /// Create an empty buffer
    pub fn empty() -> Self {
        Buffer {
            data: Bytes::new(),
        }
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a slice of the buffer data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the underlying bytes
    pub fn as_bytes(&self) -> &Bytes {
        &self.data
    }

    /// Clone the bytes (cheap, reference counted)
    pub fn clone_bytes(&self) -> Bytes {
        self.data.clone()
    }
}

/// A mutable buffer reference
pub struct BufferRef {
    data: BytesMut,
}

impl BufferRef {
    /// Create a new mutable buffer with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        BufferRef {
            data: BytesMut::with_capacity(capacity),
        }
    }

    /// Create from existing BytesMut
    pub fn new(data: BytesMut) -> Self {
        BufferRef { data }
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get mutable access to the buffer
    pub fn as_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get immutable access to the buffer
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Freeze the buffer into an immutable Buffer
    pub fn freeze(self) -> Buffer {
        Buffer {
            data: self.data.freeze(),
        }
    }

    /// Reserve capacity
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Extend the buffer with data
    pub fn extend_from_slice(&mut self, slice: &[u8]) {
        self.data.extend_from_slice(slice);
    }
}

impl Default for BufferRef {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buf = Buffer::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_buffer_empty() {
        let buf = Buffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_buffer_ref() {
        let mut buf = BufferRef::with_capacity(10);
        buf.extend_from_slice(&[1, 2, 3]);
        assert_eq!(buf.len(), 3);

        let frozen = buf.freeze();
        assert_eq!(frozen.len(), 3);
    }
}
