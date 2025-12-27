//! Reference Picture Management for H.265/HEVC
//!
//! This module implements the Decoded Picture Buffer (DPB) and reference picture
//! management according to the H.265/HEVC specification.
//!
//! # Overview
//!
//! The DPB stores decoded pictures that may be used as references for inter prediction.
//! It manages:
//! - Short-term reference pictures
//! - Long-term reference pictures
//! - Picture Order Count (POC)
//! - Reference picture lists (L0 and L1)
//!
//! # Reference Picture Types
//!
//! - **Short-term**: Recent pictures, managed by sliding window
//! - **Long-term**: Explicitly marked pictures that persist
//! - **Current**: The picture being decoded
//!
//! # Reference Lists
//!
//! - **L0**: Forward prediction (P and B slices)
//! - **L1**: Backward prediction (B slices only)

use crate::codec::h265::mv::MotionVectorField;
use crate::error::{Error, Result};
use std::collections::VecDeque;

/// Picture Order Count (POC) type
pub type Poc = i32;

/// Reference picture in the DPB
#[derive(Debug, Clone)]
pub struct ReferencePicture {
    /// Picture Order Count
    pub poc: Poc,
    /// Is this a long-term reference?
    pub is_long_term: bool,
    /// Motion vector field for this picture
    pub mv_field: MotionVectorField,
    /// Picture width
    pub width: usize,
    /// Picture height
    pub height: usize,
    /// Luma samples (placeholder - in real implementation would be full picture data)
    pub luma: Vec<u16>,
    /// Is this picture currently in use for reference?
    pub is_reference: bool,
}

impl ReferencePicture {
    /// Create a new reference picture
    pub fn new(poc: Poc, width: usize, height: usize) -> Self {
        Self {
            poc,
            is_long_term: false,
            mv_field: MotionVectorField::new(width, height),
            width,
            height,
            luma: vec![0; width * height],
            is_reference: true,
        }
    }

    /// Mark as long-term reference
    pub fn mark_long_term(&mut self) {
        self.is_long_term = true;
    }

    /// Mark as short-term reference
    pub fn mark_short_term(&mut self) {
        self.is_long_term = false;
    }

    /// Mark as unused for reference
    pub fn mark_unused(&mut self) {
        self.is_reference = false;
    }

    /// Check if picture is used for reference
    pub fn is_used(&self) -> bool {
        self.is_reference
    }
}

/// Decoded Picture Buffer (DPB)
///
/// Manages reference pictures for inter prediction
pub struct DecodedPictureBuffer {
    /// Maximum number of reference pictures
    max_num_pics: usize,
    /// Short-term reference pictures (ordered by POC)
    short_term_refs: VecDeque<ReferencePicture>,
    /// Long-term reference pictures
    long_term_refs: Vec<ReferencePicture>,
    /// Current picture POC
    current_poc: Poc,
}

impl DecodedPictureBuffer {
    /// Create a new DPB
    pub fn new(max_num_pics: usize) -> Self {
        Self {
            max_num_pics,
            short_term_refs: VecDeque::new(),
            long_term_refs: Vec::new(),
            current_poc: 0,
        }
    }

    /// Add a new reference picture
    pub fn add_picture(&mut self, mut picture: ReferencePicture) -> Result<()> {
        // Check if we need to remove old pictures (sliding window)
        while self.num_references() >= self.max_num_pics {
            self.remove_oldest_short_term()?;
        }

        // Add to appropriate list
        if picture.is_long_term {
            self.long_term_refs.push(picture);
        } else {
            // Insert in POC order
            let insert_pos = self
                .short_term_refs
                .iter()
                .position(|p| p.poc > picture.poc)
                .unwrap_or(self.short_term_refs.len());
            self.short_term_refs.insert(insert_pos, picture);
        }

        Ok(())
    }

    /// Remove oldest short-term reference picture
    fn remove_oldest_short_term(&mut self) -> Result<()> {
        if let Some(oldest) = self.short_term_refs.pop_front() {
            Ok(())
        } else {
            Err(Error::Codec(
                "No short-term references to remove".to_string(),
            ))
        }
    }

    /// Get reference picture by POC
    pub fn get_picture(&self, poc: Poc) -> Option<&ReferencePicture> {
        self.short_term_refs
            .iter()
            .find(|p| p.poc == poc && p.is_reference)
            .or_else(|| {
                self.long_term_refs
                    .iter()
                    .find(|p| p.poc == poc && p.is_reference)
            })
    }

    /// Get mutable reference picture by POC
    pub fn get_picture_mut(&mut self, poc: Poc) -> Option<&mut ReferencePicture> {
        if let Some(pic) = self
            .short_term_refs
            .iter_mut()
            .find(|p| p.poc == poc && p.is_reference)
        {
            Some(pic)
        } else {
            self.long_term_refs
                .iter_mut()
                .find(|p| p.poc == poc && p.is_reference)
        }
    }

    /// Mark picture as long-term reference
    pub fn mark_as_long_term(&mut self, poc: Poc) -> Result<()> {
        // Find in short-term refs
        if let Some(idx) = self.short_term_refs.iter().position(|p| p.poc == poc) {
            let mut pic = self.short_term_refs.remove(idx).unwrap();
            pic.mark_long_term();
            self.long_term_refs.push(pic);
            Ok(())
        } else {
            Err(Error::Codec(format!(
                "Picture with POC {} not found in short-term refs",
                poc
            )))
        }
    }

    /// Mark picture as unused for reference
    pub fn mark_as_unused(&mut self, poc: Poc) -> Result<()> {
        if let Some(pic) = self.get_picture_mut(poc) {
            pic.mark_unused();
            Ok(())
        } else {
            Err(Error::Codec(format!(
                "Picture with POC {} not found",
                poc
            )))
        }
    }

    /// Flush all reference pictures
    pub fn flush(&mut self) {
        self.short_term_refs.clear();
        self.long_term_refs.clear();
    }

    /// Get number of reference pictures
    pub fn num_references(&self) -> usize {
        self.short_term_refs
            .iter()
            .filter(|p| p.is_reference)
            .count()
            + self.long_term_refs.iter().filter(|p| p.is_reference).count()
    }

    /// Get all short-term reference POCs
    pub fn get_short_term_pocs(&self) -> Vec<Poc> {
        self.short_term_refs
            .iter()
            .filter(|p| p.is_reference)
            .map(|p| p.poc)
            .collect()
    }

    /// Get all long-term reference POCs
    pub fn get_long_term_pocs(&self) -> Vec<Poc> {
        self.long_term_refs
            .iter()
            .filter(|p| p.is_reference)
            .map(|p| p.poc)
            .collect()
    }

    /// Set current picture POC
    pub fn set_current_poc(&mut self, poc: Poc) {
        self.current_poc = poc;
    }

    /// Get current picture POC
    pub fn current_poc(&self) -> Poc {
        self.current_poc
    }
}

/// Reference picture list (L0 or L1)
#[derive(Debug, Clone)]
pub struct RefPicList {
    /// POCs of reference pictures in this list
    pocs: Vec<Poc>,
    /// Maximum list size
    max_size: usize,
}

impl RefPicList {
    /// Create a new reference picture list
    pub fn new(max_size: usize) -> Self {
        Self {
            pocs: Vec::new(),
            max_size,
        }
    }

    /// Add a reference POC to the list
    pub fn add(&mut self, poc: Poc) -> Result<()> {
        if self.pocs.len() >= self.max_size {
            return Err(Error::Codec(format!(
                "Reference list full (max {})",
                self.max_size
            )));
        }
        self.pocs.push(poc);
        Ok(())
    }

    /// Get POC at index
    pub fn get(&self, index: usize) -> Option<Poc> {
        self.pocs.get(index).copied()
    }

    /// Get number of references in list
    pub fn len(&self) -> usize {
        self.pocs.len()
    }

    /// Check if list is empty
    pub fn is_empty(&self) -> bool {
        self.pocs.is_empty()
    }

    /// Clear the list
    pub fn clear(&mut self) {
        self.pocs.clear();
    }

    /// Get all POCs
    pub fn pocs(&self) -> &[Poc] {
        &self.pocs
    }
}

/// Reference picture list construction
pub struct RefPicListBuilder {
    /// Current picture POC
    current_poc: Poc,
}

impl RefPicListBuilder {
    /// Create a new reference list builder
    pub fn new(current_poc: Poc) -> Self {
        Self { current_poc }
    }

    /// Build L0 reference list (forward prediction)
    ///
    /// L0 contains pictures with POC < current (past) and POC > current (future)
    /// Ordered by: past pictures (closest first), then future pictures
    pub fn build_l0(&self, dpb: &DecodedPictureBuffer, max_size: usize) -> RefPicList {
        let mut list = RefPicList::new(max_size);

        // Get all short-term POCs
        let mut pocs = dpb.get_short_term_pocs();

        // Separate past and future
        let mut past: Vec<_> = pocs.iter().filter(|&&p| p < self.current_poc).copied().collect();
        let mut future: Vec<_> = pocs.iter().filter(|&&p| p > self.current_poc).copied().collect();

        // Sort past by descending POC (closest first)
        past.sort_by(|a, b| b.cmp(a));

        // Sort future by ascending POC (closest first)
        future.sort();

        // Add to list: past then future
        for poc in past.iter().chain(future.iter()) {
            if list.len() >= max_size {
                break;
            }
            let _ = list.add(*poc);
        }

        // Add long-term references
        for poc in dpb.get_long_term_pocs() {
            if list.len() >= max_size {
                break;
            }
            let _ = list.add(poc);
        }

        list
    }

    /// Build L1 reference list (backward prediction)
    ///
    /// L1 is similar to L0 but with reversed order
    /// Ordered by: future pictures (closest first), then past pictures
    pub fn build_l1(&self, dpb: &DecodedPictureBuffer, max_size: usize) -> RefPicList {
        let mut list = RefPicList::new(max_size);

        // Get all short-term POCs
        let pocs = dpb.get_short_term_pocs();

        // Separate past and future
        let mut past: Vec<_> = pocs.iter().filter(|&&p| p < self.current_poc).copied().collect();
        let mut future: Vec<_> = pocs.iter().filter(|&&p| p > self.current_poc).copied().collect();

        // Sort future by ascending POC (closest first)
        future.sort();

        // Sort past by descending POC (closest first)
        past.sort_by(|a, b| b.cmp(a));

        // Add to list: future then past (reversed from L0)
        for poc in future.iter().chain(past.iter()) {
            if list.len() >= max_size {
                break;
            }
            let _ = list.add(*poc);
        }

        // Add long-term references
        for poc in dpb.get_long_term_pocs() {
            if list.len() >= max_size {
                break;
            }
            let _ = list.add(poc);
        }

        list
    }
}

/// POC (Picture Order Count) utilities
pub struct PocUtils;

impl PocUtils {
    /// Calculate POC difference
    pub fn diff(a: Poc, b: Poc) -> i32 {
        a - b
    }

    /// Check if POC a is before POC b
    pub fn is_before(a: Poc, b: Poc) -> bool {
        a < b
    }

    /// Check if POC a is after POC b
    pub fn is_after(a: Poc, b: Poc) -> bool {
        a > b
    }

    /// Get temporal distance
    pub fn temporal_distance(current: Poc, reference: Poc) -> i32 {
        (current - reference).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_picture_creation() {
        let pic = ReferencePicture::new(10, 1920, 1080);
        assert_eq!(pic.poc, 10);
        assert!(!pic.is_long_term);
        assert!(pic.is_reference);
    }

    #[test]
    fn test_reference_picture_mark_long_term() {
        let mut pic = ReferencePicture::new(10, 1920, 1080);
        pic.mark_long_term();
        assert!(pic.is_long_term);
    }

    #[test]
    fn test_reference_picture_mark_unused() {
        let mut pic = ReferencePicture::new(10, 1920, 1080);
        pic.mark_unused();
        assert!(!pic.is_used());
    }

    #[test]
    fn test_dpb_creation() {
        let dpb = DecodedPictureBuffer::new(8);
        assert_eq!(dpb.max_num_pics, 8);
        assert_eq!(dpb.num_references(), 0);
    }

    #[test]
    fn test_dpb_add_picture() {
        let mut dpb = DecodedPictureBuffer::new(8);
        let pic = ReferencePicture::new(10, 1920, 1080);
        dpb.add_picture(pic).unwrap();
        assert_eq!(dpb.num_references(), 1);
    }

    #[test]
    fn test_dpb_get_picture() {
        let mut dpb = DecodedPictureBuffer::new(8);
        let pic = ReferencePicture::new(10, 1920, 1080);
        dpb.add_picture(pic).unwrap();

        let retrieved = dpb.get_picture(10);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().poc, 10);
    }

    #[test]
    fn test_dpb_get_picture_not_found() {
        let dpb = DecodedPictureBuffer::new(8);
        let retrieved = dpb.get_picture(10);
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_dpb_sliding_window() {
        let mut dpb = DecodedPictureBuffer::new(3);

        // Add 4 pictures to trigger sliding window
        dpb.add_picture(ReferencePicture::new(0, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(1, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(2, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(3, 1920, 1080)).unwrap();

        // Should only have 3 pictures (oldest removed)
        assert_eq!(dpb.num_references(), 3);
        assert!(dpb.get_picture(0).is_none()); // Oldest removed
        assert!(dpb.get_picture(1).is_some());
        assert!(dpb.get_picture(2).is_some());
        assert!(dpb.get_picture(3).is_some());
    }

    #[test]
    fn test_dpb_mark_as_long_term() {
        let mut dpb = DecodedPictureBuffer::new(8);
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();

        dpb.mark_as_long_term(10).unwrap();

        let long_term_pocs = dpb.get_long_term_pocs();
        assert_eq!(long_term_pocs.len(), 1);
        assert_eq!(long_term_pocs[0], 10);
    }

    #[test]
    fn test_dpb_mark_as_unused() {
        let mut dpb = DecodedPictureBuffer::new(8);
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();

        dpb.mark_as_unused(10).unwrap();

        // Picture still in DPB but not counted as reference
        assert_eq!(dpb.num_references(), 0);
    }

    #[test]
    fn test_dpb_flush() {
        let mut dpb = DecodedPictureBuffer::new(8);
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(20, 1920, 1080)).unwrap();

        dpb.flush();

        assert_eq!(dpb.num_references(), 0);
    }

    #[test]
    fn test_dpb_current_poc() {
        let mut dpb = DecodedPictureBuffer::new(8);
        dpb.set_current_poc(42);
        assert_eq!(dpb.current_poc(), 42);
    }

    #[test]
    fn test_ref_pic_list_creation() {
        let list = RefPicList::new(16);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_ref_pic_list_add() {
        let mut list = RefPicList::new(16);
        list.add(10).unwrap();
        list.add(20).unwrap();

        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0), Some(10));
        assert_eq!(list.get(1), Some(20));
    }

    #[test]
    fn test_ref_pic_list_max_size() {
        let mut list = RefPicList::new(2);
        list.add(10).unwrap();
        list.add(20).unwrap();

        let result = list.add(30);
        assert!(result.is_err());
    }

    #[test]
    fn test_ref_pic_list_builder_l0() {
        let mut dpb = DecodedPictureBuffer::new(8);
        dpb.add_picture(ReferencePicture::new(5, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(15, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(20, 1920, 1080)).unwrap();

        let builder = RefPicListBuilder::new(12);
        let l0 = builder.build_l0(&dpb, 16);

        // L0 order: past (closest first), then future
        // Expected: [10, 5, 15, 20]
        assert_eq!(l0.get(0), Some(10)); // Closest past
        assert_eq!(l0.get(1), Some(5));  // Further past
        assert_eq!(l0.get(2), Some(15)); // Closest future
        assert_eq!(l0.get(3), Some(20)); // Further future
    }

    #[test]
    fn test_ref_pic_list_builder_l1() {
        let mut dpb = DecodedPictureBuffer::new(8);
        dpb.add_picture(ReferencePicture::new(5, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(15, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(20, 1920, 1080)).unwrap();

        let builder = RefPicListBuilder::new(12);
        let l1 = builder.build_l1(&dpb, 16);

        // L1 order: future (closest first), then past
        // Expected: [15, 20, 10, 5]
        assert_eq!(l1.get(0), Some(15)); // Closest future
        assert_eq!(l1.get(1), Some(20)); // Further future
        assert_eq!(l1.get(2), Some(10)); // Closest past
        assert_eq!(l1.get(3), Some(5));  // Further past
    }

    #[test]
    fn test_ref_pic_list_pocs() {
        let mut list = RefPicList::new(16);
        list.add(10).unwrap();
        list.add(20).unwrap();

        let pocs = list.pocs();
        assert_eq!(pocs, &[10, 20]);
    }

    #[test]
    fn test_poc_utils_diff() {
        assert_eq!(PocUtils::diff(20, 10), 10);
        assert_eq!(PocUtils::diff(10, 20), -10);
    }

    #[test]
    fn test_poc_utils_is_before() {
        assert!(PocUtils::is_before(10, 20));
        assert!(!PocUtils::is_before(20, 10));
    }

    #[test]
    fn test_poc_utils_is_after() {
        assert!(PocUtils::is_after(20, 10));
        assert!(!PocUtils::is_after(10, 20));
    }

    #[test]
    fn test_poc_utils_temporal_distance() {
        assert_eq!(PocUtils::temporal_distance(20, 10), 10);
        assert_eq!(PocUtils::temporal_distance(10, 20), 10);
        assert_eq!(PocUtils::temporal_distance(10, 10), 0);
    }

    #[test]
    fn test_dpb_poc_ordering() {
        let mut dpb = DecodedPictureBuffer::new(8);

        // Add in non-sequential order
        dpb.add_picture(ReferencePicture::new(20, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(15, 1920, 1080)).unwrap();

        let pocs = dpb.get_short_term_pocs();

        // Should be ordered by POC
        assert_eq!(pocs, vec![10, 15, 20]);
    }

    #[test]
    fn test_ref_pic_list_clear() {
        let mut list = RefPicList::new(16);
        list.add(10).unwrap();
        list.add(20).unwrap();

        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn test_long_term_not_removed_by_sliding_window() {
        let mut dpb = DecodedPictureBuffer::new(3);

        // Add a long-term reference
        let mut long_term = ReferencePicture::new(5, 1920, 1080);
        long_term.mark_long_term();
        dpb.add_picture(long_term).unwrap();

        // Add short-term references to trigger sliding window
        dpb.add_picture(ReferencePicture::new(10, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(15, 1920, 1080)).unwrap();
        dpb.add_picture(ReferencePicture::new(20, 1920, 1080)).unwrap();

        // Long-term should still be there
        assert!(dpb.get_picture(5).is_some());
        assert_eq!(dpb.get_long_term_pocs(), vec![5]);
    }
}
