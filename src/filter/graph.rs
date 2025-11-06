//! Filter graph for chaining filters

use super::Filter;
use crate::codec::Frame;
use crate::error::Result;

/// A node in the filter graph
pub struct FilterNode {
    filter: Box<dyn Filter>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl FilterNode {
    /// Create a new filter node
    pub fn new(filter: Box<dyn Filter>) -> Self {
        FilterNode {
            filter,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

/// A graph of connected filters
pub struct FilterGraph {
    nodes: Vec<FilterNode>,
}

impl FilterGraph {
    /// Create a new empty filter graph
    pub fn new() -> Self {
        FilterGraph { nodes: Vec::new() }
    }

    /// Add a filter to the graph
    pub fn add_filter(&mut self, filter: Box<dyn Filter>) -> usize {
        let index = self.nodes.len();
        self.nodes.push(FilterNode::new(filter));
        index
    }

    /// Connect two filters
    pub fn connect(&mut self, from: usize, to: usize) -> Result<()> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err(crate::error::Error::filter("Invalid node index"));
        }

        self.nodes[from].outputs.push(to);
        self.nodes[to].inputs.push(from);

        Ok(())
    }

    /// Process a frame through the graph
    pub fn process(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        // Simple implementation - just pass through the first filter
        // A real implementation would handle the full graph traversal
        if self.nodes.is_empty() {
            return Ok(vec![frame]);
        }

        self.nodes[0].filter.filter(frame)
    }

    /// Get the number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}
