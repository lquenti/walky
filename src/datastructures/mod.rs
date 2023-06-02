mod vecmatrix;

/// Adjacency list based on TSPLIB-XML
pub use crate::parser::{Edge, Graph, Vertex};

/// Naive Matrix implementation with `Vec<Vec<>>`
pub use crate::datastructures::vecmatrix::{GraphPath, GraphMatrix, Solution};
