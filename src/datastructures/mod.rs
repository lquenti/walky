mod vecmatrix;
mod nalgebra;

/// Adjacency list based on TSPLIB-XML
pub use crate::parser::{Edge, Graph, Vertex};

pub trait AdjacencyMatrix {
    /// The dimension of the matrix
    fn dim(&self) -> usize;

    /// Value extraction
    fn get(&self, row: usize, col: usize) -> f64;

    /// Evaluates the accumulative path cost given the current underlying graph
    fn evaluate_path(&self, path: &Path) -> f64 {
        let n = path.len();
        if n <= 1 {
            return 0.0;
        }
        let mut acc = 0.0;
        for from in 0..(n - 1) {
            let to = from + 1;
            let cost = self.get(path[from], path[to]);
            acc += cost;
        }
        acc
    }

    /// Evaluates the accumulative circle path cost given the current underlying graph
    /// This is equivalent to `GraphMatrix.evaluate_path` + returning to the initial vertex
    fn evaluate_circle(&self, path: &Path) -> f64 {
        if path.len() <= 1 {
            return 0.0;
        }
        let last_edge = self.get(*path.last().unwrap(),*path.first().unwrap());
        let path_cost = self.evaluate_path(path);
        path_cost + last_edge
    }
}

pub type Path = Vec<usize>;
pub type Solution = (f64, Vec<usize>);

/// Naive Matrix implementation with `Vec<Vec<>>`
pub use crate::datastructures::vecmatrix::VecMatrix;

/// nalgebra DMatrix based implementation
pub use crate::datastructures::nalgebra::NAMatrix;
