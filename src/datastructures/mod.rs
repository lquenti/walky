mod nalgebra;
mod vecmatrix;

/// Adjacency list based on TSPLIB-XML
pub use crate::parser::{Edge, Graph, Vertex};

#[cfg(feature = "mpi")]
use mpi::traits::*;

pub trait AdjacencyMatrix {
    /// Creates an unconnected graph with `dim` many vertices
    fn from_dim(dim: usize) -> Self;

    /// The dimension of the matrix
    fn dim(&self) -> usize;

    /// Value extraction
    fn get(&self, row: usize, col: usize) -> f64;

    /// Set the cost of an edge `row <-> col`.
    fn set(&mut self, row: usize, col: usize, cost: f64);

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
        let last_edge = self.get(*path.last().unwrap(), *path.first().unwrap());
        let path_cost = self.evaluate_path(path);
        path_cost + last_edge
    }

    /// checks, if the triangle inequality holds:
    /// `cost(i,j) <= cost(i,k) + cost(k,j)` for all i,j,k
    fn is_metric(&self) -> bool {
        let dim = self.dim();
        for i in 0..dim {
            for j in i..dim {
                let direct_cost = self.get(i, j);
                for k in 0..dim {
                    let indirect_cost = self.get(i, k) + self.get(k, j);
                    if direct_cost > indirect_cost {
                        return false;
                    }
                }
            }
        }
        true
    }
}

pub type Path = Vec<usize>;
pub type Solution = (f64, Vec<usize>);

/// Naive Matrix implementation with `Vec<Vec<>>`
pub use crate::datastructures::vecmatrix::VecMatrix;

/// nalgebra DMatrix based implementation
pub use crate::datastructures::nalgebra::NAMatrix;

/// helper for MPI calls to send both some accumulated cost and the rank who calculated it
#[derive(Default, Clone, Copy, Equivalence)]
#[cfg(feature = "mpi")]
pub struct MPICostRank(pub f64, pub i32);
