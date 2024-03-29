//! This module defines basic datastructures for this crate.
mod nalgebra;
mod vecmatrix;

/// Adjacency list based on TSPLIB-XML
pub use crate::parser::{Edge, Graph, Vertex};

#[cfg(feature = "mpi")]
use mpi::traits::*;

/// Defines properties/functions, that an adjacency matrix of a graph has.
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
                    // sometimes due to rounding errors a metric graph would be rejected.
                    // By adding 1e-10 to indirect_costs,
                    // small rounding errors are being ignored
                    if direct_cost > indirect_cost + 1e-10 {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Represents a path in a graph.
pub type Path = Vec<usize>;
/// Represents the solution to a TSP instance:
///
/// `Solution(solution_weight, solution_cycle)`
pub type Solution = (f64, Vec<usize>);

/// Naive Matrix implementation with `Vec<Vec<>>`
pub use crate::datastructures::vecmatrix::VecMatrix;

/// nalgebra DMatrix based implementation
pub use crate::datastructures::nalgebra::NAMatrix;

/// helper for MPI calls to send both some accumulated cost and the rank who calculated it
/// See either [`static_mpi_solver_generic`] or [`dynamic_mpi_solver`]
#[derive(Default, Clone, Copy, Equivalence)]
#[cfg(feature = "mpi")]
pub struct MPICostRank(pub f64, pub mpi::topology::Rank);

/// helper for MPI calls to send both the global minimum cost and the next path to compute
/// See [`dynamic_mpi_solver_generic`]
#[derive(Default, Clone, Copy, Equivalence)]
#[cfg(feature = "mpi")]
pub struct MPICostPath(pub f64, pub [usize; 3]);

#[cfg(test)]
mod test {
    use super::*;
    use ::nalgebra::DMatrix;

    /// 1 -3.- 2
    /// |     /
    /// |    /
    /// 1.  1.
    /// |  /
    /// | /
    /// 3
    ///
    /// violates the triangle inequality: 1 - 3 - 2 is shorter than 1 - 2
    #[test]
    fn test_triangle_inequality_check_fails() {
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 3., 1., 3., 0., 1., 1., 1., 0.],
        ));
        assert!(!graph.is_metric());
    }

    /// 1 -0.3- 2
    /// |     /
    /// |    /
    /// 0.1  0.2
    /// |  /
    /// | /
    /// 3
    ///
    /// triangle inequality holds
    #[test]
    fn test_triangle_inequality_check_succedes() {
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 0.3, 0.1, 0.3, 0., 0.2, 0.1, 0.2, 0.],
        ));
        assert!(graph.is_metric());
    }
}
