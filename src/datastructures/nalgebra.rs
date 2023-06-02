use crate::datastructures::{Graph, VecMatrix};
use nalgebra::{DMatrix, DVector};
use std::ops::{Deref, DerefMut};

use super::AdjacencyMatrix;

/// Wrapper/Smart Pointer around a nalgebra [`DMatrix`]
#[derive(Debug)]
pub struct NAMatrix(DMatrix<f64>);

impl Deref for NAMatrix {
    type Target = DMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for NAMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Graph> for NAMatrix {
    fn from(graph: Graph) -> Self {
        let vss: VecMatrix = graph.into();
        let rows = vss.len();
        let cols = if let Some(row) = vss.first() {
            row.len()
        } else {
            0
        };
        let mut res = DMatrix::from_element(rows, cols, 0.0);
        for (i, row) in vss.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                res[(i, j)] = x;
            }
        }
        NAMatrix(res)
    }
}

impl AdjacencyMatrix for NAMatrix {
    fn from_dim(dim: usize) -> Self {
        let mut matrix = DMatrix::from_element(dim, dim, f64::INFINITY);
        matrix.set_diagonal(&DVector::repeat(dim, 0.));
        NAMatrix(matrix)
    }
    fn dim(&self) -> usize {
        self.shape().0
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self[(row, col)]
    }

    fn set(&mut self, row: usize, col: usize, cost: f64) {
        self[(row, col)] = cost;
    }
}

impl<T> From<&T> for NAMatrix
where
    T: AdjacencyMatrix,
{
    fn from(value: &T) -> Self {
        let dim = value.dim();

        let mut adj_matr = NAMatrix::from_dim(dim);
        for i in 0..dim {
            for j in 0..dim {
                adj_matr[(i, j)] = value.get(i, j);
            }
        }
        adj_matr
    }
}
