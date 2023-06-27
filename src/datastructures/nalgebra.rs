use crate::datastructures::Graph;
use blossom::WeightedGraph;
use nalgebra::{DMatrix, DVector};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use super::{AdjacencyMatrix, Edge};

/// Wrapper/Smart Pointer around a nalgebra [`DMatrix`]
#[derive(Debug, PartialEq)]
pub struct NAMatrix(pub DMatrix<f64>);

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

impl From<&Graph> for NAMatrix {
    fn from(graph: &Graph) -> Self {
        //let vss: VecMatrix = graph.into();
        let dim = graph.num_vertices();
        let mut res = NAMatrix::from_dim(dim);
        for (i, neighbours) in graph.iter().enumerate() {
            for &Edge { to: j, cost } in neighbours.iter() {
                res[(i, j)] = cost;
            }
        }
        res
    }
}

//impl From<Graph> for NAMatrix {
//    fn from(value: Graph) -> Self {
//        <NAMatrix as From<&Graph>>::from(&value)
//    }
//}

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

impl From<&NAMatrix> for WeightedGraph {
    /// This implementation is not optimized for performance.
    fn from(value: &NAMatrix) -> Self {
        let dim = value.dim();
        let mut hash_map = HashMap::new();
        for i in 0..dim {
            let mut neighbours = Vec::with_capacity(dim);
            let mut cost = Vec::with_capacity(dim);
            for j in 0..dim {
                if i == j {
                    continue;
                }
                neighbours.push(j);
                cost.push(value[(i, j)]);
            }
            hash_map.insert(i, (neighbours, cost));
        }
        WeightedGraph::new(hash_map)
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

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_namatrix_from_graph() {
        let graph: Graph = vec![
            vec![Edge { to: 1, cost: 2.5 }],
            vec![Edge { to: 0, cost: 2.5 }],
        ]
        .into();

        let expected: NAMatrix = NAMatrix(DMatrix::from_row_slice(2, 2, &[0., 2.5, 2.5, 0.]));

        assert_eq!(expected, (&graph).into());
    }
}
