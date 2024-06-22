use crate::datastructures::Graph;
use nalgebra::{DMatrix, DVector};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::ops::{Deref, DerefMut};

use super::{AdjacencyMatrix, Edge};

/// Wrapper/Smart Pointer around a nalgebra [`DMatrix`]
#[derive(Debug, PartialEq)]
pub struct NAMatrix(pub DMatrix<f64>);

impl NAMatrix {
    /// Create from set of x/y points, using the euclidean distance
    pub fn from_points(points: &[[f64; 2]]) -> Self {
        let mut matrix = NAMatrix::from_dim(points.len());
        points.iter().enumerate().for_each(|(i, from)| {
            points.iter().skip(i + 1).enumerate().for_each(|(j, to)| {
                let j = i + j + 1;
                let weight = ((from[0] - to[0]).powi(2) + (from[1] - to[1]).powi(2)).sqrt();
                // uses the fact, that the euclidean distance is symmetric
                matrix[(i, j)] = weight;
                matrix[(j, i)] = weight;
            });
        });
        matrix
    }
    /// Create from set of x/y points, using the euclidean distance.
    /// Computation is parallelized via rayon.
    pub fn par_from_points(points: &[[f64; 2]]) -> Self {
        let mut matrix = NAMatrix::from_dim(points.len());
        points
            .par_iter()
            .zip(matrix.par_column_iter_mut())
            .for_each(|(from, mut matrix_col_j)| {
                //let from = &points[i];
                points
                    .iter()
                    .zip(matrix_col_j.iter_mut())
                    .for_each(|(to, matrix_ij)| {
                        let weight = ((from[0] - to[0]).powi(2) + (from[1] - to[1]).powi(2)).sqrt();
                        // cannot use the fact, that the euclidean distance is symmetric,
                        // due to borrowing rules for parallel iterators
                        *matrix_ij = weight;
                    });
            });
        matrix
    }
}

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

    #[test]
    fn test_from_points() {
        let points = vec![[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]];
        let matrix = NAMatrix::from_points(&points);
        for i in 0..points.len() {
            for j in 0..points.len() {
                let dist = ((points[i][0] - points[j][0]).powi(2)
                    + (points[i][1] - points[j][1]).powi(2))
                .sqrt();
                assert_eq!(dist, matrix[(i, j)]);
            }
        }
    }
}
