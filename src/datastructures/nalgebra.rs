use nalgebra::DMatrix;
use crate::datastructures::{Graph, GraphMatrix, GraphPath};

use super::AdjacencyMatrix;

pub type NAMatrix = DMatrix<f64>;

impl From<Graph> for NAMatrix {
    fn from(graph: Graph) -> Self {
        let vss: GraphMatrix = graph.into();
        let rows = vss.len();
        let cols = if let Some(row) = vss.first() {
            row.len()
        } else {
            0
        };
        let mut res = DMatrix::from_element(rows, cols, 0.0);
        for (i, row) in vss.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                res[(i,j)] = x;
            }
        }
        res
    }

}

impl AdjacencyMatrix for NAMatrix {
    fn dim(&self) -> usize {
        self.shape().0
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self[(row, col)]
    }
}
