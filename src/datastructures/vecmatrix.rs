use std::convert::From;
use std::ops::{Deref, DerefMut};

use crate::datastructures::{AdjacencyMatrix, Graph, Path};


#[derive(Debug, PartialEq)]
pub struct VecMatrix(Vec<Vec<f64>>);

impl From<Graph> for VecMatrix {
    fn from(graph: Graph) -> Self {
        let n: usize = graph.num_vertices();
        let mut matrix = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            let vi = &graph[i];
            for edge in vi.iter() {
                let j = edge.to;
                matrix[i][j] = edge.cost;
                matrix[j][i] = edge.cost;
            }
        }
        matrix.into()
    }
}

impl From<Vec<Vec<f64>>> for VecMatrix {
    fn from(matrix: Vec<Vec<f64>>) -> Self {
        VecMatrix(matrix)
    }
}

impl Deref for VecMatrix {
    type Target = Vec<Vec<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for VecMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AdjacencyMatrix for VecMatrix {
    fn dim(&self) -> usize {
        self.len()
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self[row][col]
    }

}
