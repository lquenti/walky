// TODO: Move somewhere else

use std::convert::From;
use std::ops::{Deref, DerefMut};

use crate::parser::Graph;

#[derive(Debug, PartialEq)]
pub struct GraphMatrix(Vec<Vec<f64>>);
pub type GraphPath = Vec<usize>;
pub type Solution = (f64, Vec<usize>);

impl From<Graph> for GraphMatrix {
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

impl From<Vec<Vec<f64>>> for GraphMatrix {
    fn from(matrix: Vec<Vec<f64>>) -> Self {
        GraphMatrix(matrix)
    }
}

impl Deref for GraphMatrix {
    type Target = Vec<Vec<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GraphMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GraphMatrix {
    /// Evaluates the accumulative path cost given the current underlying graph
    fn evaluate_path(&self, path: &GraphPath) -> f64 {
        let n = path.len();
        if n <= 1 {
            return 0.0;
        }
        let mut acc = 0.0;
        for from in 0..(n - 1) {
            let to = from + 1;
            let cost = self[path[from]][path[to]];
            acc += cost;
        }
        acc
    }

    /// Evaluates the accumulative circle path cost given the current underlying graph
    /// This is equivalent to `GraphMatrix.evaluate_path` + returning to the initial vertex
    pub fn evaluate_circle(&self, path: &GraphPath) -> f64 {
        if path.len() <= 1 {
            return 0.0;
        }
        let last_edge = self[*path.last().unwrap()][*path.first().unwrap()];
        let path_cost = self.evaluate_path(path);
        path_cost + last_edge
    }
}
