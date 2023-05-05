//! Exact methods to solve the TSP problem.

use std::convert::From;
use std::ops::{Deref, DerefMut};

use crate::parser::Graph;

/// Simplest possible solution: just go through all the nodes in order.
pub fn naive_solver(graph: Graph) -> Vec<usize> {
    let graph_matrix: GraphMatrix = graph.into();
    let n = graph_matrix.len();
    let mut working_vector = (0..n).collect();
    let mut curr_vec = vec![0; n];
    let curr_min = f64::INFINITY;
    traverse_graph(
        &graph_matrix,
        &mut working_vector,
        n,
        &mut curr_vec,
        curr_min,
    );
    curr_vec
}

fn traverse_graph(
    graph_matrix: &GraphMatrix,
    v: &mut GraphPath,
    size: usize,
    max_v: &mut GraphPath,
    max_cost: f64,
) {
    let n = graph_matrix.len();
    if size == 1 {
        let new_cost = graph_matrix.evaluate_circle(v);
        if new_cost < max_cost {
            // New best one
            max_v[..].copy_from_slice(&v[..]);
        }
    }
    for i in 0..n {
        traverse_graph(graph_matrix, v, size - 1, max_v, max_cost);
        if size % 2 == 1 {
            v.swap(0, size - 1);
        } else {
            v.swap(i, size - 1);
        }
    }
}

//////////////////////////////////////////

// TODO: Move somewhere else

#[derive(Debug, PartialEq)]
struct GraphMatrix(Vec<Vec<f64>>);
type GraphPath = Vec<usize>;

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
    fn evaluate_path(&self, path: &GraphPath) -> f64 {
        let n = path.len();
        if n <= 1 {
            return 0.0;
        }
        let mut acc = 0.0;
        for from in 0..(n - 1) {
            let to = from + 1;
            let cost = self[path[from]][path[to]];
            if cost == f64::INFINITY {
                return f64::INFINITY;
            }
            acc += cost;
        }
        acc
    }
    fn evaluate_circle(&self, path: &GraphPath) -> f64 {
        if path.len() <= 1 {
            return 0.0;
        }
        let last_edge = self[*path.last().unwrap()][*path.first().unwrap()];
        if last_edge == f64::INFINITY {
            return f64::INFINITY;
        }
        let path_cost = self.evaluate_path(path);
        if path_cost == f64::INFINITY {
            return f64::INFINITY;
        }
        path_cost + last_edge
    }
}
