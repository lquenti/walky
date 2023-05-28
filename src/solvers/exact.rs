//! Exact methods to solve the TSP problem.

use std::convert::From;
use std::fmt::Display;
use std::ops::{Deref, DerefMut};

use crate::parser::Graph;

use itertools::Itertools;

/// Simplest possible solution: just go through all the nodes in order.
pub fn naive_solver(graph: Graph) -> Solution {
    let graph_matrix: GraphMatrix = graph.into();
    traverse_graph(
        &graph_matrix,
    )
}

fn traverse_graph(
    graph_matrix: &GraphMatrix,
) -> Solution {
    let n = graph_matrix.len();
    (0..n).permutations(n).fold((f64::INFINITY, vec![]), |(cost_acc, v_acc), v| {
        let cost = graph_matrix.evaluate_circle(&v);
        if cost < cost_acc {
            (cost, v)
        } else {
            (cost_acc, v_acc)
        }
    }
    )
}

//////////////////////////////////////////

// TODO: Move somewhere else

#[derive(Debug, PartialEq)]
struct GraphMatrix(Vec<Vec<f64>>);
type GraphPath = Vec<usize>;
type Solution = (f64, Vec<usize>);

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
