//! Exact methods to solve the TSP problem.

use std::convert::From;
use std::mem;

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
        let mut current_cost = 0.0;
        for i in 0..(n - 1) {
            let cost = graph_matrix[v[i]][v[i + 1]];
            if cost == f64::INFINITY {
                // No chance this one is better
                return;
            }
            current_cost += cost;
        }
        if current_cost < max_cost {
            // New best one
            for i in 0..v.len() {
                max_v[i] = v[i];
            }
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

type GraphMatrix = Vec<Vec<f64>>;
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
        matrix
    }
}
