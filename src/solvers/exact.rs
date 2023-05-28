//! Exact methods to solve the TSP problem.

use std::convert::From;
use std::ops::{Deref, DerefMut};

use crate::parser::Graph;

/// Simplest possible solution: just go through all the nodes in order.
pub fn naive_solver(graph: Graph) -> Solution {
    let graph_matrix: GraphMatrix = graph.into();
    let n = graph_matrix.len();
    let mut best_permutation: GraphPath = (0..n).collect();
    let mut best_cost = f64::INFINITY;

    let mut current_permutation = best_permutation.clone();
    while next_permutation(&mut current_permutation) {
        let cost = graph_matrix.evaluate_circle(&current_permutation);
        if cost < best_cost {
            best_cost = cost;
            best_permutation = current_permutation.clone();
        }
    }
    (best_cost, best_permutation)
}

//////////////////////////////////////////

/// Finding the next permutation given an array.
/// Based on [Nayuki](https://www.nayuki.io/page/next-lexicographical-permutation-algorithm)
///
/// It ends when the array is only decreasing.
/// Thus, in order to get all permutations of [n], start with (1,2,...,n)
fn next_permutation<T: Ord>(array: &mut [T]) -> bool {
    // Find non-increasing suffix
    if array.is_empty() {
        return false;
    }
    let mut i: usize = array.len() - 1;
    while i > 0 && array[i - 1] >= array[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }

    // Find successor to pivot
    let mut j: usize = array.len() - 1;
    while array[j] <= array[i - 1] {
        j -= 1;
    }
    array.swap(i - 1, j);

    // Reverse suffix
    array[i..].reverse();
    true
}

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
            acc += cost;
        }
        acc
    }
    fn evaluate_circle(&self, path: &GraphPath) -> f64 {
        if path.len() <= 1 {
            return 0.0;
        }
        let last_edge = self[*path.last().unwrap()][*path.first().unwrap()];
        let path_cost = self.evaluate_path(path);
        path_cost + last_edge
    }
}

#[cfg(test)]
mod exact_solver {
    use super::*;

    #[test]
    fn get_all_permutations() {
        let mut starting_vec = (0..4).collect::<Vec<i32>>();
        let mut results = vec![];
        results.push(starting_vec.clone());
        while next_permutation(&mut starting_vec) {
            results.push(starting_vec.clone());
        }

        let expected = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 3, 2],
            vec![0, 2, 1, 3],
            vec![0, 2, 3, 1],
            vec![0, 3, 1, 2],
            vec![0, 3, 2, 1],
            vec![1, 0, 2, 3],
            vec![1, 0, 3, 2],
            vec![1, 2, 0, 3],
            vec![1, 2, 3, 0],
            vec![1, 3, 0, 2],
            vec![1, 3, 2, 0],
            vec![2, 0, 1, 3],
            vec![2, 0, 3, 1],
            vec![2, 1, 0, 3],
            vec![2, 1, 3, 0],
            vec![2, 3, 0, 1],
            vec![2, 3, 1, 0],
            vec![3, 0, 1, 2],
            vec![3, 0, 2, 1],
            vec![3, 1, 0, 2],
            vec![3, 1, 2, 0],
            vec![3, 2, 0, 1],
            vec![3, 2, 1, 0],
        ];
        assert_eq!(expected, results);
    }
}
