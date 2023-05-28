//! Exact methods to solve the TSP problem.

use std::convert::From;
use std::ops::{Deref, DerefMut};

use crate::parser::Graph;

/// Simplest possible solution: just go through all the nodes in order.
pub fn naive_solver(graph: Graph) -> Solution {
    todo!()
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
	array[i .. ].reverse();
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
