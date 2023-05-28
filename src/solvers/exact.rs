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

fn is_same_undirected_circle(seq1: &GraphPath, seq2: &GraphPath) -> bool {
    if seq1.len() != seq2.len() {
        return false;
    }

    let n = seq1.len();

    // Generate all possible rotations of seq1 in both directions
    let rotations = (0..n).map(
        |i| seq1[i..].iter().chain(seq1[..i].iter()).copied().collect::<GraphPath>()
    );
    let reversed_rotations = rotations.clone().map(
        |xs| xs.into_iter().rev().collect::<GraphPath>()
    );

    // Check if any rotation matches
    for rotation in rotations.chain(reversed_rotations) {
        if rotation[..] == seq2[..] {
            return true;
        }
    }

    false
}

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
    use approx::relative_eq;

    use super::*;
    use crate::parser::{Edge, Graph, Vertex};

    #[test]
    fn test_is_same_undirected_circle() {
        assert!(is_same_undirected_circle(&vec![1,2,3,4,5,6], &vec![4,3,2,1,6,5]));
    }

    #[test]
    fn test_not_same_undirected_circle() {
        assert!(!is_same_undirected_circle(&vec![1,2,3,4,5,6], &vec![4,3,2,6,1,5]));
    }

    #[test]
    fn test_float_tsp() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge {
                            to: 1,
                            cost: 13.215648444670196,
                        },
                        Edge {
                            to: 2,
                            cost: 9.674413400408712,
                        },
                        Edge {
                            to: 3,
                            cost: 1.0970596862282833,
                        },
                        Edge {
                            to: 4,
                            cost: 16.098684067859647,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 13.215648444670196,
                        },
                        Edge {
                            to: 2,
                            cost: 12.221639547131913,
                        },
                        Edge {
                            to: 3,
                            cost: 17.306826463341803,
                        },
                        Edge {
                            to: 4,
                            cost: 8.321138140452149,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 9.674413400408712,
                        },
                        Edge {
                            to: 1,
                            cost: 12.221639547131913,
                        },
                        Edge {
                            to: 3,
                            cost: 4.6376150266768885,
                        },
                        Edge {
                            to: 4,
                            cost: 15.838066781407072,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 1.0970596862282833,
                        },
                        Edge {
                            to: 1,
                            cost: 17.306826463341803,
                        },
                        Edge {
                            to: 2,
                            cost: 4.6376150266768885,
                        },
                        Edge {
                            to: 4,
                            cost: 6.102211932446107,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 16.098684067859647,
                        },
                        Edge {
                            to: 1,
                            cost: 8.321138140452149,
                        },
                        Edge {
                            to: 2,
                            cost: 15.838066781407072,
                        },
                        Edge {
                            to: 3,
                            cost: 6.102211932446107,
                        },
                    ],
                },
            ],
        };
        let (best_cost, best_permutation) = naive_solver(graph);
        assert!(relative_eq!(37.41646270666716, best_cost));
        assert!(is_same_undirected_circle(&vec![0, 3, 4, 1, 2], &best_permutation));
    }

    #[test]
    fn test_big_floating_tsp() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge {
                            to: 1,
                            cost: 5.357166694956081,
                        },
                        Edge {
                            to: 2,
                            cost: 12.673287166274285,
                        },
                        Edge {
                            to: 3,
                            cost: 15.392922519581575,
                        },
                        Edge {
                            to: 4,
                            cost: 1.8824165228898004,
                        },
                        Edge {
                            to: 5,
                            cost: 1.0673823908781577,
                        },
                        Edge {
                            to: 6,
                            cost: 8.668326879490138,
                        },
                        Edge {
                            to: 7,
                            cost: 18.956348946357103,
                        },
                        Edge {
                            to: 8,
                            cost: 5.399642479870355,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 5.357166694956081,
                        },
                        Edge {
                            to: 2,
                            cost: 11.139733539749999,
                        },
                        Edge {
                            to: 3,
                            cost: 1.661032458795486,
                        },
                        Edge {
                            to: 4,
                            cost: 18.702631945210115,
                        },
                        Edge {
                            to: 5,
                            cost: 3.847655828276122,
                        },
                        Edge {
                            to: 6,
                            cost: 15.73510598766653,
                        },
                        Edge {
                            to: 7,
                            cost: 0.24655608854276645,
                        },
                        Edge {
                            to: 8,
                            cost: 4.321598762165737,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 12.673287166274285,
                        },
                        Edge {
                            to: 1,
                            cost: 11.139733539749999,
                        },
                        Edge {
                            to: 3,
                            cost: 2.1803729313885345,
                        },
                        Edge {
                            to: 4,
                            cost: 16.313099247004377,
                        },
                        Edge {
                            to: 5,
                            cost: 5.585527987185975,
                        },
                        Edge {
                            to: 6,
                            cost: 8.932741722100753,
                        },
                        Edge {
                            to: 7,
                            cost: 12.6998544424725,
                        },
                        Edge {
                            to: 8,
                            cost: 9.05733402266841,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 15.392922519581575,
                        },
                        Edge {
                            to: 1,
                            cost: 1.661032458795486,
                        },
                        Edge {
                            to: 2,
                            cost: 2.1803729313885345,
                        },
                        Edge {
                            to: 4,
                            cost: 3.340513012587236,
                        },
                        Edge {
                            to: 5,
                            cost: 1.46551068868777,
                        },
                        Edge {
                            to: 6,
                            cost: 2.6426709551798355,
                        },
                        Edge {
                            to: 7,
                            cost: 4.492948831722041,
                        },
                        Edge {
                            to: 8,
                            cost: 13.41757522658849,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 1.8824165228898004,
                        },
                        Edge {
                            to: 1,
                            cost: 18.702631945210115,
                        },
                        Edge {
                            to: 2,
                            cost: 16.313099247004377,
                        },
                        Edge {
                            to: 3,
                            cost: 3.340513012587236,
                        },
                        Edge {
                            to: 5,
                            cost: 9.568614854660245,
                        },
                        Edge {
                            to: 6,
                            cost: 6.849461885327388,
                        },
                        Edge {
                            to: 7,
                            cost: 7.455992424446736,
                        },
                        Edge {
                            to: 8,
                            cost: 19.61866966591363,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 1.0673823908781577,
                        },
                        Edge {
                            to: 1,
                            cost: 3.847655828276122,
                        },
                        Edge {
                            to: 2,
                            cost: 5.585527987185975,
                        },
                        Edge {
                            to: 3,
                            cost: 1.46551068868777,
                        },
                        Edge {
                            to: 4,
                            cost: 9.568614854660245,
                        },
                        Edge {
                            to: 6,
                            cost: 7.516298524772413,
                        },
                        Edge {
                            to: 7,
                            cost: 17.155030102652216,
                        },
                        Edge {
                            to: 8,
                            cost: 17.46182408314527,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 8.668326879490138,
                        },
                        Edge {
                            to: 1,
                            cost: 15.73510598766653,
                        },
                        Edge {
                            to: 2,
                            cost: 8.932741722100753,
                        },
                        Edge {
                            to: 3,
                            cost: 2.6426709551798355,
                        },
                        Edge {
                            to: 4,
                            cost: 6.849461885327388,
                        },
                        Edge {
                            to: 5,
                            cost: 7.516298524772413,
                        },
                        Edge {
                            to: 7,
                            cost: 5.959449216135542,
                        },
                        Edge {
                            to: 8,
                            cost: 11.172366336098495,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 18.956348946357103,
                        },
                        Edge {
                            to: 1,
                            cost: 0.24655608854276645,
                        },
                        Edge {
                            to: 2,
                            cost: 12.6998544424725,
                        },
                        Edge {
                            to: 3,
                            cost: 4.492948831722041,
                        },
                        Edge {
                            to: 4,
                            cost: 7.455992424446736,
                        },
                        Edge {
                            to: 5,
                            cost: 17.155030102652216,
                        },
                        Edge {
                            to: 6,
                            cost: 5.959449216135542,
                        },
                        Edge {
                            to: 8,
                            cost: 8.168048838216963,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 5.399642479870355,
                        },
                        Edge {
                            to: 1,
                            cost: 4.321598762165737,
                        },
                        Edge {
                            to: 2,
                            cost: 9.05733402266841,
                        },
                        Edge {
                            to: 3,
                            cost: 13.41757522658849,
                        },
                        Edge {
                            to: 4,
                            cost: 19.61866966591363,
                        },
                        Edge {
                            to: 5,
                            cost: 17.46182408314527,
                        },
                        Edge {
                            to: 6,
                            cost: 11.172366336098495,
                        },
                        Edge {
                            to: 7,
                            cost: 8.168048838216963,
                        },
                    ],
                },
            ],
        };
        let (best_cost, best_permutation) = naive_solver(graph);
        assert!(relative_eq!(33.03008250868411, best_cost));
        assert!(is_same_undirected_circle(&vec![0, 5, 3, 2, 8, 1, 7, 6, 4], &best_permutation));
    }

    #[test]
    fn test_integer_tsp() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 5.0 },
                        Edge { to: 2, cost: 4.0 },
                        Edge { to: 3, cost: 10.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 5.0 },
                        Edge { to: 2, cost: 8.0 },
                        Edge { to: 3, cost: 5.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 4.0 },
                        Edge { to: 1, cost: 8.0 },
                        Edge { to: 3, cost: 3.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 10.0 },
                        Edge { to: 1, cost: 5.0 },
                        Edge { to: 2, cost: 3.0 },
                    ],
                },
            ],
        };
        let (best_cost, best_permutation) = naive_solver(graph);
        assert!(relative_eq!(best_cost, 17.0));
        assert!(is_same_undirected_circle(&best_permutation, &vec![0, 1, 3, 2]));
    }

    #[test]
    fn test_get_all_permutations() {
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
