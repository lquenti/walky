//! Finding an approximate solution to the MST

use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;

use crate::datastructures::{AdjacencyMatrix, NAMatrix, Solution};

/// Using the nearest neighbour algorithm for a single starting node.
///
/// The algorithm works greedily; Starting at a specific node, it first goes to the nearest node.
/// From there on, it visits the nearest previously unvisited node. Once all nodes are visited,
/// it returns to the beginning. Thats our tour.
pub fn single_nearest_neighbour(graph_matrix: &NAMatrix, index: usize) -> Solution {
    assert!(index < graph_matrix.dim());
    let num_nodes = graph_matrix.ncols();
    let mut path = Vec::with_capacity(num_nodes);
    let mut visited = vec![false; num_nodes];
    let mut current_node = index;
    let mut distance = 0.0;

    path.push(current_node);
    visited[current_node] = true;

    while path.len() < num_nodes {
        let mut nearest_neighbour = None;
        let mut min_distance = f64::INFINITY;

        for node in 0..num_nodes {
            if !visited[node] && graph_matrix[(current_node, node)] < min_distance {
                min_distance = graph_matrix[(current_node, node)];
                nearest_neighbour = Some(node);
            }
        }

        if let Some(neighbour) = nearest_neighbour {
            current_node = neighbour;
            path.push(current_node);
            distance += min_distance;
            visited[current_node] = true;
        } else {
            // This should never happen; it should be catched by the while loop
            panic!()
        }
    }

    // Add the loop closure
    distance += graph_matrix[(current_node, index)];
    (distance, path)
}

/// Generate n unique random numbers between `[0..n)`
fn n_random_numbers(min: usize, max: usize, n: usize) -> Vec<usize> {
    let mut xs: Vec<usize> = (min..max).collect();
    let mut rng = rand::thread_rng();
    xs.shuffle(&mut rng);
    xs.truncate(n);
    xs
}

/// Call [`single_nearest_neighbour`] n times, randomly.
/// Since [`single_nearest_neighbour`] is deterministic, we use n different starting nodes.
pub fn n_nearest_neighbour(graph_matrix: &NAMatrix, n: usize) -> Solution {
    assert!(n != 0);
    assert!(n <= graph_matrix.dim());
    n_random_numbers(0, graph_matrix.dim(), n)
        .into_iter()
        .map(|k| single_nearest_neighbour(graph_matrix, k))
        .min_by_key(|&(distance, _)| OrderedFloat(distance))
        .unwrap()
}

/// Call [`single_nearest_neighbour`] for every starting node.
pub fn nearest_neighbour(graph_matrix: &NAMatrix) -> Solution {
    n_nearest_neighbour(graph_matrix, graph_matrix.dim())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{datastructures::{Edge, Graph, Vertex, NAMatrix}, solvers::exact::first_improved_solver};


    #[test]
    fn integer_graph_5x5() {
        let graph: Graph = Graph {vertices: vec![Vertex { edges: vec![Edge { to: 1, cost: 6.0 }, Edge { to: 2, cost: 8.0 }, Edge { to: 3, cost: 5.0 }, Edge { to: 4, cost: 3.0 }] },Vertex { edges: vec![Edge { to: 0, cost: 6.0 }, Edge { to: 2, cost: 9.0 }, Edge { to: 3, cost: 5.0 }, Edge { to: 4, cost: 4.0 }] },Vertex { edges: vec![Edge { to: 0, cost: 8.0 }, Edge { to: 1, cost: 9.0 }, Edge { to: 3, cost: 8.0 }, Edge { to: 4, cost: 5.0 }] },Vertex { edges: vec![Edge { to: 0, cost: 5.0 }, Edge { to: 1, cost: 5.0 }, Edge { to: 2, cost: 8.0 }, Edge { to: 4, cost: 6.0 }] },Vertex { edges: vec![Edge { to: 0, cost: 3.0 }, Edge { to: 1, cost: 4.0 }, Edge { to: 2, cost: 5.0 }, Edge { to: 3, cost: 6.0 }] }]};
        let nm: NAMatrix = (&graph).into();
        // Starting Node 0
        let (distance, path) = single_nearest_neighbour(&nm, 0);
        assert_eq!(distance, 28.0);
        assert_eq!(path, vec![0,4,1,3,2]);
        // Starting Node 1
        let (distance, path) = single_nearest_neighbour(&nm, 1);
        assert_eq!(distance, 29.0);
        assert_eq!(path, vec![1,4,0,3,2]);
        // Starting Node 2
        let (distance, path) = single_nearest_neighbour(&nm, 2);
        assert_eq!(distance, 27.0);
        assert_eq!(path, vec![2,4,0,3,1]);
        // Starting Node 3
        let (distance, path) = single_nearest_neighbour(&nm, 3);
        assert_eq!(distance, 29.0);
        assert_eq!(path, vec![3,0,4,1,2]);
        // Starting Node 4
        let (distance, path) = single_nearest_neighbour(&nm, 4);
        assert_eq!(distance, 27.0);
        assert_eq!(path, vec![4,0,3,1,2]);

        // Check that total is 27 and one of the best two
        let (distance, path) = nearest_neighbour(&nm);
        assert_eq!(distance, 27.0);
        assert!(
            path == vec![2,4,0,3,1] || path == vec![4,0,3,1,2]
            );

        // That that we are below or equal to the perfect solution
        let (perfect_d, _) = first_improved_solver(&nm);
        assert!(perfect_d <= distance);
    }
}
