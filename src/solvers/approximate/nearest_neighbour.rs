//! Finding an approximate solution to the MST

use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;

use crate::datastructures::{AdjacencyMatrix, NAMatrix, Solution};

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
        let mut nearest_neighbor = None;
        let mut min_distance = f64::INFINITY;

        for node in 0..num_nodes {
            if !visited[node] && graph_matrix[(current_node, node)] < min_distance {
                min_distance = graph_matrix[(current_node, node)];
                nearest_neighbor = Some(node);
            }
        }

        if let Some(neighbor) = nearest_neighbor {
            current_node = neighbor;
            path.push(current_node);
            distance += min_distance;
            visited[current_node] = true;
        } else {
            // TODO lose the brnahcing here
            // This should never happen;
            panic!()
        }
    }
    (distance, path)
}

/// Exclusively
fn n_random_numbers(min: usize, max: usize, n: usize) -> Vec<usize> {
    let mut xs: Vec<usize> = (min..max).collect();
    let mut rng = rand::thread_rng();
    xs.shuffle(&mut rng);
    xs.truncate(n);
    xs
}

pub fn n_nearest_neighbour(graph_matrix: &NAMatrix, n: usize) -> Solution {
    assert!(n != 0);
    assert!(n <= graph_matrix.dim());
    n_random_numbers(0, graph_matrix.dim(), n)
        .into_iter()
        .map(|k| single_nearest_neighbour(graph_matrix, k))
        .min_by_key(|&(distance, _)| OrderedFloat(distance))
        .unwrap()
}

pub fn nearest_neighbour(graph_matrix: &NAMatrix) -> Solution {
    n_nearest_neighbour(graph_matrix, graph_matrix.dim())
}
