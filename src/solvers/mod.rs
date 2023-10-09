//! This is an umbrella module for all types of TSP solvers.

use crate::datastructures::AdjacencyMatrix;
pub mod approximate;
pub mod exact;

/// checks, if a given path in a graph
/// is a hamiltonian cycle
///
/// assumes: the path is a valid cycle in the graph,
/// with `path[0] == path[path.len()-1]`
pub fn is_hamiltonian_cycle<A: AdjacencyMatrix>(path: &[usize], graph: &A) -> bool {
    let num_vertices = graph.dim();
    let mut used = vec![0; num_vertices];
    for &u in path {
        if u >= num_vertices {
            return false;
        }
        used[u] += 1
    }

    let mut seen_vertex_twice = false;
    used.into_iter().all(|num_occourances| {
        if num_occourances == 2 {
            if seen_vertex_twice {
                false
            } else {
                seen_vertex_twice = true;
                true
            }
        } else {
            num_occourances == 1
        }
    })
}
