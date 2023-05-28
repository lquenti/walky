use std::collections::HashSet;

use crate::parser::{Edge, Graph, Vertex};

// TODO: make me a trait/impl

/// This function checks whether the graph is fully connected.
/// This is a prerequisite for the TSP.
pub fn is_fully_connected(g: &Graph) -> bool {
    let n = g.num_vertices();
    g.iter().all(|v| v.num_edges() == n - 1)
}

/// This function checks that whether the graph has multiple edges from one vertex to another
pub fn is_multi_edge(graph: &Graph) -> bool {
    graph.vertices.iter().any(|vertex| {
        let destinations: Vec<usize> = vertex.edges.iter().map(|edge| edge.to).collect();
        let unique_destinations: Vec<usize> = destinations.clone().into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        destinations.len() != unique_destinations.len()
    })
}

/// This function verifies that the graph is actual undirected, i.e. the weights are the same for
/// both directions.
pub fn is_undirected(g: &Graph) -> bool {
    todo!()
}

#[cfg(test)]
mod test_preconditions {
    use super::*;

    #[test]
    fn test_fully_connected_graph() {
            let graph = Graph {
        vertices: vec![
            Vertex { edges: vec![Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }] },
        ],
    };
    assert!(is_fully_connected(&graph))
    }

    #[test]
    fn test_not_fully_connected_graph() {
            let graph = Graph {
        vertices: vec![
            Vertex { edges: vec![Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }] },
        ],
    };
    assert!(!is_fully_connected(&graph));
    }

    #[test]
    fn test_not_multi_edge() {
            let graph = Graph {
        vertices: vec![
            Vertex { edges: vec![Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }] },
        ],
    };
    assert!(!is_multi_edge(&graph))
    }

    #[test]
    fn test_is_multi_edge() {
            let graph = Graph {
        vertices: vec![
            Vertex { edges: vec![Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 1, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 3, cost: 0.0 }] },
            Vertex { edges: vec![Edge { to: 0, cost: 0.0 }, Edge { to: 1, cost: 0.0 }, Edge { to: 2, cost: 0.0 }] },
        ],
    };
    assert!(is_multi_edge(&graph))
    }

    /*
    #[test]
    fn test_is_multi_edge() {

    }
    */
}
