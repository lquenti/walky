use std::collections::HashSet;

use crate::parser::Graph;

/// This function checks whether the graph is fully connected.
/// This is a prerequisite for the TSP.
pub fn is_fully_connected(g: &Graph) -> bool {
    let n = g.num_vertices();
    g.iter().all(|v| v.degree() == n - 1)
}

/// This function checks that whether the graph has multiple edges from one vertex to another
pub fn is_multi_edge(graph: &Graph) -> bool {
    graph.vertices.iter().any(|vertex| {
        let destinations: Vec<usize> = vertex.edges.iter().map(|edge| edge.to).collect();
        let unique_destinations: Vec<usize> = destinations
            .clone()
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        destinations.len() != unique_destinations.len()
    })
}

/// This function verifies that the graph is actual undirected, i.e. the weights are the same for
/// both directions.
pub fn is_undirected(graph: &Graph) -> bool {
    for i in 0..graph.num_vertices() {
        let v1 = &graph.vertices[i];
        for v1e in v1.iter() {
            // Find the Vertex connected to the edge
            let v2 = &graph.vertices[v1e.to];

            // Find reverse edge link
            for v2e in v2.iter() {
                if v2e.to == i {
                    // If not same weight => directed graph
                    if v2e.cost != v1e.cost {
                        return false;
                    }
                    break;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod test_preconditions {
    use super::*;
    use crate::parser::{Edge, Vertex};

    #[test]
    fn test_fully_connected_graph() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                    ],
                },
            ],
        };
        assert!(is_fully_connected(&graph))
    }

    #[test]
    fn test_not_fully_connected_graph() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![Edge { to: 2, cost: 0.0 }, Edge { to: 3, cost: 0.0 }],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                    ],
                },
            ],
        };
        assert!(!is_fully_connected(&graph));
    }

    #[test]
    fn test_not_multi_edge() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                    ],
                },
            ],
        };
        assert!(!is_multi_edge(&graph))
    }

    #[test]
    fn test_is_multi_edge() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                    ],
                },
            ],
        };
        assert!(is_multi_edge(&graph))
    }

    #[test]
    fn test_undirected() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 5.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 7.5 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 5.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 7.5 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                    ],
                },
            ],
        };
        assert!(is_undirected(&graph))
    }

    #[test]
    fn test_not_undirected() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 2, cost: 5.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 4.0 },
                        Edge { to: 3, cost: 0.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 0.0 },
                        Edge { to: 1, cost: 0.0 },
                        Edge { to: 2, cost: 0.0 },
                    ],
                },
            ],
        };
        assert!(!is_undirected(&graph))
    }
}
