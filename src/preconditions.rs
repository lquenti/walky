use crate::parser::{Edge, Graph, Vertex};

// TODO: make me a trait/impl

/// This function checks whether the graph is fully connected.
/// This is a prerequisite for the TSP.
pub fn is_fully_connected(g: &Graph) -> bool {
    let n = g.num_vertices();

    // Now we verify that each vertex has n-1 connections...
    // TODO write me more functional
    for v in g.iter() {
        if v.num_edges() != n-1 {
            return false
        }
    }
    true
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
}
