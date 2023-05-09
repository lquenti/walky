//! Compute a minimum spanning tree

use crate::parser::{Edge, Graph};

/// Prims algorithm for computing an MST of the given `graph`.
/// See [`prim_with_excluded_node`] for more details.
pub fn prim(graph: &Graph) -> Graph {
    prim_with_excluded_node(graph, graph.num_vertices())
}

/// greedy algorithm:
/// start at the first vertex in the graph and build an MST step by step.
///
/// `excluded_vertex`: option to exclude one vertex from the graph and thus the MST computation.
///     If you do not want to exclude a vertex from the computation, chose
///     `excluded_vertex >= graph.num_vertices()` ([`prim`] does this for you).
///
/// complexity: O(N^2)
///
/// todo: add source for the algorithm
///
/// todo: make the implementation more pretty and more rust ideomatic
pub fn prim_with_excluded_node(graph: &Graph, excluded_vertex: usize) -> Graph {
    let num_vertices = graph.num_vertices();
    let unconnected_node = num_vertices;

    // `vertex_in_mst[i] == true`: vertex i is already used in the MST
    let mut vertex_in_mst = vec![false; num_vertices + 1];

    // stores our current MST
    let mut mst_adj_list: Vec<Vec<Edge>> = vec![Vec::new(); num_vertices + 1];

    // `dist_from_mst[i]` stores the edge with that the vertex i can be connected to the MST
    // with minimal cost.
    let mut dist_from_mst: Vec<Edge> = vec![
        // base case: every vertex is "connected" to the unconnected node with cost f64::INFINITY
        Edge {
            cost: f64::INFINITY,
            to: unconnected_node,
        };
        num_vertices + 1
    ];

    // Vertex at index unconnected_node is special: it is not connected to the rest of the graph,
    // and has distance INFINITY to every other node.
    // It is used as a base case.

    // start with vertex 0
    let start_index = if excluded_vertex != 0 { 0 } else { 1 };
    dist_from_mst[start_index] = Edge {
        to: start_index,
        cost: 0.,
    };

    // at max. num_vertices many iterations, for every vertex one
    for _ in 0..=num_vertices {
        let mut next_vertex = unconnected_node;
        for i in 0..num_vertices {
            // get the index of the vertex that is currently not in the MST
            // and has minimal cost to connect to the mst
            if !vertex_in_mst[i]
                && dist_from_mst[next_vertex].cost > dist_from_mst[i].cost
                && i != excluded_vertex
            {
                next_vertex = i;
            }
        }

        // when we reach a unreachable vertex (like index num_vertices),
        // we are finished
        if dist_from_mst[next_vertex].cost == f64::INFINITY {
            break;
        }

        // add next_vertex to the mst
        vertex_in_mst[next_vertex] = true;
        if next_vertex != start_index {
            let connecting_edge = dist_from_mst[next_vertex].clone();
            let reverse_edge = Edge {
                to: next_vertex,
                cost: connecting_edge.cost,
            };
            let connection_from = connecting_edge.to;
            let connection_to = next_vertex;
            mst_adj_list[connection_to].push(connecting_edge);
            mst_adj_list[connection_from].push(reverse_edge);
        }

        // update the minimal connection costs foll all newly adjacent vertices
        for edge in graph[next_vertex].iter() {
            //dist_from_mst[edge.to] = f64::min(dist_from_mst[edge.to], edge.cost);
            if edge.cost < dist_from_mst[edge.to].cost {
                dist_from_mst[edge.to] = Edge {
                    to: next_vertex,
                    cost: edge.cost,
                }
            }
        }
    }

    // remove the last entry (for unreachable_vertex) as it is only relevant for the algorithm
    mst_adj_list.pop();
    Graph::from(mst_adj_list)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn easy_prim() {
        let graph = Graph::from(vec![
            vec![Edge { to: 1, cost: 1.0 }],
            vec![Edge { to: 0, cost: 1.0 }],
        ]);

        let mst = prim(&graph);
        assert_eq!(graph, mst);
    }

    /// graph:
    /// 0 ----- 1
    /// |\     /|
    /// | \   / |
    /// |  \ /  |
    /// |   X   |
    /// |  / \  |
    /// | /   \ |
    /// |/     \|
    /// 3 ----- 2
    ///
    /// MST:
    /// 0       1
    ///  \     /
    ///   \   /  
    ///    \ /   
    ///     X    
    ///    / \   
    ///   /   \  
    ///  /     \
    /// 3 ----- 2
    #[test]
    fn four_vertices_mst_prim() {
        let graph = Graph::from(vec![
            //vertex 0
            vec![
                Edge { to: 1, cost: 1.0 },
                Edge { to: 2, cost: 0.1 },
                Edge { to: 3, cost: 2.0 },
            ],
            //vertex 1
            vec![
                Edge { to: 0, cost: 1.0 },
                Edge { to: 2, cost: 5.0 },
                Edge { to: 3, cost: 0.1 },
            ],
            //vertex 2
            vec![
                Edge { to: 0, cost: 0.1 },
                Edge { to: 1, cost: 1.1 },
                Edge { to: 3, cost: 0.1 },
            ],
            //vertex 3
            vec![
                Edge { to: 0, cost: 2.0 },
                Edge { to: 1, cost: 0.1 },
                Edge { to: 2, cost: 0.1 },
            ],
        ]);

        let expected = Graph::from(vec![
            //vertex 0
            vec![Edge { to: 2, cost: 0.1 }],
            //vertex 1
            vec![Edge { to: 3, cost: 0.1 }],
            //vertex 2
            vec![Edge { to: 0, cost: 0.1 }, Edge { to: 3, cost: 0.1 }],
            //vertex 3
            vec![Edge { to: 2, cost: 0.1 }, Edge { to: 1, cost: 0.1 }],
        ]);

        assert_eq!(expected, prim(&graph));
    }

    /// graph:
    /// 0 ----- 1
    /// |\     /|
    /// | \   / |
    /// |  \ /  |
    /// |   X   |
    /// |  / \  |
    /// | /   \ |
    /// |/     \|
    /// 3 ----- 2
    ///
    /// exclude vertex 0 from MST computation
    ///
    /// MST:
    ///         1
    ///        /
    ///       /  
    ///      /   
    ///     /    
    ///    /     
    ///   /      
    ///  /      
    /// 3 ----- 2
    #[test]
    fn exclude_one_vertex_from_mst() {
        let graph = Graph::from(vec![
            //vertex 0
            vec![
                Edge { to: 1, cost: 1.0 },
                Edge { to: 2, cost: 0.1 },
                Edge { to: 3, cost: 2.0 },
            ],
            //vertex 1
            vec![
                Edge { to: 0, cost: 1.0 },
                Edge { to: 2, cost: 5.0 },
                Edge { to: 3, cost: 0.1 },
            ],
            //vertex 2
            vec![
                Edge { to: 0, cost: 0.1 },
                Edge { to: 1, cost: 1.1 },
                Edge { to: 3, cost: 0.1 },
            ],
            //vertex 3
            vec![
                Edge { to: 0, cost: 2.0 },
                Edge { to: 1, cost: 0.1 },
                Edge { to: 2, cost: 0.1 },
            ],
        ]);

        let expected = Graph::from(vec![
            //vertex 0 not in the MST
            vec![],
            //vertex 1
            vec![Edge { to: 3, cost: 0.1 }],
            //vertex 2
            vec![Edge { to: 3, cost: 0.1 }],
            //vertex 3
            vec![Edge { to: 1, cost: 0.1 }, Edge { to: 2, cost: 0.1 }],
        ]);

        assert_eq!(expected, prim_with_excluded_node(&graph, 0));
    }
}
