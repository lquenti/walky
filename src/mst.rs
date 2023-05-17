//! Compute a minimum spanning tree

use crate::parser::{Edge, Graph};

use rayon::prelude::*;

/// Prims algorithm for computing an MST of the given `graph`.
/// See [`prim_with_excluded_node`] for more details.
pub fn prim(graph: &Graph) -> Graph {
    prim_with_excluded_node_multi_threaded(graph, graph.num_vertices())
}

/// #TODO
/// currently only single threaded
pub fn prim_with_excluded_node_multi_threaded(graph: &Graph, excluded_vertex: usize) -> Graph {
    prim_with_excluded_node::<Vec<(Edge, bool)>>(graph, excluded_vertex)
}

/// #TODO
/// improve asymptotic performance by using a priority queue
pub fn prim_with_excluded_node_single_threaded(graph: &Graph, excluded_vertex: usize) -> Graph {
    prim_with_excluded_node::<Vec<(Edge, bool)>>(graph, excluded_vertex)
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
pub fn prim_with_excluded_node<D: FindMinCostEdge>(graph: &Graph, excluded_vertex: usize) -> Graph {
    let num_vertices = graph.num_vertices();
    let unconnected_node = num_vertices;

    // `vertex_in_mst[i] == true`: vertex i is already used in the MST
    let mut vertex_in_mst = vec![false; num_vertices + 1];

    // stores our current MST
    let mut mst_adj_list: Vec<Vec<Edge>> = vec![Vec::new(); num_vertices + 1];

    // `dist_from_mst[i]` stores the edge with that the vertex i can be connected to the MST
    // with minimal cost.
    let mut dist_from_mst = D::from_default_value(
        // base case: every vertex is "connected" to the unconnected node with cost f64::INFINITY
        Edge {
            cost: f64::INFINITY,
            to: unconnected_node,
        },
        num_vertices + 1,
    );

    // Vertex at index unconnected_node is special: it is not connected to the rest of the graph,
    // and has distance INFINITY to every other node.
    // It is used as a base case.

    // start with vertex 0, or with vertex 1 if vertex 0 shall be excluded
    let start_index = if excluded_vertex != 0 { 0 } else { 1 };
    dist_from_mst.set_cost(
        start_index,
        Edge {
            to: start_index,
            cost: 0.,
        },
    );
    dist_from_mst.set_excluded_vertex(excluded_vertex);

    // iterate over maximally `num_vertices` many iterations (for every vertex one)
    for _ in 0..=num_vertices {
        //let mut next_vertex = unconnected_node;
        //for i in 0..num_vertices {
        //    // get the index of the vertex that is currently not in the MST
        //    // and has minimal cost to connect to the mst
        //    if !vertex_in_mst[i]
        //        && dist_from_mst[next_vertex].cost > dist_from_mst[i].cost
        //        && i != excluded_vertex
        //    {
        //        next_vertex = i;
        //    }
        //}
        let (next_vertex, next_edge) = dist_from_mst.find_edge_with_minimal_cost(Edge {
            to: num_vertices,
            cost: f64::INFINITY,
        });

        // when we reach an unreachable vertex (like index num_vertices),
        // we are finished
        if next_edge.cost == f64::INFINITY {
            break;
        }

        // add next_vertex to the mst
        dist_from_mst.mark_vertex_as_used(next_vertex);
        if next_vertex != start_index {
            //let connecting_edge = dist_from_mst[next_vertex].clone();
            let reverse_edge = Edge {
                to: next_vertex,
                cost: next_edge.cost,
            };
            let connection_from = next_edge.to;
            let connection_to = next_vertex;
            mst_adj_list[connection_to].push(next_edge);
            mst_adj_list[connection_from].push(reverse_edge);
        }

        // update the minimal connection costs foll all newly adjacent vertices
        for edge in graph[next_vertex].iter() {
            dist_from_mst.update_minimal_cost(next_vertex, *edge);
        }
    }

    // remove the last entry (for unreachable_vertex) as it is only relevant for the algorithm
    mst_adj_list.pop();
    Graph::from(mst_adj_list)
}

/// This trait reflects a datastructure,
/// that holds Edges and can give back the edge with minimal cost,
/// as well as update the cost of edges.
pub trait FindMinCostEdge {
    fn from_default_value(default_val: Edge, size: usize) -> Self;
    fn find_edge_with_minimal_cost(&self, base_case: Edge) -> (usize, Edge);
    /// update the connection cost of `edge_to.to`.
    /// If `edge_to.cost` is less than the current cost, the cost decreases to
    /// `edge_to.cost` and `from` gets saved as the connecting vertex.
    /// If it is higher, the cost does *not* increase.
    /// If provided with the edge `from --> edge_to.to`,
    /// the structure will then possibly remember the reverse edge `from <-- edge_to.to`
    fn update_minimal_cost(&mut self, from: usize, edge_to: Edge);

    /// sets the cost of connecting from `from` to `edge_to.to` to the value `edge_to.cost`.
    fn set_cost(&mut self, from: usize, edge_to: Edge);

    /// sets which vertex to exclude/ignore in the computations
    fn set_excluded_vertex(&mut self, excluded_vertex: usize);

    fn mark_vertex_as_used(&mut self, used_vertex: usize);
}

/// Edge: holds the (currently minimal) connection cost,
/// and the vertex to which to connect to the MST
///
/// bool: true, if the Vertex is in the MST, false if the vertex is not in the MST.
impl FindMinCostEdge for Vec<(Edge, bool)> {
    fn from_default_value(default_val: Edge, size: usize) -> Self {
        vec![(default_val, false); size]
    }

    fn find_edge_with_minimal_cost(&self, base_case: Edge) -> (usize, Edge) {
        let mut next_vertex = base_case.to;
        let mut reverse_edge = base_case;
        for (i, edge) in
            self.iter().enumerate().filter_map(
                |(i, &(edge, used_in_mst))| if used_in_mst { None } else { Some((i, edge)) },
            )
        {
            // get the index of the vertex that is currently not in the MST
            // and has minimal cost to connect to the mst
            if reverse_edge.cost > edge.cost {
                next_vertex = i;
                reverse_edge = edge;
            }
        }
        (next_vertex, reverse_edge)
    }

    fn update_minimal_cost(&mut self, from: usize, edge_to: Edge) {
        //self[edge_to.to] = f64::min(self[edge_to.to], edge.cost);
        if edge_to.cost < self[edge_to.to].0.cost {
            self[edge_to.to].0 = Edge {
                to: from,
                cost: edge_to.cost,
            };
        }
    }

    fn set_cost(&mut self, from: usize, edge_to: Edge) {
        self[from].0 = edge_to;
    }

    fn mark_vertex_as_used(&mut self, used_vertex: usize) {
        self[used_vertex].1 = true;
    }

    fn set_excluded_vertex(&mut self, excluded_vertex: usize) {
        self.mark_vertex_as_used(excluded_vertex);
    }
}

#[cfg(test)]
mod test {
    use quickcheck_macros::quickcheck;

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

        assert_eq!(expected, prim_with_excluded_node_multi_threaded(&graph, 0));
    }

    #[test]
    fn prim_single_and_multi_threaded_agree() {
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
        let excluded_vertex = 0;
        assert_eq!(
            prim_with_excluded_node_single_threaded(&graph, excluded_vertex),
            prim_with_excluded_node_multi_threaded(&graph, excluded_vertex)
        );
    }
}
