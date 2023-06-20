//! Compute a minimum spanning tree

use core::panic;
use std::{
    cmp::Reverse,
    ops::{Deref, DerefMut},
};

use crate::{
    computation_mode::*,
    datastructures::{AdjacencyMatrix, Edge, Graph, NAMatrix},
};

use delegate::delegate;
use nalgebra::{Dyn, U1};
use ordered_float::OrderedFloat;
use priority_queue::PriorityQueue;
use rayon::prelude::*;

/// Prims algorithm for computing an MST of the given `graph`.
///
/// `MODE`: constant parameter, choose one of the values from [`crate::computation_mode`]
///
/// See [`prim_with_excluded_node`] for more details.
pub fn prim<const MODE: usize>(graph: &NAMatrix) -> Graph {
    match MODE {
        SEQ_COMPUTATION => prim_with_excluded_node_single_threaded(graph, graph.dim()),
        PAR_COMPUTATION => prim_with_excluded_node_multi_threaded(graph, graph.dim()),
        #[cfg(feature = "mpi")]
        MPI_COMPUTATION => prim::<SEQ_COMPUTATION>(graph),
        _ => panic_on_invaid_mode::<MODE>(),
    }
}

/// multithreaded version of [`prim_with_excluded_node_single_threaded`].
///
/// If you have multiple calls to prims algorithm, use a single threaded version
/// and make the calls in parallel.
pub fn prim_with_excluded_node_multi_threaded(graph: &NAMatrix, excluded_vertex: usize) -> Graph {
    prim_with_excluded_node::<MultiThreadedVecWrapper>(graph, excluded_vertex)
}

/// naive version using only vectors as data structures.
/// For small enough (might not have to be very small) inputs
/// this is faster than a priority queue due to
/// less branching and better auto-vectorization potential.
/// Asymptotic performance: O(N^2)
pub fn prim_with_excluded_node_single_threaded(graph: &NAMatrix, excluded_vertex: usize) -> Graph {
    prim_with_excluded_node::<Vec<(Edge, bool)>>(graph, excluded_vertex)
}

/// improve asymptotic performance by using a priority queue
pub fn prim_with_excluded_node_priority_queue(graph: &NAMatrix, excluded_vertex: usize) -> Graph {
    prim_with_excluded_node::<VerticesInPriorityQueue>(graph, excluded_vertex)
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
fn prim_with_excluded_node<D: FindMinCostEdge>(graph: &NAMatrix, excluded_vertex: usize) -> Graph {
    let num_vertices = graph.dim();
    let unconnected_node = num_vertices;

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
        let (next_vertex, next_edge) = dist_from_mst.find_edge_with_minimal_cost();

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
        //for edge in graph[next_vertex].iter() {
        //for (to, &cost) in graph.row(next_vertex).iter().enumerate() {
        //    dist_from_mst.update_minimal_cost(next_vertex, Edge { to, cost });
        //}
        dist_from_mst.update_minimal_cost(next_vertex, graph.row(next_vertex))
    }

    // remove the last entry (for unreachable_vertex) as it is only relevant for the algorithm
    mst_adj_list.pop();
    Graph::from(mst_adj_list)
}

type NAMatrixRowView<'a> =
    nalgebra::Matrix<f64, U1, Dyn, nalgebra::ViewStorage<'a, f64, U1, Dyn, U1, Dyn>>;

/// This trait reflects a datastructure,
/// that holds Edges and can give back the edge with minimal cost,
/// as well as update the cost of edges.
trait FindMinCostEdge {
    fn from_default_value(default_val: Edge, size: usize) -> Self;

    /// Get the index of the vertex that is currently not in the MST
    /// and has minimal cost to connect to the mst, as well as the
    /// corresponding connecting edge to the MST.
    fn find_edge_with_minimal_cost(&self) -> (usize, Edge);
    /// update the connection cost of `edge_to.to`.
    /// If `edge_to.cost` is less than the current cost, the cost decreases to
    /// `edge_to.cost` and `from` gets saved as the connecting vertex.
    /// If it is higher, the cost does *not* increase.
    /// If provided with the edge `from --> edge_to.to`,
    /// the structure will then possibly remember the reverse edge `from <-- edge_to.to`
    fn update_minimal_cost(&mut self, from: usize, new_neighbours: NAMatrixRowView);

    /// sets the cost of connecting from `from` to `edge_to.to` to the value `edge_to.cost`.
    fn set_cost(&mut self, from: usize, edge_to: Edge);

    /// sets which vertex to exclude/ignore in the computations
    fn set_excluded_vertex(&mut self, excluded_vertex: usize);

    fn mark_vertex_as_used(&mut self, used_vertex: usize);
}

#[derive(Clone, Debug, PartialEq)]
struct VerticesInPriorityQueue {
    /// stores the vertices that are not currently in the MST,
    /// can efficiently find the vertex with minimal connection cost to the MST
    cost_queue: PriorityQueue<usize, Reverse<OrderedFloat<f64>>>,
    /// implements the following map:
    /// given a vertex `i`, the minimal cost edge to the
    /// MST is to the vertex `j == connection_to_mst[i]`
    connection_to_mst: Vec<usize>,
    /// `used[i]`: vertex `i` is already part of the MST
    used: Vec<bool>,
}
impl FindMinCostEdge for VerticesInPriorityQueue {
    fn from_default_value(default_val: Edge, size: usize) -> Self {
        VerticesInPriorityQueue {
            cost_queue: PriorityQueue::from(
                (0..size)
                    .map(|i| (i, Reverse(OrderedFloat(default_val.cost))))
                    .collect::<Vec<(usize, Reverse<OrderedFloat<f64>>)>>(),
            ),
            connection_to_mst: vec![default_val.to; size],
            used: vec![false; size],
        }
    }

    fn find_edge_with_minimal_cost(&self) -> (usize, Edge) {
        let base_case = Edge {
            to: self.connection_to_mst.len(),
            cost: f64::INFINITY,
        };
        let (&next_vertex, &Reverse(OrderedFloat(cost))) = self
            .cost_queue
            .peek()
            .unwrap_or((&base_case.to, &Reverse(OrderedFloat(base_case.cost))));
        let to = self.connection_to_mst[next_vertex];

        (next_vertex, Edge { to, cost })
    }

    fn update_minimal_cost(&mut self, from: usize, new_neighbours: NAMatrixRowView) {
        for (to, &cost) in new_neighbours.iter().enumerate() {
            if self.used[to] {
                continue;
            }
            let Reverse(OrderedFloat(old_cost)) = self.cost_queue
            .push_increase(to, Reverse(OrderedFloat(cost)))
            .unwrap_or_else(|| panic!("Every unused unused vertex shall be contained in the queue from the beginning. Missing vertex: {}", to));
            if cost <= old_cost {
                self.connection_to_mst[to] = from;
            }
        }
    }

    fn set_cost(&mut self, from: usize, edge_to: Edge) {
        self.cost_queue
            .change_priority(&from, Reverse(OrderedFloat(edge_to.cost)));

        self.connection_to_mst[from] = edge_to.to;
    }

    fn set_excluded_vertex(&mut self, excluded_vertex: usize) {
        self.mark_vertex_as_used(excluded_vertex);
    }

    fn mark_vertex_as_used(&mut self, used_vertex: usize) {
        self.cost_queue.remove(&used_vertex);
        self.used[used_vertex] = true;
    }
}

/// Edge: holds the (currently minimal) connection cost,
/// and the vertex to which to connect to the MST
///
/// bool: true, if the Vertex is in the MST, false if the vertex is not in the MST.
impl FindMinCostEdge for Vec<(Edge, bool)> {
    fn from_default_value(default_val: Edge, size: usize) -> Self {
        vec![(default_val, false); size]
    }

    fn find_edge_with_minimal_cost(&self) -> (usize, Edge) {
        let base_case = Edge {
            to: self.len(),
            cost: f64::INFINITY,
        };
        let (next_vertex, reverse_edge) = self
            .iter()
            .enumerate()
            // skip all used vertices
            .filter_map(
                |(i, &(edge, used_in_mst))| if used_in_mst { None } else { Some((i, edge)) },
            )
            // find the next vertex via the corresponding edge with minimal cost
            .min_by(|&(_, edg_i), &(_, edg_j)| {
                OrderedFloat(edg_i.cost).cmp(&OrderedFloat(edg_j.cost))
            })
            // unwrap, or give back the base case
            .unwrap_or((base_case.to, base_case));
        (next_vertex, reverse_edge)
    }

    fn update_minimal_cost(&mut self, from: usize, new_neighbours: NAMatrixRowView) {
        //self[to] = f64::min(self[to], edge.cost);
        for (to, &cost) in new_neighbours.iter().enumerate() {
            if cost < self[to].0.cost {
                self[to].0 = Edge { to: from, cost };
            }
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

#[derive(Debug, PartialEq)]
struct MultiThreadedVecWrapper(Vec<(Edge, bool)>);

impl Deref for MultiThreadedVecWrapper {
    type Target = Vec<(Edge, bool)>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MultiThreadedVecWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl FindMinCostEdge for MultiThreadedVecWrapper {
    fn from_default_value(default_val: Edge, size: usize) -> Self {
        MultiThreadedVecWrapper(Vec::from_default_value(default_val, size))
    }
    delegate! {
        to self.0 {
            fn set_cost(&mut self, from: usize, edge_to: Edge);
            fn set_excluded_vertex(&mut self, excluded_vertex: usize);
            fn mark_vertex_as_used(&mut self, used_vertex: usize);
        }
    }

    fn update_minimal_cost(&mut self, from: usize, new_neighbours: NAMatrixRowView) {
        //self[to] = f64::min(self[to], edge.cost);
        let dim = new_neighbours.shape().1;
        //for (to, &cost) in new_neighbours.par_iter().enumerate()
        (0..dim).into_par_iter().for_each(|to| {
            let neighbour_prt = new_neighbours.as_ptr() as *mut f64;
            // safety: the data exists, we do not leave the range
            // of the underlying NAMatrix (we add at most dim*(dim-1),
            // and the pointer to the row has at most offset dim-1 from the cell at index (0,0).
            // Therefore we stay within an offset of (dim*dim)-1
            let cost = unsafe { *neighbour_prt.add(dim * to) };
            let to_dist_ptr = self.as_ptr() as *mut (Edge, bool);
            if cost < self[to].0.cost {
                // safety:
                //  - no race conditions, since the parallel iterator visits each value of to
                //    exactly once
                //  - we do not exeed the length of the vector self.0
                unsafe {
                    (*to_dist_ptr.add(to)).0 = Edge { to: from, cost };
                }
            }
        });
    }

    fn find_edge_with_minimal_cost(&self) -> (usize, Edge) {
        let base_case = Edge {
            to: self.0.len(),
            cost: f64::INFINITY,
        };
        let (next_vertex, reverse_edge) = self
            .0
            .par_iter()
            .enumerate()
            // skip all used vertices
            .filter_map(
                |(i, &(edge, used_in_mst))| if used_in_mst { None } else { Some((i, edge)) },
            )
            // find the next vertex via the corresponding edge with minimal cost
            .min_by(|&(_, edg_i), &(_, edg_j)| {
                OrderedFloat(edg_i.cost).cmp(&OrderedFloat(edg_j.cost))
            })
            // unwrap, or give back the base case
            .unwrap_or((base_case.to, base_case));
        (next_vertex, reverse_edge)
    }
}

#[cfg(test)]
mod test {
    use std::assert_eq;

    use nalgebra::DMatrix;

    use super::*;

    #[test]
    fn easy_prim() {
        let graph = Graph::from(vec![
            vec![Edge { to: 1, cost: 1.0 }],
            vec![Edge { to: 0, cost: 1.0 }],
        ]);

        let mst = prim::<SEQ_COMPUTATION>(&(&graph).into());
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

        assert_eq!(expected, prim::<SEQ_COMPUTATION>(&(&graph).into()));
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

        assert_eq!(
            expected,
            prim_with_excluded_node_multi_threaded(&(&graph).into(), 0)
        );
    }

    #[test]
    fn prim_all_versions_agree() {
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
        let res_st = prim_with_excluded_node_single_threaded(&(&graph).into(), excluded_vertex);
        let res_mt = prim_with_excluded_node_multi_threaded(&(&graph).into(), excluded_vertex);
        let res_prio = prim_with_excluded_node_priority_queue(&(&graph).into(), excluded_vertex);
        assert_eq!(
            res_st, res_mt,
            "single_threaded should agree with multi_threaded"
        );
        assert_eq!(
            res_st, res_prio,
            "single_threaded should agree with priority queue version"
        );
    }

    #[test]
    fn test_vertices_in_priority_queue_from_default_value() {
        let default_val = Edge {
            to: 3,
            cost: f64::INFINITY,
        };

        let size = 5;

        let vert = VerticesInPriorityQueue::from_default_value(default_val, size);

        let mut queue = PriorityQueue::new();
        for i in 0..size {
            queue.push(i, Reverse(OrderedFloat(f64::INFINITY)));
        }

        assert_eq!(vert.cost_queue, queue);
        assert_eq!(vert.cost_queue.into_vec(), vec![0, 1, 2, 3, 4]);
        assert_eq!(vert.connection_to_mst, vec![3; 5])
    }

    #[test]
    fn test_vertices_in_priority_queue_increase_priority() {
        let default_val = Edge {
            to: 4,
            cost: f64::INFINITY,
        };

        let size = 5;

        let mut vert = VerticesInPriorityQueue::from_default_value(default_val, size);

        let res = vert.cost_queue.push_increase(0, Reverse(OrderedFloat(1.0)));
        assert_eq!(res, Some(Reverse(OrderedFloat(f64::INFINITY))));
    }

    #[test]
    fn test_vertices_in_priority_queue_update_priority_does_not_panic() {
        let default_val = Edge {
            to: 4,
            cost: f64::INFINITY,
        };

        let size = 5;

        let mut vert = VerticesInPriorityQueue::from_default_value(default_val, size);
        let mat = DMatrix::from_row_slice(1, size, &[1.0; 5]);

        vert.update_minimal_cost(0, mat.row(0));
    }

    #[test]
    fn test_vertices_in_priority_queue_update_priority_works() {
        let default_val = Edge {
            to: 4,
            cost: f64::INFINITY,
        };

        let size = 5;

        let mut vert = VerticesInPriorityQueue::from_default_value(default_val, size);
        let mat = DMatrix::from_row_slice(1, size, &[0.0, 1.0, 0.0, 0.0, 0.0]);

        vert.update_minimal_cost(0, mat.row(0));
        assert_eq!(vert.connection_to_mst[1], 0);
        assert_eq!(
            vert.cost_queue.get_priority(&1),
            Some(&Reverse(OrderedFloat(1.0f64)))
        );
    }
}
