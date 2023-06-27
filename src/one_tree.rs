//! This module is about computing the 1-tree lower bound for the TSP.
//! See [this article](https://doi.org/10.1287/opre.18.6.1138) for further information.

use rayon::prelude::*;

#[cfg(feature = "mpi")]
use crate::ROOT_RANK;
#[cfg(feature = "mpi")]
use mpi::collective::*;
#[cfg(feature = "mpi")]
use mpi::topology::*;

use crate::{
    computation_mode,
    datastructures::{AdjacencyMatrix, Edge, Graph, NAMatrix},
    mst::prim_with_excluded_node_single_threaded,
};

/// Creates a 1-tree from a given graph.
///
/// The `special_vertex` is the vertex, that is excluded during MST computation.
///
/// # Panics
/// **in debug mode**: if `special_vertex >= graph.num_vertices()`, since then the vertex is not
/// valid.
pub fn one_tree(graph: &NAMatrix, special_vertex: usize) -> Graph {
    debug_assert!(
        special_vertex < graph.dim(),
        "The special vertex has to be a valid vertex in the graph."
    );

    let mut mst = prim_with_excluded_node_single_threaded(graph, &[special_vertex]);
    let mut fst_min_edg = Edge {
        to: special_vertex,
        cost: f64::INFINITY,
    };
    let mut snd_min_edg = Edge {
        to: special_vertex,
        cost: f64::INFINITY,
    };

    // find the two edges with minimal cost in the graph, that connect
    // `special_vertex` to the rest of the graph
    for (to, &cost) in graph.row(special_vertex).iter().enumerate() {
        // skip this case:
        // the algorithm should not consider a vertex its nearest neighbour,
        // it should only consider proper neighbours
        if to == special_vertex {
            continue;
        }

        let edge = Edge { to, cost };
        if edge.cost < fst_min_edg.cost {
            // edge is the best edge
            snd_min_edg = fst_min_edg;
            fst_min_edg = edge;
        } else if edge.cost < snd_min_edg.cost {
            // edge is worse than fst_min_edg, but better than snd_min_edg
            snd_min_edg = edge;
        }
    }

    // add the two edges to the mst, making it a 1-tree
    mst.add_undirected_edge(special_vertex, fst_min_edg.to, fst_min_edg.cost);
    mst.add_undirected_edge(special_vertex, snd_min_edg.to, snd_min_edg.cost);

    mst
}

/// cumputes the 1-tree lower bound
///
/// `MODE`: the level of parallelization, see also [`computation_mode`]
///
/// # Panics
/// if the graph is empty.
pub fn one_tree_lower_bound<const MODE: usize>(graph: &NAMatrix) -> f64 {
    match MODE {
        computation_mode::SEQ_COMPUTATION => one_tree_lower_bound_seq(graph),
        computation_mode::PAR_COMPUTATION => one_tree_lower_bound_par(graph),
        #[cfg(feature = "mpi")]
        computation_mode::MPI_COMPUTATION => one_tree_lower_bound_mpi(graph),
        _ => computation_mode::panic_on_invaid_mode::<MODE>(),
    }
}

/// cumputes the 1-tree lower bound
///
/// # Panics
/// if the graph is empty.
fn one_tree_lower_bound_seq(graph: &NAMatrix) -> f64 {
    (0..graph.dim())
        .map(|special_vertex| one_tree(graph, special_vertex).undirected_edge_weight())
        .min_by(|x, y| {
            x.partial_cmp(y)
                .expect("Tried to compare NaN value. Your data seems currupt.")
        })
        .expect("Cannot compute the 1-tree lower bound of the empty graph")
}

/// cumputes the 1-tree lower bound in parallel
///
/// # Panics
/// if the graph is empty.
fn one_tree_lower_bound_par(graph: &NAMatrix) -> f64 {
    (0..graph.dim())
        .into_par_iter()
        .map(|special_vertex| one_tree(graph, special_vertex).undirected_edge_weight())
        .min_by(|x, y| {
            x.partial_cmp(y)
                .expect("Tried to compare NaN value. Your data seems currupt.")
        })
        .expect("Cannot compute the 1-tree lower bound of the empty graph")
}

/// cumputes the 1-tree lower bound in parallel
///
/// # Panics
/// if the graph is empty.
#[cfg(feature = "mpi")]
fn one_tree_lower_bound_mpi(graph: &NAMatrix) -> f64 {
    fn one_tree_lower_bound_mpi_with_world(graph: &NAMatrix, world: SystemCommunicator) -> f64 {
        let rank = world.rank() as usize;
        let size = world.size() as usize;
        let root_process = world.process_at_rank(ROOT_RANK);

        // take the start_vertex with the indices i, s.t. `i % size == rank`
        let local_lower_bound = (rank..graph.dim())
            .step_by(size)
            .map(|special_vertex| one_tree(graph, special_vertex).undirected_edge_weight())
            .min_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Tried to compare NaN value. Your data seems currupt.")
            })
            .expect("Cannot compute the 1-tree lower bound of the empty graph");

        let mut global_lower_bound = f64::INFINITY;
        if rank as i32 == ROOT_RANK {
            root_process.reduce_into_root(
                &local_lower_bound,
                &mut global_lower_bound,
                SystemOperation::min(),
            );
        } else {
            root_process.reduce_into(&local_lower_bound, SystemOperation::min());
        }
        global_lower_bound
    }

    // do not drop the Universe until the lower bound function has finished!
    if let Some(universe) = mpi::initialize() {
        let world = universe.world();
        one_tree_lower_bound_mpi_with_world(graph, world)
    } else {
        let world = SystemCommunicator::world();
        one_tree_lower_bound_mpi_with_world(graph, world)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;

    use computation_mode::*;

    use super::*;

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
    /// special vertex: 0
    ///
    /// MST:
    /// 0       1
    ///        /
    ///       /  
    ///      /   
    ///     /    
    ///    /     
    ///   /      
    ///  /      
    /// 3 ----- 2
    ///
    /// 1-tree:
    /// 0 ----- 1
    ///  \     /
    ///   \   /  
    ///    \ /   
    ///     X    
    ///    / \   
    ///   /   \  
    ///  /     \
    /// 3 ----- 2
    #[test]
    fn compute_a_1_tree() {
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
            vec![Edge { to: 2, cost: 0.1 }, Edge { to: 1, cost: 1.0 }],
            //vertex 1
            vec![Edge { to: 3, cost: 0.1 }, Edge { to: 0, cost: 1.0 }],
            //vertex 2
            vec![Edge { to: 3, cost: 0.1 }, Edge { to: 0, cost: 0.1 }],
            //vertex 3
            vec![Edge { to: 1, cost: 0.1 }, Edge { to: 2, cost: 0.1 }],
        ]);
        let special_vertex = 0;
        assert_eq!(expected, one_tree(&(&graph).into(), special_vertex));
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
    /// special vertex: 0
    ///
    /// MST:
    /// 0       1
    ///        /
    ///       /  
    ///      /   
    ///     /    
    ///    /     
    ///   /      
    ///  /      
    /// 3 ----- 2
    ///
    /// 1-tree:
    /// 0 ----- 1
    /// |      /
    /// |     /  
    /// |    /   
    /// |   /    
    /// |  /     
    /// | /      
    /// |/      
    /// 3 ----- 2
    #[test]
    fn compute_1_tree_lower_bound() {
        let graph = Graph::from(vec![
            //vertex 0
            vec![
                Edge { to: 1, cost: 1.0 },
                Edge { to: 2, cost: 0.1 },
                Edge { to: 3, cost: 0.01 },
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
                Edge { to: 0, cost: 0.01 },
                Edge { to: 1, cost: 0.1 },
                Edge { to: 2, cost: 0.1 },
            ],
        ]);
        for i in 0..graph.num_vertices() {
            println!(
                "special vertex: {}, resulting sum: {}",
                i,
                one_tree(&(&graph).into(), i).undirected_edge_weight()
            );
        }

        assert_abs_diff_eq!(
            0.31,
            one_tree_lower_bound::<SEQ_COMPUTATION>(&(&graph).into())
        );
    }

    #[test]
    fn compare_1_tree_lower_bound_seq_vs_par() {
        let graph = Graph::from(vec![
            //vertex 0
            vec![
                Edge { to: 1, cost: 1.0 },
                Edge { to: 2, cost: 0.1 },
                Edge { to: 3, cost: 0.01 },
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
                Edge { to: 0, cost: 0.01 },
                Edge { to: 1, cost: 0.1 },
                Edge { to: 2, cost: 0.1 },
            ],
        ]);
        for i in 0..graph.num_vertices() {
            println!(
                "special vertex: {}, resulting sum: {}",
                i,
                one_tree(&(&graph).into(), i).undirected_edge_weight()
            );
        }

        assert_abs_diff_eq!(
            one_tree_lower_bound::<PAR_COMPUTATION>(&(&graph).into()),
            one_tree_lower_bound::<SEQ_COMPUTATION>(&(&graph).into())
        );
    }
}
