use blossom::WeightedGraph;
use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    computation_mode::*,
    datastructures::{AdjacencyMatrix, Edge, Graph, NAMatrix, Solution},
    mst::prim,
    solvers::approximate::matching::approx_min_cost_matching,
};

/// Computes an approximation to the TSP, using Christofides algorithm.
/// This implementation uses for performance reasons a randomized approximation to
/// the min-const matching problem (see [`super::matching::approx_min_cost_matching`].
///
/// For an implementation with exact min-cost matching implementation see [`christofides_exact`].
///
/// For further description, see also [`christofides_generic`].
pub fn christofides<const MODE: usize>(graph: &Graph) -> Solution {
    christofides_generic::<MODE>(graph, compute_approx_matching::<MODE>)
}

/// Computes an approximation to the TSP, using Christofides algorithm.
/// This implementation is much slower than [`christofides`], since it uses
/// an exact solution to the min-cost matching problem,
/// which is also not parallelized.
///
/// For an implementation with approximated min-cost matching implementation see [`christofides_exact`].
///
/// For further description, see also [`christofides_generic`].
pub fn christofides_exact<const MODE: usize>(graph: &Graph) -> Solution {
    christofides_generic::<MODE>(graph, compute_exact_matching)
}

/// See [the original paper from
/// Christofides](https://apps.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf)
/// for a good overview of the algorithm.
///
/// `MODE`: constant parameter, choose one of the values from [`crate::computation_mode`]
///

/// `matching_computer` does the following:
///  1. compute subgraph of `graph` only with vertices that have odd degree in the MST,
///  2. then compute a minimum-weight maximum matching for the subgraph
///
/// The algorithm should be performant, therefore instead of the generic [`crate::datastructures::AdjacencyMatrix`]
/// trait,
/// the type [`NAMatrix`] is used.
///
/// When `MODE == MPI_COMPUTATION`, then the function expects that
/// [mpi::initialize](https://rsmpi.github.io/rsmpi/mpi/fn.initialize.html) has been
/// called before, and that the result of the MPI initialization will not be dropped until the
/// function is finished. See als [this issue](https://github.com/lquenti/walky/issues/30)
pub fn christofides_generic<const MODE: usize>(
    graph: &Graph,
    matching_computer: fn(&Graph, &NAMatrix) -> Vec<[usize; 2]>,
) -> Solution {
    // create an adjacency matrix from the adjacency list
    let graph_matr: NAMatrix = graph.into();

    // validate that the triangle ineq. holds
    debug_assert!(
        graph_matr.is_metric(),
        "The given Graph is not metric, but Christofides Algorithm
    // requires the triangle inequality"
    );

    // 1. find MST
    let mst = prim::<MODE>(&graph_matr);

    // 2. compute subgraph of `graph` only with vertices that have odd degree in the MST,
    // then compute a minimum-weight maximum matching for the subgraph
    let matching = matching_computer(&mst, &graph_matr);

    // 3. union the perfect matching with the MST into a multigraph
    let multigraph = fill_multigraph_with_mst_and_matching::<MODE>(&graph_matr, &mst, matching);

    // 4. compute a eulerian cycle through the multigraph
    let mut euler_cycle = eulerian_cycle_from_multigraph(multigraph);

    // 5. compute a hamiltonian cylce from the eulerian cycle -- the approximate TSP solution
    hamiltonian_from_eulerian_cycle(graph_matr.dim(), &mut euler_cycle);
    let hamilton_cycle = euler_cycle;
    let sum_cost: f64 = hamilton_cycle
        .windows(2)
        .map(|window| graph_matr[(window[0], window[1])])
        .sum();
    (sum_cost, hamilton_cycle)
}

/// This function is being passed to [`christofides_generic`] in the `matching_computer` argument,
/// see there for more info on what this function does.
///
/// This function uses the [`blossom`] crate and it's function [`WeightedGraph::maximin_matching`]
/// to calculate a perfect matching of minimal weight.
#[inline]
fn compute_exact_matching(mst: &Graph, graph: &NAMatrix) -> Vec<[usize; 2]> {
    // 2. compute subgraph of `graph` only with vertices that have odd degree in the MST,
    // then compute a minimum-weight maximum matching for the subgraph
    let subgraph: WeightedGraph = Into::<WeightedGraph>::into(graph)
        .filter_vertices(|&vertex| mst.vertex_degree(vertex) % 2 == 1);

    //note: the maximal matching is perfect
    let matching = subgraph
        .maximin_matching()
        .expect("Something went wrong: could not compute the maximal minimum weight matching");

    matching.edges().into_iter().map(|(a, b)| [a, b]).collect()
}

/// This function is being passed to [`christofides_generic`] in the `matching_computer` argument,
/// see there for more info on what this function does.
///
/// The generated matching will be perfect, but it will not necessairily be the matching with
/// minimal cost. This function uses [`approx_min_cost_matching`] to find a matching with
/// approximately minimal cost.
#[inline]
fn compute_approx_matching<const MODE: usize>(mst: &Graph, graph: &NAMatrix) -> Vec<[usize; 2]> {
    let subgraph: Vec<_> = mst
        .iter()
        .enumerate()
        .filter_map(|(i, vertex)| {
            if vertex.degree() % 2 == 1 {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // make the algorithm at most linear, in terms of edges
    let tries = subgraph.len().pow(2);

    approx_min_cost_matching::<MODE>(graph, subgraph, tries)
}

/// finds a eulerian cycle in the given multigraph,
/// using Hierholzer's algorithm
fn eulerian_cycle_from_multigraph(mut multigraph: DMatrix<(f64, usize)>) -> Vec<usize> {
    // every vertex has at least degree 2
    let dim = multigraph.shape().0;
    let mut euler_cycle = Vec::with_capacity(2 * dim + 1);
    let mut degree: Vec<usize> = multigraph
        .column_iter()
        .map(|col| col.into_iter().map(|&(_, times)| times).sum())
        .collect();

    // trivial cycle: from vertex 0 to itself
    euler_cycle.push(0);

    // find a cycle and incorporate it into the euler_cycle
    while let Some((vertex_idx_in_cycle, &vertex)) = euler_cycle
        .iter()
        .enumerate()
        .find(|(_idx, &vertex)| degree[vertex] != 0)
    {
        // the previous cycle has been closed, without visiting all edged in the graph
        // now find another cycle
        let cycle = find_cycle(vertex, &mut multigraph, &mut degree);
        // split the euler_cycle into 2 parts:
        // [0..vertex_idx_in_cycle] and [vertex_idx_in_cycle+1..]
        let tail = euler_cycle.split_off(vertex_idx_in_cycle + 1);
        euler_cycle.extend_from_slice(&cycle[1..]);
        euler_cycle.extend_from_slice(&tail);
    }
    euler_cycle
}

/// computes a cycle in the given multigraph
///
/// assumption:
/// `degree[i] == multigraph.column(
fn find_cycle(
    start_vertex: usize,
    multigraph: &mut DMatrix<(f64, usize)>,
    degree: &mut [usize],
) -> Vec<usize> {
    let mut cycle = Vec::new();

    let mut vertex = start_vertex;
    cycle.push(vertex);
    while let Some((idx, neighbour)) = multigraph
        .row_mut(vertex)
        .iter_mut()
        .enumerate()
        .find(|(_idx, &mut (_cost, times))| times != 0)
    {
        // find the next vertex in the eulerian cycle
        cycle.push(idx);

        debug_assert!(
            neighbour.1 > 0,
            "There should exist an edge from vertex {} to {}",
            vertex,
            idx
        );
        neighbour.1 -= 1;

        debug_assert!(
            multigraph[(idx, vertex)].1 > 0,
            "There should exist an edge from vertex {} to {}",
            vertex,
            idx
        );
        multigraph[(idx, vertex)].1 -= 1;

        debug_assert!(
            degree[vertex] > 0,
            "The degree of the vertex {} should be more than 0",
            vertex
        );
        degree[vertex] -= 1;

        debug_assert!(
            degree[idx] > 0,
            "The degree of the vertex {} should be more than 0",
            idx
        );
        degree[idx] -= 1;
        vertex = idx;
    }
    cycle
}

/// assumption: the underlying graph is complete and metric (the triangle inequality holds)
///
/// computes from an euclidean cycle the hamiltonian cycle, by skipping
/// already visited vertices
///
/// `dim`: number of vertices in the base graph
/// (at least needed: `dim >= euler_cycle.iter().max().unwrap()`). May panic, if this does not
/// hold.
fn hamiltonian_from_eulerian_cycle(dim: usize, euler_cycle: &mut Vec<usize>) {
    debug_assert!(
        dim >= *euler_cycle
            .iter()
            .max()
            .expect("The eulerian cycle shall not be empty")
    );
    let mut visited = vec![false; dim];
    let mut is_in_ham_cycle: Vec<(usize, bool)> = euler_cycle.iter().map(|&i| (i, false)).collect();

    for pair in is_in_ham_cycle.iter_mut() {
        let i = pair.0;
        let in_cycle = &mut pair.1;
        if !visited[i] {
            visited[i] = true;
            *in_cycle = true;
        }
    }

    let mut is_in_cycle_iter = is_in_ham_cycle.into_iter();
    euler_cycle.retain(|&_| {
        let (_, in_cycle) = is_in_cycle_iter.next().unwrap();
        in_cycle
    });
    euler_cycle.push(
        *euler_cycle
            .first()
            .expect("the hamiltonian cycle should not be empty"),
    );
}

/// This function is taylored to be applied in the christofides algorithm.
/// In general it unions a [`Graph`] (`mst`) with an Edgelist (`matching`) into a multigraph.
///
/// `base_graph`: the underlying graph for the MST, as well as the matching.
///
/// `mst`: MST of `base_graph`
///
/// `matching` a matching of `base_graph`
///
/// # Returns: a multigraph:
///
/// `multigraph[(i, j)] == (cost, num_edges)` with `cost`: the cost of the edge, and `num_edges`:
/// the number of times the edge occours between vertex `i` and vertex `j`.
#[inline]
fn fill_multigraph_with_mst_and_matching<const MODE: usize>(
    base_graph: &NAMatrix,
    mst: &Graph,
    matching: Vec<[usize; 2]>,
) -> DMatrix<(f64, usize)> {
    #[cfg(feature = "mpi")]
    if MODE == MPI_COMPUTATION {
        return fill_multigraph_with_mst_and_matching::<SEQ_COMPUTATION>(base_graph, mst, matching);
    }

    // base the multigraph on the original graph
    let mut multigraph = base_graph.map(|cost| (cost, 0));

    // populate the matix with the edges of the mst

    match MODE {
        SEQ_COMPUTATION => {
            mst.iter()
                .zip(multigraph.row_iter_mut())
                .for_each(|(vertex, mut neighbours_vec)| {
                    vertex
                        .iter()
                        .for_each(|&Edge { to, cost }| neighbours_vec[(0, to)] = (cost, 1))
                })
        }
        PAR_COMPUTATION => {
            // sadly, NAlgebra doesn`t provide a par_row_iter_mut() method,
            // so we use the fact that all edges are undirected, and therefore
            // the adjacency matrix is symmetric, i.e. we can exchange rows with columns
            mst.par_iter()
                .zip(multigraph.par_column_iter_mut())
                .for_each(|(vertex, mut neighbours_vec)| {
                    vertex
                        .iter()
                        .for_each(|&Edge { to, cost }| neighbours_vec[(to, 0)] = (cost, 1))
                });
        }
        #[cfg(feature = "mpi")]
        MPI_COMPUTATION => unreachable!("On MPI the SEQ_COMPUTATION variant is used"),
        _ => panic_on_invaid_mode::<MODE>(),
    }

    // add into the multigraph the edges from the matching
    match MODE {
        SEQ_COMPUTATION => matching.into_iter().for_each(|[i, j]| {
            multigraph[(i, j)].1 += 1;
            multigraph[(j, i)].1 += 1;
        }),
        PAR_COMPUTATION => {
            let dim = multigraph.shape().0;
            // cannot directly create a *mut pointer,
            // since *mut pointers do not implement std::marker::Sync
            let multigraph_slice = multigraph.as_slice();
            // safety: each cell is only written to once,
            // since the matching contains each vertex at most once
            unsafe {
                matching.into_par_iter().for_each(|[i, j]| {
                    let multigraph_prt = multigraph_slice.as_ptr() as *mut (f64, usize);
                    //multigraph[edge].1 += 1;
                    (*multigraph_prt.add(j * dim + i)).1 += 1;
                    //multigraph[(j, i)].1 += 1;
                    (*multigraph_prt.add(i * dim + j)).1 += 1;
                })
            }
        }
        #[cfg(feature = "mpi")]
        MPI_COMPUTATION => unreachable!("On MPI the SEQ_COMPUTATION variant is used"),
        _ => panic_on_invaid_mode::<MODE>(),
    }
    multigraph
}

#[cfg(test)]
mod test {
    use nalgebra::DMatrix;

    use crate::{
        computation_mode::*,
        datastructures::{Edge, Graph, NAMatrix},
        mst::prim,
        solvers::{
            self,
            approximate::christofides::{
                christofides, eulerian_cycle_from_multigraph,
                fill_multigraph_with_mst_and_matching, find_cycle, hamiltonian_from_eulerian_cycle,
            },
        },
    };

    /// graph: (with edge cost annotated)
    /// 0 ----1.--- 1
    ///  \         /
    ///   \       /
    ///    2.    3.
    ///     \   /
    ///      \ /
    ///       2
    ///
    /// resulting mst: (without edge cost annotated)
    /// 0 - 1
    ///  \
    ///   2
    #[test]
    fn multigraph_can_represent_the_mst_with_empty_matching() {
        let matching = vec![];
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 1., 2., 1., 0., 3., 2., 3., 0.],
        ));

        let mst = prim::<SEQ_COMPUTATION>(&graph);

        let expected = DMatrix::from_row_slice(
            3,
            3,
            &[
                (0., 0),
                (1., 1),
                (2., 1),
                (1., 1),
                (0., 0),
                (3., 0),
                (2., 1),
                (3., 0),
                (0., 0),
            ],
        );
        assert_eq!(
            expected,
            fill_multigraph_with_mst_and_matching::<SEQ_COMPUTATION>(&graph, &mst, matching)
        );
    }

    /// graph: (with edge cost annotated)
    /// 0 ----1.--- 1
    ///  \         /
    ///   \       /
    ///    2.    3.
    ///     \   /
    ///      \ /
    ///       2
    ///
    /// matching (not a proper matching, but for the test that is okay):
    /// 0 - 1
    ///  \
    ///   2
    #[test]
    fn multigraph_can_represent_empty_mst_with_matching() {
        let matching = vec![[1, 0], [2, 0]];
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 1., 2., 1., 0., 3., 2., 3., 0.],
        ));

        let mst = Graph::from(vec![]);

        let expected = DMatrix::from_row_slice(
            3,
            3,
            &[
                (0., 0),
                (1., 1),
                (2., 1),
                (1., 1),
                (0., 0),
                (3., 0),
                (2., 1),
                (3., 0),
                (0., 0),
            ],
        );
        assert_eq!(
            expected,
            fill_multigraph_with_mst_and_matching::<SEQ_COMPUTATION>(&graph, &mst, matching)
        );
    }

    /// graph: (with edge cost annotated)
    /// 0 ----1.--- 1
    ///  \         /
    ///   \       /
    ///    2.    3.
    ///     \   /
    ///      \ /
    ///       2
    ///
    /// resulting mst: (without edge cost annotated)
    /// 0 - 1
    ///  \
    ///   2
    /// matching (not a proper matching, but for the test that is okay):
    /// 0 - 1
    ///  \
    ///   2
    ///
    /// resulting multigraph:
    /// 0 - 1
    ///  \
    ///   2
    /// with every edge existing twice
    #[test]
    fn multigraph_can_represent_the_mst_with_matching() {
        let matching = vec![[1, 0], [2, 0]];
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 1., 2., 1., 0., 3., 2., 3., 0.],
        ));

        let mst = prim::<SEQ_COMPUTATION>(&graph);

        let expected = DMatrix::from_row_slice(
            3,
            3,
            &[
                (0., 0),
                (1., 2),
                (2., 2),
                (1., 2),
                (0., 0),
                (3., 0),
                (2., 2),
                (3., 0),
                (0., 0),
            ],
        );
        assert_eq!(
            expected,
            fill_multigraph_with_mst_and_matching::<SEQ_COMPUTATION>(&graph, &mst, matching)
        );
    }

    /// graph: (with edge cost annotated)
    /// 0 ----1.--- 1
    ///  \         /
    ///   \       /
    ///    2.    3.
    ///     \   /
    ///      \ /
    ///       2
    ///
    /// resulting mst: (without edge cost annotated)
    /// 0 - 1
    ///  \
    ///   2
    /// matching (not a proper matching, but for the test that is okay):
    /// 0 - 1
    ///  \
    ///   2
    ///
    /// resulting multigraph:
    /// 0 - 1
    ///  \
    ///   2
    /// with every edge existing twice
    #[test]
    fn fill_multigraph_with_mst_and_matching_seq_and_par_agree() {
        let matching = vec![[1, 0], [2, 0]];
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 1., 2., 1., 0., 3., 2., 3., 0.],
        ));

        let mst = prim::<SEQ_COMPUTATION>(&graph);

        let expected = DMatrix::from_row_slice(
            3,
            3,
            &[
                (0., 0),
                (1., 2),
                (2., 2),
                (1., 2),
                (0., 0),
                (3., 0),
                (2., 2),
                (3., 0),
                (0., 0),
            ],
        );
        assert_eq!(
            fill_multigraph_with_mst_and_matching::<SEQ_COMPUTATION>(
                &graph,
                &mst,
                matching.clone()
            ),
            fill_multigraph_with_mst_and_matching::<PAR_COMPUTATION>(
                &graph,
                &mst,
                matching.clone()
            )
        );
        assert_eq!(
            expected,
            fill_multigraph_with_mst_and_matching::<SEQ_COMPUTATION>(&graph, &mst, matching,),
        );
    }

    /// graph with one cycle:
    /// 0 - 1
    /// |   |
    /// 3 - 2
    #[test]
    fn test_find_cycle() {
        let mut multigraph = DMatrix::from_row_slice(
            4,
            4,
            &[
                (0., 0),
                (1., 1),
                (0., 0),
                (1., 1),
                (1., 1),
                (0., 0),
                (1., 1),
                (0., 0),
                (0., 0),
                (1., 1),
                (0., 0),
                (1., 1),
                (1., 1),
                (0., 0),
                (1., 1),
                (0., 0),
            ],
        );
        let mut degree = vec![2; 4];
        let expected = vec![0, 1, 2, 3, 0];
        assert_eq!(expected, find_cycle(0, &mut multigraph, &mut degree))
    }

    /// multigraph with eulerian cycle:
    ///   ___
    ///  /   \
    /// 0-----1
    /// |\   /|
    /// | \ / |
    /// |  X  |
    /// | / \ |
    /// |/   \|
    /// 3-----2
    ///  \___/
    ///
    ///  a eulerian cycle: 0-1-2-3-0-1-3-2-0
    ///  (is not the only one)
    #[test]
    fn test_find_eulerian_cycle() {
        let multigraph = DMatrix::from_row_slice(
            4,
            4,
            &[
                (0., 0),
                (1., 2),
                (1., 1),
                (1., 1),
                (1., 2),
                (0., 0),
                (1., 1),
                (1., 1),
                (1., 1),
                (1., 1),
                (0., 0),
                (1., 2),
                (1., 1),
                (1., 1),
                (1., 2),
                (0., 0),
            ],
        );
        // the graph contains 9 edges
        let expected = 9;
        assert_eq!(expected, eulerian_cycle_from_multigraph(multigraph).len());
    }

    /// multigraph with eulerian cycle:
    ///   ___
    ///  /   \
    /// 0-----1
    /// |\   /|
    /// | \ / |
    /// |  X  |
    /// | / \ |
    /// |/   \|
    /// 3-----2
    ///  \___/
    ///
    ///  eulerian cycle: 0-1-2-3-0-1-3-2-0
    ///
    ///  hamiltonian cycle: 0-1-2-3-0
    #[test]
    fn test_hamiltonian_from_eulerian_cycle() {
        let mut euler_cycle = vec![0, 1, 2, 3, 0, 1, 3, 2, 0];
        let expected = vec![0, 1, 2, 3, 0];

        hamiltonian_from_eulerian_cycle(4, &mut euler_cycle);
        assert_eq!(expected, euler_cycle);
    }

    /// metric graph:
    /// 0-----1
    /// |\   /|
    /// | \ / |
    /// |  X  |
    /// | / \ |
    /// |/   \|
    /// 3-----2
    ///
    /// edge weights: vertical and horizontal edges: 1.,
    ///               diagonal edges: 2.
    ///
    /// let `s` be the accumulated cost of an exact solution of the TSP,
    /// then Christofides algorithm finds a solution with cost of at most `1.5 * s`
    #[test]
    fn test_christofides_against_exact_solver() {
        let graph: Graph = vec![
            vec![
                Edge { to: 0, cost: 0. },
                Edge { to: 1, cost: 1. },
                Edge { to: 2, cost: 2. },
                Edge { to: 3, cost: 1. },
            ],
            vec![
                Edge { to: 0, cost: 1. },
                Edge { to: 1, cost: 0. },
                Edge { to: 2, cost: 1. },
                Edge { to: 3, cost: 2. },
            ],
            vec![
                Edge { to: 0, cost: 2. },
                Edge { to: 1, cost: 1. },
                Edge { to: 2, cost: 0. },
                Edge { to: 3, cost: 1. },
            ],
            vec![
                Edge { to: 0, cost: 1. },
                Edge { to: 1, cost: 2. },
                Edge { to: 2, cost: 1. },
                Edge { to: 3, cost: 0. },
            ],
        ]
        .into();

        let exact_solution = solvers::exact::first_improved_solver::<NAMatrix>(&(&graph).into());
        let result = christofides::<SEQ_COMPUTATION>(&graph);
        assert!(
            exact_solution.0 <= result.0,
            "Christofides algorithm cannot outperfrom the exact solution"
        );
        assert!(
            1.5 * exact_solution.0 >= result.0,
            "Christofides algorithm is at most 50% worse than the exact solution"
        );
    }
}
