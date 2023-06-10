use blossom::WeightedGraph;
use nalgebra::DMatrix;

use crate::{
    datastructures::{AdjacencyMatrix, Edge, Graph, NAMatrix, Solution},
    mst::prim,
};

/// See [the original paper from
/// Christofides](https://apps.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf)
/// for a good overview of the algorithm
///
/// should be performant, therefore instead of the generic [`datastructures::AdjacencyMatrix`]
/// trait,
/// the type [`datastructures::NAMatrix`] is used.
pub fn christofides(graph: &Graph) -> Solution {
    // create an adjacency matrix from the adjacency list
    let graph_matr: NAMatrix = graph.into();

    // validate that the triangle ineq. holds
    debug_assert!(
        graph_matr.is_euclidean(),
        "The given Graph is not euclidean, but Christofides Algorithm
    // requires the triangle inequality"
    );

    // 1. find MST
    let mst = prim(&graph_matr);

    // 2. compute subgraph of `graph` only with vertices that have odd degree in the MST
    let subgraph: WeightedGraph = Into::<WeightedGraph>::into(graph)
        .filter_vertices(|&vertex| mst.vertex_degree(vertex) % 2 == 1);

    // 3. compute a minimum-weight maximum matching for the graph
    // note: the maximal matching is perfect
    let matching = subgraph
        .maximin_matching()
        .expect("Something went wrong: could not compute the maximal minimum weight matching");

    // 4. union the perfect matching with the MST into a multigraph
    let matching_edges = matching.edges();
    let multigraph = fill_multigraph_with_mst_and_matching(&graph_matr, &mst, matching_edges);

    //let graphMat: NAMatrix = graph.into();

    // 5. compute a eulerian cycle through the multigraph
    let mut euler_cycle = eulerian_cycle_from_multigraph(multigraph);

    // 6. compute a hamiltonian cylce from the eulerian cycle -- the approximate TSP solution
    hamiltonian_from_eulerian_cycle(graph_matr.dim(), &mut euler_cycle);
    let hamilton_cycle = euler_cycle;
    let sum_cost: f64 = hamilton_cycle
        .windows(2)
        .map(|window| graph_matr[(window[0], window[1])])
        .sum();
    (sum_cost, hamilton_cycle)
}

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
        .column_mut(vertex)
        .iter_mut()
        .enumerate()
        .find(|(_idx, &mut (_cost, times))| times != 0)
    {
        // find the next vertex in the eulerian cycle
        cycle.push(idx);

        debug_assert!(
            neighbour.1 > 0,
            "There should exist an edge from vertex {} to {} ",
            vertex,
            idx
        );
        neighbour.1 -= 1;

        debug_assert!(
            multigraph[(vertex, idx)].1 > 0,
            "There should exist an edge from vertex {} to {} ",
            vertex,
            idx
        );
        multigraph[(vertex, idx)].1 -= 1;

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

/// assumption: the underlying graph is complete and euclidean
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
fn fill_multigraph_with_mst_and_matching(
    base_graph: &NAMatrix,
    mst: &Graph,
    matching: Vec<(usize, usize)>,
) -> DMatrix<(f64, usize)> {
    // base the multigraph on the original graph
    let mut multigraph = base_graph.map(|cost| (cost, 0));

    // populate the matix with the edges of the mst
    for (i, vertex) in mst.iter().enumerate() {
        for &Edge { to: j, cost } in vertex.iter() {
            multigraph[(i, j)] = (cost, 1);
        }
    }
    // add into the multigraph the edges from the matching
    for edge @ (i, j) in matching.into_iter() {
        //let cost = graph_matr[edge];
        //multigraph[edge].0 = cost;
        multigraph[edge].1 += 1;
        multigraph[(j, i)].1 += 1;
    }
    multigraph
}

/// an undirected Edge with no cost
#[derive(Debug, Copy, Clone)]
struct UndirectedEdge {
    a: usize,
    b: usize,
}

impl PartialEq for UndirectedEdge {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b || self.a == other.b && self.b == other.a
    }
}

#[cfg(test)]
mod test {
    use nalgebra::DMatrix;

    use crate::{
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

    use super::UndirectedEdge;

    #[test]
    fn test_same_edge_eq() {
        let e1 = UndirectedEdge { a: 1, b: 0 };
        let e2 = UndirectedEdge { a: 1, b: 0 };
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_reversed_edge_eq() {
        let e1 = UndirectedEdge { a: 1, b: 0 };
        let e2 = UndirectedEdge { a: 0, b: 1 };
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_not_equal_edges() {
        let e1 = UndirectedEdge { a: 1, b: 0 };
        let e2 = UndirectedEdge { a: 1, b: 1 };
        assert_ne!(e1, e2);
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
    #[test]
    fn multigraph_can_represent_the_mst_with_empty_matching() {
        let matching = vec![];
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 1., 2., 1., 0., 3., 2., 3., 0.],
        ));

        let mst = prim(&graph);

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
            fill_multigraph_with_mst_and_matching(&graph, &mst, matching)
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
        let matching = vec![(1, 0), (2, 0)];
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
            fill_multigraph_with_mst_and_matching(&graph, &mst, matching)
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
        let matching = vec![(1, 0), (2, 0)];
        let graph = NAMatrix(DMatrix::from_row_slice(
            3,
            3,
            &[0., 1., 2., 1., 0., 3., 2., 3., 0.],
        ));

        let mst = prim(&graph);

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
            fill_multigraph_with_mst_and_matching(&graph, &mst, matching)
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

    /// euclidean graph:
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
        let result = christofides(&graph);
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
