use blossom::WeightedGraph;
use nalgebra::DMatrix;

use crate::{
    datastructures::{Edge, Graph, NAMatrix},
    mst::prim,
};

/// See [the original paper from
/// Christofides](https://apps.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf)
/// for a good overview of the algorithm
///
/// should be performant, therefore instead of the generic [`datastructures::AdjacencyMatrix`]
/// trait,
/// the type [`datastructures::NAMatrix`] is used.
pub fn christofides(graph: &Graph) {
    // #TODO: validate that the triangle ineq. holds
    // assert!(graph.is_euclidean(), "The given Graph is not euclidean, but Christofides Algorithm
    // requires the triangle inequality");

    // 1. find MST
    let graph_matr: NAMatrix = graph.into();
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

    // 6. compute a hamiltonian cylce from the eulerian cycle -- the approximate TSP solution
    todo!()
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
        datastructures::{Graph, NAMatrix},
        mst::prim,
        solvers::approximate::christofides::fill_multigraph_with_mst_and_matching,
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
}
