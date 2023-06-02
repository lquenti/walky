use blossom::WeightedGraph;

use crate::{mst::prim, parser::Graph};

/// See [the original paper from
/// Christofides](https://apps.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf)
/// for a good overview of the algorithm
pub fn christofides(graph: &Graph) {
    // #TODO: validate that the triangle ineq. holds
    // assert!(graph.is_euclidean(), "The given Graph is not euclidean, but Christofides Algorithm
    // requires the triangle inequality");

    // 1. find MST
    let mst = prim(graph);

    // 2. compute subgraph of `graph` only with vertices that have odd degree in the MST
    let subgraph: WeightedGraph = Into::<WeightedGraph>::into(graph)
        .filter_vertices(|&vertex| mst.vertex_degree(vertex) % 2 == 1);

    // 3. compute a minimum-weight maximum matching for the graph
    // note: the maximal matching is perfect
    let matching = subgraph
        .maximin_matching()
        .expect("Something went wrong: could not compute the maximal minimum weight matching");

    // 4. union the perfect matching with the MST

    // 5. ??

    // 6. compute a hamiltonian cylce -- the approximate TSP solution
    todo!()
}
