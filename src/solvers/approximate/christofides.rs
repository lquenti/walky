use blossom::WeightedGraph;

use crate::{mst::prim, parser::Graph};

pub fn christofides(graph: &Graph) {
    // #TODO: validate that the triangle ineq. holds
    // assert!(graph.is_euclidean(), "The given Graph is not euclidean, but Christofides Algorithm
    // requires the triangle inequality");

    let mst = prim(graph);
    let mst = WeightedGraph::new(
        mst.iter()
            .enumerate()
            .map(|(from, vertices)| todo!())
            .collect(),
    );
    todo!()
}
