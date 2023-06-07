use blossom::WeightedGraph;

use crate::{
    datastructures::{Graph, NAMatrix},
    mst::prim,
};

/// See [the original paper from
/// Christofides](https://apps.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf)
/// for a good overview of the algorithm
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

    // 4. union the perfect matching with the MST
    let matching_edges = matching.edges();
    //let graphMat: NAMatrix = graph.into();

    // 5. ??

    // 6. compute a hamiltonian cylce -- the approximate TSP solution
    todo!()
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
}
