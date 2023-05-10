//! This module is about computing the 1-tree lower bound for the TSP.
//! See [this article](https://doi.org/10.1287/opre.18.6.1138) for further information.

use serde::Serialize;

use crate::{
    mst::prim_with_excluded_node,
    parser::{Edge, Graph},
};

/// Creates a 1-tree from a given graph.
///
/// The `special_vertex` is the vertex, that is excluded during MST computation.
///
/// # Panics
/// **in debug mode**: if `special_vertex >= graph.num_vertices()`, since then the vertex is not
/// valid.
pub fn one_tree(graph: &Graph, special_vertex: usize) -> Graph {
    debug_assert!(
        special_vertex < graph.num_vertices(),
        "The special vertex has to be a valid vertex in the graph."
    );

    let mut mst = prim_with_excluded_node(graph, special_vertex);
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
    for edge in graph[special_vertex].iter() {
        if edge.cost < fst_min_edg.cost {
            // edge is the best edge
            snd_min_edg = fst_min_edg;
            fst_min_edg = edge.clone();
        } else if edge.cost < snd_min_edg.cost {
            // edge is worse than fst_min_edg, but better than snd_min_edg
            snd_min_edg = edge.clone();
        }
    }

    // add the two edges to the mst, making it a 1-tree
    mst.add_undirected_edge(special_vertex, fst_min_edg.to, fst_min_edg.cost);
    mst.add_undirected_edge(special_vertex, snd_min_edg.to, snd_min_edg.cost);

    mst
}

/// cumputes the 1-tree lower bound
///
/// # Panics
/// if the graph is empty.
pub fn one_tree_lower_bound(graph: &Graph) -> f64 {
    (0..graph.num_vertices())
        .map(|special_vertex| one_tree(graph, special_vertex).undirected_edge_weight())
        .min_by(|x, y| {
            x.partial_cmp(y)
                .expect("Tried to compare NaN value. Your data seems currupt.")
        })
        .expect("Cannot compute the 1-tree lower bound of the empty graph")
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;

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
        assert_eq!(expected, one_tree(&graph, special_vertex));
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
                one_tree(&graph, i).undirected_edge_weight()
            );
        }

        assert_abs_diff_eq!(0.31, one_tree_lower_bound(&graph));
    }
}