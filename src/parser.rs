use std::{ops::Index, slice::SliceIndex};

use delegate::delegate;
use quick_xml::de::from_str;
use serde::{Deserialize, Serialize};

/// Can be parsed from an xml document with the
/// [XML-TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/XML-TSPLIB/Description.pdf)
/// format.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TravellingSalesmanProblemInstance {
    pub name: String,
    pub source: String,
    pub description: String,
    pub double_precision: u32,
    pub ignored_digits: u32,
    pub graph: Graph,
}

impl TravellingSalesmanProblemInstance {
    /// Parse a TSP instance from an xml `str`.
    ///
    /// This parsing does not check whether the graph is a valid TSP instance
    /// as long as it is a valid xml document.
    pub fn parse_from_xml(xml: &str) -> Result<Self, quick_xml::DeError> {
        from_str(xml)
    }

    /// #TODO: specify the conditions
    ///
    /// Check if the defined conditions on the graph apply.
    pub fn do_conditions_hold(&self) -> bool {
        todo!()
    }
}

/// This represents a graph, with the collection of all its edges and vertices.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Graph {
    #[serde(rename = "$value")]
    vertices: Vec<Vertex>,
}

impl Graph {
    delegate! {
        to self.vertices {
            /// Yields an Iterator over all vertices in this graph.
            /// The vertices are traversend in increasing order, starting from index `0`.
            pub fn iter(&self) -> std::slice::Iter<Vertex>;

            /// return the number of vertices in the graph
            #[call(len)]
            pub fn num_vertices(&self) -> usize;
        }
    }

    /// Returns the sum over all (directed) edges in the graph.
    /// For undirected graphs the graph contains 2 directed edged per undirected edge,
    /// hence the sum will be twice as big.
    pub fn directed_edge_weight(&self) -> f64 {
        self.iter()
            .map(|vertex| vertex.iter().map(|edge| edge.cost).sum::<f64>())
            .sum()
    }

    /// # Precondition
    /// The graph shall be undirected, meaning: for vertices `u` and `v`
    /// the graph contains two directed edges: `u -> v` and `u <- v`, both with the same cost.
    ///
    /// # Returns
    /// Return the sum of cost over all **undirected** edges in the graph.
    pub fn undirected_edge_weight(&self) -> f64 {
        self.directed_edge_weight() * 0.5
    }

    /// adds an undirected edge to the graph.
    /// There is no check, whether the edge might already exist in the graph.
    pub fn add_undirected_edge(&mut self, from: usize, to: usize, cost: f64) {
        debug_assert!(
            from < self.num_vertices(),
            "The vertex 'from' has to be a valid vertex in the graph"
        );
        debug_assert!(
            to < self.num_vertices(),
            "The vertex 'to' has to be a valid vertex in the graph"
        );

        let edge = Edge { to, cost };
        let edge_reverse = Edge { to: from, cost };
        self.vertices[from].add_edge(edge);
        self.vertices[to].add_edge(edge_reverse);
    }
}

impl<I> Index<I> for Graph
where
    I: SliceIndex<[Vertex]>,
{
    type Output = <Vec<Vertex> as Index<I>>::Output;
    delegate! {
        to self.vertices {
            fn index(&self, index: I) -> &<Self as Index<I>>::Output;
        }
    }
}

impl From<Vec<Vec<Edge>>> for Graph {
    fn from(value: Vec<Vec<Edge>>) -> Self {
        let vec = value.into_iter().map(Vertex::from).collect();
        Graph { vertices: vec }
    }
}

/// This representes a vertex and contains the collection of edges from this vertex
/// to all adjacent vertices.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Vertex {
    #[serde(rename = "$value")]
    edges: Vec<Edge>,
}

impl Vertex {
    delegate! {
        to self.edges {
            /// Yields an Iterator over all edges from this vertex to the adjacent edges.
            pub fn iter(&self) -> std::slice::Iter<Edge>;

            /// Adds an edge to the vertex
            #[call(push)]
            pub fn add_edge(&mut self, edge: Edge);
        }
    }
}

impl From<Vec<Edge>> for Vertex {
    fn from(value: Vec<Edge>) -> Self {
        Vertex { edges: value }
    }
}

/// Represents a directed edge from a known node to the node `to`,
/// the edge has cost/weight/distance `cost`.
#[derive(Debug, Serialize, Deserialize, PartialEq, Copy, Clone)]
pub struct Edge {
    #[serde(rename = "$text")]
    pub to: usize,
    #[serde(rename = "@cost")]
    pub cost: f64,
}

#[cfg(test)]
mod test_graph_methods {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn build_vertex_from_slice() {
        let edges = vec![Edge { to: 0, cost: 1.0 }, Edge { to: 30, cost: 0.34 }];
        let expected = Vertex {
            edges: vec![Edge { to: 0, cost: 1.0 }, Edge { to: 30, cost: 0.34 }],
        };
        assert_eq!(expected, Vertex::from(edges));
    }

    #[test]
    fn build_graph_from_slice() {
        let vertices = vec![
            vec![Edge { to: 0, cost: 1.0 }, Edge { to: 30, cost: 0.34 }],
            vec![Edge { to: 10, cost: 5.34 }],
        ];
        let expected = Graph {
            vertices: vec![
                Vertex::from(vertices[0].clone()),
                Vertex::from(vertices[1].clone()),
            ],
        };
        assert_eq!(expected, Graph::from(vertices));
    }

    #[test]
    fn directed_edge_weight() {
        let graph = Graph::from(vec![
            vec![Edge { to: 0, cost: 1.0 }, Edge { to: 30, cost: 0.34 }],
            vec![Edge { to: 10, cost: 5.34 }],
        ]);
        let expected = 1.0 + 0.34 + 5.34;
        assert_abs_diff_eq!(expected, graph.directed_edge_weight());
    }

    /// maybe here woul be a good place to do property based testing with
    /// e.g. qucikcheck
    #[test]
    fn undirected_edge_weight() {
        let graph = Graph::from(vec![
            vec![Edge { to: 2, cost: 1.0 }, Edge { to: 1, cost: 0.34 }],
            vec![Edge { to: 0, cost: 0.34 }],
            vec![Edge { to: 0, cost: 1.0 }],
        ]);
        assert_abs_diff_eq!(
            graph.directed_edge_weight() / 2.0,
            graph.undirected_edge_weight()
        );
    }
}

#[cfg(test)]
mod test_parsing {
    use quick_xml::de::from_str;

    use super::*;

    #[test]
    fn parse_edge() {
        let edge_str = r#"<edge cost="2.000e+01">1</edge>"#;
        let expected = Edge { to: 1, cost: 2e1 };
        assert_eq!(expected, from_str(edge_str).unwrap());
    }

    #[test]
    fn parse_vertex() {
        let vertex_str =
            r#"<vertex><edge cost="3.000e+01">1</edge><edge cost="2.000e+01">0</edge></vertex>"#;
        let expected = Vertex {
            edges: vec![Edge { to: 1, cost: 3e1 }, Edge { to: 0, cost: 2e1 }],
        };
        assert_eq!(expected, from_str(vertex_str).unwrap());
    }

    #[test]
    fn parse_graph() {
        let vertex_str = r#"
            <graph>
            <vertex><edge cost="3.000e+01">1</edge><edge cost="2.000e+01">0</edge></vertex>
            <vertex><edge cost="3.000e+01">0</edge><edge cost="2.000e+01">2</edge></vertex>
            </graph>"#;
        let expected = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![Edge { to: 1, cost: 3e1 }, Edge { to: 0, cost: 2e1 }],
                },
                Vertex {
                    edges: vec![Edge { to: 0, cost: 3e1 }, Edge { to: 2, cost: 2e1 }],
                },
            ],
        };
        assert_eq!(expected, from_str(vertex_str).unwrap());
    }

    #[test]
    fn parse_small_tsp_example() {
        let xml = r#"
<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<travellingSalesmanProblemInstance>
  <name>test</name>
  <source>Johann</source>
  <description>small example</description>
  <doublePrecision>15</doublePrecision>
  <ignoredDigits>0</ignoredDigits>
  <graph>
    <vertex>
        <edge cost="1.123456789012345e+00">1</edge>
    </vertex>
    <vertex>
        <edge cost="1.000000000000000e+01">0</edge>
    </vertex>
  </graph>
</travellingSalesmanProblemInstance>
"#;
        let expected = TravellingSalesmanProblemInstance {
            name: String::from("test"),
            source: String::from("Johann"),
            description: String::from("small example"),
            double_precision: 15,
            ignored_digits: 0,
            graph: Graph {
                vertices: vec![
                    Vertex {
                        edges: vec![Edge {
                            to: 1,
                            cost: 1.123456789012345,
                        }],
                    },
                    Vertex {
                        edges: vec![Edge { to: 0, cost: 1e1 }],
                    },
                ],
            },
        };
        assert_eq!(expected, from_str(xml).unwrap());
    }
}
