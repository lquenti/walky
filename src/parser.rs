use std::{ops::Index, slice::SliceIndex};

use delegate::delegate;
use quick_xml::de::from_str;
use serde::{Deserialize, Serialize};

/// Can be parsed from an xml document with
/// [this](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/XML-TSPLIB/Description.pdf)
/// format.
///
/// Here, we impose no further restrictions, such as the graph being undirected.
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
        }
    }

    /// Returns the sum over all (directed) edges in the graph.
    /// For undirected graphs the graph contains 2 directed edged per undirected edge,
    /// hence the sum will be twice as big.
    pub fn directed_edge_weight(&self) -> f64 {
        todo!()
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
        }
    }
}

/// Represents a directed edge from a known node to the node `to`,
/// the edge has cost/weight/distance `cost`.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    #[serde(rename = "$text")]
    pub to: usize,
    #[serde(rename = "@cost")]
    pub cost: f64,
}

#[cfg(test)]
mod test {
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
