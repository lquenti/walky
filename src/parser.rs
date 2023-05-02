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
    name: String,
    source: String,
    description: String,
    double_precision: u32,
    ignored_digits: u32,
    graph: Graph,
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

/// This representes a vertex and contains the collection of edges from this vertex
/// to all adjacent vertices.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Vertex {
    #[serde(rename = "$value")]
    edges: Vec<Edge>,
}

/// Represents a directed edge from a known node to the node `to`,
/// the edge has cost/weight/distance `cost`.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    #[serde(rename = "$text")]
    to: usize,
    #[serde(rename = "@cost")]
    cost: f64,
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
