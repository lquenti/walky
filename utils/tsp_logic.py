import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def create_matrix(size):
    """Creates metric fully connected graph with diagonal cost zero."""
    points = np.random.rand(size, 2)
    n = len(points)
    graph_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_distance(points[i], points[j])
            graph_matrix[i, j] = distance
            graph_matrix[j, i] = distance

    return graph_matrix


def convert_to_rust_code(cost_matrix):
    """Convert a numpy cost matrix to rust code"""
    n = len(cost_matrix)
    vertices = []
    for i in range(n):
        edges = []
        for j in range(n):
            if i != j and cost_matrix[i][j] != 0.0:
                edges.append(f"Edge {{ to: {j}, cost: {cost_matrix[i][j]} }}")
        vertices.append(f"Vertex {{ edges: vec![{', '.join(edges)}] }}")
    rust_code = f"let graph = Graph {{vertices: vec![{','.join(vertices)}]}};"
    return rust_code

def convert_to_xml(cost_matrix):
    root = ET.Element("travellingSalesmanProblemInstance")
    name_elem = ET.SubElement(root, "name")
    name_elem.text = "RandomGraph"
    source_elem = ET.SubElement(root, "source")
    source_elem.text = "gh:lquenti/walky"
    description_elem = ET.SubElement(root, "description")
    description_elem.text = "random example"
    double_precision_elem = ET.SubElement(root, "doublePrecision")
    double_precision_elem.text = str(15)
    ignored_digits_elem = ET.SubElement(root, "ignoredDigits")
    ignored_digits_elem.text = str(0)
    graph_elem = ET.SubElement(root, "graph")

    n = cost_matrix.shape[0]
    for i in range(n):
        vertex_elem = ET.SubElement(graph_elem, "vertex")
        for j in range(n):
            if i != j:
                edge_elem = ET.SubElement(vertex_elem, "edge")
                edge_elem.set("cost", "{:.15e}".format(cost_matrix[i, j]))
                edge_elem.text = str(j)
    return ET.tostring(root, encoding='utf-8', method="xml").decode("utf-8")
