import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

import sys

MATRIX_SIZE = 5

def create_matrix(size):
    """Creates random symmetric fully connected graph with diagonal cost zero."""
    matrix = np.random.rand(size, size) * 20
    matrix = np.triu(matrix, 1) + np.triu(matrix, 1).T
    np.fill_diagonal(matrix, 0)
    return matrix

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

if __name__ == "__main__":
    matrix = create_matrix(MATRIX_SIZE)
    print(matrix)
    permutation, distance = solve_tsp_dynamic_programming(matrix)
    print(f"{permutation=}, {distance=}")

    if sys.argv[1].lower() == "xml":
        print("NOT YET IMPLEMENTED", file=sys.stderr)
        sys.exit(1)
    elif sys.argv[1].lower() == "rust":
        print(convert_to_rust_code(matrix))
    else:
        print("Use argv[1]==rust for rust format!")
