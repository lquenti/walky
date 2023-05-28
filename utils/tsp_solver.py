import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

MATRIX_SIZE = 5

def create_matrix(size):
    """Creates random symmetric fully connected graph with diagonal cost zero."""
    matrix = np.random.rand(MATRIX_SIZE, MATRIX_SIZE) * 20
    matrix = np.triu(matrix, 1) + np.triu(matrix, 1).T
    np.fill_diagonal(matrix, 0)
    return matrix

if __name__ == "__main__":
    matrix = create_matrix(MATRIX_SIZE)
    print(matrix)
    permutation, distance = solve_tsp_dynamic_programming(matrix)
    print(f"{permutation=}, {distance=}")
