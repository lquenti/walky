import numpy as np
import random

from tsp_logic import convert_to_rust_code

def generate_graph_matrix(n):
    # Generate a random graph matrix with positive integers
    graph_matrix = np.random.randint(1, 10, size=(n, n))
    np.fill_diagonal(graph_matrix, 0) # go to itself is always 0
    graph_matrix = (graph_matrix + graph_matrix.T) // 2  # Make the matrix symmetric
    return graph_matrix


def tsp_nearest_neighbour(graph_matrix, start_node):
    num_nodes = len(graph_matrix)
    path = [start_node]
    unvisited = set(range(num_nodes))
    print("Starting at node", start_node)
    unvisited.remove(start_node)
    current_node = start_node

    while unvisited:
        nearest_neighbour = min(unvisited, key=lambda x: graph_matrix[current_node][x])
        print(f"{path=}\t{unvisited=}\tcosts={graph_matrix[current_node]} {nearest_neighbour=}")
        path.append(nearest_neighbour)
        unvisited.remove(nearest_neighbour)
        current_node = nearest_neighbour

    cost = sum(graph_matrix[path[i]][path[i + 1]] for i in range(num_nodes-1))
    cost += graph_matrix[path[-1]][path[0]]  # Return to the start node

    return path, cost


def main(seed=1337, size=10):
    random.seed(seed)
    np.random.seed(seed)
    graph_matrix = generate_graph_matrix(size)
    print("Graph Matrix:")
    print(graph_matrix)
    for start_node in range(0, size):
        path, cost = tsp_nearest_neighbour(graph_matrix, start_node)
        print("Path:", path)
        print("Cost:", cost, "\n")

    print(convert_to_rust_code(graph_matrix))

if __name__ == "__main__":
    main()
