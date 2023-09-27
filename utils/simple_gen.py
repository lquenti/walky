from python_tsp.exact import solve_tsp_dynamic_programming

from tsp_logic import *

import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use argv[1]==rust for rust format, xml for xml! argv[2] == matrix size")
        sys.exit(1)

    matrix = create_matrix(int(sys.argv[2]))
    permutation, distance = solve_tsp_dynamic_programming(matrix)
    #print(f"{permutation=}, {distance=}")
    if sys.argv[1].lower() == "xml":
        print(convert_to_xml(matrix))
    elif sys.argv[1].lower() == "rust":
        print(convert_to_rust_code(matrix))
    else:
        print("Use argv[1]==rust for rust format, xml for xml!")
        sys.exit(1)
