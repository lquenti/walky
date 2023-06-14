import os
import sys
import shutil

from tsp_logic import *

def usage_and_die():
    print("USAGE: python3 benchmark_gen.py <min_benchmark_size> <max_benchmark_size>", file=sys.stderr)
    print("benchmark size i == complete graph with i nodes", file=sys.stderr)
    sys.exit(1)

def principal_minor(matrix, k):
    minor = np.zeros((k, k), dtype=matrix.dtype)

    for i in range(k):
        for j in range(k):
            minor[i][j] = matrix[i][j]

    return minor


if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage_and_die()
    l,r = sys.argv[1:]
    l,r = int(l), int(r)

    # Step 1: Generate matrix of maximum size
    matrix = create_matrix(r)

    # Step 2: get principal minors of size [l:r-1]
    res = [(k, principal_minor(matrix, k)) for k in range(l, (r-1)+1)]
    res.append((r,matrix))

    # Step 3: Save the results
    folder_name = "results"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    for (k, m) in res:
        with open(f"{folder_name}{os.sep}{k}.xml", 'w') as fp:
            fp.write(convert_to_xml(m))
