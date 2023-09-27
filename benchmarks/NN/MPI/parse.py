import os
import re
import statistics

import matplotlib.pyplot as plt

FOLDER_PATH = './'
MATCH_FOR_STARTING = "elapsed seconds: "

def extract_nodes_and_processes(file_path):
    pattern = r'output_(\d+)_nodes_(\d+)_taskspernode\.txt'
    match = re.search(pattern, file_path)
    if match:
        nodes = int(match.group(1))
        processes = int(match.group(2))
        return (nodes, processes)
    else:
        return None

def get_measurements(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if line.startswith(MATCH_FOR_STARTING):
                measurement = float(line.split(":")[1].strip())
                measurements.append(measurement)
    return measurements

def find_matched_files(folder_path):
    files = os.listdir(folder_path)

    matched_files = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        nodes_processes = extract_nodes_and_processes(file)
        if nodes_processes:
            measurements = get_measurements(file_path)
            matched_files.append((nodes_processes, measurements))

    return matched_files

xs = find_matched_files(FOLDER_PATH)

# do plot logic here
single_process_per_node = [(x[0][0], statistics.median(x[1])) for x in xs if x[0][1] == 1]
single_process_per_node.sort(key=lambda x: x[0])

multi_processes_per_node = [(x[0][0] * x[0][1], statistics.median(x[1])) for x in xs if
               not (x[0][0] > 1 and x[0][1] == 1)]
multi_processes_per_node.sort(key=lambda x: x[0])

print(single_process_per_node)
print(multi_processes_per_node)

plt.plot(*map(list, zip(*single_process_per_node)), label = "single node", c="blue")
plt.plot(*map(list, zip(*multi_processes_per_node)), label = "multi nodes", c="red")
plt.legend()
plt.xlabel("number of processes")
plt.ylabel("t")
plt.title("Nearest Neighbour MPI (n=3000)")

#plt.show()
plt.savefig("nn-mpi.pdf")
