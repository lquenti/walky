import os
import re

FOLDER_PATH = './'
MATCH_FOR_STARTING = "Christofides solution weight: "

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

if __name__ == "__main__":
    xs = find_matched_files(FOLDER_PATH)
    
    # do plot logic here
    
    print(xs)


