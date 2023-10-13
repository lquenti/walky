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

for x in xs:
    if len(x[1]) == 0:
        print(x)

# do plot logic here
single_process_per_node = [(x[0][0], statistics.median(x[1])) for x in xs if x[0][1] == 1]
single_process_per_node.sort(key=lambda x: x[0])

multi_processes_per_node = [(x[0][0] * x[0][1], statistics.median(x[1])) for x in xs if
               not (x[0][0] > 1 and x[0][1] == 1)]
multi_processes_per_node.sort(key=lambda x: x[0])

print(single_process_per_node)
print(multi_processes_per_node)

x_data, y_data = map(list, zip(*single_process_per_node))
#plt.plot(*map(list, zip(*single_process_per_node)), label = "single node", c="blue")
#plt.plot(*map(list, zip(*multi_processes_per_node)), label = "multi nodes", c="red")
import numpy as np
import scipy as sp
# ~~ use this when plotting on logarithmic axes
x_log = np.log10(x_data)
y_log = np.log10(y_data)
xref=np.logspace(x_log.min(),x_log.max(),num=50)
reg = sp.stats.linregress(x_log, y_log)
p = reg.slope
a = np.power(10, reg.intercept)

plt.plot(xref, a*np.power(xref, p), "b--", alpha=.5,\
        label="regression: $y \\in \\mathcal{O}(" + " x^{" + f"{p:.1f}" + "})$")

# ~~ use this when plotting on non-logarithmix axes
#def powerlaw(x, a,p):
#    return a*(x**p)
#popt, pcov = sp.optimize.curve_fit(powerlaw, *single_threaded)
#a, p = popt
#plt.plot(xref, a*np.power(xref, p), "b--", alpha=.5,\
#        label="regression: $y \\in \\mathcal{O}(" + " x^{" + f"{p:.1f}" + "})$")

plt.plot(x_data, y_data, label = "single node measurement", marker="o", markersize=3)

x_data, y_data = map(list, zip(*multi_processes_per_node))
x_log = np.log10(x_data)
y_log = np.log10(y_data)
reg = sp.stats.linregress(x_log, y_log)
xref=np.logspace(x_log.min(),x_log.max(),num=50)
p = reg.slope
a = np.power(10, reg.intercept)
plt.plot(xref, a*np.power(xref, p), "--", alpha=.5, c="red",\
         label="regression: $y \\in \\mathcal{O}(" + " x^{" + f"{p:.1f}" + "})$")
plt.plot(x_data, y_data, label = "multi nodes measurement", marker="o", markersize=3, c="red")
plt.legend()
plt.xlabel("number of processes")
plt.ylabel('time [s]')
plt.title("1-tree Lower Bound MPI (n=3000)")

plt.xscale("log")
plt.yscale("log")

#plt.show()
plt.savefig("1-tree-mpi.pdf")
