import numpy as np 
import matplotlib.pyplot as plt 

from parse import find_matched_files

if __name__ == "__main__":
    xs = find_matched_files(".")
    
    xs = [d for d in xs if not (d[0] == (2,1) or d[0] == (4,1) or d[0] == (8,1))]
    xs = sorted(xs, key=lambda d: d[0][0] * d[0][1])

    x = len(xs)
    y = len(xs[0][1])
    measurements = np.zeros((x,y))
    num_tasks = np.zeros(x)
    for (i, data) in enumerate(xs):
        (idx, datapoints) = data
        measurements[i] = np.array(datapoints)
        num_tasks[i] = idx[0] * idx[1]

    medians = np.median(measurements, axis=1)

    #plt.plot(num_tasks, medians)
    num_tasks_2d = np.array([num_tasks for i in range(y)]).T
    plt.scatter(num_tasks_2d, measurements, color="orange", s=4.)

    plt.plot(num_tasks, medians, linewidth=2.5, label = "median")

    plt.xticks(num_tasks)

    plt.title("Christofides randomized solution weight, parallelized with MPI (n=2500)")
    plt.xlabel("number of MPI tasks")
    plt.ylabel("solution weight")
    plt.legend()
    #plt.show()
    plt.savefig("christofides-mpi.pdf")
