import json
import numpy as np
import matplotlib.pyplot as plt

y_min = 0
y_max = 5*60 # 5min

# TODO load in the data
def load_from_file(path):
    with open(path, 'r') as fp:
        parsed_data = json.load(fp)
    results = parsed_data['results']
    xs = [result["parameters"]["N"] for result in results]
    ys = [result["median"] for result in results]
    return xs,ys

for (algorithm, label) in [
        ("v0", "naive"),
        ("v1", "Fixed Stating Node"),
        ("v2", "Prefix Sum Caching"),
        ("v3", "Naive prune"),
        ("v4", "NN prune"),
        ("v5", "MST prune"),
        ("v6", "MST cache"),
        ("multithreaded", "Multithreaded")
        ]:
    xs,ys = load_from_file(f"./results/results_{algorithm}.json")
    if algorithm == "v4": # TODO fix me
        xs = xs[:18]
        ys = ys[:18]
    plt.plot(xs,ys, label=algorithm)


plt.ylim(y_min, y_max)
plt.xlabel('Graph Size')
plt.ylabel('t')
plt.title('Exact Solver Single Node')
ax = plt.gca()
ax.set_ylim([0, 10])
ax.set_xticks(np.round(np.linspace(0, 50, 11), 2))
plt.legend()
#plt.show()
plt.savefig("exact-stmt.pdf")
