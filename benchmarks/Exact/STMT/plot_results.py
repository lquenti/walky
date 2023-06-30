import json
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

for algorithm in ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "multithreaded"]:
    xs,ys = load_from_file(f"./results/results_{algorithm}.json")
    plt.plot(xs,ys, label=algorithm)

plt.ylim(y_min, y_max)
plt.xlabel('Graph Size')
plt.ylabel('t')
plt.title('Exact Solver Single Node')
plt.legend()
plt.show()

