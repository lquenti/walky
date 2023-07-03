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

for (algorithm, label) in [
        ("v0", "Naive"),
        ("v1", "Fixed Starting Node"),
        ("v2", "Prefix Sum Caching"),
        ("v3", "Naive prune"),
        ("v4", "NN prune"),
        #("v5", "MST prune"),
        #("v6", "MST cache"),
        #("multithreaded", "Multithreaded")
        ]:
    xs,ys = load_from_file(f"./results/results_{algorithm}.json")
    plt.plot(xs,ys, label=label)

plt.ylim(y_min, y_max)
plt.xlabel('Graph Size')
plt.ylabel('t')
plt.title('Exact Solver Single Node')
ax = plt.gca()
ax.set_xlim([0,13])
#ax.set_xlim([0,20])
ax.set_ylim([0,100])
plt.legend()
#plt.show()
plt.savefig("exact-stmt.pdf")
