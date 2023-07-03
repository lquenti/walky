import json
import matplotlib.pyplot as plt

BASE_FOLDER = "results"
MIN_SIZE = 100
MAX_SIZE = 3000
STEP = 100

single_threaded = ([], [])
multi_threaded = ([], [])
for size in range(MIN_SIZE, MAX_SIZE+1, STEP):
    # single threaded
    with open(f"./{BASE_FOLDER}/result_{size}_single-threaded.json") as fp:
        obj = json.load(fp)
    single_threaded[0].append(size)
    single_threaded[1].append(obj["median"]) # could also use mean

    # multi threaded
    with open(f"./{BASE_FOLDER}/result_{size}_multi-threaded.json") as fp:
        obj = json.load(fp)
    multi_threaded[0].append(size)
    multi_threaded[1].append(obj["median"]) # could also use mean

plt.plot(*single_threaded, label = "single threaded")
plt.plot(*multi_threaded, label = "multi threaded")

#plt.ylim(y_min, y_max)
plt.xlabel('Graph Size')
plt.ylabel('time in seconds')
plt.title('Exact Solver Single Node')
plt.legend()
#plt.show()
plt.savefig("christofides-stmt.pdf")

