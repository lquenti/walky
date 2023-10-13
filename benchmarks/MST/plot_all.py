import json
import matplotlib.pyplot as plt

# ~~~ load STMT values ~~~
BASE_FOLDER = "STMT/results"
MIN_SIZE = 100
MAX_SIZE = 3000
STEP = 100

MIN_XXL = 4000
MAX_XXL = 10000
STEP_XXL = 1000

single_threaded = ([], [])
multi_threaded = ([], [])
sizes = [s for s in range(MIN_SIZE, MAX_SIZE+1, STEP)] + [s for s in range(MIN_XXL, MAX_XXL+1, STEP_XXL)]
for size in sizes:
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

# ~~~ load QUEUE values ~~~
BASE_FOLDER = "QUEUE/results"

queue_results = ([], [])
for size in sizes:
    # single threaded
    with open(f"./{BASE_FOLDER}/result_{size}_.json") as fp:
        obj = json.load(fp)
    queue_results[0].append(size)
    queue_results[1].append(obj["median"]) # could also use mean

import numpy as np
import scipy as sp

# ~~ use this when plotting on logarithmic axes
x_log = np.log10(single_threaded[0])
y_log = np.log10(single_threaded[1])
xref=np.logspace(x_log.min(),x_log.max(),num=50)
reg = sp.stats.linregress(x_log, y_log)
p = reg.slope
a = np.power(10, reg.intercept)

plt.plot(xref, a*np.power(xref, p), "b--", alpha=.5,\
        label="regression: $y \\in \\Theta(" + " x^{" + f"{p:.1f}" + "})$")

# ~~ use this when plotting on non-logarithmix axes
#def powerlaw(x, a,p):
#    return a*(x**p)
#popt, pcov = sp.optimize.curve_fit(powerlaw, *single_threaded)
#a, p = popt
#plt.plot(xref, a*np.power(xref, p), "b--", alpha=.5,\
#        label="regression: $y \\in \\mathcal{O}(" + " x^{" + f"{p:.1f}" + "})$")

plt.plot(*single_threaded, label = "sequential vector measurement", marker="o", markersize=3)

x_log = np.log10(multi_threaded[0])
y_log = np.log10(multi_threaded[1])
reg = sp.stats.linregress(x_log, y_log)
xref=np.logspace(x_log.min(),x_log.max(),num=50)
p = reg.slope
a = np.power(10, reg.intercept)
plt.plot(xref, a*np.power(xref, p), "--", alpha=.5, c="orange",\
         label="regression: $y \\in \\Theta(" + " x^{" + f"{p:.1f}" + "})$")
plt.plot(*multi_threaded, label = "multi threaded measurement", marker="o", markersize=3)

x_log = np.log10(queue_results[0])
y_log = np.log10(queue_results[1])
reg = sp.stats.linregress(x_log, y_log)
xref=np.logspace(x_log.min(),x_log.max(),num=50)
p = reg.slope
a = np.power(10, reg.intercept)
plt.plot(xref, a*np.power(xref, p), "--", alpha=.5, c="green",\
         label="regression: $y \\in \\Theta(" + " x^{" + f"{p:.1f}" + "})$")
plt.plot(*queue_results, label = "sequential queue measurement", marker="o", markersize=3)

plt.xscale("log")
plt.yscale("log")

#plt.ylim(y_min, y_max)
plt.xlabel('Graph Size (number of vertices)')
plt.ylabel('time [s]')
plt.title('MST Lower Bound Single Node Performance')
plt.legend(fontsize="small")
#plt.show()
plt.savefig("MST-all.pdf")

