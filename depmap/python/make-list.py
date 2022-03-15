import pandas as pd
import matplotlib.pyplot as plotter


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return "{p:.2f}%  ({v:d})".format(p=pct, v=val)

    return my_autopct


x = pd.read_csv(r"../H50mod.csv", index_col=0, header=0, na_values="NaN")
x = x.T
clusters = {}
for c in x.columns:
    clusters[c] = []
    for r in x.index:
        if x[c][r] == 1:
            clusters[c].append(r.split(".", 1)[0])

for key, val in clusters.items():
    z = clusters.get(key)

    uniqueWords = []
    num = []
    for i in z:
        if not i in uniqueWords:
            uniqueWords.append(i)

    for i in range(len(uniqueWords)):
        num.append(z.count(uniqueWords[i]))

    figureObject, axesObject = plotter.subplots()
    axesObject.pie(num, labels=uniqueWords, autopct=make_autopct(num), startangle=90)
    axesObject.axis("equal")
    plotter.suptitle("Cluster number %d" % key, size=16)
    plotter.savefig("../pie-charts/rank=50,cluster=%d" % key)
    plotter.clf()
