import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from fn_NMF_nm import *
from fn_consensus import *

X = pd.read_csv(
    r"../../inputs/portal-Avana-2018-06-08-n.csv",
    index_col=0,
    header=0,
    na_values="NaN",
)
X = X.fillna(0)

iterations = 10000
trials = 15

print("\nNumber of trials = %i" % trials)
print("Number of iterations in each trial = %i\n" % iterations)

for rank in range(50, 66):
    if rank % 5 != 0:
        continue
    cmatrix = CMatrix(X)

    error = 0

    print("Running NMF for rank = %i\n" % rank)

    for run in range(trials):
        print("\nTrial = %i for rank = %i\n" % (run, rank))
        nmf = NMF(X, rank, iterations)
        nmf.runNMF()
        error += nmf.err
        print("\nCalculating connectivity matrix for W")
        connW = cmatrix.calcConnectivityW(nmf.W)
        print("Calculating connectivity matrix for H")
        connH = cmatrix.calcConnectivityH(nmf.H)
        print("\nAdding connectivity matrix to consensus")
        cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH)

    print("\nError for rank %d is %f" % (rank, error))

    print("\nMaking consensus matrix")
    cmatrix.finalizeConsensusMatrix()

    print("\nReordering connectivity matrix W")
    cmW = cmatrix.reorderConsensusMatrix(cmatrix.cmW)
    print("Reordering connectivity matrix H")
    cmH = cmatrix.reorderConsensusMatrix(cmatrix.cmH)

    plt.subplot(221)
    plt.imshow(cmW, cmap="hot", interpolation="nearest")
    plt.title("Consensus matrix for W")
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(cmH, cmap="hot", interpolation="nearest")
    plt.title("Consensus matrix for H")
    plt.colorbar()

    plt.suptitle("Consensus matrices for rank = %d by NMF" % (rank), size=16)

    savedir = "../consensus-matrices"
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    print("\nSaving plots\n")

    plt.savefig(savedir + "/k = %d" % (rank))
    plt.clf()
