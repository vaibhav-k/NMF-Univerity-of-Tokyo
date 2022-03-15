import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from fn_SparseNMF import *
from fn_consensus import *

X = pd.read_csv(
    r"../../inputs/portal-Avana-2018-06-08-n.csv",
    index_col=0,
    header=0,
    na_values="NaN",
)
X = X.fillna(0)

iterations = 100
trials = 5

print("\nNumber of trials = %i" % trials)
print("Number of iterations in each trial = %i\n" % iterations)

beta = 3
sparsity = 5

for rank in range(3, 31):
    error = 0

    consMat = ConsensusMatrix(X)

    print("Running Sparse NMF for rank = %i" % rank)

    for run in range(trials):
        print("\nTrial = %i for rank = %i\n" % (run, rank))
        nmf = SparseNMF(X, rank, iterations, beta, sparsity)
        nmf.runNMF()
        error += nmf.error
        print("\nCalculating connectivity matrix W")
        connW = consMat.calcConnMatW(nmf.W)
        print("Calculating connectivity matrix H")
        connH = consMat.calcConnMatH(nmf.H)
        print("\nAdding connectivity matrix")
        consMat.calcConsMat(connW, connH)

    print("\nError for rank %d is %f" % (rank, error))

    print("\nMaking consensus matrix")
    consMat.calcAvConsMat()

    print("\nReordering connectivity matrix W")
    cmW = consMat.reorderConsensusMatrix(consMat.consMatW)
    print("Reordering connectivity matrix H")
    cmH = consMat.reorderConsensusMatrix(consMat.consMatH)

    plt.subplot(211)
    plt.imshow(cmW, cmap="hot", interpolation="nearest")
    plt.title("Consensus matrix for W")
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(cmH, cmap="hot", interpolation="nearest")
    plt.title("Consensus matrix for H")
    plt.colorbar()

    plt.suptitle("Consensus matrices for rank = %d by Sparse NMF" % (rank), size=16)

    savedir = "../consensus-matrices"
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    print("\nSaving plots\n")

    plt.savefig(savedir + "/k = %d, beta= %d" % (rank, beta))
    plt.clf()
