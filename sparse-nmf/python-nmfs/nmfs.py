import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fn_nmfs import *
from fn_consensus import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/projects/sparse-nmf/input/input_CCLE_drug_IC50_zero-one.csv', index_col=0, header=0, na_values='NaN')
X = X.fillna(0)

rank = 3

iterations = 1

beta = 3
sparsity = 5

runs = 1
consMat = ConsensusMatrix(X)

for run in range(runs):
	print(run)
	snmf = SparseNMF(X, rank, iterations, beta, sparsity)
	connmatW = consMat.calcConnMatW(snmf.W)
	connmatH = consMat.calcConnMatH(snmf.H)
	consMat.calcConsMat(connmatW, connmatH)

consMat.calcAvConsMat()

cmW = consMat.reorderConsensusMatrix(consMat.consMatW)
cmH = consMat.reorderConsensusMatrix(consMat.consMatH)

plt.subplot(211)
plt.imshow(cmW, cmap='hot', interpolation='nearest')
plt.title('Consensus matrix for W')
plt.colorbar()
plt.subplot(212)
plt.imshow(cmH, cmap='hot', interpolation='nearest')
plt.title('Consensus matrix for H')
plt.colorbar()
plt.show()