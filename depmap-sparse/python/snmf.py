import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fn_SparseNMF import *
from fn_consensus import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/portal-Avana-2018-06-08-n.csv', index_col=0, header=0)
X = X.fillna(0)

rank = 3
iterations = 100

beta = 2
sparsity = 40

runs = 50
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