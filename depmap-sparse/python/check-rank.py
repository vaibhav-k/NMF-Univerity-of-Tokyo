import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fn_SparseNMF import *
from fn_consensus import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/input/portal-Avana-2018-06-08-n.csv', index_col=0, header=0)
X = X.fillna(0)

iterations = 100

beta = 4
sparsity = 10
rank = 3
runs = 5

error = 0
consMat = ConsensusMatrix(X)

for run in range(runs):
	snmf = SparseNMF(X, rank, iterations, beta, sparsity)
	error += snmf.err
	connmatW = consMat.calcConnMatW(snmf.W)
	connmatH = consMat.calcConnMatH(snmf.H)
	consMat.calcConsMat(connmatW, connmatH)

consMat.calcAvConsMat()

print("Error for rank %d is %f" % (rank, error))

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
#plt.show()
plt.savefig("C:/Users/Vaibhav/nmf/projects/depmap-sparse/consensus-matrices/k = %d, beta= %d" % (rank, beta))