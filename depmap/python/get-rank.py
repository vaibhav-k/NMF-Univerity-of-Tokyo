import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fn_consensus import *
from fn_nmf import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/portal-Avana-2018-06-08-n.csv', index_col=0, header=0, na_values='NaN')
X = X.fillna(0)
mX = 1 - X.isnull()

maxit = 1
runs = 1

for rank in range (3,31):
	cmatrix = CMatrix(X)

	error = 0

	for run in range(runs):
		jnmf = NMF(X, mX, rank, maxit)
		error += jnmf.err
		connW = cmatrix.calcConnectivityW(jnmf.W)
		connH = cmatrix.calcConnectivityH(jnmf.H)
		cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH)
	
	print("Error for rank %d is %f" % (rank, error))

	cmatrix.finalizeConsensusMatrix()

	cmW = cmatrix.reorderConsensusMatrix(cmatrix.cmW)
	cmH = cmatrix.reorderConsensusMatrix(cmatrix.cmH)

	plt.subplot(221)
	plt.imshow(cmW, cmap='hot', interpolation='nearest')
	plt.title('Consensus matrix for W')
	plt.colorbar()

	plt.subplot(222)
	plt.imshow(cmH, cmap='hot', interpolation='nearest')
	plt.title('Consensus matrix for H')
	plt.colorbar()

	plt.suptitle("Consensus matrices for rank = %d" % (rank), size=16)
	savedir = '../consensus-matrices'
	if not os.path.exists(savedir):
		os.mkdir(savedir)
	plt.savefig(savedir + "/k = %d" % (rank))
	plt.clf()
	#plt.show()