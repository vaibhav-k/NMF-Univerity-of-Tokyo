import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fn_nmf import *
from fn_consensus import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/input_small_data_DRUG.zero-one.csv', index_col=0, header=0, na_values='NaN')
X = X.fillna(0)

Y = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/input_small_data_MUT.zero-one.csv', index_col=0, header=0, na_values='NaN')
Y = X.fillna(0)

Z = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/input_small_data_PROT.zero-one.csv', index_col=0, header=0, na_values='NaN', encoding="ISO-8859-1")
Z = X.fillna(0)

mX = 1 - X.isnull()
mY = 1 - Y.isnull()
mZ = 1 - Z.isnull()

maxit = 100
runs = 5

for rank in range (2,31):
	cmatrix = JConsensus(X, Y, Z)

	error = 0

	for run in range(runs):
		jnmf = JNMF(X, mX, Y, mY, Z, mZ, rank, maxit)
		error += jnmf.toterr
		connW = cmatrix.calcConnectivityW(jnmf.W)
		connHx = cmatrix.calcConnectivityH(jnmf.Hx)
		connHy = cmatrix.calcConnectivityH(jnmf.Hy)
		connHz = cmatrix.calcConnectivityH(jnmf.Hz)
		cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connHx, connHy, connHz)
	
	error /= runs
	print("Error for rank %d is %f" % (rank, error))

	cmatrix.finalizeConsensusMatrix()

	cmW = cmatrix.reorderConsensusMatrix(cmatrix.cmW)
	cmHx = cmatrix.reorderConsensusMatrix(cmatrix.cmHx)
	cmHy = cmatrix.reorderConsensusMatrix(cmatrix.cmHy)
	cmHz = cmatrix.reorderConsensusMatrix(cmatrix.cmHz)

	plt.subplot(221)
	plt.imshow(cmW, cmap='hot', interpolation='nearest')
	plt.title('Consensus matrix for W')
	plt.colorbar()

	plt.subplot(222)
	plt.imshow(cmHx, cmap='hot', interpolation='nearest')
	plt.title('Consensus matrix for Hx')
	plt.colorbar()

	plt.subplot(223)
	plt.imshow(cmHy, cmap='hot', interpolation='nearest')
	plt.title('Consensus matrix for Hy')
	plt.colorbar()

	plt.subplot(224)
	plt.imshow(cmHz, cmap='hot', interpolation='nearest')
	plt.title('Consensus matrix for Hz')
	plt.colorbar()

	plt.suptitle("Consensus matrices for rank = %d" % (rank), size=16)
	'''
	plt.savefig('C:/Users/Vaibhav/NMF/vaibhav_NMF/JointNMF/figures-CCLE/jnmf_consensus_matrices_rank=%d.png' % rank)
	plt.clf()
	'''
	plt.show()
	
