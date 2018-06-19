import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fn_consensus import *
from fn_nmf import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/projects/depmap/input/portal-Avana-2018-06-08-n.csv', index_col=0, header=0, na_values='NaN')
#X = pd.read_csv(r'C:\Users\Vaibhav\nmf\projects\jfcr39\input/input_small_data_DRUG.zero-one.csv', index_col=0, header=0, na_values='NaN')
X = X.fillna(0)
mX = 1 - X.isnull()

maxit = 1
runs = 1

for rank in range (2,31):
	cmatrix = ConsensusMatrix(X)

	error = 0

	for run in range(runs):
		'''
		jnmf = JointNMF_mask(X, mX, rank, maxit)
		jnmf.check_nonnegativity()
		jnmf.initialize_W_H()
		#jnmf.update_euclidean_multiplicative()
		jnmf.wrapper_calc_euclidean_multiplicative_update()
		jnmf.print_distance_of_HW_to_X(i)
		#error += nmf.toterr
		'''
		jnmf = NMF(X, mX, rank, maxit)
		error += jnmf.err
		connW = cmatrix.calcConnectivityW(jnmf.W)
		connH = cmatrix.calcConnectivityH(jnmf.H)
		cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH)
	
	print("Error for rank %d is %f" % (rank, error))
	#error /= runs
	#print("Error for rank %d is %f" % (rank, error))

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

	plt.show()