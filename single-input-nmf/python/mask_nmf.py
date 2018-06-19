import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fn_mask_nmf import *
from fn_consensus import *
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.pyplot import savefig, imshow, set_cmap

K = 3
if len(sys.argv) > 1:
	K = int(sys.argv[1])
	print ("K =  %d" % K)

Xori = pd.read_csv(r'C:/Users/Vaibhav/Downloads/portal-Avana-2018-06-08.csv', header=0, index_col=0, na_values='NaN')
X = Xori[Xori.notnull().any(axis=1)]
mX = 1 - X.isnull()
X = X.fillna(0)

maxiter = 1

consMat = ConsensusMatrix(X)

for run in range(1):
	nmf = NMF(X, mX, K, maxiter)
	nmf.initializeWH()
	nmf.wrapper_update()
	nmf.calc_error()
	connmatW = consMat.calcConnMatW(nmf.W)
	connmatH = consMat.calcConnMatH(nmf.H)
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
