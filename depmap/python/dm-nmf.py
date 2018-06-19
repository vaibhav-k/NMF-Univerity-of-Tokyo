import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from fn_dm-nmf import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/portal-Avana-2018-06-08.csv', index_col=0, na_values='NaN')
mX = 1 - X.isnull()

maxit = 100
runs = 50
rank = 3

W = pd.DataFrame(np.zeros((X.shape[0], rank)), index=list(X.index))
H = pd.DataFrame(np.zeros((rank, X.shape[1])), columns=list(X))

Wb = pd.DataFrame(np.zeros((X.shape[0], rank)), index=list(X.index))
Hb = pd.DataFrame(np.zeros((rank, X.shape[1])), columns=list(X))

minerr = sys.maxsize
error = 0

for run in range(runs):
	jnmf = JNMF(X, mX, Y, mY, Z, mZ, rank, maxit)
	if(jnmf.toterr<minerr):
		Wb = jnmf.W
		Hb = jnmf.H
	error += jnmf.toterr
	W += jnmf.W
	H += jnmf.H

W /= runs
H /= runs

W = reorderMatrixW(W)
H = reorderMatrixH(H)

error /= runs
print("Error for rank %d is %f" % (rank, error))