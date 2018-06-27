import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from fn_NMF_nm import *
from fn_consensus import *
from fn_cophenet import *

X = pd.read_csv(r'../../inputs/portal-Avana-2018-06-08-n.csv', index_col=0, header=0, na_values='NaN')
X = X.fillna(0)

iterations = 1
trials = 1
report = pd.DataFrame(columns = ["K", "Cophenetic Corr"])

print("\nNumber of trials = %i" % trials)
print("Number of iterations in each trial = %i\n" % iterations)

for rank in range (2,5):
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

	print("\nFinding Cophenetic corelation for rank = %d" % rank)
	coph = CophCoeff(cmatrix.cmW, rank, report)
	coph.calcCoph()

print("Saving CSV")
report.to_csv('new_cophenetic_corr.csv')