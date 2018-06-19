import pandas as pd
import numpy as np
from operator import itemgetter

def reorderMatrixH(X):
	X = clusterH(X)
	X = getTogetherH(X)
	return X

def clusterH(X):
	a = X
	a = (a == a.max(axis=0)[None,:]).astype(int)
	return a

def getTogetherH(X):
	X = np.transpose(X)
	for i in range(X.shape[1]):
		X = sorted(X, key=itemgetter(i))
	return np.transpose(X)

def reorderMatrixW(X):
	X = clusterW(X)
	X = getTogetherW(X)
	return X

def clusterW(X):
	a = X
	a = (a == a.max(axis=1)[:,None]).astype(int)
	return a

def getTogetherW(X):
	for i in range(X.shape[1]):
		X = sorted(X, key=itemgetter(i))
	return X