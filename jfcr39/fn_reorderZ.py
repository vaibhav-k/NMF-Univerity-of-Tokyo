import pandas as pd
import numpy as np
from operator import itemgetter


def reorderMatrixH(X):
    nX = (X - X.mean()) / X.std()
    X = clusterH(nX)
    X = getTogetherH(X)
    return X


def clusterH(X):
    a = X
    print(X)
    a = (a > 0).astype(int)
    return a


def getTogetherH(X):
    X = np.transpose(X)
    X = X.sort_values(by=list(X))
    return np.transpose(X)


def reorderMatrixW(X):
    nX = (X - X.mean()) / X.std()
    X = clusterW(nX)
    X = getTogetherW(X)
    return X


def clusterW(X):
    a = X
    a = (a > -1).astype(int)
    return a


def getTogetherW(X):
    X = X.sort_values(by=list(X))
    return X
