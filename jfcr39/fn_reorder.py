import pandas as pd
import numpy as np
from operator import itemgetter


def reorderMatrixH(X):
    X = clusterH(X)
    X = getTogetherH(X)
    return X


def clusterH(X):
    a = X
    a = (a == a.max()).astype(int)
    return a


def getTogetherH(X):
    X = np.transpose(X)
    X = X.sort_values(by=list(X))
    return np.transpose(X)


def reorderMatrixW(X):
    X = clusterW(X)
    X = getTogetherW(X)
    return X


def clusterW(X):
    a = X
    a = pd.DataFrame(a.values == a.max(axis=1)[:, None], dtype=int, index=list(X.index))
    return a


def getTogetherW(X):
    X = X.sort_values(by=list(X))
    return X
