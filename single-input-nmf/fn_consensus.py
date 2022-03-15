import numpy as np
import pandas as pd
import numpy.matlib
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class ConsensusMatrix:
    def __init__(self, X):
        self.consMatW = pd.DataFrame(np.zeros((X.shape[0], X.shape[0])))
        self.consMatH = pd.DataFrame(np.zeros((X.shape[1], X.shape[1])))
        self.cmNum = 0

    def calcConnMatW(self, W):
        maxW = W.values.max(axis=1)
        maxW[maxW == 0] = 1
        maxMatW = (np.tile(maxW, (W.shape[1], 1))).transpose()
        binaryW = W == maxMatW
        connMatW = np.dot(binaryW, binaryW.transpose())
        return pd.DataFrame(connMatW)

    def calcConnMatH(self, H):
        maxH = H.values.max(axis=0)
        maxH[maxH == 0] = 1
        maxMatH = np.tile(maxH, (H.shape[0], 1))
        binaryH = H == maxMatH
        connMatH = np.dot(binaryH.transpose(), binaryH)
        return pd.DataFrame(connMatH)

    def calcConsMat(self, connMatW, connMatH):
        self.consMatW = self.consMatW + connMatW
        self.consMatH = self.consMatH + connMatH
        self.cmNum = self.cmNum + 1

    def calcAvConsMat(self):
        self.consMatW = self.consMatW / self.cmNum
        self.consMatH = self.consMatH / self.cmNum

    def reorderConsensusMatrix(self, M):
        Y = 1 - M
        p = squareform(Y)
        Z = linkage(p, method="average")
        ivl = leaves_list(Z)
        ivl = ivl[::-1]
        reorderM = pd.DataFrame(
            M.values[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl]
        )
        return reorderM
