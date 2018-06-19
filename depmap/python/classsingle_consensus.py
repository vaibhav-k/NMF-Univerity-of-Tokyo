'''
Created on 2016/10/21

@author: taiho
'''
import numpy as np
import pandas as pd
import numpy.matlib
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class ConsensusMatrix(object):
	'''
	Connectivity matrix (single JNMF run)
	Consensus matrix (multiple JNMF run)
	Reordring consensus matrix
	Cophenetic correlation
	
	'''


	def __init__(self, X1):
		'''
		Constructor
		'''
		self.cmW = pd.DataFrame(np.zeros((X1.shape[0], X1.shape[0])))
		self.cmH1 = pd.DataFrame(np.zeros((X1.shape[1], X1.shape[1])))
		self.cmNum = 0

	def calcConnectivityW(self, W):
		maxW = W.as_matrix().max(axis=1)
		maxW[maxW == 0] = 1
		argmaxW = W.as_matrix().argmax(axis=1)
		maxMatW = (np.tile(maxW, (W.shape[1], 1))).transpose()
		binaryW = W == maxMatW
		connMatW = np.dot(binaryW, binaryW.transpose())
		return connMatW
	
	def calcConnectivityH(self, H):
		maxH = H.as_matrix().max(axis=0)
		maxH[maxH == 0] = 1
		argmaxH = H.as_matrix().argmax(axis=0)
		maxMatH = np.tile(maxH, (H.shape[0], 1))
		binaryH = H == maxMatH
		connMatH = np.dot(binaryH.transpose(), binaryH)
		return connMatH
		
	def addConnectivityMatrixtoConsensusMatrix(self, connW, connH1):
		self.cmW += connW
		self.cmH1 += connH1
		self.cmNum += 1
		
	def finalizeConsensusMatrix(self):
		self.cmW /= self.cmNum
		self.cmH1 /= self.cmNum

	def reorderConsensusMatrix(self, M):
		Y = 1 - M
		Z = linkage(squareform(Y), method='average')
		ivl = leaves_list(Z)
		ivl = ivl[::-1]
		reorderM = pd.DataFrame(M.as_matrix()[:, ivl][ivl, :], index = M.columns[ivl], columns = M.columns[ivl])
		return reorderM
		