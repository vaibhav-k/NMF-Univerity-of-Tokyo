import pandas as pd
import numpy as np

class NMF():
	def __init__(self, X, mX, K, maxit):
		self.X = X
		self.mX = mX
		self.K = K
		self.maxit = maxit
		self.toterr = 0
		self.err = 0

		self.initializeWH()

	def initializeWH(self):
		self.W = pd.DataFrame(np.random.randint(0,100,size=(self.X.shape[0], self.K)), dtype=float)
		self.H = pd.DataFrame(np.random.randint(0,100,size=(self.K, self.X.shape[1])), dtype=float)
		self.eps = np.finfo(self.W.values.dtype).eps

		self.update()
		self.err = self.toterr/self.maxit

	def update(self):
		for i in range(self.maxit):
			self.H = np.multiply(self.H, np.divide(np.dot(self.W.T, np.multiply(self.mX, self.X)), np.dot(self.W.T, np.multiply(self.mX, np.dot(self.W, self.H) + self.eps))))
			self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(self.mX, self.X), self.H.T), (np.dot(np.multiply(self.mX, np.dot(self.W, self.H)), self.H.T) + self.eps)))

			self.error(i)

	def error(self, i):
		self.err = np.mean(np.mean(np.abs(self.X-np.dot(self.W, self.H))))
		self.toterr += self.err 
