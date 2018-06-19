import pandas as pd
import numpy as np

class JNMF():
	def __init__(self, X, mX, Y, mY, Z, mZ, K, maxit):
		self.X = X
		self.Y = Y
		self.Z = Z
		self.mX = mX
		self.mY = mY
		self.mZ = mZ
		self.K = K
		self.maxit = maxit

		self.toterr = 0
		self.err = 0

		self.initializeWH()

	def initializeWH(self):
		self.W = pd.DataFrame(np.random.randint(0,100,size=(self.X.shape[0], self.K)))
		self.Hx = pd.DataFrame(np.random.randint(0,100,size=(self.K, self.X.shape[1])))
		self.Hy = pd.DataFrame(np.random.randint(0,100,size=(self.K, self.Y.shape[1])))
		self.Hz = pd.DataFrame(np.random.randint(0,100,size=(self.K, self.Z.shape[1])))

		self.update()

		self.toterr = self.toterr/self.maxit
		#print("error=%f" % (self.toterr))

	def update(self):
		for i in range(self.maxit):
			self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(self.mX, self.X), self.Hx.T) + np.dot(np.multiply(self.mY, self.Y), self.Hy.T) + np.dot(np.multiply(self.mZ, self.Z), self.Hz.T), np.dot(np.multiply(self.mX, np.dot(self.W, self.Hx)), self.Hx.T) + np.dot(np.multiply(self.mY, np.dot(self.W, self.Hy)), self.Hy.T) + np.dot(np.multiply(self.mZ, np.dot(self.W, self.Hz)), self.Hz.T)))
			self.Hx = np.multiply(self.Hx, np.divide(np.dot(self.W.T, np.multiply(self.mX, self.X)), np.dot(self.W.T, np.multiply(self.mX, np.dot(self.W, self.Hx)))))
			self.Hy = np.multiply(self.Hy, np.divide(np.dot(self.W.T, np.multiply(self.mY, self.Y)), np.dot(self.W.T, np.multiply(self.mY, np.dot(self.W, self.Hy)))))
			self.Hz = np.multiply(self.Hz, np.divide(np.dot(self.W.T, np.multiply(self.mZ, self.Z)), np.dot(self.W.T, np.multiply(self.mZ, np.dot(self.W, self.Hz)))))

			self.error(i)

	def error(self, i):
		self.err = np.mean(np.mean(np.abs(self.X-np.dot(self.W, self.Hx)))) + np.mean(np.mean(np.abs(self.Y-np.dot(self.W, self.Hy)))) + np.mean(np.mean(np.abs(self.Z-np.dot(self.W, self.Hz))))
		self.toterr += self.err 