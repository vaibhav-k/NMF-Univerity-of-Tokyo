import numpy as np
import pandas as pd
import numpy.matlib
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

class NMF():

		def __init__(self, X, K, iterations):
			self.X = X
			self.K = K
			self.iterations = iterations

		def initializeWH(self):
			self.H = pd.DataFrame(np.random.normal(size = (self.K, np.shape(self.X)[1])))
			self.W = pd.DataFrame(np.random.normal(size = (np.shape(self.X)[0], self.K)))
			self.X1 = np.dot(self.W, self.H)
			self.eps = np.finfo(self.W.values.dtype).eps

		def update(self):
			self.H = np.multiply(self.H, np.divide(np.dot(np.transpose(self.W), self.X), np.dot(np.transpose(self.W), np.dot(self.W, self.H))+self.eps))
			self.W = np.multiply(self.W, np.divide(np.dot(self.X, np.transpose(self.H)), np.dot(np.dot(self.W, self.H), np.transpose(self.H))+self.eps))

		def wrapper_update(self):
			for x in range(self.iterations):
				self.update()
				self.calc_error()
				self.print_details(x)

		def calc_error(self):
			self.X2 = np.dot(self.W, self.H)
			#self.diff = np.sum(np.sum(np.abs(self.X2 - self.X1)))
			#self.eucl_dist = self.calc_edist(self.X, self.X2)
			self.error = np.mean(np.mean(np.abs(self.X - self.X2)))

		def calc_edist(self, X, Y):
			dist = np.sum(np.sum(np.power(X-Y, 2)))
			return dist

		def print_details(self, run):
			#print("[%s] diff = %f, eucl_dist = %f, error = %f" % (run, self.diff, self.eucl_dist, self.error))
			print("[%s] error = %f" % (run, self.error))