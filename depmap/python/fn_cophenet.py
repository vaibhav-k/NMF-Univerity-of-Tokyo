import pandas as pd
import matplotlib.pyplot as plt
import time
import fastcluster as fc
from scipy.cluster.hierarchy import cophenet
from matplotlib.pyplot import imshow, set_cmap

class CophCoeff():
	def __init__(self, W, K, report):
		self.report = report
		self.M = W
		self.K = K

	def calcCoph(self):
		print("Finding pdist")
		ori_dists = fc.pdist(self.M)
		print("Doing fast clustering")
		Z = fastcluster.linkage(ori_dists, method='average')
		print("Finding cophenetic correlation")
		[coph_corr, coph_dists] = cophenet(Z, ori_dists)
		print("rank k = %d, cophenetic corr. =  %d" % (self.K, coph_corr))
		self.report.append(pd.DataFrame([[self.K,coph_corr]], columns = ["K", "Cophenetic Corr"]), ignore_index=True)