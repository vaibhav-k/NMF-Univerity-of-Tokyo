import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from fn_NMF_nm import *
from fn_reorder import *

X = pd.read_csv(
    r"../../inputs/portal-Avana-2018-06-08-n.csv", index_col=0, na_values="NaN"
)
mX = 1 - X.isnull()

rank = 50
iterations = 20000

W = pd.DataFrame(np.zeros((X.shape[0], rank)), index=list(X.index))
H = pd.DataFrame(np.zeros((rank, X.shape[1])), columns=list(X))

error = 0

nmf = NMF(X, rank, iterations)
nmf.runNMF()

error = nmf.err

W = nmf.W
H = nmf.H

# print('Reordering output matrices')
W.to_csv("../W50uno.csv")
H.to_csv("../H50uno.csv")
"""
W = reorderMatrixW(W)
H = reorderMatrixH(H)

print("Error for rank %d is %f" % (rank, error))
"""
"""
print('Making plots')

plt.figure()
plt.figure(figsize=(X.shape[1]/6, 10))

plt.subplot(211)
sns.heatmap(W, xticklabels=True, yticklabels=True, cmap="YlGnBu")
#plt.imshow(W)
plt.title('Output matrix for W')

plt.subplot(212)
sns.heatmap(H, xticklabels=True, yticklabels=True, cmap="YlGnBu")
#plt.imshow(H)
plt.title('Output matrix for H')

plt.suptitle("Reordered output matrices for rank = 3", size=16)
#plt.savefig('test')
plt.show()
"""
"""
W.to_csv('../W55.csv')
H.to_csv('../H55.csv')
"""
