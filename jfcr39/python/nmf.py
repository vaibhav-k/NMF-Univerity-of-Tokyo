import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from mpl_toolkits.mplot3d import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
pca = sklearnPCA(n_components=2)

from fn_nmf import *
from fn_reorder import *

X = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/input_small_data_DRUG.zero-one.csv', index_col=0, na_values='NaN')
X = X.fillna(0)

Y = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/input_small_data_MUT.zero-one.csv', index_col=0, na_values='NaN')
Y = Y.fillna(0)

Z = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/input_small_data_PROT.zero-one.csv', index_col=0, na_values='NaN', encoding="ISO-8859-1")
Z = Z.fillna(0)

'''
plt.subplot(222)
plt.imshow(reorderMatrixH(X), cmap='hot', interpolation='nearest')
plt.title('input matrix for drugs')
plt.colorbar()

plt.subplot(223)
plt.imshow(reorderMatrixH(Y), cmap='hot', interpolation='nearest')
plt.title('input matrix for mutations')
plt.colorbar()

plt.subplot(224)
plt.imshow(reorderMatrixH(Z), cmap='hot', interpolation='nearest')
plt.title('input matrix for proteins')
plt.colorbar()

plt.suptitle("Reordered input matrices", size=16)
plt.show()
'''

mX = 1 - X.isnull()
mY = 1 - Y.isnull()
mZ = 1 - Z.isnull()

maxit = 100
runs = 1000
rank = 3

W = pd.DataFrame(np.zeros((X.shape[0], rank)), index=list(X.index))
Hx = pd.DataFrame(np.zeros((rank, X.shape[1])), columns=list(X))
Hy = pd.DataFrame(np.zeros((rank, Y.shape[1])), columns=list(Y))
Hz = pd.DataFrame(np.zeros((rank, Z.shape[1])), columns=list(Z))

Wb = pd.DataFrame(np.zeros((X.shape[0], rank)), index=list(X.index))
Hxb = pd.DataFrame(np.zeros((rank, X.shape[1])), columns=list(X))
Hyb = pd.DataFrame(np.zeros((rank, Y.shape[1])), columns=list(Y))
Hzb = pd.DataFrame(np.zeros((rank, Z.shape[1])), columns=list(Z))

minerr = sys.maxsize
error = 0

for run in range(runs):
	jnmf = JNMF(X, mX, Y, mY, Z, mZ, rank, maxit)
	if(jnmf.toterr<minerr):
		Wb = jnmf.W
		Hxb = jnmf.Hx
		Hyb = jnmf.Hy
		Hzb = jnmf.Hz
	error += jnmf.toterr
	W += jnmf.W
	Hx += jnmf.Hx
	Hy += jnmf.Hy
	Hz += jnmf.Hz

W /= runs
Hx /= runs
Hy /= runs
Hz /= runs
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = Hx.T.values[:, 0:1]
y_vals = Hx.T.values[:, 1:2]
z_vals = Hx.T.values[:, 2:3]

ax.scatter(x_vals, y_vals, z_vals, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = Hy.T.values[:, 0:1]
y_vals = Hy.T.values[:, 1:2]
z_vals = Hy.T.values[:, 2:3]

ax.scatter(x_vals, y_vals, z_vals, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = Hz.T.values[:, 0:1]
y_vals = Hz.T.values[:, 1:2]
z_vals = Hz.T.values[:, 2:3]

ax.scatter(x_vals, y_vals, z_vals, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
'''
W = reorderMatrixW(W)
Hx = reorderMatrixH(Hx)
Hy = reorderMatrixH(Hy)
Hz = reorderMatrixH(Hz)

error /= runs
print("Error for rank %d is %f" % (rank, error))

plt.subplot(221)
sns.heatmap(W, xticklabels=True, yticklabels=True, cmap="YlGnBu")
plt.title('Output matrix for W')

plt.subplot(222)
sns.heatmap(Hx, xticklabels=True, yticklabels=True, cmap="YlGnBu")
plt.title('Output matrix for Hx')

plt.subplot(223)
sns.heatmap(Hy, xticklabels=True, yticklabels=True, cmap="YlGnBu")
plt.title('Output matrix for Hy')

plt.subplot(224)
sns.heatmap(Hz, xticklabels=True, yticklabels=True, cmap="YlGnBu")
plt.title('Output matrix for Hz')

plt.suptitle("Clustered output matrices for rank = 3", size=16)
plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = Hx.T.values[:, 0:1]
y_vals = Hx.T.values[:, 1:2]
z_vals = Hx.T.values[:, 2:3]

ax.scatter(x_vals, y_vals, z_vals, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = Hy.T.values[:, 0:1]
y_vals = Hy.T.values[:, 1:2]
z_vals = Hy.T.values[:, 2:3]

ax.scatter(x_vals, y_vals, z_vals, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_vals = Hz.T.values[:, 0:1]
y_vals = Hz.T.values[:, 1:2]
z_vals = Hz.T.values[:, 2:3]

ax.scatter(x_vals, y_vals, z_vals, marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
'''


'''
plt.subplot(222)
plt.imshow(np.dot(Wb,Hxb).astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for drugs')
plt.colorbar()

plt.subplot(223)
plt.imshow(np.dot(Wb,Hyb).astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for mutations')
plt.colorbar()

plt.subplot(224)
plt.imshow(np.dot(Wb,Hzb).astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for proteins')
plt.colorbar()

plt.suptitle("Best output matrices for rank = 3", size=16)
plt.show()
'''
'''
plt.subplot(222)
plt.imshow(np.dot(W,Hx).astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for drugs')
plt.colorbar()

plt.subplot(223)
plt.imshow(np.dot(W,Hy).astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for mutations')
plt.colorbar()

plt.subplot(224)
plt.imshow(np.dot(W,Hz).astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for proteins')
plt.colorbar()

plt.suptitle("Unordered output matrices for rank = 3", size=16)
plt.show()
'''
'''
plt.subplot(221)
plt.imshow(W.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for W')
plt.colorbar()

plt.subplot(222)
plt.imshow(Hx.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for Hx')
plt.colorbar()

plt.subplot(223)
plt.imshow(Hy.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for Hy')
plt.colorbar()

plt.subplot(224)
plt.imshow(Hz.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for Hz')
plt.colorbar()

plt.suptitle("Unordered output matrices for rank = 3", size=16)
plt.show()
'''

'''
plt.subplot(221)
plt.imshow(W, cmap='hot', interpolation='nearest')
plt.title('Output matrix for W')
plt.colorbar()

plt.subplot(222)
plt.imshow(Hx, cmap='hot', interpolation='nearest')
plt.title('Output matrix for drugs')
plt.colorbar()

plt.subplot(223)
plt.imshow(Hy, cmap='hot', interpolation='nearest')
plt.title('Output matrix for mutations')
plt.colorbar()

plt.subplot(224)
plt.imshow(Hz, cmap='hot', interpolation='nearest')
plt.title('Output matrix for proteins')
plt.colorbar()

plt.suptitle("Clustered output matrices for rank = 3", size=16)
plt.show()
'''
'''
plt.subplot(221)
plt.imshow(W.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for cells')
plt.colorbar()

plt.subplot(222)
plt.imshow(Hx.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for drugs')
plt.colorbar()

plt.subplot(223)
plt.imshow(Hy.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for mutations')
plt.colorbar()

plt.subplot(224)
plt.imshow(Hz.astype(float), cmap='hot', interpolation='nearest')
plt.title('Output matrix for Proteins')
plt.colorbar()

plt.suptitle("Clustered output matrices for rank = 3", size=16)
plt.show()
'''
'''
plt.subplot(222)
plt.imshow(np.dot(W,Hx), cmap='hot', interpolation='nearest')
plt.title('Output matrix for drugs')
plt.colorbar()

plt.subplot(223)
plt.imshow(np.dot(W,Hy), cmap='hot', interpolation='nearest')
plt.title('Output matrix for mutations')
plt.colorbar()

plt.subplot(224)
plt.imshow(np.dot(W,Hz), cmap='hot', interpolation='nearest')
plt.title('Output matrix for proteins')
plt.colorbar()

plt.suptitle("Clustered output matrices for rank = 3", size=16)
plt.show()
'''
'''
sns.heatmap(W, xticklabels=True, yticklabels=True, cmap="YlGnBu")
plt.show()
'''
#f,(ax1,ax2,ax3, ax4) = plt.subplots(1,4)
#g1 = sns.heatmap(W, xticklabels=True, yticklabels=True, cmap="YlGnBu",ax=ax1)
#g2 = sns.heatmap(Hx, xticklabels=True, yticklabels=True, cmap="YlGnBu",ax=ax2)
#g3 = sns.heatmap(Hy, xticklabels=True, yticklabels=True, cmap="YlGnBu",ax=ax3)
#g4 = sns.heatmap(Hz, xticklabels=True, yticklabels=True, cmap="YlGnBu",ax=ax4)
#plt.show()