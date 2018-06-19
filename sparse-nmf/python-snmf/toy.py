import numpy as np
import matplotlib.pyplot as plt

w=np.ones((1,10))
h1=np.ones((1,30))
h2=np.ones((1,40))
h3=np.ones((1,50))

W=np.zeros((30,3))
H1=np.zeros((3,90))
H2=np.zeros((3,120))
H3=np.zeros((3,150))

W[0:10,0:1]=w.T
W[10:20,1:2]=w.T
W[20:30,2:3]=w.T

H1[0:1,0:30]=h1
H1[1:2,30:60]=h1
H1[2:3,60:90]=h1

H2[0:1,0:40]=h2
H2[1:2,40:80]=h2
H2[2:3,80:120]=h2

H3[0:1,0:50]=h3
H3[1:2,50:100]=h3
H3[2:3,100:150]=h3

X1=np.dot(W,H1)
X2=np.dot(W,H2)
X3=np.dot(W,H3)
np.random.rand(3,2)

XX1=X1+0.5*np.random.rand(30,90)
XX2=X2+0.5*np.random.rand(30,120)
XX3=X3+0.5*np.random.rand(30,150)

plt.imshow(W, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
plt.imshow(H1, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
