import pandas as pd
import numpy as np


class NMF:
    def __init__(self, X, K, iterations):
        self.X = X
        self.X_np = X.values
        self.K = K
        self.iterations = iterations
        self.err = 0

    def initializeWH(self):
        self.W = pd.DataFrame(
            np.random.randint(0, 100, size=(self.X.shape[0], self.K)),
            dtype=float,
            index=list(self.X.index),
        )
        self.H = pd.DataFrame(
            np.random.randint(0, 100, size=(self.K, self.X.shape[1])),
            dtype=float,
            columns=list(self.X),
        )
        self.eps = np.finfo(self.W.values.dtype).eps

    def update(self):
        for run in range(self.iterations):
            # print("Update iteration = %i for rank = %i" % (run, self.K))
            self.H = np.multiply(
                self.H,
                np.divide(
                    np.dot(self.W.T, self.X),
                    np.dot(self.W.T, np.dot(self.W, self.H) + self.eps),
                ),
            )
            self.W = np.multiply(
                self.W,
                np.divide(
                    np.dot(self.X, self.H.T),
                    (np.dot(np.dot(self.W, self.H), self.H.T) + self.eps),
                ),
            )

    def error(self):
        self.err = np.mean(np.mean(np.abs(self.X - np.dot(self.W, self.H))))

    def runNMF(self):
        self.initializeWH()
        self.update()
        self.error()
