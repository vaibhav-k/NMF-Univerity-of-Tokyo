import pandas as pd
import numpy as np
from sklearn import preprocessing


class SparseNMF:
    def __init__(self, X, rank, iterations, beta, sparsity):
        self.X = X
        self.K = rank
        self.iterations = iterations
        self.beta = beta
        self.sparsity = sparsity

    def initializeWH(self):
        self.W = np.random.randint(100, size=(self.X.shape[0], self.K))
        self.H = np.random.randint(0, 100, size=(self.K, self.X.shape[1]))
        self.W = preprocessing.normalize(self.W)
        self.A = self.W.dot(self.H)
        self.one = np.ones((self.X.shape[0], 1))

    def update(self):
        for run in range(self.iterations):
            print("Update iteration = %i for rank = %i" % (run, self.K))
            self.H = np.multiply(
                self.H,
                np.divide(
                    np.dot(
                        preprocessing.normalize(self.W).T,
                        np.multiply(self.X, np.power(self.A, self.beta - 2)),
                    ),
                    np.dot(self.W.T, np.power(self.A, self.beta - 1)) + self.sparsity,
                ),
            )
            self.A = self.W.dot(self.H)
            self.W = np.multiply(
                self.W,
                np.divide(
                    np.dot(
                        np.multiply(np.power(self.A, self.beta - 2), self.X), self.H.T
                    )
                    + np.multiply(
                        preprocessing.normalize(self.W),
                        np.dot(
                            np.dot(self.one, self.one.T),
                            np.multiply(
                                preprocessing.normalize(self.W),
                                np.dot(np.power(self.A, self.beta - 1), self.H.T),
                            ),
                        ),
                    ),
                    np.dot(np.power(self.A, self.beta - 1), self.H.T)
                    + np.multiply(
                        preprocessing.normalize(self.W),
                        np.dot(
                            np.dot(self.one, self.one.T),
                            np.multiply(
                                preprocessing.normalize(self.W),
                                np.dot(
                                    np.multiply(
                                        np.power(self.A, self.beta - 2), self.X
                                    ),
                                    self.H.T,
                                ),
                            ),
                        ),
                    ),
                ),
            )
            self.W = preprocessing.normalize(self.W)
            self.A = self.W.dot(self.H)

    def calc_error(self):
        self.error = np.mean(np.mean(np.abs(self.X - self.A)))

    def runNMF(self):
        self.initializeWH()
        self.update()
        self.calc_error()
