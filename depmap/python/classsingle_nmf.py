'''
Created on 2016/10/19

@author: taiho
'''
import numpy as np
import pandas as pd

class JointNMF_mask(object):
    '''
    Joint NMF
    '''


    def __init__(self, X1, maskX1, rank, maxiter):
        '''
        Constructor
        '''
        self.X1 = X1
        self.maskX1 = maskX1
        self.rank = rank
        self.maxiter = maxiter
        
    def check_nonnegativity(self):
        if(self.X1.min().min()):
            raise Exception('non negativity')
        
        
    def initialize_W_H(self):
        self.W = pd.DataFrame(np.random.rand(self.X1.shape[0], self.rank), index = list(self.X1.index), columns = map(str, range(1, self.rank+1)))
        self.H1 = pd.DataFrame(np.random.rand(self.rank, self.X1.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X1.columns)
        self.X1r_pre = np.dot(self.W, self.H1)
        self.eps = np.finfo(self.W.as_matrix().dtype).eps
    
    def calc_euclidean_multiplicative_update(self):
        self.H1 = np.multiply(self.H1, np.divide(np.dot(self.W.T, np.multiply(self.maskX1, self.X1)), np.dot(self.W.T, np.multiply(self.maskX1, np.dot(self.W, self.H1)+self.eps))))
        self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(np.c_[self.maskX1], np.c_[self.X1]), np.transpose(np.c_[self.H1])), (np.dot(np.multiply(np.c_[self.maskX1], np.dot(self.W, np.c_[self.H1])), np.transpose(np.c_[self.H1]))+self.eps)))
        
    def wrapper_calc_euclidean_multiplicative_update(self):
        for run in range(self.maxiter):
            self.calc_euclidean_multiplicative_update()
            self.calc_distance_of_HW_to_X()
            #self.print_distance_of_HW_to_X(run)       
    
    def calc_distance_of_HW_to_X(self):
        self.X1r = np.dot(self.W, self.H1)
        self.diff = np.sum(np.sum(np.abs(self.X1r_pre-self.X1r)))
        self.X1r_pre = self.X1r
        self.eucl_dist1 = self.calc_euclidean_dist(self.X1, self.X1r)
        self.eucl_dist = self.eucl_dist1
        self.error1 = np.mean(np.mean(np.abs(self.X1-self.X1r)))/np.mean(np.mean(self.X1))
        self.error = self.error1 
        
        
    def print_distance_of_HW_to_X(self, text):
        print("[%s] diff = %f, eucl_dist = %f, error = %f" % (text, self.diff, self.eucl_dist, self.error))
               
         
    def calc_euclidean_dist(self, X, Y):
        dist = np.sum(np.sum(np.power(X-Y, 2)))
        return dist
            
#    def run(self):
        