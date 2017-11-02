import numpy as np
from cvxopt import solvers, matrix
from kernelFactory import KernelFactory

class SVMLearn(object):
    ''' Implementation of SVM for binary cases

    This is the implementation described in Bishop's Machine Learning book.
    It assumes that the labels are already {-1, 1} and that the training data
    is a numpy array.

    After creating an object, it's necessary to call the fit method so that
    the model is computed and then new data can be classified using the predict method.

    Attributes:
        X : training data
        t : traning labels
        kernel : kernel to be used. it's decided by the kernelType argument
        C : constant used to penalize missclassified inputs. If no value is
        given, it is assumed to be 1
        N : training data size
        multipliers : Lagrange multipliers computed by the quadric optmization solver
        b : constant used for calculating y. It's obtained using the multipliers and
        the labels

    '''

    def __init__(self, X, t, kernelType="", C=1, *args, **kwargs):
        '''
        Creates an instance of the SVMLearn
        '''
        self._X = X
        self._t = t
        self._kernel = KernelFactory.create_kernel(kernelType, *args, **kwargs)
        self._C = C
        self._N = X.shape[0]
        self._multipliers = np.zeros(self._N)
        self._b = 0
    
    def compute_y(self, x):
        '''
        Calculates the predicted label y. 
        Y is calculated using the formula
        y = \sum_{i = 0}^N multiplier_i*t_i*K(X_i, x) + b
        '''
        partial_sum = 0
        for i in range(self._N):
            partial_sum += self._multipliers[i]*self._t[i]*self._kernel.apply(self._X[i], x)
        return partial_sum + b
    
    def gram_matrix(self):
        '''
        Computes the gram matrix. The matrix is 
        such that K(x, y) = kernel(x, y)
        '''
        return np.array([[self._kernel(x, y) for x in self._X] for y in self._X])

    def fit(self):
        K = self.gram_matrix()
        T = np.outer(t, t)
        P = matrix(np.dot(-1*K, T))
        q = matrix(np.ones(self._N))
        G_C = np.eye(self._N)
        G_zero = np.diag((-1)*np.ones(self._N))
        h_C = np.ones(self._N) * self._C
        h_zero = np
        G = matrix(np.vstack((G_zero, G_C)))
        h = matrix(np.vstack((h_C, h_zero)))
