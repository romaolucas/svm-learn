import numpy as np
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
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
        y = self._b
        S = self.support_vectors()
        for m in S:
            y += self._multipliers[m]*self._t[m]*self._kernel.apply(self._X[m], x)
        return y
    
    def gram_matrix(self):
        '''
        Computes the gram matrix. The matrix is 
        such that K(x, y) = kernel(x, y)
        '''
        return np.array([[self._kernel.apply(x, y) for x in self._X] for y in self._X])

    def support_vectors(self):
        '''
        returns a set of indexes for the support vectors, that is
        the multipliers non-zero value
        '''
        S = []
        for index, multiplier in enumerate(self._multipliers):
            if multiplier > 0:
                S.append(index)
        return S

    def multipliers_box_constraint(self):
        '''
        returns a set of indexes of the multipliers that satisfy
        0 < multiplier < C
        '''
        M = []
        for index, multiplier in enumerate(self._multipliers):
            if multiplier > 0 and multiplier < self._C:
                M.append(index)
        return M

    def calculate_b(self):
        '''
        Calculates the b, such that y = w'x + b
        b = 1/N^{M} \sum_{n \in M} (t_n - \sum_{m \in S} multiplier_m*t_m*k(x_n, x_m)
        S = set of support vectors, i.e. multiplier > 0
        M = set of multipliers with 0 < multiplier < C
        '''
        M = self.multipliers_box_constraint()
        S = self.support_vectors()
        K = self.gram_matrix()
        b = 0
        for n in M:
            partial_sum = 0
            for m in S:
                partial_sum += self._multipliers[m]*self._t[m]*K[n][m]
            b += self._t[n] - partial_sum
        b = b/len(M)
        return b

    def fit(self):
        '''
        Computes the lagrange multipliers solving the quadric problem
        G is the matrix representing the box constraints
        A is the matrix representing \sum_{n} multipliers_n*t_n = 0
        P is the matrix with entrances t_i*t_j*k(x_i, x_j)
        '''
        K = self.gram_matrix()
        T = np.outer(self._t, self._t)
        P = matrix(T * K)
        q = matrix(np.ones(self._N))
        G_C = matrix(np.eye(self._N))
        G_zero = matrix(np.diag((-1)*np.ones(self._N)))
        h_C = matrix(np.ones(self._N) * self._C)
        h_zero = matrix(np.zeros(self._N))
        G = matrix(np.vstack((G_C, G_zero)))
        h = matrix(np.vstack((h_C, h_zero)))
        A = matrix(self._t.astype(np.double), (1, self._N))
        b = matrix(0.0)
        solution = solvers.qp(P, q, G, h, A, b)
        self._multipliers = np.ravel(solution['x'])
        print("multipliers ", self._multipliers)
        self._b = self.calculate_b()

    def train(self, X):
        '''
        classify a set of inputs in X using the formula
        y(x) = b + \sum_{i \in S} t[i]*multipliers[i]*k(x, x[i])
        with S being the set of the support vectors indexes
        the label for x is the signal of y(x)
        '''
        S = self.support_vectors()
        y = []
        for x in X:
            y_x = self._b
            for m in S:
                y_x += self._multipliers[m]*self._t[m]*self._kernel.apply(x, self._X[m])
            y.append(1 if y_x >= 0 else -1)
        return y

    def compute_accuracy(self, X, t):
        y = self.train(X)
        correctly_classified = 0
        for y_i, t_i in zip(y, t):
            if y_i == t_i:
                correctly_classified += 1
        return correctly_classified / len(t)

def main():
    import csv
    X = []
    t = []
    with open('clean2.data', mode='r') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        for row in data_reader:
            X.append(row[0:166])
            t.append(row[-1])
    X = np.array(X, dtype=float)
    t = np.array(t, dtype=float)
    for index, lbl in enumerate(t):
        if lbl == 0:
            t[index] = -1
    X_train, X_test, t_train, t_test = model_selection.train_test_split(X, t)
    print("X train shape:", X_train.shape)
    svm = SVMLearn(X_train, t_train)
    svm.fit()
    print("finished training")
    print("Accuracy: ", svm.compute_accuracy(X_test, t_test))

if __name__ == '__main__':
    main()
