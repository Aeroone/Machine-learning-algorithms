######################################################################
#The codes are based on Python2.7. 
#Please install numpy, cvxopt packages before using.
#Thank you for your suggestions!
#
#@version 1.0
######################################################################
import numpy as np
from cvxopt import matrix, solvers

def svm_function(X, Y, C):
    
    # number of samples
    N = X.shape[0]
    # get NxN kernel matrix (here is linear kernel)
    K = X.dot(X.T)
    
    # used for quadratic term in obj function
    G = (Y.dot(Y.T)) * K
    # used for linear term in obj function
    b = - np.ones((N, 1))
    
    # lower bound of alpha
    lb = np.zeros((N, 1))
    # upper bound of alpha
    ub = C * np.ones((N, 1))
           
    P = matrix(G)
    q = matrix(b)
    Gm = matrix(np.vstack([-np.eye(G.shape[0]), np.eye(G.shape[0])]))
    h = matrix(np.vstack([-lb, ub]))
    A = matrix(Y.T)
    b = matrix(0.0)
    
    # quadratic optimization
    # solve the dual problem
    opt = solvers.qp(P,q,Gm,h,A,b)
    alpha = np.array(opt['x']).squeeze()
    
    # find out the support vector indexes, i.e., the vectors whose alpha is
    # larger than zero
    sv = np.where(alpha > np.power(0.1, 10))[0]
    
    # Calculate the decision bound
    # beta (i.e., w) is a linear combination of support vectors
    beta = np.sum((alpha[sv].reshape(alpha[sv].shape[0], 1) * Y[sv]) * X[sv, :], axis = 0)
    beta = beta.reshape(1, beta.shape[0])
    
    # beta0 (i.e., bias) is calculated using KKT condition
    beta0 = np.sum(alpha[sv].reshape(alpha[sv].shape[0], 1).T * Y[sv].T * K[sv, :][:, sv], axis = 1)
    beta0 = beta0.reshape(beta0.shape[0], 1)
    beta0 = np.mean(Y[sv] - beta0)
    
    return beta, beta0