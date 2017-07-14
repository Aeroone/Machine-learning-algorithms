############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, cvxopt packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from cvxopt import matrix, solvers
from scipy import spatial

def kernel_svm(X, Y, C, r):
    
    N = X.shape[0]
   
    ## Gaussian Kernel
    
    # K(x,y) = exp(-(x-y)^2/(2*sigma^2))
    # K is a symmetric Kernel
    tmpdist2 = np.power(spatial.distance.pdist(X), 2)
    # median trick for selecting kernel bandwidth 
    sigma2 = np.median(tmpdist2)
    
    
    K = np.exp(-spatial.distance.squareform(tmpdist2) * 1.0 / (2 * sigma2 * r))   
    
    G = (Y.dot(Y.T)) * K
    b = - np.ones((N, 1))
    
    # lower bound of dual variables (alpha)
    lb = np.zeros((N, 1))
    # upper bound of dual variables (alpha)
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
    sv = np.where((alpha > np.power(0.1, 10)) & (alpha < C))[0]
    
    # Calculate the decision bound
    beta0 = np.sum(alpha[sv].reshape(alpha[sv].shape[0], 1).T * Y[sv].T * K[sv, :][:, sv], axis = 1)
    beta0 = beta0.reshape(beta0.shape[0], 1)
    beta0 = np.mean(Y[sv] - beta0)
        
    return alpha, sv, beta0, sigma2