############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from scipy import spatial

## predict for kernel ridge regression
def fitkernel(Xtrain, Xtest, Ytrain, lam, r):
    
    dismat = np.power(spatial.distance.pdist(Xtrain.T), 2)
    s0 = np.median(dismat)
    sigma = s0 * r
    K = np.exp(-spatial.distance.squareform(dismat) * 1.0 / (2 * sigma))
    dismattest = spatial.distance.cdist(Xtrain.T, Xtest.T)
    Ktest = np.exp(-np.power(dismattest, 2) / (2 * sigma))
    pred = Ytrain.T.dot(np.linalg.inv(K + lam * np.eye(K.shape[0])).dot(Ktest))
    return pred