############################################################################
# The codes are based on Python2.7. 
# Please install numpy package before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
    
## predict for linear/ polynomial regression
def fitreg(Xtrain, Xtest, Y):
    
    regparam = np.power(0.1, 10)
    tmp = Xtrain.dot(Xtrain.T) + regparam * np.eye(Xtrain.shape[0])
    theta = np.linalg.inv(tmp).dot(Xtrain.dot(Y))
    pred = Xtest.T.dot(theta)    
    return pred, theta