############################################################################
# Multinomial logistic regression 
############################################################################
# The codes are based on Python2.7. 
# Please install numpy package before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np

def logistic_regression(X, Y):
    # X is n x d 
    # Y is n x k
    # B is d x k
    n = X.shape[0]
    d = X.shape[1]
    k = Y.shape[1]
    B = np.zeros((d, k))
    max_iter = 200
    step_size = 0.1
    
    for i in range(0, max_iter):
        pred = softmax_fn(X.dot(B))
        err = Y - pred 
        grad = X.T.dot(err) / n
        B = B + step_size * grad         
            
    return B

# Numerically stable softmax
def softmax_fn(y):
    
    max_y = y.max(axis = 1)
    max_y = max_y.reshape(max_y.shape[0], 1)
    ny = np.exp(y - max_y)
    tmp = np.sum(ny, axis = 1)
    tmp = tmp.reshape(tmp.shape[0], 1)
    p = ny / tmp
        
    if(np.any(np.isnan(p))):
        raise ValueError('Logistic Regression - Run into NaN!')
    
    return p