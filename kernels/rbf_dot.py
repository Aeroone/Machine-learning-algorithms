############################################################################
# The codes are based on Python2.7. 
# Please install numpy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
# Radial basis function inner product

# Pattern input format : [pattern1 ; pattern2 ; ...]
# Output : p11*p21 p11*p22 ... ; p12*p21 ...
# Deg is kernel size
import numpy as np

def rbf_dot(patterns1,patterns2,deg):
    
    #Note : patterns are transposed for compatibility with C code.
    
    size1 = patterns1.shape
    size2 = patterns2.shape
    
    # new version
    G = np.sum((patterns1 * patterns1), axis = 1)
    H = np.sum((patterns2 * patterns2), axis = 1)
    
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    
    H = Q + R - 2 * patterns1.dot(patterns2.T)
    H = np.exp(-H / 2 / np.power(deg, 2))
    
    return H