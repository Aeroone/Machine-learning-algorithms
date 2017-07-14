############################################################################
# The codes are based on Python2.7. 
# Please install numpy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def eval_stump(stump_ind, stump_x0, stump_s, X):
    h  = np.sign( stump_s * (X[:, stump_ind] - stump_x0) )
    return h
