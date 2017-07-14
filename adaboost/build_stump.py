############################################################################
# The codes are based on Python2.7. 
# Please install numpy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
# --------------------------------------
# build a stump from each component and return the best
import numpy as np

def build_stump(X, y, w):
    
    d = X.shape[1]
    w = w / np.sum(w) # normalized the weights (if not already)
    
    werr = np.zeros((1, d)) 
    stump_ind_array = np.zeros((1, d))
    stump_x0_array = np.zeros((1, d))
    stump_s_array = np.zeros((1, d))
    for i in range(0, d):
        stump_i_werr, stump_i_x0, stump_i_s = build_onedim_stump(X[:, i], y, w)
        stump_ind_array[0, i] = i
        stump_x0_array[0, i] = stump_i_x0
        stump_s_array[0, i] = stump_i_s
        werr[0, i] = stump_i_werr
    
    ind = np.argmin(werr.squeeze()) # return the best stump 
                
    return stump_ind_array[0, ind], stump_x0_array[0, ind], stump_s_array[0, ind], werr[0, ind]
# --------------------------------------
# build a stump from a single input component
def build_onedim_stump(x, y, w):
    
    xsorted = np.sort(x) # ascending 
    I = np.argsort(x)
    Ir = I[::-1] # descending

    score_left = np.cumsum(w[0, I] * y[0, I]) # left to right sums
    score_right = np.cumsum(w[0, Ir] * y[0, Ir]) # right to left sums
    
    # score the -1 -> 1 boundary between successive points 
    score = -score_left[0:score_left.shape[0] - 1] + score_right[::-1][1:score_right.shape[0]]
    
    # find distinguishable points (possible boundary locations)
    Idec = np.where(xsorted[0:xsorted.shape[0] - 1] < xsorted[1:xsorted.shape[0]])[0]
    
    # locate the boundary or give up
    if (Idec.shape[0] > 0):
        maxscore = np.max(np.abs(score[Idec])) # maximum weighted agreement
        ind = np.argmax(np.abs(score[Idec]))
        ind = Idec[ind]
        
        stump_werr = 0.5 - 0.5 * maxscore # weighted error
        stump_x0 = (xsorted[ind] + xsorted[ind + 1]) * 1.0 / 2 # threshold
        stump_s = np.sign(score[ind]) # direction of -1 -> 1 change
            
    else:
        stump_werr = 0.5
        stump_x0 = 0
        stump_s = 1
        
    return stump_werr, stump_x0, stump_s