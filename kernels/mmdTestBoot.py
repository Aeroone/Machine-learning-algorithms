############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
# This function implements the MMD two-sample test using a bootstrap
# approach to compute the test threshold.

# Inputs: 
# X contains dx columns, m rows. Each row is an i.i.d sample
# Y contains dy columns, m rows. Each row is an i.i.d sample
# alpha is the level of the test
# params.sig is kernel size. If -1, use median distance heuristic.
# params.shuff is number of bootstrap shuffles used to
#        estimate null CDF
# params.bootForce: if this is 1, do bootstrap, otherwise
#        look for previously saved threshold


# Outputs: 
#        thresh: test threshold for level alpha test
#        testStat: test statistic: m * MMD_b (biased)

import numpy as np
import rbf_dot
import scipy.io as sio 

def mmdTestBoot(X,Y,alpha,params_sig,params_shuff,params_bootForce):
    
    m = X.shape[0]
    
    # Set kernel size to median distance between points in aggregate sample
    if params_sig == -1:
        Z = np.concatenate((X, Y), axis = 0) # aggregate the sample
        size1 = Z.shape[0]
        if size1 > 100:
            Zmed = Z[0:100, :]
            size1 = 100
        else:
            Zmed = Zmed
        
        G = np.sum(Zmed * Zmed, axis = 1)
        Q = np.tile(G, (1, size1))
        R = np.tile(G.T, (size1, 1))
        dists = Q + R - 2 * Zmed.dot(Zmed.T)
        dists = dists - np.tril(dists)
        dists = dists.T.reshape(np.power(size1, 2), 1)
        params_sig = np.sqrt(0.5 * np.median(dists[dists>0]))
        # rbf_dot has factor two in kernel
        
    K = rbf_dot.rbf_dot(X, X, params_sig)
    L = rbf_dot.rbf_dot(Y, Y, params_sig)
    KL = rbf_dot.rbf_dot(X, Y, params_sig)
    
    # MMD statistic. Here we use biased 
    # v-statistic. NOTE: this is m * MMD_b
    testStat = 1.0 / m * np.sum(np.sum(K + L - KL - KL.T))
        
    ################################################################
    threshFileName = 'mmdTestTresh' + str(m)
        
    Kz1 = np.concatenate((K, KL), axis = 1)
    Kz2 = np.concatenate((KL.T, L), axis = 1)
    Kz = np.concatenate((Kz1, Kz2), axis = 0)
    
    if params_bootForce == 1:
        
        MMDarr = np.zeros((params_shuff, 1))
        for whichSh in range(0, params_shuff):
            indShuff = np.random.permutation(2 * m)
            
            KzShuff = Kz[indShuff, :][:, indShuff]
            K = KzShuff[0:m, 0:m]
            L = KzShuff[m:2*m, m:2*m]
            KL = KzShuff[0:m, m:2*m]
            
            MMDarr[whichSh, 0] = 1.0 / m * np.sum(np.sum(K + L - KL - KL.T))
            
        MMDarr = np.sort(MMDarr.squeeze())
        thresh = MMDarr[int(np.round((1 - alpha) * params_shuff))]
        
        tmp = np.array([np.sum(np.where(MMDarr > testStat)[0]) * 1.0 / params_shuff, 1.0 / params_shuff]) 
        pval = np.max(tmp)
        
        #sio.savemat(threshFileName, 'thresh')
        #sio.savemat(threshFileName, 'MMDarr')
    else:
        load = sio.loadmat(threshFileName)
 
    return testStat,thresh,pval,params_sig,params_shuff,params_bootForce 