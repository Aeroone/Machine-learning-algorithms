############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import fitreg
import fitkernel

def cross_validation(X, Y, flag, lam, r):
# flag == 1: linear or polynomial regression
# flag == 2: kernel rigde regression    
    m = X.shape[1]
    fold = m
    avgerr = np.zeros((1, fold))
    for l in range(0, fold):
        
        testindex = np.arange(l, m, fold)
        totalindex = np.arange(0, m)
        trainindex = np.setdiff1d(totalindex,testindex)     
        
        Xtrain = X[:, trainindex]
        Ytrain = Y[trainindex]
        Xtest = X[:, testindex]
        
        if flag == 1:
            predtest = fitreg.fitreg(Xtrain, Xtest, Ytrain)
        elif flag == 2:
            predtest1 = fitkernel.fitkernel(Xtrain, Xtest, Ytrain, lam, r)
            predtest = predtest1.T
        
        avgerr[0, l] = np.mean(np.power(predtest - Y[testindex], 2))
    avgerr = np.mean(avgerr)
    return avgerr  