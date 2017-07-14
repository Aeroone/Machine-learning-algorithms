############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mmdTestBoot

def kernel_two_sample_test(N, alpha, kernel_param):
    
    mu = 0
    var = 0.2
    # Generate two samples
    x = np.zeros((1, N))
    y = np.zeros((1, N))
    for i in range(0, N):
        # Generate Gaussian random variables
        x[0, i] = np.random.normal(mu, var)
        
        # Generate Laplacian noise
        u = np.random.uniform(0, 1)-0.5
        b = var / np.sqrt(2)
        y[0, i] = mu - b * np.sign(u) * np.log(1 - 2 * np.abs(u))
                
    z = np.concatenate((x, y), axis = 1).squeeze()   
    # show the signal
    plt.figure()
    plt.plot(z)
    plt.hold(True)        
    #plt.show()    
    
    # Setting all the parameters
    params_sig = kernel_param
    params_shuff = 100
    params_bootForce = 1
    # two sample test using B-stats 
    testStat,thresh,pval,params_sig,params_shuff,params_bootForce = \
        mmdTestBoot.mmdTestBoot(x, y, alpha, params_sig,params_shuff,params_bootForce)
    
    print 'threshold:'
    print thresh
    
    print 'The test statistic for Kernel Test is %f' % testStat
    
    testStat_python, pvalue = stats.ttest_ind(x.squeeze(), y.squeeze())
        
    print 'The test statistic for Paired T-Test is %f' % testStat_python
    
    return testStat, testStat_python
        
    
N = 1000
alpha = 0.05
kernel_param = 0.1
testStat_kernel, testStat_python = kernel_two_sample_test(N, alpha, kernel_param)