#######################################################################
#
# EXPECTATION MAXIMIZATION FOR COIN TOSS
# =======================================
#
# Expectation maximization applied to a coin toss example
# Assume you have five observations of 10 coin flips from two coins
# but you dont know from which coin each of the observations is from
#
# The EM algorithm starts by initializing a random prior
# then it calculates the expected log probability distribution over
# the observations, and based on the log probability updates the prior
#
# 1st:  Coin B, {HTTTHHTHTH}, 5H,5T
# 2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
# 3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
# 4th:  Coin B, {HTHTTTHHTT}, 4H,6T
# 5th:  Coin A, {THHHTHHHTH}, 7H,3T#
#
# From MLE: pA(heads) = 0.80 and pB(heads)=0.45
#######################################################################
#The codes are based on Python2.7. 
#Please install numpy, scipy packages before using.
#Thank you for your suggestions!
#
#@version 1.0
#######################################################################
import numpy as np
import scipy
from scipy import misc

# function for returning log likelihood of multinomial data 
def pll_multinomial(obs, param):

    data_fact = 0
    data_log_prob = 0
    col = obs.shape[0]
    
    for k in range(0, col):
        data_fact = data_fact + np.log(misc.factorial(obs[k]))
        data_log_prob = data_log_prob + obs[k] * np.log(param[k])
    
    pll = np.log(misc.factorial(np.sum(obs))) - data_fact + data_log_prob
    return pll

head_count = np.array([5,9,8,4,7])
tail_count = 10 - head_count
len_experiments = 5
head_count = head_count.reshape(1, head_count.shape[0])
tail_count = tail_count.reshape(1, tail_count.shape[0])
observations = np.concatenate((head_count.T, tail_count.T), axis = 1)

iterno = 1000
# Initializing the priors
coin1_prior = np.zeros((1, iterno + 1))
coin2_prior = np.zeros((1, iterno + 1))

coin1_prior[0,0] = 0.6
coin2_prior[0,0] = 0.5

# EM - Algorithm
for i in range(0, iterno):
    expectation_coin1 = np.zeros((5,2))
    expectation_coin2 = np.zeros((5,2))
    # E_step
    for j in range(0, len_experiments):
        temp = observations[j,:]
        # Update log likelihood
        coin1_pll = pll_multinomial(temp, np.array([coin1_prior[0,i], 1-coin1_prior[0,i]]))
        coin2_pll = pll_multinomial(temp, np.array([coin2_prior[0,i], 1-coin2_prior[0,i]]))
    
        # Update the weights
        weight_coin1 = np.exp(coin1_pll)/(np.exp(coin1_pll)+np.exp(coin2_pll))
        weight_coin2 = np.exp(coin2_pll)/(np.exp(coin1_pll)+np.exp(coin2_pll))
    
        expectation_coin1[j,:] = weight_coin1 * observations[j,:]
        expectation_coin2[j,:] = weight_coin2 * observations[j,:]
    
    # M_step
    coin1_prior[0,i+1] = np.sum(expectation_coin1[:,0])/np.sum(expectation_coin1)
    coin2_prior[0,i+1] = np.sum(expectation_coin2[:,0])/np.sum(expectation_coin2)

print 'observations: ' 
print observations
print 'coin1_prior: %f \n' % coin1_prior[0,iterno]
print 'coin2_prior: %f \n' % coin2_prior[0,iterno]