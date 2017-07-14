############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from scipy import spatial

def viterbi(input_seq, model_p0, model_A, model_obs_mu, model_obs_std):
# Viterbi decoding algorithm
# produces the assignment that maximizes the joint distribution
# p(x_1, x_2, ..., x_T) using dynamic programming
# using the recursion
#  p(i, t+1) = max_j [ p(j, t) A(i, j) ] o(x_{t+1}|i)
#
# where 
#      p(i, t) = max_{s_1,...,s_{t-1}}
#      P(x_1, ..., x_t, h_1=s_1,..,h_{t-1}=s_{t-1},h_t=i)  
# and
#      o(x_{t+1}|i) is the probability of observing x_{t+1} given
# the hidden state is i
#
# input_seq:
#  d_obs x T, observation sequence
# model:
#  a struct that contains current model parameters
#  p0: d_hid x 1, prior distribution
#  A:  d_hid x d_hid, transition probability
#  obs_mu:  d_obs x d_hid, observation/emission Gaussian means
#  obs_std: d_hid x 1, observation/emission Gaussian standard deviations
#
# decoded_seq:
#  1 x T, decoded hidden states

    T = input_seq.shape[1]

    # to avoid underflow, we work with logarithms,
    # and multiplication becomes additions
    logp0 = np.log(model_p0)
    logA = np.log(model_A)
    obs_mu = model_obs_mu
    obs_std = model_obs_std
    d_hid = model_p0.shape[0]
    
    # dynamic programming intermediate results
    p = -np.inf * np.ones((d_hid, T))
    maxp = np.zeros((d_hid, T))
    
    # the first step P(x_1, h_1 = i) = P(h_1=i) * P(x_1|h_1=i)
    # P(x_1|h_1 = i) is a Gaussian distribution
    # log P(x_1|h_1 = i) = - ||x-mu||^2/(2 sigma^2) - log(sigma) - 0.5 log(2 pi)
    logobs = - np.power(spatial.distance.cdist(obs_mu.T, input_seq[:, 0].reshape(1, input_seq[:, 0].shape[0])), 2)
    logobs = logobs / (2 * np.power(obs_std, 2)) - np.log(obs_std)
    p[:, 0] = (logp0 + logobs).squeeze()
    for t in range(0, T - 1):
        # A(i,j), i corresponds to t+1's time and j corresponds to t's time
        # transpose p(:, t) to add on the matching j's indice
        tmp = logA + p[:, t].reshape(1, p[:, t].shape[0])
        # now maximize over j
        p[:, t + 1] = np.max(tmp, axis = 1)
        maxp[:, t + 1] = np.argmax(tmp, axis = 1)
        
        # log emission probability at t+1's time stamp
        logobs = - np.power(spatial.distance.cdist(obs_mu.T, input_seq[:, t + 1].reshape(1, input_seq[:, t + 1].shape[0])), 2)
        logobs = logobs / (2 * np.power(obs_std, 2)) - np.log(obs_std)
        p[:, t + 1] = p[:, t + 1] + logobs.squeeze()
        
    
    # Now maximize the hidden state at T
    max_state = np.argmax(p[:, T - 1])
    decoded_seq = np.zeros((1, T))
    decoded_seq[0, T - 1] = max_state
    
    # start backtracking
    for t in range(T - 2, -1, -1):
        decoded_seq[0, t] = maxp[int(decoded_seq[0, t + 1]), t + 1]
    
    return decoded_seq 