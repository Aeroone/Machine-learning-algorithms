############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from scipy import stats

# Perform inference on the hidden variables given an input sequence
# also known as the forward-backward algorithm
#
# Calculate pairwise conditional distributions, that is,
#   P(H_t, H_{t+1}|X=(x_1,x_2,...,x_T))
#
# input_seq: d_obs x T
# model: a struct of model parameters
# expected_pairwise: d_hid x d_hid x T-1

def inference(input_seq, model_p0, model_A, model_obs_mu, model_obs_std):
        
    d_obs = input_seq.shape[0]
    T = input_seq.shape[1]
    p0 = model_p0
    A = model_A
    d_hid = model_p0.shape[0]
    obs_mu = model_obs_mu
    obs_std = model_obs_std
               
    obs_std = obs_std.T.reshape(1, 1, d_hid)
    obs_std = np.tile(obs_std, (1, d_obs, 1))
        
    alpha = np.zeros((d_hid, T)) # forward variables
    cnorm = np.zeros((1, T)) #  normalization coefficients
    
    obs_likelihood = np.zeros((obs_mu.shape[1], 1))
    obs_likelihood[0, 0] = stats.multivariate_normal.pdf(input_seq[:, 0], \
        mean = obs_mu.T[0, :], cov = np.diag(obs_std[:, :, 0].squeeze()))
    obs_likelihood[1, 0] = stats.multivariate_normal.pdf(input_seq[:, 0], \
        mean = obs_mu.T[1, :], cov = np.diag(obs_std[:, :, 1].squeeze()))
    obs_likelihood[2, 0] = stats.multivariate_normal.pdf(input_seq[:, 0], \
        mean = obs_mu.T[2, :], cov = np.diag(obs_std[:, :, 2].squeeze()))
    
    
    alpha[:, 0] = (p0 * obs_likelihood).squeeze()
    cnorm[0, 0] = np.sum(alpha[:, 0])
    alpha[:, 0] = alpha[:, 0] * 1.0 / cnorm[0, 0]
    for t in range(1, T):
        alpha[:, t] = A.dot(alpha[:, t - 1])
        obs_likelihood = np.zeros((obs_mu.shape[1], 1))
        obs_likelihood[0, 0] = stats.multivariate_normal.pdf(input_seq[:, t], \
            mean = obs_mu.T[0, :], cov = np.diag(obs_std[:, :, 0].squeeze()))
        obs_likelihood[1, 0] = stats.multivariate_normal.pdf(input_seq[:, t], \
            mean = obs_mu.T[1, :], cov = np.diag(obs_std[:, :, 1].squeeze()))
        obs_likelihood[2, 0] = stats.multivariate_normal.pdf(input_seq[:, t], \
            mean = obs_mu.T[2, :], cov = np.diag(obs_std[:, :, 2].squeeze()))
        alpha[:, t] = alpha[:, t] * obs_likelihood.squeeze()
        cnorm[0, t] = np.sum(alpha[:, t])
        alpha[:, t] = alpha[:, t] * 1.0 / cnorm[0, t]
    
    beta = np.zeros((d_hid, T))  # backward variables
    cnorm2 = np.zeros((1, T))
    beta[:, T - 1] = 1
    cnorm2[0, T- 1] = 1
    for t in range(T - 2, -1, -1):
        obs_likelihood = np.zeros((obs_mu.shape[1], 1))
        obs_likelihood[0, 0] = stats.multivariate_normal.pdf(input_seq[:, t + 1], \
            mean = obs_mu.T[0, :], cov = np.diag(obs_std[:, :, 0].squeeze()))
        obs_likelihood[1, 0] = stats.multivariate_normal.pdf(input_seq[:, t + 1], \
            mean = obs_mu.T[1, :], cov = np.diag(obs_std[:, :, 1].squeeze()))
        obs_likelihood[2, 0] = stats.multivariate_normal.pdf(input_seq[:, t + 1], \
            mean = obs_mu.T[2, :], cov = np.diag(obs_std[:, :, 2].squeeze()))
        beta[:, t] = (A.T.dot(beta[:, t + 1].reshape(beta[:, t + 1].shape[0], 1) * obs_likelihood)).squeeze()
        cnorm2[0, t] = np.sum(beta[:, t])
        beta[:, t] = beta[:, t] * 1.0 / cnorm2[0, t]
        
    expected_pairwise = np.zeros((d_hid, d_hid, T-1))
    for t in range(0, T - 1):
        tmp = A * (alpha[:, t].reshape(alpha[:, t].shape[0], 1).T)
        obs_likelihood = np.zeros((obs_mu.shape[1], 1))
        obs_likelihood[0, 0] = stats.multivariate_normal.pdf(input_seq[:, t + 1], \
            mean = obs_mu.T[0, :], cov = np.diag(obs_std[:, :, 0].squeeze()))
        obs_likelihood[1, 0] = stats.multivariate_normal.pdf(input_seq[:, t + 1], \
            mean = obs_mu.T[1, :], cov = np.diag(obs_std[:, :, 1].squeeze()))
        obs_likelihood[2, 0] = stats.multivariate_normal.pdf(input_seq[:, t + 1], \
            mean = obs_mu.T[2, :], cov = np.diag(obs_std[:, :, 2].squeeze()))
        tmp = tmp * (beta[:, t + 1].reshape(beta[:, t + 1].shape[0], 1) * obs_likelihood)
        tmp = tmp * 1.0 / np.sum(np.sum(tmp)) # normalize to be (joint) pairwise probability
        expected_pairwise[:, :, t] = tmp        
            
    return expected_pairwise