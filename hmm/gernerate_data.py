############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import scipy.sparse
import scipy.io as sio

# generate data
d_hid = 3 # hidden state dimension
d_obs = 4 # observation state dimension

np.random.seed (592404) # fix an arbitrary random seed
p0 = np.random.rand(d_hid, 1)
p0 = p0 * 1.0 / np.sum(p0) # prior distribution

# create a strong transition dependence
rand_perm = np.random.permutation(d_hid)
A = scipy.sparse.csc_matrix( (np.ones((1,d_hid)).squeeze(),(np.arange(0, d_hid), rand_perm)) )
A = np.array(A.todense()) + 1.0 * np.random.rand(d_hid, d_hid)

# a_{ij} = P(next = i | current = j)
A = A / np.sum(A, axis = 0) # transition probability table

# observation models are Gaussian
# there are a total of d_hid Gaussian distributions
obs_mu = np.random.randn(d_obs, d_hid) # means; each column is for a hidden state
# standard deviations; we assume the covariances are scaled identity, so only scalars are needed.
obs_std = np.abs(np.random.randn(d_hid, 1))

T = 100 # sequence length
N = 600 # number of such sequences

hid_sequences = np.empty((1, N), dtype=object) # hidden state sequences
obs_sequences = np.empty((1, N), dtype=object) # observation sequences
for i in range(0, N):
    if i % 40 == 0:
        print 'Generated %i / %i sequences\n' % (i + 40, N) 
    
    sequence = np.zeros((d_obs, T))
    hid_sequence = np.zeros((1, T))
    # sample the first hidden state according to the prior distribution
    hid_state = np.where(np.random.multinomial(1, p0.squeeze(), size=1)[0] > 0)[0][0]
    
    hid_sequence[0, 0] = hid_state
    # sample an observation state according to a Gaussian
    # (indexed by hid_state)   
    obs_state = obs_std[hid_state, 0] * np.random.randn(d_obs, 1) + obs_mu[:, hid_state].reshape(obs_mu[:, hid_state].shape[0], 1)
    sequence[:, 0] = obs_state.squeeze()
    for t in range(1, T):
        # sample the hidden state according to transition probability
        hid_state = np.where(np.random.multinomial(1, A[:, hid_state].squeeze(), size=1)[0] > 0)[0][0]
        hid_sequence[0, t] = hid_state
        # sample an observation state
        obs_state = obs_std[hid_state, 0] * np.random.randn(d_obs, 1) + obs_mu[:, hid_state].reshape(obs_mu[:, hid_state].shape[0], 1)
        sequence[:, t] = obs_state.squeeze()
    
    hid_sequences[0][i] = hid_sequence + 1
    obs_sequences[0][i] = sequence

sio.savemat('synthetic_data_new.mat', {'p0': p0, 'A': A, 'obs_mu': obs_mu, \
                                       'obs_std': obs_std, 'hid_sequences': hid_sequences, \
                                       'obs_sequences': obs_sequences})