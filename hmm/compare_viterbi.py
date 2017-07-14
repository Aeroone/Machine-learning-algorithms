############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import scipy.io as sio
import viterbi

# test
load = sio.loadmat('synthetic_data.mat')

# Viterbi decoding
model_A = load['A']
model_p0 = load['p0']
model_obs_mu = load['obs_mu']
model_obs_std = load['obs_std']
obs_sequences = load['obs_sequences']
hid_sequences = load['hid_sequences']

N = obs_sequences.shape[1]

viterbi_err = 0
for i in range(0, N):
    decoded_seq = viterbi.viterbi(obs_sequences[0][i], model_p0, model_A, model_obs_mu, model_obs_std)
    diff = np.abs(decoded_seq - (hid_sequences[0][i].astype(float) - 1.0))
    viterbi_err = viterbi_err + np.sum(diff != 0)
viterbi_err = viterbi_err * 1.0 / N
print 'Error for Viterbi decoding %f\n' % viterbi_err

# Marginal decoding: decoding by maximizing
# marginal distribution, that is, ignore transition
# equivalent to uniform transition
model_A = np.ones(model_A.shape)
marg_err = 0
for i in range(0, N):
    decoded_seq = viterbi.viterbi(obs_sequences[0][i], model_p0, model_A, model_obs_mu, model_obs_std)
    diff = np.abs(decoded_seq - (hid_sequences[0][i].astype(float) - 1.0))
    marg_err = marg_err + np.sum(diff != 0)
marg_err = marg_err * 1.0 / N
print 'Error for marginal decoding %f\n' % marg_err  