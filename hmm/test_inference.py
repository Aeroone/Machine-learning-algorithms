############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import scipy.io as sio
import inference

# test inference
load = sio.loadmat('synthetic_data.mat')

# Viterbi decoding
model_A = load['A']
model_p0 = load['p0']
model_obs_mu = load['obs_mu']
model_obs_std = load['obs_std']
obs_sequences = load['obs_sequences']
hid_sequences = load['hid_sequences']

N = obs_sequences.shape[1]

expected_pairwise = inference.inference(obs_sequences[0][0], model_p0, model_A, model_obs_mu, model_obs_std)