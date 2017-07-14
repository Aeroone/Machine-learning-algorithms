############################################################################
# Outlier detection of usps digits.
############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib, cvxopt packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

matFile = sio.loadmat('usps_all.mat')
data = matFile['data']

n = data.shape[1]

# Only work with a subset of digit six
# orig_data is n x d, where n is the number of data points
# and d is the dimension.
subsample_n = 400
orig_data = data[:, np.random.choice(n, subsample_n, replace=False), 5].T
orig_data = orig_data.astype(float) / 255

mydata = orig_data
X = mydata

G = X.dot(X.T)
b = - np.diag(G)

C = 0.1
E = np.ones((1, mydata.shape[0]))
lb = np.zeros((X.shape[0], 1))
ub = C * np.ones((X.shape[0], 1))

# min 0.5*x'*H*x + f'*x   subject to:  A*x <= b
P = matrix(2 * G)
q = matrix(b)
G = matrix(np.vstack([-np.eye(G.shape[0]), np.eye(G.shape[0])]))
h = matrix(np.vstack([-lb, ub]))
A = matrix(E)
b = matrix(1.0)

sol = solvers.qp(P,q,G,h,A,b)
alpha = np.array(sol['x']).squeeze()

idx_support_vectors = np.where((alpha > np.finfo(float).eps) & (alpha < C))[0]

center = X.T.dot(alpha.reshape(alpha.shape[0], 1)).squeeze()
R = np.linalg.norm(X[idx_support_vectors[0], :] - center)

center = center.reshape(center.shape[0], 1)
x_dist = X.T - center
x_dist = np.sum(x_dist * x_dist, axis = 0)

sorted_x_dist = np.argsort(x_dist)[::-1]

n_display = 8
# draw normal images
for i in range(0, n_display):
    plt.subplot(2,n_display,i+1)
    plt.axis('off')
    plt.imshow( orig_data[sorted_x_dist[subsample_n-i-1], :].reshape(16, 16).T, cmap='gray')

# draw abnormal images
for i in range(0, n_display):
    plt.subplot(2,n_display,i+1+n_display)
    plt.axis('off')
    plt.imshow( orig_data[sorted_x_dist[i], :].reshape(16, 16).T, cmap='gray')
    
plt.show()