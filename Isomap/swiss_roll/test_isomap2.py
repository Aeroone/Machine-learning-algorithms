#####################################################################
#
# ISOMAP (Isometric Feature Mapping) analysis example on Swiss Roll data.
# =======================================================================
#
# DATASET URL: http://isomap.stanford.edu/datasets.html
# ------------
#
# The following files are used to run the analysis here:
#
# swiss_roll_data.mat: Data coordinates are contained in X_data and Y_data
# variables
# --------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Update the matplotlib package to the newest edition to enable 3-D plot
# Thank you for your suggestions!
#
# @version 1.0
# --------------------------------------
#####################################################################
import scipy.io as sio 
import numpy as np
from scipy.sparse import csc_matrix, csgraph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

matFile = sio.loadmat('swiss_roll_data.mat')

x = matFile['X_data']
x = np.array(x)
x = x[:, np.arange(0, x.shape[1], 10)]

y = matFile['Y_data']
y = np.array(y)
y = y[:, np.arange(0, y.shape[1], 10)]

# number of data points to work with 
m = x.shape[1]

# Plot 3D scatter of the original data points. Rotate the figure using the
# rotate 3D button in the above panel for figure window to see the 3D swiss
# roll.
plt.figure(1)
plt.ion()
ax = Axes3D(plt.gcf())
ax.scatter(x[0,:], x[1,:], x[2,:], s = 18 * np.ones((1, m)), c = y[0,:])
plt.show()
plt.ioff()

raw_input('press any key to continue\n')

## Step 1: Create neigborhood graph 
# Find neighbors of each data point within distance epsilon (e).
# G is the adjacency matrix recording neighbor Euclidean distance 
G1 = np.sum(np.power(x,2),axis = 0).T.reshape(m,1)
G1 = G1.dot(np.ones((1,m)))
G2 = np.sum(np.power(x,2),axis = 0).reshape(1,m)
G2 = np.ones((m, 1)).dot(G2)
G3 = 2 * x.T.dot(x)
G = G1 + G2 - G3
G[G < 0] = 0
G = np.sqrt(G)

e = 0.2 * np.median(G)
G[G > e] = 0

# Get rid of effectively Infinite distance for simplicity
sG = np.sum(G, axis = 0)
idx = np.where(sG != 0)[0]

G = G[idx,:][:,idx]
m = G.shape[0]

## Step 2: Using all pair shortest path algorithm, construct graph distance
#  matrix  
D = csgraph.shortest_path(csc_matrix(G))
D2 = np.power(D, 2) # Using square for inner product
H = np.eye(m) - np.ones((m,1)).dot(np.ones((1,m)))/m # Construct special centring matrix H
Dt = -0.5 * H.dot(D2) # Apply H to both sides of D2
Dt = Dt.dot(H)

## Step 3: Low dim. representation that preserves distance information
k = 10
V, S, U = np.linalg.svd(Dt) # computes the k largest singular values and 
                            # associated singular vectors of distance matrix


# Use the eigenvectors corresponding to the largest eigenvalue as 1st
# coordinate and second larges eignevalue as 2nd coordinate
dim1_new = V[:,0] * np.sqrt(S[0])
dim2_new = V[:,1] * np.sqrt(S[1])

# Plot scatter of the swiss roll dataset in reudced dimensions after isomap
# analysis.
plt.figure(2)
plt.ion()
plt.scatter(-dim1_new, -dim2_new, s= 18 * np.ones((1, 698)), c = y[1,:])
plt.show()
plt.ioff()
plt.show()