#############################################################
#
# ISOMAP and PCA codes used to generate input files for visualiztion.
# =============================================================
#
# DATASET URL: http://isomap.stanford.edu/datasets.html
# ------------
#
# Note: Please refer to file usps_isomap_pca_visualize.m to generate the
# visualization for the results obtaine dby code in this file.
#
# This file contains the code for PCA and ISOMAP analysis. It proivdes 
# results to be used in usps_isomap_pca_visualize.m
#
# The following files are used to run the analysis:
#
# USPS_2digits.mat: Original USPS data samples
#
# OUTPUT FILES
# ============
# Note: All the below files are not used by visualization code but they are
# created with a 'NEW' added in end of their name to exemplify run of this 
# code.
#
# Vx_USPS_new.mat: Created by PCA code. Similar to Vx_USPS.mat.  
# dim1_usps_new.mat: Created by Isomap code. Similar to dim1_usps.mat.
# dim2_usps_new.mat: Created by Isomap code. Similar to dim2_usps.mat. 
# --------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib, h5py packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# -------------------------------------- 
#############################################################

# This is how we extract digits 2 from original dataset.
# This is not runnable here as original files are missing.
import h5py
import scipy.io as sio 
import numpy as np
from scipy.sparse import csc_matrix, csgraph
import math

matFile = h5py.File('USPS_2digits.mat')
# Original data samples stored in variable 'xx'.

xx = matFile['xx']
xx = np.array(xx)
xx = xx.T

## Isomap code
m = xx.shape[1] # number of data points to work with

## Step 1: Create neigborhood graph 
# Find neighbors of each data point within distance epsilon (e).
# G is the adjacency matrix recording neighbor Euclidean distance
G1 = np.sum(np.power(xx,2),axis = 0).T.reshape(m,1)
G1 = G1.dot(np.ones((1,m)))
G2 = np.sum(np.power(xx,2),axis = 0).reshape(1,m)
G2 = np.ones((m, 1)).dot(G2)
G3 = 2 * xx.T.dot(xx)
G = G1 + G2 - G3
G[G < 0] = 0
G = np.sqrt(G)

e = 0.8 * np.median(G)
G[G > e] = 0

# Get rid of effectively Infinite distance for simplicity
sG = np.sum(G, axis = 0)
idx = np.where(sG != 0)[0]

G = G[idx,:][:,idx]
m = G.shape[0]

## Step 2: Using all pair shortest path algorithm, construct graph distance 
# matrix
D = csgraph.shortest_path(csc_matrix(G))

D[D > math.pow(10,20)] = 0
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

# Save new representations to file for use in visualization
sio.savemat('dim1_USPS_new.mat', {'dim1_new':dim1_new})
sio.savemat('dim2_USPS_new.mat', {'dim2_new':dim2_new})

# End Isomap

## PCA code

center = np.mean(xx, axis = 1) # Mean of dataset
sz = xx.shape[1] # size of dataset
x = xx - center.reshape(center.shape[0],1).dot( np.ones((1,sz)))
# subtract the mean of dataset

covariance = np.cov(x) # Compute the covariance
# computes eigenvectors of covariance
S, V = np.linalg.eig(covariance)
sortidx = S.argsort()[::-1] 
V = V[:,sortidx][:,0:2]

# Project x to the principal direction (top 2 principal components)
Vx_new = V.T.dot(x)
# Save new representation to file for use in visualization
sio.savemat('Vx_USPS_new.mat', {'Vx_new':Vx_new})
# End PCA 