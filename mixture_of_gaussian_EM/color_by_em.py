#######################################################################
#
# EXPECTATION MAXIMIZATION  applied to an image.
# Credits to Mohammad Asif Khan for optimized EM
#
#######################################################################
#The codes are based on Python2.7. 
#Please install numpy, scipy packages before using.
#Thank you for your suggestions!
#
#@version 1.0
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans

# function for gaussian distribution
def gauss_dist(x,mu,sigma):
    
    exp_term = np.exp(-(np.power(x - mu, 2)/(2 * sigma)))
    y = (1 / np.sqrt(2 * np.pi * sigma)) * exp_term
        
    return y

# Perform GMM based expectation maximization algorithm  
# initialization be randomization
def EM(data, k):
    C = kmeans(data.T, k)
    C = C[0]
    
    wt = (1.0/k) * np.ones((k, 1))
    mu = C
    sigma = np.array([[np.var(data)],[np.var(data)]])
    
    
    iterno = 200
    log_likelihood = np.zeros((1, iterno))
    log_likelihood_old=0
    for i in range(0, iterno):
        # E_step
        numerator = np.zeros((1, data.shape[0]))
        num_gauss = gauss_dist(data, mu, sigma)
        numerator = wt.T.dot(num_gauss)
        
        log_likelihood[0, i] = np.sum(np.log(numerator))
        gamma = (wt * num_gauss) / numerator
        
        # M Step
        temp = np.sum(gamma, axis = 1)
        temp = temp.reshape(temp.shape[0], 1)        
        mu = gamma.dot(data.T) / temp
        
        sigma = np.sum((gamma * np.power(data - mu, 2)).T, axis = 0) / np.sum(gamma.T, axis = 0) 
        sigma = sigma.reshape(sigma.shape[0], 1)
        
        wt = np.sum(gamma.T, axis = 0) / data.shape[1]
        wt = wt.reshape(wt.shape[0], 1)
       
        if (np.abs( log_likelihood_old - log_likelihood[0, i]) < np.power(10.0, -5)):
            break
        else:
            log_likelihood_old = log_likelihood[0, i]       
    
    plt.figure(1)
    plt.plot(log_likelihood[0,0:i])
    plt.xlabel('Iteration')
    plt.ylabel('Observed Data Log-likelihood')
    return mu, sigma, wt

# To perform this operation on an image  
im_1 = plt.imread('tire.tif')
imc_1 = im_1.astype(float)
U = imc_1.shape[0]
V = imc_1.shape[1]

data = imc_1.reshape(1, U * V)

k = 2 # Number of classes
[mu, sigma, wt1] = EM(data, k)

# To calculate probabilities for data with respect to all classes 
# using optimized EM
y1 = gauss_dist(data, mu, sigma)
prob_new = np.max(y1, axis = 0).reshape(U, V)
# Labeling the elements of data into classes on the basis of 
# maximum probabilty
label = prob_new - y1[0,:].reshape(U, V)
label[label != 0] = 2
label[label == 0] = 1
plt.figure(2)
plt.imshow(im_1, cmap='Greys_r')
plt.figure(3)
plt.imshow(label)
plt.show()