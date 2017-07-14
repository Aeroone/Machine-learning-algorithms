############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import kernel_svm

def kernel_circular_data(C, r):
    
    # generate two gaussians with same center, but different covariance
    mean = np.array([0,0])
    cov1 = np.array([[0.2, 0], [0, 0.2]])
    cov2 = np.array([[0.8, 0], [0, 0.8]])
    
    # generate 200 samples for first gaussian
    count = 0
    dist1 = np.zeros((200, 2))
    label1 = np.ones((200, 1))
    while count < 200:
        # using matlab's multivariate gaussian random generator
        temp = np.random.multivariate_normal(mean, cov1)
        if np.linalg.norm(temp) < 0.5:
            dist1[count, :] = temp
            count = count + 1
    
    # generate 200 samples for second gaussian
    count = 0
    dist2 = np.zeros((200, 2))
    label2 = -1 * np.ones((200, 1))
    while count < 200:
        temp = np.random.multivariate_normal(mean, cov2)
        # to be more distinct, we make another class distributed on the
        # outer side of the first class        
        if np.linalg.norm(temp) > 0.7 and np.linalg.norm(temp) < 1:
            dist2[count, :] = temp
            count = count + 1
    
    plt.figure()
    plt.scatter(dist1[:, 0], dist1[:, 1], color='blue')
    plt.hold(True)        
    plt.scatter(dist2[:, 0], dist2[:, 1], color='red')        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Set plotted')       
        
    X = np.concatenate((dist1, dist2), axis = 0)
    Y = np.concatenate((label1, label2), axis = 0)
           
    # randomly select 300 samples for training, and the rest for test
    p = np.random.permutation(400)
    X_train = X[p[0:300], :]
    X_test = X[p[300:400], :]
    Y_train = Y[p[0:300], :]
    Y_test = Y[p[300:400], :]
    
    alpha, sv, beta, sigma = kernel_svm.kernel_svm(X_train, Y_train, C, r)
    
    # Establish the meshgrid of test points
    test_x = np.linspace(-1, 1, 20)
    test_x = test_x.reshape(test_x.shape[0], 1)
    test_y = np.linspace(-1, 1, 20)
    test_y = test_y.reshape(test_y.shape[0], 1)
    
    # Compute the values
    Z = np.zeros((20, 20))
    for i in range(0, 20):
        for j in range(0, 20):
            temp = np.array([test_x[i, 0], test_y[j, 0]])
            temp = temp.reshape(1, temp.shape[0])
            Ktest = - 1.0 * np.power(spatial.distance.cdist(X_train[sv, :], temp), 2) / (2 * sigma * r)
            Ktest = np.exp(Ktest)
            Z[i, j] = np.sum(alpha[sv].reshape(alpha[sv].shape[0], 1) * Y_train[sv, :] * Ktest, axis = 0)[0]

    gridx, gridy = np.meshgrid(test_x.T, test_y.T)
    plt.figure()
    pos_class = np.where(Y_train == 1)
    plt.scatter(X_train[pos_class, 0], X_train[pos_class, 1], color = 'blue')
    plt.hold(True)        
    neg_class = np.where(Y_train == -1)
    plt.scatter(X_train[neg_class, 0], X_train[neg_class, 1], color = 'red')
    plt.hold(True)  
    plt.contour(gridx, gridy, Z, 20)       
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian kernel on a circular dataset')       
    plt.show()
            
C = 10
r = 1           
kernel_circular_data(C, r)         