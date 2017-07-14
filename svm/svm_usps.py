######################################################################
#The codes are based on Python2.7. 
#Please install numpy, scipy, matplotlib, cvxopt packages before using.
#Thank you for your suggestions!
#
#@version 1.0
######################################################################
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import show_image
import svm
import warnings
warnings.filterwarnings("ignore")

# read data
load1 = sio.loadmat('usps_all.mat')
data = load1['data'].astype(float)

# numbers
X = np.concatenate((data[:,:,1].T, data[:,:,2].T), axis = 0)
X1 = data[:,:,1].T
X2 = data[:,:,2].T
Y = np.concatenate((np.ones((1100, 1)), -np.ones((1100, 1))), axis = 0)

H = 16
W = 16

# Create a training set
Xtrain = np.concatenate((X[0:int(1100 * 0.8), :], X[1100:1980, :]), axis = 0)
Ytrain = np.concatenate((Y[0:int(1100 * 0.8), :], Y[1100:1980, :]), axis = 0)

# test set
Xtest = np.concatenate((X[int(1100 * 0.8) : 1100, :], X[1980:2200, :]), axis = 0)
Ytest = np.concatenate((Y[int(1100 * 0.8) : 1100, :], Y[1980:2200, :]), axis = 0)

plt.figure()
show_image.show_image_function(Xtest, H, W)
plt.axis('off')
plt.title('Test set')


train_size = Ytrain.shape[0]
test_size = Ytest.shape[0]

Xtrain = Xtrain.astype(float)
Xtest = Xtest.astype(float)

print '--running svm\n'
beta, beta0 = svm.svm_function(Xtrain, Ytrain, 1)


Y_hat_train = np.sign(Xtrain.dot(beta.T) + beta0)
        
precision1 = np.sum( (Ytrain == Y_hat_train) & (Ytrain == 1) ) * 1.0 / np.sum(Ytrain == 1)
precision2 = np.sum( (Ytrain == Y_hat_train) & (Ytrain == -1) ) * 1.0 / np.sum(Ytrain == -1)
precision = ( np.sum( (Ytrain == Y_hat_train) & (Ytrain == 1) ) + \
                np.sum( (Ytrain == Y_hat_train) & (Ytrain == -1) ) ) * 1.0 / \
                Ytrain.shape[0]
print 'train precision on class1 %f\n' % precision1
print 'train precision on class2 %f\n' % precision2
print 'train precision on all %f\n' % precision
                               
                
Y_hat_test = np.sign(Xtest.dot(beta.T) + beta0)
       
precision1 = np.sum( (Ytest == Y_hat_test) & (Ytest == 1) ) * 1.0 / np.sum(Ytest == 1)
precision2 = np.sum( (Ytest == Y_hat_test) & (Ytest == -1) ) * 1.0 / np.sum(Ytest == -1)
precision = ( np.sum((Ytest == Y_hat_test) & (Ytest == 1)) + \
                np.sum((Ytest == Y_hat_test) & (Ytest == -1)) ) * 1.0 / \
                Ytest.shape[0]
print 'test precision on class1 %f\n' % precision1
print 'test precision on class2 %f\n' % precision2
print 'test precision on all %f\n' % precision

m = Xtest.shape[0]
n = Xtest.shape[1]
incorrect_image_estimation = np.zeros((m, n))
ind_disp = (Y_hat_test == 1).squeeze()
incorrect_image_estimation[ind_disp, :] = Xtest[ind_disp, :]

plt.figure()
show_image.show_image_function(incorrect_image_estimation, H, W)
plt.axis('off')
plt.title('Being classified as 2')

incorrect_image_estimation = np.zeros((m, n))
ind_disp = (Y_hat_test == -1).squeeze()
incorrect_image_estimation[ind_disp, :] = Xtest[ind_disp, :]

plt.figure()
show_image.show_image_function(incorrect_image_estimation, H, W)
plt.axis('off')
plt.title('Being classified as 3')

plt.show()