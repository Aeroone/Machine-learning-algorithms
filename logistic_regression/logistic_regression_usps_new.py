############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
# This demo shows how to build logistic regression model to classify usps hand-writing digits.
# 1. Implement the logistic regression model with batch gradient descent method
# 2. Show the error for both training and testing data set
############################################################################
import scipy.io as sio 
import numpy as np
import show_image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

## Load USPS dataset
# (1) This dataset only contains digit 1 and 2. 
# (2) Both of them have 1100 data points.

# read data
load1 = sio.loadmat('usps_all.mat')
data = load1['data'].astype(float)

# class 1: digit 1
# class 2: digit 2
X = np.concatenate((data[:, :, 0].T, data[:, : ,1].T), axis = 0)
X_row = X.shape[0]
X_col = X.shape[1]
H = 16
W = 16

plt.figure()
show_image.show_image_function(X, H, W)
plt.axis('off')
plt.title('Whole Dataset of Digit 1 and Digit 2')

## Separate the dataset into training and testing
nclass1 = data[:,:,0].shape[1] # 1100
nclass2 = data[:,:,1].shape[1] # 1100
Y = np.concatenate((np.ones((nclass1, 1)), 2 * np.ones((nclass2, 1))), axis = 0)

# Use p percent data as training data
p = 0.8
nclass1_train = np.round(nclass1 * p).astype(int)
nclass1_test = nclass1 - nclass1_train

nclass2_train = np.round(nclass2 * p).astype(int)
nclass2_test = nclass2 - nclass2_train

# Training set
Xtrain = np.concatenate((X[0: nclass1_train, :], X[nclass1: nclass1 + nclass2_train, :]), axis = 0)
Ytrain = np.concatenate((Y[0: nclass1_train, :], Y[nclass1: nclass1 + nclass2_train, :]), axis = 0)

# Testing set
Xtest = np.concatenate((X[nclass1_train: nclass1, :], X[nclass1 + nclass2_train: nclass1 + nclass2, :]), axis = 0)
Ytest = np.concatenate((Y[nclass1_train: nclass1, :], Y[nclass1 + nclass2_train: nclass1 + nclass2, :]), axis = 0)

train_size = Ytrain.shape[0]
test_size = Ytest.shape[0]

Xtrain = Xtrain.astype(float)
Xtest = Xtest.astype(float)

## Building classifier using batch gradient descent method

# Parameters
mu = 0.001 # learning rate
threshold = 3
max_iter = 50

# Random initialization
theta = np.random.randn(1, Xtrain.shape[1])

currnorm = 10 # some value larger than threshold
round = 1

while ((currnorm > threshold) and (round <= max_iter)):
    
    # Vectorized version
    # Take the direction of the negative gradient
    temp = Xtrain.dot(theta.T)
    sgn = Ytrain * (-2) + 3
    diff = ((-1 / (1 + np.exp(sgn * temp))) * sgn).T.dot(Xtrain)
    
    # Update the parameters
    theta = theta - mu * diff
    currnorm = np.linalg.norm(diff).astype(float) / Xtrain.shape[0]
    print 'Round = %d: \t %f\n' % (round, currnorm)
    
    round = round + 1

# Evaluate the classifier on the training data set
# (1) Calculate the classification error for training data
# (2) Show the images which are incorrect classified
# +1: Label 1(digit 1)
# -1: Label 2(digit 2)
cc_train = 0
cr_train = 0
rc_train = 0
rr_train = 0
cc_test = 0
cr_test = 0
rc_test = 0
rr_test = 0

m = Xtrain.shape[0]
n = Xtrain.shape[1]
incorrect_image_prediction = np.zeros((m, n))
for i in range(0, nclass1_train):
    # Training set of Class 1(digit 1)
    tmp = Xtrain[i, :].reshape(1, Xtrain[i, :].shape[0])
    logloss1 = np.log( 1 + np.exp(-1 * tmp.dot(theta.T)))
    logloss2 = np.log( 1 + np.exp(tmp.dot(theta.T)))
    
    if ((logloss1 == np.inf) and (logloss2 < np.inf)):
        cr_train = cr_train + 1
        incorrect_image_prediction[i, :] = Xtrain[i, :]
    elif ((logloss1 < np.inf) and (logloss2 == np.inf)):
        cc_train = cc_train + 1
    elif (logloss1 < logloss2):
        cc_train = cc_train + 1
    elif (logloss1 > logloss2):
        cr_train = cr_train + 1
        incorrect_image_prediction[i, :] = Xtrain[i, :]

for i in range(nclass1_train, nclass1_train + nclass2_train):
    # Training set of Class 2(digit 2)
    tmp = Xtrain[i, :].reshape(1, Xtrain[i, :].shape[0])
    logloss1 = np.log( 1 + np.exp(-1 * tmp.dot(theta.T)))
    logloss2 = np.log( 1 + np.exp(tmp.dot(theta.T)))
    
    if ((logloss1 == np.inf) and (logloss2 < np.inf)):
        rr_train = rr_train + 1
    elif ((logloss1 < np.inf) and (logloss2 == np.inf)):
        rc_train = rc_train + 1
        incorrect_image_prediction[i, :] = Xtrain[i, :]
    elif (logloss1 < logloss2):
        rc_train = rc_train + 1
        incorrect_image_prediction[i, :] = Xtrain[i, :]  
    elif (logloss1 > logloss2):
        rr_train = rr_train + 1

plt.figure()
show_image.show_image_function(incorrect_image_prediction, H, W)
plt.axis('off')
plt.title('Incorrect Classification for Training Data')

## For testing data, we use the classifier we have built to do the classification and evaluate the error       
# (1) Calculate the classification error for testing data
# (2) Show the digits which are incorrect classified

m = Xtest.shape[0]
n = Xtest.shape[1]
incorrect_image_estimation = np.zeros((m, n))
for i in range(0, nclass1_test):
    # Training set of Class 1(digit 1)
    tmp = Xtest[i, :].reshape(1, Xtest[i, :].shape[0])
    logloss1 = np.log( 1 + np.exp(-1 * tmp.dot(theta.T)))
    logloss2 = np.log( 1 + np.exp(tmp.dot(theta.T)))
    
    if ((logloss1 == np.inf) and (logloss2 < np.inf)):
        cr_test = cr_test + 1
        incorrect_image_estimation[i, :] = Xtest[i, :]
    elif ((logloss1 < np.inf) and (logloss2 == np.inf)):
        cc_test = cc_test + 1
    elif (logloss1 < logloss2):
        cc_test = cc_test + 1
    elif (logloss1 > logloss2):
        cr_test = cr_test + 1
        incorrect_image_estimation[i, :] = Xtest[i, :]

for i in range(nclass1_test, nclass1_test + nclass2_test):
    # Training set of Class 2(digit 2)
    tmp = Xtest[i, :].reshape(1, Xtest[i, :].shape[0])
    logloss1 = np.log( 1 + np.exp(-1 * tmp.dot(theta.T)))
    logloss2 = np.log( 1 + np.exp(tmp.dot(theta.T)))
    
    if ((logloss1 == np.inf) and (logloss2 < np.inf)):
        rr_test = rr_test + 1
    elif ((logloss1 < np.inf) and (logloss2 == np.inf)):
        rc_test = rc_test + 1
        incorrect_image_estimation[i, :] = Xtest[i, :]
    elif (logloss1 < logloss2):
        rc_test = rc_test + 1
        incorrect_image_estimation[i, :] = Xtest[i, :]  
    elif (logloss1 > logloss2):
        rr_test = rr_test + 1
        
plt.figure()
show_image.show_image_function(incorrect_image_estimation, H, W)
plt.axis('off')
plt.title('Incorrect Classification for Testing Data')
  
train_err = (cr_train + rc_train) * 1.0 / (cc_train + cr_train + rc_train + rr_train)
test_err = (cr_test + rc_test) * 1.0 / (cc_test + cr_test + rc_test + rr_test)

print 'Training Error = %f \n' % train_err
print 'Testing Error = %f \n' % test_err
plt.show() 