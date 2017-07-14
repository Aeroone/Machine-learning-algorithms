############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
####################
# This demo shows how to use different regression model to predict the head acceleration based on the time.
# 1. This demo uses linear regression, polynomial regression with different
# degrees and kernel ridge regression with different parameters to do the
# prediction.
# 2. This demo use 10-fold cross validation to evaluate the errors.
############################################################################
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import warnings
warnings.filterwarnings("ignore")

## Data Description
# A data frame giving a series of measurements of head acceleration in a
# simulated motorcycle accident, used to test crash helmets.
# Y: Acceleration
# X: Time

## Predict by fit the linear or polynomial regression model.
def fitreg(Xtrain, Xtest, Y):
    # coefficient theta
    tmp = Xtrain.dot(Xtrain.T)
    theta = np.linalg.inv(tmp).dot(np.eye(tmp.shape[0])).dot(Xtrain).dot(Y)
    pred = Xtest.T.dot(theta)  # predict on test dataset   
    return pred

## Predict by fit the kernel ridge regression model.
def fitkernel(Xtrain, Xtest, Ytrain, lam, r):
    
    dismat = np.power(spatial.distance.pdist(Xtrain.T), 2) # pairwise distance
    s0 = np.median(dismat)
    sigma = s0 * r
    K = np.exp(-spatial.distance.squareform(dismat) * 1.0 / (2 * sigma))
    dismattest = spatial.distance.cdist(Xtrain.T, Xtest.T)
    Ktest = np.exp(-np.power(dismattest, 2) / (2 * sigma))
    pred = Ytrain.T.dot(np.linalg.inv(K + lam * np.eye(K.shape[0])).dot(Ktest))
    return pred

## 10-fold Cross validation function
# flag == 1: linear or polynomial regression
# flag == 2: kernel ridge regression
def cross_validation(X, Y, flag, lam, r, m):
    
    fold = 10
    avgerr = np.zeros((1, fold))
    for l in range(0, fold):
        
        testindex = np.arange(l, m, fold)
        totalindex = np.arange(0, m)
        trainindex = np.setdiff1d(totalindex,testindex)     
        
        Xtrain = X[:, trainindex]
        Ytrain = Y[trainindex]
        Xtest = X[:, testindex]
        
        # m is the length of the data which is defined in the main function.
        if flag == 1:
            predtest = fitreg(Xtrain, Xtest, Ytrain)
        elif flag == 2:
            predtest1 = fitkernel(Xtrain, Xtest, Ytrain, lam, r)
            predtest = predtest1.T
        
        avgerr[0, l] = np.mean(np.power(predtest - Y[testindex], 2))
    avgerr = np.mean(avgerr)
    return avgerr    

## Main Function
load1 = sio.loadmat('motor.mat')
data = load1['motor'].astype(float)
Time = data[:, 0]
Time = Time.reshape(Time.shape[0], 1)
Acceleration = data[:, 1]
Acceleration = Acceleration.reshape(Acceleration.shape[0], 1)

# Show the scatter plot of the dataset
plt.figure()
plt.ion()
plt.scatter(Time, Acceleration)
plt.xlim(0,60)  
plt.ylim(-200, 200)  
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()
plt.ioff()

# Prepare the time data for plotting
Time = Time.T
xgrid = np.arange(0, 60 + 0.01, 0.01)
n = xgrid.shape[0]
xgrid = xgrid.reshape(1, n)
X_p = np.concatenate((np.ones((1, n)), xgrid), axis = 0)


## Linear Regression
raw_input('press key to run linear regression ...\n') 
plt.close("all")
plt.figure()
plt.ion()
plt.scatter(Time, Acceleration)
plt.xlim(0,60)  
plt.ylim(-200, 200) 
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Linear Regression')

m = Time.shape[1]
X = np.concatenate((np.ones((1, m)), Time), axis = 0)
predgrid1 = fitreg(X, X_p, Acceleration)
plt.plot(xgrid.squeeze(), predgrid1.T.squeeze(), 'r')
plt.show() # plot the first order linear regression line
plt.ioff()

# Evaluate the error by 10-fold cross validation
avgerr = cross_validation(X, Acceleration, 1, 100, 100, m)
print 'Linear Regression, Average Error: %f\n' % avgerr


## polynomial regression
raw_input('press key to run polynomial regression ...\n')
plt.close("all")

plt.ion()
plt.figure()
plt.scatter(Time, Acceleration)
plt.xlim(0,60)  
plt.ylim(-200, 200) 
plt.xlabel('Time')
plt.ylabel('Acceleration')

dmax = 7; # maximal degree
for d in range(2, dmax + 1):
    X = np.concatenate((X, np.power(Time, d)), axis = 0)
    X_p = np.concatenate((X_p, np.power(xgrid, d)), axis = 0)
    predgrid2 = fitreg(X, X_p, Acceleration)
    if d == 6: # Corresponding to the lowest error
        plt.plot(xgrid.squeeze(), predgrid2.T.squeeze(), 'r')
    else:
        plt.plot(xgrid.squeeze(), predgrid2.T.squeeze(), 'b')

    plt.title('Polynomial Regression')
    plt.hold(True)
    plt.show()
    plt.pause(0.5)   
        
    # Evaluate the error by 10-fold cross validation
    avgerr = cross_validation(X, Acceleration, 1, 100, 100, m)
    print 'Linear Regression, Average Error: %f\n' % avgerr

plt.hold(True)
plt.show()


## kernel ridge regression 
raw_input('press key to run kernel ridge regression ...\n')
plt.ioff()
plt.close("all")

# Show the dataset in the scatter plot
plt.figure()
plt.ion()
plt.scatter(Time, Acceleration, label="data points")
plt.xlim(0,60)  
plt.ylim(-200, 200) 
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Kernel Ridge Regression')
plt.legend()

# bandwidth parameter; 
r = np.array([0.01, 1, 100])
# Parameters
lam = np.array([0.001, 0.01, 0.1, 1, 10])

X = np.concatenate((np.ones((1, m)), Time), axis = 0)
X_p = np.concatenate((np.ones((1, n)), xgrid), axis = 0)

# Try kernel ridge regression with different parameters
for i in range(0, lam.shape[0]):
    for j in range(0, r.shape[0]):
        
        predgrid3 = fitkernel(X, X_p, Acceleration, lam[i], r[j])
        # Only plot the following situations:
        # (1) large sigma, large lambda; 
        # (2) samll sigma, small lambda; 
        # (3) small sigma, large lambda; 
        # (4) regression line with lowest error
        if (r[j] == r.min() and lam[i] == lam.min()):
            plt.plot(xgrid.squeeze(), predgrid3.T.squeeze(), 'k', label="small $\sigma$, small $\lambda$")
            plt.legend()
            plt.hold(True) # plot the kernel ridge regression
            plt.legend()
            plt.show()
            plt.pause(0.5)
        elif (r[j] == r.min() and lam[i] == lam.max()):
            plt.plot(xgrid.squeeze(), predgrid3.T.squeeze(), 'b', label="small $\sigma$, large $\lambda$")
            plt.legend()
            plt.hold(True) # plot the kernel ridge regression
            plt.show()
            plt.pause(0.5)
        elif (r[j] == r.max() and lam[i] == lam.max()):
            plt.plot(xgrid.squeeze(), predgrid3.T.squeeze(), 'g', label="large $\sigma$, large $\lambda$")
            plt.legend()
            plt.hold(True) # plot the kernel ridge regression
            plt.show()
            plt.pause(0.5)
        elif (r[j] == 1 and lam[i] == 0.001): # corresponding to the smallest error
            plt.plot(xgrid.squeeze(), predgrid3.T.squeeze(), 'r', label="Regression line with lowest error")
            plt.legend()
            plt.hold(True)
            plt.show()
            plt.pause(0.5)
            
        # Evaluate error by 10-fold cross validation
        avgerr = cross_validation(X, Acceleration, 2, lam[i], r[j], m)
        print 'lambda = %.3f, r = %.2f, Average Error: %.4f\n' % (lam[i], r[j], avgerr)

plt.hold(True)
plt.show()

raw_input('press key to exit ...\n')
plt.ioff()