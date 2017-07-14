############################################################################
# This demo is related to the leaf dataset, including 2 parts.
# 1. Implement the feature selection algorithm.
# 2. Based on the previous feature sorting, repeat add 1 feature at a
# time to do the classification and estimate the error.
############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import logistic_regression
from scipy import spatial
from scipy.sparse import csc_matrix

# leaf dataset
data = sio.loadmat('PCA_leaf_example/leaf.mat')['M']

# Dataset description
# The provided data comprises the following shape (attributes 3 to 9) and 
# texture (attributes 10 to 16) features:
# 1. Class (Species)
# 2. Specimen Number
# 3. Eccentricity
# 4. Aspect Ratio
# 5. Elongation
# 6. Solidity
# 7. Stochastic Convexity
# 8. Isoperimetric Factor
# 9. Maximal Indentation Depth
# 10. Lobedness
# 11. Average Intensity
# 12. Average Contrast
# 13. Smoothness
# 14. Third moment
# 15. Uniformity
# 16. Entropy

feature_names = np.array(['Eccentricity', 'Aspect Ratio', 'Elongation', 'Solidity', 'Stochastic Convexity',\
    'Isoperimetric Factor', 'Maximal Indentation Depth', 'Lobedness', 'Average Intensity',\
    'Average Contrast', 'Smoothness', 'Third moment', 'Uniformity', 'Entropy'])

# Extract attributes from the raw data.
X = data[:, 2:16]
n = X.shape[0]
d = X.shape[1]

Y = data[:, 0]

# Map the class indexes to 1 to n_classes 
# (Class indexes are not continuous in this dataset)
uniq_symbols, Y = np.unique(Y, return_inverse=True)
n_classes = uniq_symbols.shape[0]


## Feature Selection 
n_samples = 100
sigma = 1
normalization_const = 1 / np.sqrt(2 * np.pi * np.power(sigma, 2))
mi = np.zeros((d, 1))

# For each value of the label y = k, estimate density P(y = k)
class_prior = np.histogram(Y, bins=np.arange(0,n_classes + 1))[0]
class_prior = class_prior.astype(float) / np.sum(class_prior)

# For each feature x_i
for i in range(0, d):
    
    joint_distr = np.zeros((n_samples, n_classes))
    feat_val_min = np.min(X[:, i])
    feat_val_max = np.max(X[:, i])
    
    # Estimate the feature density
    # Discretize KDE
    sample_points = np.linspace(feat_val_min, feat_val_max, n_samples)
    sample_points = sample_points.reshape(sample_points.shape[0], 1)
    
    tmp = X[:, i].reshape(X[:, i].shape[0], 1)    
    sample_train_data_dist = spatial.distance.cdist(sample_points, tmp)
    
    densities = normalization_const * np.exp(- np.power(sample_train_data_dist, 2) / (2 * np.power(sigma, 2)))
    
    # For each value of the label y = j, estimate p(x_i/y = j) 
    for j in range(0, n_classes):
        class_density = np.mean(densities[:, Y == j], axis = 1)
        joint_distr[:, j] = class_density * class_prior[j]
        
    # Normalize joint distribution
    joint_distr = joint_distr / np.sum(joint_distr)
    
    # Marginal feature distribution P(X)
    feat_distr = np.sum(joint_distr, axis = 1)
    # Marginal class distribution P(Y)
    class_distr = np.sum(joint_distr, axis = 0)
    
    # Cross product P(X) * P(Y)
    feat_distr = feat_distr.reshape(feat_distr.shape[0], 1)
    class_distr = class_distr.reshape(1, class_distr.shape[0])
    cross_prod = feat_distr.dot(class_distr)
    
    # Mutual information \sum_x,y P(X, Y) log( P(X, Y) / (P(X)P(Y)) )
    np.seterr(divide='ignore', invalid='ignore')    
    tmp = joint_distr * np.log(joint_distr / cross_prod)
    # We define 0 * log 0 to be 0
    tmp = np.nan_to_num(tmp) 
    # Score feature x_i
    mi[i, 0] = np.sum(tmp)
    
# Sort features based on the scores and show the top 5 features  
sorted_mi_idx = np.argsort(mi, axis = 0)[::-1]
print 'Top 5 informative features\n'
for i in range(0, 5):
    print '%d. %s\n' % (i + 1, feature_names[sorted_mi_idx[i]])

plt.figure(1)
x = np.arange(1, mi.shape[0] + 1)
plt.stem(x, mi[:,0])
plt.xlabel('Features')
plt.ylabel('Mutual information')

## Classification
# Use the logistic regression model to do the classification
# Use 10-fold cross validation to evaluate the model
n_total = X.shape[0]
full_data = X 
trueY = Y
full_label = csc_matrix( (np.ones(n_total), (np.arange(n_total), trueY)), shape=(n_total, n_classes))

feat_block = 1
total_blocks = 14
feat_select_err_list = np.zeros((1, total_blocks))
data_rand_idx = np.random.permutation(n_total)
n_folds = 10
cv_size = np.ceil(n_total * 1.0 / n_folds).astype(int)

# Choose different numbers of features to build models
for f_i in range(0, total_blocks):
    # Choose top f_i features
    print 'feature selection block: %d\n' % (f_i + 1)
    top_feat_idx = sorted_mi_idx[0:(feat_block * (f_i + 1))].squeeze()

    # Leave-one-out error
    cv_err = np.zeros((1, n_folds))
    for i in range(0, n_folds):
        # Seperate the training and testing data
        
        test_idx = data_rand_idx[i*cv_size: min(n_total, (i+1)*cv_size)]
        data_idx = np.setdiff1d(np.arange(0,n_total), test_idx)
        cv_trainX = full_data[data_idx][:, top_feat_idx]
        cv_trainY = full_label[data_idx].todense()
                
        # Building logistic regression model 
        if f_i == 0: 
            cv_trainX = cv_trainX.reshape(cv_trainX.shape[0], 1)
        B = logistic_regression.logistic_regression(cv_trainX, cv_trainY)
                
        # Test the model on testing data and compute the error
        tmp = full_data[test_idx][:, top_feat_idx]
        if f_i == 0:
            tmp = tmp.reshape(tmp.shape[0], 1)
        tmp = tmp.dot(B)
        pred = np.argmax(tmp, axis = 1)
        
        cv_err[0, i] = np.mean(np.asarray(pred).squeeze() != (trueY[test_idx]))
    feat_select_err_list[0, f_i] = np.mean(cv_err)  
    
plt.figure(2)
x = np.arange(1, mi.shape[0] + 1)
plt.plot(x, feat_select_err_list[0,:])
plt.xlabel('Number of Features')    
plt.ylabel('Cross validation error') 
plt.show()