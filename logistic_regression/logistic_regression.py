############################################################################
# logistic regression 
############################################################################
# The codes are based on Python2.7. 
# Please install numpy package before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
from numpy.linalg import inv

def log_regress_train(trainX, trainY, reg_param, iters, show_error):
    
    train_truey = np.argmax(np.array(trainY.todense()), axis = 1)
    
    ntr = trainY.shape[0]
    numlables = trainY.shape[1]
    
    inv_term = inv(trainX.T.dot(trainX) + reg_param * np.eye(trainX.shape[1]))
    train_preds = np.zeros((trainY.shape[0], trainY.shape[1]))
    w = np.zeros((trainX.shape[1], numlables))
    
    for i in range(0, iters):
        
        train_haty = np.exp(train_preds)
        train_haty = train_haty.astype(float) / np.sum(train_haty, axis = 1).reshape(train_haty.shape[0], 1)
        
        xy_term = trainX.T.dot( np.array(trainY - train_haty) )
        w = w +inv_term.dot(xy_term)
        
        train_preds = trainX.dot(w)
        train_predy = np.argmax(train_preds, axis = 1)
        
        train_error = np.sum(train_predy != train_truey) * 1.0 / ntr
        
        if show_error == 1:
            print '---- %d : train %f\n' % (i + 1,train_error)
    
    return w, train_error