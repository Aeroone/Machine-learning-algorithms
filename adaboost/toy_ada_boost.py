############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import matplotlib.pyplot as plt
import build_stump
import eval_stump
import warnings
warnings.filterwarnings("ignore")

## Toy problem

# Data Points

x = np.array([[2,8],[3,4],[5,2],[6,9],[7,12],[9,8],[9,14],[11,11],[13,12],[14,4]]).T
y = np.array([1,1,-1,-1,1,-1,1,1,-1,-1])
y = y.reshape(1, -1)

# plotting the data
plt.figure()
plt.plot(x[0, np.where(y.squeeze() == 1)], x[1, np.where(y.squeeze() == 1)], 'r+')
plt.plot(x[0, np.where(y.squeeze() == -1)], x[1, np.where(y.squeeze() == -1)], 'bo')
plt.xlim([0, 15])
plt.ylim([0, 15])

# Setting number of iterations and applying decision stump for that number
# of times
t = 3

num_iter = t
w = np.ones((1, y.shape[1]))
p = w * 1.0 / np.sum(w)
x_t = []
y_t = []
h_t = []
alpha_t = []
ind = []
x0 = []
err_t = []
while t > 0:
    
    stump_ind, stump_x0, stump_s, stump_werr = build_stump.build_stump(x.T, y, w)
    ind.append(stump_ind)
    x0.append(stump_x0)
    h = eval_stump.eval_stump(stump_ind, stump_x0, stump_s, x.T)
    h_t.append(list(h))
    alpha = 0.5 * np.log((1 - stump_werr) * 1.0 / stump_werr)
    err_t.append(stump_werr)
    alpha_t.append(alpha)
    w =  w * np.exp(-alpha * y * h.reshape(1, -1))
    t = t - 1

final_error = []
h_t = np.array(h_t)
alpha_t = np.array(alpha_t).reshape(1, -1)
for i in range(1, num_iter + 1):
    final_labeling = np.sign(alpha_t[0, 0:i].dot(h_t[0:i, :]))
    f = np.sum(final_labeling != y) * 1.0 / y.shape[1]
    final_error.append(f)
    print 'The total classification error after round', str(i),' is equal to : ',str(final_error[i - 1])
     
# Plotting each individual decision stump on a separate figure.     

t = np.arange(0, 15 + 0.01, 0.01)
t = t.reshape(1, -1)
for i in range(0, len(ind)):
    if (ind[i] ==  0):
        plt.figure()
        plt.fill([0, 0, x0[i], x0[i]],[0, 15, 15, 0], facecolor=[0.75,0.75,0.25])   
        plt.hold(True)
        plt.fill([x0[i], x0[i], 15, 15],[0, 15, 15, 0], facecolor=[0.25,0.75,0.75])   
        plt.hold(True)
        
        index = np.logical_and((y.squeeze() == 1), h_t[i, :] == -1)
        plt.plot(x[0, index], x[1, index], 'r*', markersize = np.floor(20 * alpha_t[0, i]))
        index = np.logical_and((y.squeeze() == 1), h_t[i, :] == 1)
        plt.plot(x[0, index], x[1, index], 'r*')
        index = np.logical_and((y.squeeze() == -1), h_t[i, :] == 1)
        plt.plot(x[0, index], x[1, index], '*', markersize = np.floor(20 * alpha_t[0, i]))
        index = np.logical_and((y.squeeze() == -1), h_t[i, :] == -1)
        plt.plot(x[0, index], x[1, index], '*')
        plt.xlim([0, 15])
        plt.ylim([0, 15])
        plt.title('Round' + str(i) + ' : error = ' + str(err_t[i]) + ', alpha = ' + str(alpha_t[0, i]))
        #raw_input('--press key to continue ...\n')
        
    elif (ind[i] == 1):
        plt.figure()
        plt.fill([0, 0, 15, 15],[0, x0[i], x0[i], 0], facecolor=[0.75,0.75,0.25])   
        plt.hold(True)
        plt.fill([0, 0, 15, 15],[x0[i], x0[i] + 15, x0[i] + 15, x0[i]], facecolor=[0.25,0.75,0.75])   
        plt.hold(True)
                
        index = np.logical_and((y.squeeze() == 1), h_t[i, :] == -1)
        plt.plot(x[0, index], x[1, index], 'r*', markersize = np.floor(20 * alpha_t[0, i]))
        index = np.logical_and((y.squeeze() == 1), h_t[i, :] == 1)
        plt.plot(x[0, index], x[1, index], 'r*')
        index = np.logical_and((y.squeeze() == -1), h_t[i, :] == 1)
        plt.plot(x[0, index], x[1, index], '*', markersize = np.floor(20 * alpha_t[0, i]))
        index = np.logical_and((y.squeeze() == -1), h_t[i, :] == -1)
        plt.plot(x[0, index], x[1, index], '*')
        plt.xlim([0, 15])
        plt.ylim([0, 15])
        plt.title('Round' + str(i) + ' : error = ' + str(err_t[i]) + ', alpha = ' + str(alpha_t[0, i]))
        #raw_input('--press key to continue ...\n')

# Plotting the final decision boundary 

plt.figure()
plt.fill([4,4,12,12,15,15,0],[0,10,10,15,15,0,0],facecolor=[0.25,0.75,0.75])
plt.hold(True)
plt.fill([0,4,4,12,12,0],[0,0,10,10,15,15],facecolor=[0.75,0.75,0.25])
plt.hold(True)
plt.plot(x[0, np.where(y.squeeze() == 1)], x[1, np.where(y.squeeze() == 1)], 'r+')
plt.plot(x[0, np.where(y.squeeze() == -1)], x[1, np.where(y.squeeze() == -1)], 'o')
plt.hold(True)
for i in range(0, len(ind)):
    if (ind[i] == 0):
        plt.plot(x0[i] * np.ones((1, t.shape[1])), t)
        plt.xlim([0, 15])
        plt.ylim([0, 15])
        plt.hold(True)
    elif (ind[i] == 1):
        plt.plot(t, x0[i] * np.ones((1, t.shape[1])))
        plt.xlim([0, 15])
        plt.ylim([0, 15])

# The total classification error after each iteration.

h_t = np.array(h_t)
alpha_t = np.array(alpha_t).reshape(1, -1)
final_error = []
for i in range(1, num_iter + 1):
    final_labeling = np.sign(alpha_t[0, 0:i].dot(h_t[0:i, :]))
    f = np.sum(final_labeling != y) * 1.0 / y.shape[1]
    final_error.append(f)
    print 'The total classification error after round ' + str(i) + ' is equal to : ' + str(final_error[i - 1])
plt.show()