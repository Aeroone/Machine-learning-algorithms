############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################

## Circular Data problem
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import build_stump
import eval_stump
import warnings
warnings.filterwarnings("ignore")

p = np.arange(-70, 70 + 0.5, 0.5)
q = np.arange(-70, 70 + 0.5, 0.5)
z = np.zeros((2, p.shape[0] * q.shape[0]))
for i in range(0, p.shape[0]):
    
    z[0, i * p.shape[0] + np.arange(0, q.shape[0])] = p[i] * np.ones((1, q.shape[0]))
    z[1, i * p.shape[0] + np.arange(0, q.shape[0])] = q

z = z.T
# load grid.mat
# alpha_t: The vector of alphas calculated for a specific number of iterations 'num_iter'.

# load error_trend.mat
load1 = sio.loadmat('circular_data.mat')
x = load1['X']

load2 = sio.loadmat('circular_label.mat')
y = load2['Y']

## 
# Show the data
plt.ion()
plt.figure()
plt.axis('equal')
plt.plot(x[0:200, 0], x[0:200, 1], 'bo')
plt.hold(True)
plt.plot(x[200:400, 0], x[200:400, 1], 'ro')
plt.hold(True)
plt.draw()
plt.pause(0.1) 


y_p = y + 2
# Setting number of iterations and applying decision stump for that number
# of times
t = 100

num_iter = t
final_error = np.zeros((1, num_iter))
w = np.ones((1, y.shape[1]))
x_t = []
y_t = []
alpha_t = []
r = np.arange(-100, 100 + 0.01, 0.01)
ind = []
x0 = []
h_t = []
j_t = []
err_t = []
while t > 0:
    
    stump_ind, stump_x0, stump_s, stump_werr = build_stump.build_stump(x, y, w)
    ind.append(stump_ind)
    x0.append(stump_x0)
    h = eval_stump.eval_stump(stump_ind, stump_x0, stump_s, x)
    j = eval_stump.eval_stump(stump_ind, stump_x0, stump_s, z)
    h_t.append(list(h))
    j_t.append(list(j))
    alpha = 0.5 * np.log((1 - stump_werr) * 1.0 / stump_werr)
    err_t.append(stump_werr)
    alpha_t.append(alpha)
    w =  w * np.exp(-alpha * y * h.reshape(1, -1))
    t = t - 1
    
    print '--iteration %d' % t
    # Plotting decision stumps for specific t.  
    
    if (ind[-1] == 0):
        plt.plot([x0[-1], x0[-1]], [-60, 60], 'b')
        plt.hold(True)
    elif (ind[-1] == 1):
        plt.plot([-70, 70], [x0[-1], x0[-1]], 'b')
        plt.hold(True)
    plt.draw()
    plt.pause(0.1) 
    
plt.ioff()

# The total classification error after each iteration.
error = []
h_t = np.array(h_t)
j_t = np.array(j_t)
alpha_t = np.array(alpha_t).reshape(1, -1)
for i in range(1, num_iter + 1):
    final_labeling = np.sign(alpha_t[0, 0:i].dot(h_t[0:i, :]))
    draw_labeling = np.sign(alpha_t[0, 0:i].dot(j_t[0:i, :]))
    f = np.sum(final_labeling != y) * 1.0 / y.shape[1]
    error.append(f)
    
print 'Please wait for the time-consuming plot!'
plt.figure
plt.plot(z[np.where(draw_labeling == -1), 0], z[np.where(draw_labeling == -1), 1], 'g.')
plt.hold(True)
plt.plot(z[np.where(draw_labeling == 1), 0], z[np.where(draw_labeling == 1), 1], 'y.')
plt.plot(x[0:200, 0], x[0:200, 1], 'bo')
plt.hold(True)
plt.plot(x[200:400, 0], x[200:400, 1], 'ro')

for i in range(0, len(ind)):
    if (ind[i] == 0):
        plt.plot([x0[i], x0[i]], [-60, 60], 'b')
        plt.hold(True)
    elif (ind[i] == 1):
        plt.plot([-70, 70], [x0[i], x0[i]], 'b')
        plt.hold(True)
plt.ylim(min(x0), max(x0))
plt.xlim([-70, 70])

plt.figure()
plt.plot(np.array(error))
plt.title('adaboosting error')
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.show()