############################################################################
# The codes are based on Python2.7. 
# Please install numpy, matplotlib packages before using.
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

x = np.array([[2,8],[3,4],[5,2],[6,9],[7,12],[9,8],[9,14],[11,11],[13,32],[14,4]]).T
y = np.array([1,1,-1,-1,1,-1,1,1,-1,-1])
w = np.ones((1, y.shape[0]))


stump_ind, stump_x0, stump_s, werr = build_stump.build_stump(x.T, y.reshape(1, -1), w)
h = eval_stump.eval_stump(stump_ind, stump_x0, stump_s, x.T)

t = np.arange(-15, 15 + 0.01, 0.01)
z = np.arange(-15, 15 + 0.01, 0.01)
# plotting the data
plt.figure()
plt.plot(x[0,np.where(y == 1)[0]], x[1,np.where(y == 1)[0]], 'r+')
plt.plot(x[0,np.where(y == -1)[0]], x[1,np.where(y == -1)[0]], 'o')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.show()
