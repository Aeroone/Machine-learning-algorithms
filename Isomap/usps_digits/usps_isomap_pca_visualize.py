#####################################################################
#
# PCA ISOMAP Visualization for USPS dataset
# ==========================================
#
# Note: Please refer to file usps_isomap_pca.m for PCA and ISOMAP code 
# used to run on analysis.
#
# This file plots the data samples and then shows the PCA and ISOMAP
# analysis on the 2D plot.
#
# This file uses already available analysis results generated earlier using
# the code file usps_isomap_pca.m
#
# The following files are used to plot the data shown here:
#
# USPS_2digits.mat: Original USPS data samples
# dim1_usps.mat, dim2_usps.mat: Results from earlier run of Isomap
# Vx_USPS.mat: Results from earlier run of PCA.
#
# The results may change each time we run the code in usps_isomap_pca.m . 
# Hence, we use the results from earlier run as we are assigning images to 
# the points in 2D manually in this file.
# --------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib, h5py packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# -------------------------------------- 
#####################################################################
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi

## Plots the data samples.

# The data is loaded from faces.mat
matFile = h5py.File('USPS_2digits.mat')

xx = matFile['xx']
xx = np.array(xx)
xx = xx.T

faceW = 16
faceH = 16
numPerLine = 16
ShowLine = 8

Y = np.zeros((faceH*ShowLine, faceW*numPerLine))
for i in range(0, ShowLine):
    for j in range(0, numPerLine):
        
        Y[i*faceH:(i+1)*faceH, j*faceW:(j+1)*faceW] = \
            xx[:, i*numPerLine+j].reshape(faceH, faceW)

plt.figure(1)
plt.ion()
plt.imshow(Y, cmap='gray')
plt.axis('off')
plt.axis('equal')
plt.show()
plt.ioff()

raw_input('press any key to continue\n')

## PCA Visualziation
# The data is loaded from Ux_face.mat 

matFile = h5py.File('Vx_USPS.mat')
Vx = matFile['Vx']
Vx = np.array(Vx)
Vx = Vx.T

# In this block we fit images of the faces to the points in 2D.

th = np.arange(0,2*pi+pi/50,pi/50)

xunit_a = 0.25 * cos(th) + -6.771
yunit_a = 0.25 * sin(th) + 3.463

xunit_b = 0.25 * cos(th) + -4.481
yunit_b = 0.25 * sin(th) + 3.729

xunit_c = 0.25 * cos(th) + -2.494
yunit_c = 0.25 * sin(th) + 4.434

xunit_d = 0.25 * cos(th) + 0.1177
yunit_d = 0.25 * sin(th) + 4.298

xunit_e = 0.25 * cos(th) + 2.309
yunit_e = 0.25 * sin(th) + 4.504

xunit_f = 0.25 * cos(th) + 4.395
yunit_f = 0.25 * sin(th) + 4.905

xunit_g = 0.25 * cos(th) + -6.727
yunit_g = 0.25 * sin(th) + 1.272

xunit_h = 0.25 * cos(th) + -4.345
yunit_h = 0.25 * sin(th) + 1.48

xunit_i = 0.25 * cos(th) + -2.247
yunit_i = 0.25 * sin(th) + 1.581

xunit_j = 0.25 * cos(th) + -0.4442
yunit_j = 0.25 * sin(th) + 1.485

xunit_k = 0.25 * cos(th) + 1.822
yunit_k = 0.25 * sin(th) + 1.647

xunit_l = 0.25 * cos(th) + 4.121
yunit_l = 0.25 * sin(th) + 1.935

xunit_m = 0.25 * cos(th) + 6.378
yunit_m = 0.25 * sin(th) + 2.077

xunit_n = 0.25 * cos(th) + -7.32
yunit_n = 0.25 * sin(th) + -1.07

xunit_o = 0.25 * cos(th) + -4.91
yunit_o = 0.25 * sin(th) + -1.036

xunit_p = 0.25 * cos(th) + -2.897
yunit_p = 0.25 * sin(th) + -0.7345

xunit_q = 0.25 * cos(th) + -0.6758
yunit_q = 0.25 * sin(th) + -0.9005

xunit_r = 0.25 * cos(th) + 1.392
yunit_r = 0.25 * sin(th) + -0.8453

xunit_s = 0.25 * cos(th) + 3.66
yunit_s = 0.25 * sin(th) + -0.563

xunit_t = 0.25 * cos(th) + 5.919
yunit_t = 0.25 * sin(th) + -0.2616

xunit_u = 0.25 * cos(th) + -2.73
yunit_u = 0.25 * sin(th) + -2.903

xunit_w = 0.25 * cos(th) + -0.578
yunit_w = 0.25 * sin(th) + -2.922

xunit_z = 0.25 * cos(th) + 1.199
yunit_z = 0.25 * sin(th) + -3.687

xunit_z1 = 0.25 * cos(th) + 2.7
yunit_z1 = 0.25 * sin(th) + -3.016

xunit_z2 = 0.25 * cos(th) + -1.794
yunit_z2 = 0.25 * sin(th) + -5.453

xunit_z3 = 0.25 * cos(th) + -1.253
yunit_z3 = 0.25 * sin(th) + 5.489

xunit_z4 = 0.25 * cos(th) + 2.248
yunit_z4 = 0.25 * sin(th) + 6.479

xunit_z5 = 0.25 * cos(th) + 4.497
yunit_z5 = 0.25 * sin(th) + -2.404

a_img = xx[:,118].reshape(16, 16)
b_img = xx[:,422].reshape(16, 16)
c_img = xx[:,414].reshape(16, 16)
d_img = xx[:,71].reshape(16, 16)
e_img = xx[:,120].reshape(16, 16)
f_img = xx[:,172].reshape(16, 16)
g_img = xx[:,417].reshape(16, 16)
h_img = xx[:,17].reshape(16, 16)
i_img = xx[:,66].reshape(16, 16)
j_img = xx[:,386].reshape(16, 16)
k_img = xx[:,357].reshape(16, 16)
l_img = xx[:,34].reshape(16, 16)
m_img = xx[:,285].reshape(16, 16)
n_img = xx[:,146].reshape(16, 16)
o_img = xx[:,164].reshape(16, 16)
p_img = xx[:,443].reshape(16, 16)
q_img = xx[:,69].reshape(16, 16)
r_img = xx[:,63].reshape(16, 16)
s_img = xx[:,424].reshape(16, 16)
t_img = xx[:,151].reshape(16, 16)
u_img = xx[:,33].reshape(16, 16)
w_img = xx[:,316].reshape(16, 16)
z_img = xx[:,210].reshape(16, 16)
z1_img = xx[:,207].reshape(16, 16)
z2_img = xx[:,181].reshape(16, 16)
z3_img = xx[:,1].reshape(16, 16)
z4_img = xx[:,378].reshape(16, 16)
z5_img = xx[:,13].reshape(16, 16)

plt.figure(2)

plt.ion()
plt.scatter(Vx[0,:], Vx[1,:], s= 18 * np.ones((473,1)), zorder=-1)
plt.hold(True)

plt.imshow(a_img, extent=[-7.171, -6.371, 3.163, 2.363], origin='lower', cmap='gray')
plt.plot(xunit_a, yunit_a,'red')

plt.imshow(b_img, extent=[-4.881, -4.081, 3.429, 2.629], origin='lower', cmap='gray')
plt.plot(xunit_b, yunit_b,'red')

plt.imshow(c_img, extent=[-2.894, -2.094, 4.134, 3.334], origin='lower', cmap='gray')
plt.plot(xunit_c, yunit_c,'red')

plt.imshow(d_img, extent=[-0.2823, 0.5177, 3.998, 3.198], origin='lower', cmap='gray')
plt.plot(xunit_d, yunit_d,'red')

plt.imshow(e_img, extent=[1.909, 2.709, 4.204, 3.404], origin='lower', cmap='gray')
plt.plot(xunit_e, yunit_e,'red')

plt.imshow(f_img, extent=[3.995, 4.795, 4.605, 3.805], origin='lower', cmap='gray')
plt.plot(xunit_f, yunit_f,'red')

plt.imshow(g_img, extent=[-7.127, -6.327, 0.972, 0.172], origin='lower', cmap='gray')
plt.plot(xunit_g, yunit_g,'red')

plt.imshow(h_img, extent=[-4.745, -3.945, 1.18, 0.38], origin='lower', cmap='gray')
plt.plot(xunit_h, yunit_h,'red')

plt.imshow(i_img, extent=[-2.647, -1.847, 1.281, 0.481], origin='lower', cmap='gray')
plt.plot(xunit_i, yunit_i,'red')

plt.imshow(j_img, extent=[-0.8442, 0.0442, 1.185, 0.385], origin='lower', cmap='gray')
plt.plot(xunit_j, yunit_j,'red')

plt.imshow(k_img, extent=[1.422, 2.222, 1.347, 0.547], origin='lower', cmap='gray')
plt.plot(xunit_k, yunit_k,'red')

plt.imshow(l_img, extent=[3.721, 4.521, 1.635, 0.835], origin='lower', cmap='gray')
plt.plot(xunit_l, yunit_l,'red')

plt.imshow(m_img, extent=[5.978, 6.778, 1.777, 0.977], origin='lower', cmap='gray')
plt.plot(xunit_m, yunit_m,'red')

plt.imshow(n_img, extent=[-7.72, -6.92, -1.47, -2.27], origin='lower', cmap='gray')
plt.plot(xunit_n, yunit_n,'red')

plt.imshow(o_img, extent=[-5.31, -4.51, -1.336, -2.136], origin='lower', cmap='gray')
plt.plot(xunit_o, yunit_o,'red')

plt.imshow(p_img, extent=[-3.297, -2.497, -1.0345, -1.8345], origin='lower', cmap='gray')
plt.plot(xunit_p, yunit_p,'red')

plt.imshow(q_img, extent=[-1.0758, -0.2758, -1.2005, -2.005], origin='lower', cmap='gray')
plt.plot(xunit_q, yunit_q,'red')

plt.imshow(r_img, extent=[0.992, 1.792, -1.1453, -1.9453], origin='lower', cmap='gray')
plt.plot(xunit_r, yunit_r,'red')

plt.imshow(s_img, extent=[3.26, 4.06, -0.863, -1.663], origin='lower', cmap='gray')
plt.plot(xunit_s, yunit_s,'red')

plt.imshow(t_img, extent=[5.519, 6.319, -0.5616, -1.3616], origin='lower', cmap='gray')
plt.plot(xunit_t, yunit_t,'red')

plt.imshow(u_img, extent=[-3.13, -2.33, -3.203, -4.003], origin='lower', cmap='gray')
plt.plot(xunit_u, yunit_u,'red')

plt.imshow(w_img, extent=[-0.978, -0.178, -3.222, -4.022], origin='lower', cmap='gray')
plt.plot(xunit_w, yunit_w,'red')

plt.imshow(z_img, extent=[0.799, 1.599, -3.987, -4.787], origin='lower', cmap='gray')
plt.plot(xunit_z, yunit_z,'red')

plt.imshow(z1_img, extent=[2.3, 3.1, -3.316, -4.116], origin='lower', cmap='gray')
plt.plot(xunit_z1, yunit_z1,'red')

plt.imshow(z2_img, extent=[-2.894, -2.094, -4.653, -5.453], origin='lower', cmap='gray')
plt.plot(xunit_z2, yunit_z2,'red')

plt.imshow(z3_img, extent=[-1.653, -0.853, 5.189, 4.389], origin='lower', cmap='gray')
plt.plot(xunit_z3, yunit_z3,'red')

plt.imshow(z4_img, extent=[1.848, 1.048, 6.179, 5.379], origin='lower', cmap='gray')
plt.plot(xunit_z4, yunit_z4,'red')

plt.imshow(z5_img, extent=[4.097, 4.897, -2.704, -3.504], origin='lower', cmap='gray')
plt.plot(xunit_z5, yunit_z5,'red')

#plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.title('PCA')
plt.show()
plt.ioff()

raw_input('press any key to continue\n')

## Isomap Visualziation
# The data is loaded from dim1_usps.mat and dim2_usps.mat 

matFile = h5py.File('dim1_usps.mat')
dim1 = matFile['dim1']
dim1 = np.array(dim1)
dim1 = dim1.T

matFile = h5py.File('dim2_usps.mat')
dim2 = matFile['dim2']
dim2 = np.array(dim2)
dim2 = dim2.T

# In this block we fit images of the faces to the points in 2D.

th = np.arange(0,2*pi+pi/50,pi/50)

xunit_a = 0.5 * cos(th) + 13.19
yunit_a = 0.5 * sin(th) + 8.177

xunit_b = 0.5 * cos(th) + -10.01
yunit_b = 0.5 * sin(th) + -5.913

xunit_c = 0.5 * cos(th) + 7.486
yunit_c = 0.5 * sin(th) + -7.533

xunit_d = 0.5 * cos(th) + -10.33
yunit_d = 0.5 * sin(th) + 5.519

xunit_e = 0.5 * cos(th) + -5.192
yunit_e = 0.5 * sin(th) + 5.242

xunit_f = 0.5 * cos(th) + -16.66
yunit_f = 0.5 * sin(th) + 4.828

xunit_g = 0.5 * cos(th) + -10.04
yunit_g = 0.5 * sin(th) + 0.2263

xunit_h = 0.5 * cos(th) + -1.15
yunit_h = 0.5 * sin(th) + 5.008

xunit_i = 0.5 * cos(th) + 3.745
yunit_i = 0.5 * sin(th) + 5.016

xunit_j = 0.5 * cos(th) + 8.256
yunit_j = 0.5 * sin(th) + 5.347

xunit_k = 0.5 * cos(th) + 6.095
yunit_k = 0.5 * sin(th) + 9.156

xunit_l = 0.5 * cos(th) + 1.032
yunit_l = 0.5 * sin(th) + 9.053

xunit_n = 0.5 * cos(th) + -2.437
yunit_n = 0.5 * sin(th) + 9.062

xunit_o = 0.5 * cos(th) + 6.589
yunit_o = 0.5 * sin(th) + 11.77

xunit_p = 0.5 * cos(th) + -5.171
yunit_p = 0.5 * sin(th) + 0.39

xunit_q = 0.5 * cos(th) + -0.6051
yunit_q = 0.5 * sin(th) + 0.1347

xunit_r = 0.5 * cos(th) + 4.465
yunit_r = 0.5 * sin(th) + -0.002556

xunit_s = 0.5 * cos(th) + 9.08
yunit_s = 0.5 * sin(th) + 0.5817

xunit_t = 0.5 * cos(th) + -6.013
yunit_t = 0.5 * sin(th) + -5.145

xunit_u = 0.5 * cos(th) + -1.059
yunit_u = 0.5 * sin(th) + -4.821

xunit_v = 0.5 * cos(th) + 3.941
yunit_v = 0.5 * sin(th) + -4.683

xunit_w = 0.5 * cos(th) + 0.01467
yunit_w = 0.5 * sin(th) + -9.82

xunit_z = 0.5 * cos(th) + -6.262
yunit_z = 0.5 * sin(th) + -9.524

xunit_z1 = 0.5 * cos(th) + -15
yunit_z1 = 0.5 * sin(th) + -4.335

xunit_z2 = 0.5 * cos(th) + 10.9
yunit_z2 = 0.5 * sin(th) + -5.869

xunit_z3 = 0.5 * cos(th) + -13.72
yunit_z3 = 0.5 * sin(th) + 1.335

xunit_z4 = 0.5 * cos(th) + 11.6
yunit_z4 = 0.5 * sin(th) + 2.763

xunit_z5 = 0.5 * cos(th) + 15.54
yunit_z5 = 0.5 * sin(th) + -11.3


a_img = xx[:,134].reshape(16, 16)
b_img = xx[:,327].reshape(16, 16)
c_img = xx[:,409].reshape(16, 16)
d_img = xx[:,365].reshape(16, 16)
e_img = xx[:,170].reshape(16, 16)
f_img = xx[:,70].reshape(16, 16)
g_img = xx[:,128].reshape(16, 16)
h_img = xx[:,297].reshape(16, 16)
i_img = xx[:,326].reshape(16, 16)
j_img = xx[:,214].reshape(16, 16)
k_img = xx[:,293].reshape(16, 16)
l_img = xx[:,337].reshape(16, 16)
n_img = xx[:,209].reshape(16, 16)
o_img = xx[:,398].reshape(16, 16)
p_img = xx[:,226].reshape(16, 16)
q_img = xx[:,283].reshape(16, 16)
r_img = xx[:,175].reshape(16, 16)
s_img = xx[:,360].reshape(16, 16)
t_img = xx[:,287].reshape(16, 16)
u_img = xx[:,340].reshape(16, 16)
v_img = xx[:,221].reshape(16, 16)
w_img = xx[:,374].reshape(16, 16)
z_img = xx[:,109].reshape(16, 16)
z1_img = xx[:,256].reshape(16, 16)
z2_img = xx[:,13].reshape(16, 16)
z3_img = xx[:,84].reshape(16, 16)
z4_img = xx[:,381].reshape(16, 16)
z5_img = xx[:,151].reshape(16, 16)

plt.figure(3)

plt.ion()
plt.scatter(dim1, dim2, s= 18 * np.ones((473, 1)), zorder=-1)
plt.hold(True)

plt.imshow(a_img, extent=[12.19, 14.19, 7.577, 5.577], origin='lower', cmap='gray')
plt.plot(xunit_a, yunit_a,'red')

plt.imshow(b_img, extent=[-11.01, -9.01, -6.513, -8.513], origin='lower', cmap='gray')
plt.plot(xunit_b, yunit_b,'red')

plt.imshow(c_img, extent=[6.486, 8.486, -8.133, -10.133], origin='lower', cmap='gray')
plt.plot(xunit_c, yunit_c,'red')

plt.imshow(d_img, extent=[-11.33, -9.33, 4.919, 2.919], origin='lower', cmap='gray')
plt.plot(xunit_d, yunit_d,'red')

plt.imshow(e_img, extent=[-6.192, -4.192, 4.642, 2.642], origin='lower', cmap='gray')
plt.plot(xunit_e, yunit_e,'red')

plt.imshow(f_img, extent=[-17.66, -15.66, 4.228, 2.228], origin='lower', cmap='gray')
plt.plot(xunit_f, yunit_f,'red')

plt.imshow(g_img, extent=[-11.04, -9.04, -0.5837, -2.5873], origin='lower', cmap='gray')
plt.plot(xunit_g, yunit_g,'red')

plt.imshow(h_img, extent=[-2.15, -0.15, 4.408, 2.408], origin='lower', cmap='gray')
plt.plot(xunit_h, yunit_h,'red')

plt.imshow(i_img, extent=[2.745, 4.745, 4.016, 2.016], origin='lower', cmap='gray')
plt.plot(xunit_i, yunit_i,'red')

plt.imshow(j_img, extent=[7.256, 9.256, 4.747, 2.747], origin='lower', cmap='gray')
plt.plot(xunit_j, yunit_j,'red')

plt.imshow(k_img, extent=[5.095, 7.095, 8.556, 6.556], origin='lower', cmap='gray')
plt.plot(xunit_k, yunit_k,'red')

plt.imshow(l_img, extent=[0.032, 2.032, 8.453, 6.453], origin='lower', cmap='gray')
plt.plot(xunit_l, yunit_l,'red')

plt.imshow(n_img, extent=[-1.437, -3.437, 8.462, 6.462], origin='lower', cmap='gray')
plt.plot(xunit_n, yunit_n,'red')

plt.imshow(o_img, extent=[7.189, 9.189, 12.77, 10.77], origin='lower', cmap='gray')
plt.plot(xunit_o, yunit_o,'red')

plt.imshow(p_img, extent=[-6.171, -4.171, -0.21, -2.21], origin='lower', cmap='gray')
plt.plot(xunit_p, yunit_p,'red')

plt.imshow(q_img, extent=[-1.6051, 0.3949, -0.4653, -2.4653], origin='lower', cmap='gray')
plt.plot(xunit_q, yunit_q,'red')

plt.imshow(r_img, extent=[3.465, 5.465, -0.597444, -2.597444], origin='lower', cmap='gray')
plt.plot(xunit_r, yunit_r,'red')

plt.imshow(s_img, extent=[8.08, 10.08, -0.0183, -2.0183], origin='lower', cmap='gray')
plt.plot(xunit_s, yunit_s,'red')

plt.imshow(t_img, extent=[-7.013, -5.013, -5.745, -7.745], origin='lower', cmap='gray')
plt.plot(xunit_t, yunit_t,'red')

plt.imshow(u_img, extent=[-2.059, -0.059, -5.421, -7.421], origin='lower', cmap='gray')
plt.plot(xunit_u, yunit_u,'red')

plt.imshow(v_img, extent=[2.941, 4.491, -5.283, -7.283], origin='lower', cmap='gray')
plt.plot(xunit_v, yunit_v,'red')

plt.imshow(w_img, extent=[-0.98533, 1.01467, -10.42, -12.42], origin='lower', cmap='gray')
plt.plot(xunit_w, yunit_w,'red')

plt.imshow(z_img, extent=[-7.262, -5.262, -10.124, -12.124], origin='lower', cmap='gray')
plt.plot(xunit_z, yunit_z,'red')

plt.imshow(z1_img, extent=[-16, -14, -4.995, -6.995], origin='lower', cmap='gray')
plt.plot(xunit_z1, yunit_z1,'red')

plt.imshow(z2_img, extent=[9.9, 11.9, -6.469, -8.469], origin='lower', cmap='gray')
plt.plot(xunit_z2, yunit_z2,'red')

plt.imshow(z3_img, extent=[-14.72, -12.72, 0.335, -1.665], origin='lower', cmap='gray')
plt.plot(xunit_z3, yunit_z3,'red')

plt.imshow(z4_img, extent=[10.6, 12.6, 2.163, 0.163], origin='lower', cmap='gray')
plt.plot(xunit_z4, yunit_z4,'red')

plt.imshow(z5_img, extent=[14.54, 16.54, -11.9, -13.9], origin='lower', cmap='gray')
plt.plot(xunit_z5, yunit_z5,'red')

plt.gca().invert_yaxis()

plt.title('isomap')

plt.show()
plt.ioff()
plt.show()