#####################################################################
#
# PCA ISOMAP Visualization for Faces dataset
# ==========================================
#
# Note: Please refer to file faces_isomap_pca.m for PCA and ISOMAP code 
# used to run on analysis.
#
# This file plots the data samples and then shows the PCA and ISOMAP
# analysis on the 2D plot.
#
# This file uses already available analysis results generated earlier using
# the code file faces_isomap_pca.m
#
# The following files are used to plot the data shown here:
#
# faces.mat: Original USPS data samples
# dim1.mat, dim2.mat: Results from earlier run of Isomap
# Ux_face.mat: Results from earlier run of PCA.
#
# The results may change each time we run the code in face_isomap_pca.m . 
# Hence, we use the results from earlier run as we are assigning images to 
# the points in 2D manually in this file.
# -----------------------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib, h5py packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# -----------------------------------------------------
#####################################################################
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi

## Plots the data samples.

# The data is loaded from faces.mat
matFile = h5py.File('faces.mat')
images = matFile['images']
images = np.array(images)
images = images.T

faceW = 64
faceH = 64
numPerLine = 16
ShowLine = 8

Y = np.zeros((faceH*ShowLine, faceW*numPerLine))
for i in range(0, ShowLine):
    for j in range(0, numPerLine):
        
        Y[i*faceH:(i+1)*faceH, j*faceW:(j+1)*faceW] = \
            images[:, i*numPerLine+j].reshape(faceH, faceW).T

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

matFile = h5py.File('Ux_face.mat')
Ux = matFile['Ux']
Ux = np.array(Ux)
Ux = Ux.T

# In this block we fit images of the faces to the points in 2D.

th = np.arange(0,2*pi+pi/50,pi/50)

xunit_a = 0.5 * cos(th) + -12.87
yunit_a = 0.5 * sin(th) + 7.131

xunit_b = 0.5 * cos(th) + -7.01
yunit_b = 0.5 * sin(th) + 7.961

xunit_c = 0.5 * cos(th) + -2.608
yunit_c = 0.5 * sin(th) + 9.03

xunit_d = 0.5 * cos(th) + 2.216
yunit_d = 0.5 * sin(th) + 7.823

xunit_e = 0.5 * cos(th) + 6.688
yunit_e = 0.5 * sin(th) + 8.025

xunit_f = 0.5 * cos(th) + -12.67
yunit_f = 0.5 * sin(th) + -0.3996

xunit_g = 0.5 * cos(th) + -7.849
yunit_g = 0.5 * sin(th) + -0.07263

xunit_h = 0.5 * cos(th) + -2.001
yunit_h = 0.5 * sin(th) + -0.1126

xunit_i= 0.5 * cos(th) + 2.761
yunit_i = 0.5 * sin(th) + 0.07668

xunit_j= 0.5 * cos(th) + 7.538
yunit_j = 0.5 * sin(th) + 0.3146

xunit_k= 0.5 * cos(th) + -6.446
yunit_k = 0.5 * sin(th) + -8.875

xunit_l= 0.5 * cos(th) + -3.5
yunit_l = 0.5 * sin(th) + -6.264

xunit_m= 0.5 * cos(th) + 1.104
yunit_m = 0.5 * sin(th) + -7.177

xunit_n= 0.5 * cos(th) + 4.282
yunit_n = 0.5 * sin(th) + -7.097

xunit_o= 0.5 * cos(th) + 7.841
yunit_o = 0.5 * sin(th) + -6.649

xunit_p= 0.5 * cos(th) + -4.69
yunit_p = 0.5 * sin(th) + -15.14

xunit_q= 0.5 * cos(th) + -0.4317
yunit_q = 0.5 * sin(th) + -13.08

xunit_r= 0.5 * cos(th) + 4.29
yunit_r = 0.5 * sin(th) + -13.01

xunit_s= 0.5 * cos(th) + 10.07
yunit_s = 0.5 * sin(th) + 4.429

xunit_t= 0.5 * cos(th) + 13.64
yunit_t = 0.5 * sin(th) + 3.27

xunit_z= 0.5 * cos(th) + -10.55
yunit_z = 0.5 * sin(th) + 11.3

xunit_z1= 0.5 * cos(th) + -4.926
yunit_z1 = 0.5 * sin(th) + 12.72


a_img = images[:,223].reshape(64, 64).T
b_img = images[:,70].reshape(64, 64).T
c_img = images[:,382].reshape(64, 64).T
d_img = images[:,337].reshape(64, 64).T
e_img = images[:,399].reshape(64, 64).T
f_img = images[:,165].reshape(64, 64).T
g_img = images[:,266].reshape(64, 64).T
h_img = images[:,214].reshape(64, 64).T
i_img = images[:,139].reshape(64, 64).T
j_img = images[:,55].reshape(64, 64).T
k_img = images[:,356].reshape(64, 64).T
l_img = images[:,156].reshape(64, 64).T
m_img = images[:,3].reshape(64, 64).T
n_img = images[:,270].reshape(64, 64).T
o_img = images[:,668].reshape(64, 64).T
p_img = images[:,489].reshape(64, 64).T
q_img = images[:,431].reshape(64, 64).T
r_img = images[:,448].reshape(64, 64).T
s_img = images[:,155].reshape(64, 64).T
t_img = images[:,501].reshape(64, 64).T
z_img = images[:,626].reshape(64, 64).T
z1_img = images[:,622].reshape(64, 64).T



plt.figure(2)

plt.ion()
plt.scatter(Ux[0,:], Ux[1,:], s= 18 * np.ones((1, 698)), zorder=-1)
plt.hold(True)

plt.imshow(a_img, extent=[-11.87, -13.87, 6.531, 3.531], origin='lower', cmap='gray')
plt.plot(xunit_a, yunit_a,'red')

plt.imshow(b_img, extent=[-6.01, -8.01, 7.361, 4.361], origin='lower', cmap='gray')
plt.plot(xunit_b, yunit_b,'red')

plt.imshow(c_img, extent=[-1.608, -3.608, 8.03, 5.03], origin='lower', cmap='gray')
plt.plot(xunit_c, yunit_c,'red')

plt.imshow(d_img, extent=[1.216, 3.216, 7.223, 4.223], origin='lower', cmap='gray')
plt.plot(xunit_d, yunit_d,'red')

plt.imshow(e_img, extent=[5.668, 7.668, 7.425, 4.425], origin='lower', cmap='gray')
plt.plot(xunit_e, yunit_e,'red')

plt.imshow(f_img, extent=[-13.67, -11.67, -0.9996, -3.9996], origin='lower', cmap='gray')
plt.plot(xunit_f, yunit_f,'red')

plt.imshow(g_img, extent=[-8.849, -6.849, -0.67263, -3.67263], origin='lower', cmap='gray')
plt.plot(xunit_g, yunit_g,'red')

plt.imshow(h_img, extent=[-1.001, -3.001, -0.7126, -3.7126], origin='lower', cmap='gray')
plt.plot(xunit_h, yunit_h,'red')

plt.imshow(i_img, extent=[1.761, 3.761, -0.52332, -3.52332], origin='lower', cmap='gray')
plt.plot(xunit_i, yunit_i,'red')

plt.imshow(j_img, extent=[6.538, 8.538, -0.2954, -3.2953], origin='lower', cmap='gray')
plt.plot(xunit_j, yunit_j,'red')

plt.imshow(k_img, extent=[-7.446, -5.446, -9.475, -12.475], origin='lower', cmap='gray')
plt.plot(xunit_k, yunit_k,'red')

plt.imshow(l_img, extent=[-2.5, -4.5, -6.864, -9.864], origin='lower', cmap='gray')
plt.plot(xunit_l, yunit_l,'red')

plt.imshow(m_img, extent=[0.104, 2.104, -7.777, -10.777], origin='lower', cmap='gray')
plt.plot(xunit_m, yunit_m,'red')

plt.imshow(n_img, extent=[3.282, 5.282, -7.697, -10.697], origin='lower', cmap='gray')
plt.plot(xunit_n, yunit_n,'red')

plt.imshow(o_img, extent=[6.841, 8.841, -7.249, -10.249], origin='lower', cmap='gray')
plt.plot(xunit_o, yunit_o,'red')

plt.imshow(p_img, extent=[-3.69, -5.69, -15.74, -18.74], origin='lower', cmap='gray')
plt.plot(xunit_p, yunit_p,'red')

plt.imshow(q_img, extent=[-1.4317, 0.5683, -13.68, -16.68], origin='lower', cmap='gray')
plt.plot(xunit_q, yunit_q,'red')

plt.imshow(r_img, extent=[3.29, 5.29, -13.61, -16.61], origin='lower', cmap='gray')
plt.plot(xunit_r, yunit_r,'red')

plt.imshow(s_img, extent=[9.07, 11.07, 3.629, 0.629], origin='lower', cmap='gray')
plt.plot(xunit_s, yunit_s,'red')

plt.imshow(t_img, extent=[12.64, 14.64, 2.67, -0.33], origin='lower', cmap='gray')
plt.plot(xunit_t, yunit_t,'red')

plt.imshow(z_img, extent=[-11.55, -9.55, 10.7, 7.7], origin='lower', cmap='gray')
plt.plot(xunit_z, yunit_z,'red')

plt.imshow(z1_img, extent=[-3.926, -5.926, 12.12, 9.12], origin='lower', cmap='gray')
plt.plot(xunit_z1, yunit_z1,'red')

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.xlim([-20,20])
plt.ylim([-20,15])
plt.title('PCA')
plt.show()
plt.ioff()

#-----------------------------------------------------------
raw_input('press any key to continue\n')

## Isomap visualization
# The data is loaded from dim1.mat and dim2.mat 

matFile = h5py.File('dim1.mat')
dim1 = matFile['dim1']
dim1 = np.array(dim1)
dim1 = dim1.T

matFile = h5py.File('dim2.mat')
dim2 = matFile['dim2']
dim2 = np.array(dim2)
dim2 = dim2.T

# In this block we fit images of the faces to the points in 2D.

th = np.arange(0,2*pi+pi/50,pi/50)
xunit_a = 1 * cos(th) + 40.1
yunit_a = 1 * sin(th) + -21.74

xunit_b = 1 * cos(th) + -45.98
yunit_b = 1 * sin(th) + -3.783

xunit_c = 1 * cos(th) + -7.17
yunit_c = 1 * sin(th) + 10.47

xunit_d = 1 * cos(th) + -12.01
yunit_d = 1 * sin(th) + -26.69

xunit_e = 1 * cos(th) + 28.32
yunit_e = 1 * sin(th) + 12.04

xunit_f = 1 * cos(th) + 3.361
yunit_f = 1 * sin(th) + -6.631

xunit_g = 1 * cos(th) + 18.92
yunit_g = 1 * sin(th) + -20.51

xunit_h = 1 * cos(th) + -27.76
yunit_h = 1 * sin(th) + 4.455

xunit_i = 1 * cos(th) + -28.14
yunit_i = 1 * sin(th) + -15.11

xunit_j = 1 * cos(th) + 9.127
yunit_j = 1 * sin(th) + 8.518

xunit_k = 1 * cos(th) + -15.1
yunit_k = 1 * sin(th) + -7.862

xunit_l = 1 * cos(th) + 2.498
yunit_l = 1 * sin(th) + -24.67

xunit_m = 1 * cos(th) + 20.44
yunit_m = 1 * sin(th) + -4.112

xunit_n = 1 * cos(th) + -19.02
yunit_n = 1 * sin(th) + 14.3

xunit_o = 1 * cos(th) + -2.943
yunit_o = 1 * sin(th) + 21.79

xunit_p = 1 * cos(th) + 14.6
yunit_p = 1 * sin(th) + 18.3

xunit_q = 1 * cos(th) + 39.57
yunit_q = 1 * sin(th) + -7.296

xunit_r = 1 * cos(th) + 30.11
yunit_r = 1 * sin(th) + -17.24

xunit_s = 1 * cos(th) + -39.55
yunit_s = 1 * sin(th) + 5.48

a_img = images[:,613].reshape(64, 64).T
b_img = images[:,571].reshape(64, 64).T
c_img = images[:,245].reshape(64, 64).T
d_img = images[:,652].reshape(64, 64).T
e_img = images[:,243].reshape(64, 64).T
f_img = images[:,339].reshape(64, 64).T
g_img = images[:,577].reshape(64, 64).T
h_img = images[:,357].reshape(64, 64).T
i_img = images[:,24].reshape(64, 64).T
j_img = images[:,290].reshape(64, 64).T
k_img = images[:,84].reshape(64, 64).T
l_img = images[:,593].reshape(64, 64).T
m_img = images[:,524].reshape(64, 64).T
n_img = images[:,181].reshape(64, 64).T
o_img = images[:,590].reshape(64, 64).T
p_img = images[:,197].reshape(64, 64).T
q_img = images[:,499].reshape(64, 64).T
r_img = images[:,92].reshape(64, 64).T
s_img = images[:,645].reshape(64, 64).T

plt.figure(3)

plt.ion()
plt.scatter(dim1, dim2, s= 18 * np.ones((698, 1)), zorder=-1)
plt.hold(True)

plt.imshow(a_img, extent=[37, 43, -25, -31], origin='lower', cmap='gray')
plt.plot(xunit_a, yunit_a,'red')

plt.imshow(b_img, extent=[-48, -42, -5, -11], origin='lower', cmap='gray')
plt.plot(xunit_b, yunit_b,'red')

plt.imshow(c_img, extent=[-10, -4, 8, 2], origin='lower', cmap='gray')
plt.plot(xunit_c, yunit_c,'red')

plt.imshow(d_img, extent=[-15, -9, -28, -34], origin='lower', cmap='gray')
plt.plot(xunit_d, yunit_d,'red')

plt.imshow(e_img, extent=[25, 31, 11, 5], origin='lower', cmap='gray')
plt.plot(xunit_e, yunit_e,'red')

plt.imshow(f_img, extent=[0, 6, -7.7, -13.5], origin='lower', cmap='gray')
plt.plot(xunit_f, yunit_f,'red')

plt.imshow(g_img, extent=[15, 21, -21.6, -27.6], origin='lower', cmap='gray')
plt.plot(xunit_g, yunit_g,'red')

plt.imshow(h_img, extent=[-30.76, -24.76, 3.3, -2.7], origin='lower', cmap='gray')
plt.plot(xunit_h, yunit_h,'red')

plt.imshow(i_img, extent=[-31.14, -25.14, -16.2, -22.2], origin='lower', cmap='gray')
plt.plot(xunit_i, yunit_i,'red')

plt.imshow(j_img, extent=[6.127, 12.127, 7.418, 1.418], origin='lower', cmap='gray')
plt.plot(xunit_j, yunit_j,'red')

plt.imshow(k_img, extent=[-18.1, -12.1, -8.96, -14.96], origin='lower', cmap='gray')
plt.plot(xunit_k, yunit_k,'red')

plt.imshow(l_img, extent=[-0.51, 5.49, -25.77, -31.77], origin='lower', cmap='gray')
plt.plot(xunit_l, yunit_l,'red')

plt.imshow(m_img, extent=[17.44, 23.44, -5.21, -11.21], origin='lower', cmap='gray')
plt.plot(xunit_m, yunit_m,'red')

plt.imshow(n_img, extent=[-22.02, -16.02, 13.2, 7.2], origin='lower', cmap='gray')
plt.plot(xunit_n, yunit_n,'red')

plt.imshow(o_img, extent=[-5.94, 0.06, 20.69, 14.69], origin='lower', cmap='gray')
plt.plot(xunit_o, yunit_o,'red')

plt.imshow(p_img, extent=[11.6, 17.6, 17.2, 11.2], origin='lower', cmap='gray')
plt.plot(xunit_p, yunit_p,'red')

plt.imshow(q_img, extent=[36.57, 42.57, -8.39, -14.39], origin='lower', cmap='gray')
plt.plot(xunit_q, yunit_q,'red')

plt.imshow(r_img, extent=[27.11, 33.11, -18.34, -24.34], origin='lower', cmap='gray')
plt.plot(xunit_r, yunit_r,'red')

plt.imshow(s_img, extent=[-42.55, -37.55, 4.38, -1.62], origin='lower', cmap='gray')
plt.plot(xunit_s, yunit_s,'red')

plt.gca().invert_yaxis()

plt.title('isomap')

plt.show()
plt.ioff()
plt.show()