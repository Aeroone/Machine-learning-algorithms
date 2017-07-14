############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
############################################################################
import numpy as np
import matplotlib.pyplot as plt

def kernel_pca(k):
    
    # Simple 2d data with y = x^2 + gaussian(0,0.2)
    # x_1 = linspace(-1,1,100);
    # noise = mvnrnd(0,0.2,100);
    # x_1 = x_1';
    # x_2 = x_1.^2 + noise;
    # x = [x_1, x_2];
    x = np.random.multivariate_normal(np.array([0,0]), 0.1 * np.eye(2), 30)
    x = np.concatenate((x, np.random.multivariate_normal(np.array([2,2]), 0.1 * np.eye(2), 30)), axis = 0)
    x = np.concatenate((x, np.random.multivariate_normal(np.array([2,-2]), 0.1 * np.eye(2), 30)), axis = 0)
    m = x.shape[0]
    print x    
        
    # Calculate Kernel Matrix
    poly_kernel = np.power((x.dot(x.T)), k)
    poly_kernel = 1.0 / m * poly_kernel
    
    # Calculate Gram matrix
    one_matrix = 1.0 / m * np.ones((m, m))
    poly_kernel_hat = poly_kernel - one_matrix.dot(poly_kernel) - poly_kernel.dot(one_matrix) + \
                      one_matrix.dot(poly_kernel).dot(one_matrix)
    
    
    d, v = np.linalg.eig(poly_kernel_hat)
    d = d[0:6].real
    v = v[:, 0:6].real
    
    print 'The Eigenvalues are'
    print d    
    
    # Obtain maximum eigenvector
    eigenval = d
    C = np.max(eigenval)
    
    for l in range(0, 6):
        
        # normalize the eigenvector
        norm_eigenvec = np.linalg.norm(v[:, l])
        eigenvec = v[:, l] / norm_eigenvec
        
        eigenvec = eigenvec.reshape(eigenvec.shape[0], 1)

        # Establish the mesh grid of test points
        test_x = np.linspace(-3, 3, 50)
        test_x = test_x.reshape(test_x.shape[0], 1)
        test_y = np.linspace(-3, 3, 50)
        test_y = test_y.reshape(test_y.shape[0], 1)
                        
        # Compute the values
        Z = np.zeros((50, 50))
        for i in range(0, 50):
            for j in range(0, 50):
                temp = np.array([test_x[i, 0], test_y[j, 0]])
                temp = temp.reshape(1, temp.shape[0])
                Z[i, j] = np.sum(eigenvec * np.power(x.dot(temp.T), k), axis = 0)[0]
         
        gridx, gridy = np.meshgrid(test_x.T, test_y.T)

        plt.figure()
        plt.scatter(x[:, 0], x[:, 1])
        plt.hold(True)        
        plt.contour(gridx, gridy, Z, 20)        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Eigenfunction' + str(l))       
        
    return C

k = 2
c = kernel_pca(k)
print c   
plt.show() 