import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
A1 = np.array([[1,3,4],[2,4,9],[0,5,7]])
# AR = np.array([[7,9],[a,b],[c,d]])
A1_inv = inv(A1)
res = np.matmul(A1,A1_inv)
print('Matrix:' , A1)
print('Eigenvalues and Eigenvectors: ',LA.eig(A1))
print('Determinant ',LA.det(A1))
print('Inverse: ', A1_inv)
print('A1.A1^(-1)= I',res)