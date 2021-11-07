#Matrix and linear algebra operations
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

#by default numpy doesnt differentiate between row and column vector, it is important that we tell 
#in most cases it is ok. but if one needs, this is a way to do it.
r1 = np.array([[1,2,3,4]]) #(1,4)
r2 = np.array([[5],[6],[7],[8]]) #(4.1)
print(r1,'\n', r2)
print(np.matmul(r2,r1)) # gives a matrix
print(np.matmul(r1,r2)) # gives a number