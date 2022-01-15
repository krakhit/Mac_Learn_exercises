# This is a first principle code for predicting housing price using
#1Normal equation
# This is an exercise from Coursera's Machine Learning course bt Andrew Ng.
# I like to work with vectors, and sometimes adding an extra dimension in the arrays allows one to differentiate
#between a vector and its transpose.

import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from numpy import linalg as LA, matmul
import matplotlib.pyplot as plt
import time

# read dataset

d_set = pd.read_csv('ex1data2.txt', sep =',', names= ['Size','# rooms','price'])
d_arr = d_set.to_numpy()
#get the columns (dim,)
siz = np.array(d_arr[:,0])
dim = np.size(siz)
room = np.array(d_arr[:,1])
price = np.array(d_arr[:,2])
ons = np.ones(dim)

#this reads it into a row vector with  (1,dim)
siz_r = np.array([siz])
room_r = np.array([room])
price_r = np.array([price])
ons_r = np.array([ons])

#this converts it into a column vector with shape (dim,1)  elements are like [ [1], [2], [3], ... ,dim]
siz_c = np.transpose(siz_r)
room_c = np.transpose(room_r)
price_c = np.transpose(price_r)
ons_c = np.transpose(ons_r)

# X matrix (dim, features+1)
X_arr = np.transpose(np.array([ons,siz,room]))
# X matrix (features+1,dim)
X_arrt= np.transpose(X_arr)


#Normal equation,  will come out like siz_c (dim,1)
t_in = time.time()
# X' Y
Xty = np.matmul(X_arrt,price_c)
# (X' X)^{-1}
Xtx = np.matmul(X_arrt,X_arr)
Xtx_in = LA.inv(Xtx)
#theta = (X' X)^{-1}. X'. y
theta_c = np.matmul(Xtx_in,Xty)
t_out =time.time()
print('The fitting parameters from the normal eq are ',theta_c)
print('Time taken: ',t_out-t_in)
#print(np.shape(theta_c))
# # # test data for prediction, passes test
#test = np.array([[1, 1650, 3]])
#print(np.shape(test))
#print(np.matmul(test,theta_c))
#passes test
