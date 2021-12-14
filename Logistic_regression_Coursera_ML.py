from matplotlib import colors
import numpy as np
from numpy.core.numeric import identity
import pandas as pd
from numpy import exp, linalg as LA, matmul, singlecomplex
import matplotlib.pyplot as plt
import time

## Define Sigmoid function for Logistic regression
def sigmoid(X_arr,theta_c):
     #X_arr (dim, features+1), theta = features+1,1 
     z = np.squeeze(np.matmul(X_arr,theta_c),1)
     return 1/(1+exp(-z))

## define the cost function for LR
def cost(X_arr,theta_c,y_c):
    epsilon = 0.0000000001
    m = np.shape(y_c)[0]
    h = sigmoid(X_arr,theta_c)
    t1 = np.matmul(np.transpose(y_c), np.log(h+epsilon))
    t2 = np.matmul(np.transpose(1-y_c),np.log(1-h+epsilon))
    # print(h,t1,t2)
    JJ = 1/m * (-t1 - t2) 
    return np.squeeze(JJ).item()

## gradient descent function
def grad_desc(X_arr,theta_c,y_c,alpha,iter):
    m = np.shape(y_c)[0]
    cost_arr=[]
    for i in range(iter):
        h = np.transpose(np.array([sigmoid(X_arr,theta_c)])) 
        diff = h - y_c #row vector of (m,1)
        #X_arr is (m,f)
        delta =  np.matmul(np.transpose(X_arr),diff) 
        theta_c = theta_c - alpha/m * delta
        cost_f=cost(X_arr,theta_c,y_c)
        cost_arr.append(cost_f)

    return theta_c,cost_arr 


# read dataset
d_set = pd.read_csv('ex2data1.txt', sep =',', names= ['Exam_1','Exam_2','Admission Status'])
d_arr = d_set.to_numpy()
#get the columns
ex1 = np.array(d_arr[:,0])
dim = np.size(ex1)
ex2 = np.array(d_arr[:,1])
ad_stat = np.array(d_arr[:,2])
ons = np.ones(dim)

#this reads it into a row vector with n columns -> this extra [] , makes the shaoe (1,n)
ex1_r = np.array([ex1])
ex2_r = np.array([ex2])
ad_stat_r = np.array([ad_stat])
ons_r = np.array([ons])

#this converts it into a column vector with 1 columns ->shape (n,1) 
ex1_c = np.transpose(ex1_r)
ex2_c = np.transpose(ex2_r)
ad_stat_c = np.transpose(ad_stat_r)
ons_c = np.transpose(ons_r)

# X matrix (dim, features+1)
X_arr = np.transpose(np.array([ons,ex1,ex2]))

# #Visualize data set
# #selected people
# sel= d_set[d_set['Admission Status'] == True]
# sel_arr= sel.to_numpy()
# #Not selected people
# n_sel= d_set[d_set['Admission Status'] == False]
# n_sel_arr= n_sel.to_numpy()
# #selected people for ex1 and ex2
# sel_ex1= np.array(sel_arr[:,0])
# sel_ex2= np.array(sel_arr[:,1])
# # Not selected people for ex1 and ex2
# n_sel_ex1= np.array(n_sel_arr[:,0])
# n_sel_ex2= np.array(n_sel_arr[:,1])
# plt.scatter(sel_ex1,sel_ex2, color='Green',label='Selected')
# plt.scatter(n_sel_ex1,n_sel_ex2, color='Red',label='Not Selected')
# plt.xlabel('Exam 1')
# plt.ylabel('Exam 2')
# plt.title('Admissions comittee data')
# plt.legend()
# plt.show()


## Sigmoid check: works
# X_arr = np.array([[1,0, 0],[0,1,0],[0,0,1]])
# theta_c = np.array([[100],[100],[100]])
# h = np.transpose(np.array([sigmoid(X_arr,theta_c)]))
# print(np.shape(h)) 

## cost function checks - works
# theta_c = np.array([[0],[0],[0]])
# print(cost(X_arr,theta_c,ad_stat_c))

# gradient descent algorithm for classifier
theta_in = np.array([[0],[0],[0]])
print(cost(X_arr,theta_in,ad_stat_c))
theta_c,cost_arr = grad_desc(X_arr,theta_in,ad_stat_c,0.0001,400)
print(theta_c)
print(cost(X_arr,theta_c,ad_stat_c))
