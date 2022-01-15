# This is a first principle code for predicting housing price using
#1. Gradient descent
# This is an exercise from Coursera's Machine Learning course bt Andrew Ng.
from matplotlib import colors
import numpy as np
import pandas as pd
from numpy import linalg as LA, matmul
import matplotlib.pyplot as plt
import time
    
#definition of cost function:
def J_CF(X,y,teta):
    di = np.size(y)
    diff =  np.matmul(X,teta) - y
    difft = np.transpose(diff)
    jf = 1/(2*di) * np.matmul(difft,diff).item()
    return jf

# read dataset

d_set = pd.read_csv('ex1data2.txt', sep =',', names= ['Size','# rooms','price'])
d_arr = d_set.to_numpy()
#get the columns
siz = np.array(d_arr[:,0])
dim = np.size(siz)
room = np.array(d_arr[:,1])
price = np.array(d_arr[:,2])
ons = np.ones(dim)

#this reads it into a row vector with n columns -> this extra [] , makes the shaoe (1,n)
siz_r = np.array([siz])
room_r = np.array([room])
price_r = np.array([price])
ons_r = np.array([ons])

#this converts it into a column vector with 1 columns ->shape (n,1) 
siz_c = np.transpose(siz_r)
room_c = np.transpose(room_r)
price_c = np.transpose(price_r)
ons_c = np.transpose(ons_r)

# X matrix (dim, features+1)
X_arr = np.transpose(np.array([ons,siz,room]))
## visualize data set
#create empty 3d plot
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# #add data
# ax.scatter(siz,room,price,color='Red')
# #add axes
# ax.set_xlabel('House Size')
# ax.set_ylabel('# of rooms')
# ax.set_zlabel('Price')
# ax.set_title('Housing prices')
# plt.show()

# categorical plotting
# fis,axo = plt.subplots(1,2,sharey=True)
# axo[0].scatter(siz,price)
# axo[0].set_xlabel('House size')
# axo[1].scatter(room,price)
# axo[1].set_xlabel('# of rooms')
# fis.suptitle('price vs each feature')
# plt.show()

## Gradient descent
#mean normalize data for gradient descent
mu_siz =  np.mean(siz)
sd_siz =  np.std(siz)
mu_rm =  np.mean(room)
sd_rm =  np.std(room)
siz_mn =  (siz - mu_siz)/(sd_siz)
room_mn =  (room - mu_rm)/(sd_rm)
X_arr_mn = np.transpose(np.array([ons,siz_mn,room_mn]))
X_arr_mnt= np.transpose(X_arr_mn)
theta = np.zeros(3)
theta_r = np.array([theta])
theta_c = np.transpose(theta_r)

alpha  = 0.01
Jarr =  []
t_in = time.time()
it_er = 500
for i in range(it_er):
    delta_c =  1/dim * (np.matmul(np.matmul(X_arr_mnt,X_arr_mn),theta_c) - np.matmul(X_arr_mnt,price_c))
    theta_c =  theta_c - alpha * delta_c
    jf = J_CF(X_arr_mn,price_c,theta_c)
    Jarr.append(jf)
t_out = time.time()
time_taken =  t_out - t_in
print(theta_c)
##hypothesis
hp = np.squeeze(np.matmul(X_arr_mn,theta_c),1)
##convergence of the cost function

fis,axo = plt.subplots(1,3,sharey=False)
axo[0].scatter(siz,price,color='Blue')
axo[0].plot(X_arr[:,1],hp,color='Red')
axo[0].set_xlabel('House size')
axo[0].set_ylabel('Price')
axo[1].scatter(room,price,color='Blue')
axo[1].plot(X_arr[:,2],hp,color='Red')
axo[1].set_xlabel('Number of rooms')
axo[1].set_ylabel('Price')
axo[2].plot(Jarr)
axo[2].set_ylabel('Cost function')
axo[2].set_xlabel('Iterations')
fis.suptitle('Data vs prediction')
plt.show()
print('The house size is an appropriate feature, that indicates the price variation as compared to number of rooms.')
print('Gradient descent took '+ str(time_taken) + 'seconds to converge in ' + str(it_er) + ' iterations')


# # test data for prediction, passes test
# test = np.array([[1, (1650-mu_siz)/sd_siz, (3-mu_rm)/sd_rm]])
# print(np.matmul(test,theta_c))
