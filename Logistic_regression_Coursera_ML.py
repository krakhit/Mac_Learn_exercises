from matplotlib import colors
import numpy as np
import pandas as pd
from numpy import linalg as LA, matmul
import matplotlib.pyplot as plt
import time

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

#plot data set
#selected people
sel= d_set[d_set['Admission Status'] == True]
sel_arr= sel.to_numpy()
#Not selected people
n_sel= d_set[d_set['Admission Status'] == False]
n_sel_arr= n_sel.to_numpy()
#selected people for ex1 and ex2
sel_ex1= np.array(sel_arr[:,0])
sel_ex2= np.array(sel_arr[:,1])
# Not selected people for ex1 and ex2
n_sel_ex1= np.array(n_sel_arr[:,0])
n_sel_ex2= np.array(n_sel_arr[:,1])
plt.scatter(sel_ex1,sel_ex2, color='Green',label='Selected')
plt.scatter(n_sel_ex1,n_sel_ex2, color='Red',label='Not Selected')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.title('Admissions comittee data')
plt.legend()
plt.show()