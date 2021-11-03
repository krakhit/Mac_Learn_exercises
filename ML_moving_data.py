import numpy as np
from numpy.core.fromnumeric import size
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from numpy.core.fromnumeric import size
#from numpy.linalg import inv
#from numpy import linalg as LA
#A1 = np.array([[1,3,4],[2,4,9],[0,5,7]])
#print("The Matrix" , A1, " is of dimensions ", size(A1,0) , "x" ,size(A1,1))
## Extracting data from a csv or data file
rnt = np.genfromtxt('/Users/karthikinbasekar/Desktop/ML_Stanford/octave_trial/d1.csv',delimiter=',')
print("The dimensions are: ", size(rnt,0),"x",size(rnt,1))
## Extracting part of data from data array
rnex = np.array(rnt[0:2,:])
## adding the identity array to the existing array
r1= np.array([1,1])
rnew= np.vstack((r1,rnex))
print(rnex)
## Extracting various columns into one dimensional vectors
x_ax = rnt[:,[0]]
y_ax = rnt[:,[1]]
## Ploting data point value (m) vs data value
#plt.plot(y_ax)
#plt.plot(x_ax)
#plt.show()