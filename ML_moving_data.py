import numpy as np
from numpy.core.fromnumeric import size
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