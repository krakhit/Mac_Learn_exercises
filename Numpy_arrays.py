# Array operations 
import numpy as np
#print a random matrix of numbers of size 10)
r1 = np.array([1,2,3])
r2 = np.array([21,22,23])
r3 = np.array([9,5,8])
# array indexing begins from 0
# matrix from arrays
A_mat = np.array([r1,r2,r3])
#matrix indexing begins from A_{11}= A[0,0]
# np.empty() makes an empty array
# np.arange(0,9,2) makes an array from 0 - 9 in steps of 2
# default data types are np.float64, np.ones(2,dtype=np.int64)
# 'A sorted array: np.sort(r1)), unfortunately this only does ascending
# descending sort is done by reversing r2 = np.sort(r1) then r2[::-1], 
# remember that array indexing from left is 0,1,... whereas the reverse is -1,-2,...
# 'A joined array:', np.concatenate((r1,r2))
# number of axes of the matrix is: ', A_mat.ndim
# 'Shape of the matrix is ', A_mat.shape)
#'total nymber of elements of the array is ', A_mat.size)
print('One can also reshape the array\n ', A_mat.reshape(9,1))
A1_new = np.expand_dims(A_mat, axis =1)
print('One can also add a new axis', A1_new.shape)
print('The slice of data from r1(0,1) is: ', r1[0:2])
print('The slice of data from r1(1,2) is: ', r1[-2:])
print('You can also pick elements inside with a condition!', A_mat[(A_mat>5) & (A_mat<8)])
print('You can also put boolean conditions\n', (A_mat >5) | (A_mat<2))
n_st = np.vstack((r1,r2,r3))
print('Stacking arrays: or building matrices' , n_st)
n_svt = np.hstack((r1,r2,r3))
print('Stacking arrays: or building matrices' , n_svt)
print('Arrays can be added or subtracted:', r1+r2)
print('max of an array: ',A_mat.max())
print('min of an array: ',A_mat.min())
print('Sum of an array (all elements): ',A_mat.sum())
print('Sum of an array (on x axis): ',A_mat.sum(axis=0))
print('Sum of an array (on x axis): ',A_mat.sum(axis=1))
rng = np.random.default_rng(4) #random no generator
print('Here is a random array: ', rng.random(3))
print('Here is a random integer array: ', rng.integers(3, size =(3,3)))
de_f = [np.zeros(3),np.ones(3), rng.random(3)]
print('Uniques: ', np.unique(de_f,return_index=True))
print('Occurence count: ', np.unique(de_f,return_counts=True))
print('Transpose', A_mat.T)
print('RMS')
t_pre= np.array([1,5,7])
t_lab= np.array([9,8,2])
RM_S = 1/3 * np.sum(np.square(t_pre - t_lab))
print('RMS:', RM_S )
# save an array np.save('filename', arrayname) or np.savez for multiple
# load an array np.load('filename.npy')
# CSV  format np.savetxt('filename.csv', arrayname)  and np.loadtxt
