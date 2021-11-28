import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from sklearn import datasets, linear_model, preprocessing
import pandas as pd
import time

d_set = pd.read_csv('ex1data2.txt', sep =',', names= ['Size','# rooms','price'])
d_X = d_set[['Size','# rooms']]
d_y = d_set['price']
t1=time.time()
reg = linear_model.LinearRegression()
reg.fit(d_X,d_y)
print('Coefficients:', reg.coef_)
t2=time.time()
df_test = pd.DataFrame(np.array([[1650, 3]]), columns=['Size','# rooms'])
y_pred=reg.predict(d_X)
y_pred_test = reg.predict(df_test)
print(y_pred_test)
print('Time taken: ', t2-t1)
fis,axo = plt.subplots(1,2,sharey=False)
axo[0].scatter(d_X[['Size']],d_y,color='Black')
axo[0].plot(d_X[['Size']],y_pred,color='Blue')
axo[0].scatter(df_test[['Size']],y_pred_test,color='Red')
axo[0].set_xlabel('House size')
axo[0].set_ylabel('Price')
axo[1].scatter(d_X[['# rooms']],d_y,color='Black')
axo[1].plot(d_X[['# rooms']],y_pred,color='Blue')
axo[1].scatter(df_test[['# rooms']],y_pred_test,color='Red')
axo[1].set_xlabel('Number of rooms')
axo[1].set_ylabel('Price')
fis.suptitle('Data vs prediction')
plt.show()
