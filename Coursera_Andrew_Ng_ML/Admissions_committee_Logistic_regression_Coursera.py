import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
## this is one of the methods that can avoid overflow or underflow, the second method is to use a slightly different 
# rep each for positive or negative argument 
# def exp_normalize(z):
#     z_max= np.amax(z)
#     return np.exp(z-z_max)

## Define Sigmoid function for Logistic regression
def hypo(X_arr,theta_c):
    return np.dot(X_arr,theta_c)

def sigmoid(z):
     #X_arr (dim, features+1), theta = features+1,1 
    return 1.0/(1.0+ np.exp(-1.0*z))

## define the cost function for LR
def cost_f(X_arr,theta_c,y_c):
    dim =  np.shape(X_arr)[0]
    hyp = sigmoid(hypo(X_arr,theta_c))
    ## the standard cost function is - (y log h + (1-y) log (1-h))
    err=[]
    ep = 1e-5
    for i in range(dim):
        if y_c[i] == 1:
            err.append(-y_c[i] * np.log(hyp[i]))
        elif y_c[i] == 0:
            err.append(-(1-y_c[i]) * np.log(1-hyp[i]))

    cost = 1/dim * sum(err)
    grad = 1/dim * np.dot(np.transpose(X_arr),(hyp-y_c))
    return cost,grad

def grad_desc(X_arr,y_c,theta_c,alpha,iter):
    cost_arr = []
    for i in range(iter):
        cost,grad = cost_f(X_arr,theta_c,y_c)
        theta_c = theta_c - (alpha* grad)
        cost_arr.append(cost)

    return theta_c, cost_arr

# read dataset
df = pd.read_csv('ex2data1.txt', sep =',', names= ['Exam_1','Exam_2','Admission_Status'])
df_selected = df[df['Admission_Status'] == 1][{'Exam_1','Exam_2'}]
df_not_selected = df[df['Admission_Status'] == 0][{'Exam_1','Exam_2'}]
plt.scatter(df_selected['Exam_1'],df_selected['Exam_2'], color='Green',label='Selected')
plt.scatter(df_not_selected['Exam_1'], df_not_selected[['Exam_2']], color='Red',label='Not Selected')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.title('Admissions comittee data')
plt.legend()
plt.show()

dim = df.shape[0]
ones = [1]*dim
# un-normalized values cause gradient descent to be very slow
sd_Ex_1= df['Exam_1'].std()
sd_Ex_2= df['Exam_2'].std()
mean_Ex_1 = df['Exam_1'].mean()
mean_Ex_2 = df['Exam_2'].mean()
norm_Ex_1 = (df['Exam_1'] - mean_Ex_1)/sd_Ex_1
norm_Ex_2 = (df['Exam_2'] - mean_Ex_2)/sd_Ex_2
x_dat_raw = pd.DataFrame({'x_0' : ones, 'x_1': df['Exam_1'], 'x_2': df['Exam_2']})
x_dat = pd.DataFrame({'x_0' : ones, 'x_1': norm_Ex_1, 'x_2': norm_Ex_2})

y_dat = df['Admission_Status']
x_arr = x_dat.to_numpy()
y_arr = y_dat.to_numpy()
rows,features =  np.shape(x_arr)[0],np.shape(x_arr)[1]
#seed theta
theta_in = np.zeros(shape=features, dtype=np.float32)
test= np.array([1,45,85])
ite =400
theta_c,cost_arr= grad_desc(x_arr,y_arr,theta_in,1,ite)

plt.plot(cost_arr)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')
plt.title('Convergence of Logistic regression')
plt.show()

print('Initial cost function value: ',cost_arr[0])
print('Final cost function value: ',cost_arr[ite-1])
print('Final Gradients:', theta_c)

#plot the decision boundary
intercept =  -theta_c[0]/theta_c[2]
slope = -theta_c[1]/theta_c[2]

# Plot the data and the classification with the decision boundary.
# we calculated the theta for normalized values we have to write the x and y variables in
#the un normalized form in order to plot on the same scale
# (x_2 - x_2m)/x_2sd = slope (x_1-x_1m)x_1sd - intercept x_0
# x_2/x_2sd = slope* x_1/x_1sd - intercept x_0 + x_2m/x_2sd - slope * x_1m/x_1sd
# x_2 = slope * x_1  x_2sd/x_1sd - intercept x_2sd * x_0 + x_2m + slope * + x_1m/x_1sd * x_2 sd

xd = np.array([x_dat_raw['x_1'].min(),x_dat_raw['x_1'].max()])
yd = slope * sd_Ex_2/sd_Ex_1 * xd + intercept * sd_Ex_2 + mean_Ex_2 - slope * mean_Ex_1/sd_Ex_1 * sd_Ex_2

plt.scatter(df_selected['Exam_1'],df_selected['Exam_2'], color='Green',label='Selected')
plt.scatter(df_not_selected['Exam_1'], df_not_selected[['Exam_2']], color='Red',label='Not Selected')
plt.plot(xd, yd)
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.title('Decision-boundary for Logistic regression (Admission Committee data)')
plt.legend()
plt.show()

## accuracy
y_model = sigmoid(hypo(x_dat,theta_c))
y_pred=[]
for i in range(len(y_model)):
    if y_model[i]>=0.5:
        y_pred.append(1.0)
    else:
        y_pred.append(0.0)

data =  np.transpose([y_arr,y_pred])
compare = pd.DataFrame(data, columns=['Dataset','Predictions'])

cm11 = compare[(compare['Dataset'] == 1.0) & (compare['Predictions'] == 1.0)].Predictions.count()
cm10 = compare[(compare['Dataset'] == 1.0) & (compare['Predictions'] == 0.0)].Predictions.count()
cm01 = compare[(compare['Dataset'] == 0.0) & (compare['Predictions'] == 1.0)].Predictions.count()
cm00 = compare[(compare['Dataset'] == 0.0) & (compare['Predictions'] == 0.0)].Predictions.count()
cm =  [[cm11,cm10],[cm01,cm00]]
confusion_matrix = pd.DataFrame(cm,index=['Actual 1', 'Actual 0'], columns=['Predicted 1','Predicted 0'])
print('The confusion matrix is:\n',confusion_matrix)
model_accuracy =  (cm11+cm00)/(cm11+cm00+cm01+cm10)
print('Model accuracy is: {:.1%}'.format(model_accuracy))
# # test data for prediction, passes test

test = np.array([[1, (45-mean_Ex_1)/sd_Ex_1, (85-mean_Ex_2)/sd_Ex_2]])
prob  = float(sigmoid(hypo(test,theta_c)))
print('Prediction: The probability that a student with a score of 45 in Exam_1 and 85 in Exam_2, gets selected by the committee is: \n{:.1%}'.format(prob))