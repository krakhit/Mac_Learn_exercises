import matplotlib
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt

## Define Sigmoid function for Logistic regression
def hypo(X_arr,theta_c):
    return np.dot(X_arr,theta_c)

def sigmoid(z):
     #X_arr (dim, features+1), theta = features+1,1 
    return 1.0/(1.0+ np.exp(-1.0*z))

def cost_f(X_arr,theta_c,y_c,reg_param):
    dim =  np.shape(X_arr)[0]
    hyp = sigmoid(hypo(X_arr,theta_c))
    ## the standard cost function is - (y log h + (1-y) log (1-h))
    err=[]
    # grad=theta_c
    ep = 1e-5
    for i in range(dim):
        if y_c[i] == 1:
            err.append(-y_c[i] * np.log(hyp[i]))
        elif y_c[i] == 0:
            err.append(-(1-y_c[i]) * np.log(1-hyp[i]))

    cost = 1/dim * sum(err) + reg_param/(2*dim) * np.dot(theta_c,theta_c)
    grad_0 = np.array(dim * np.dot(np.transpose(X_arr[:,0]),(hyp-y_c)))
    grad_rest = np.array(1/dim * np.dot(np.transpose(X_arr[:,1:]),(hyp-y_c)) + reg_param/dim * theta_c[1:]).tolist()
    grad = np.hstack((grad_0,grad_rest))
    return cost, grad

def grad_desc(X_arr,y_c,theta_c,alpha,iter,reg_param):
    cost_arr = []
    for i in range(iter):
        cost,grad = cost_f(X_arr,theta_c,y_c,reg_param)
        theta_c = theta_c - (alpha* grad)
        cost_arr.append(cost)

    return theta_c, cost_arr

def pow(x,n):
    return x**n

def map_feature(x_1,x_2,n):
    temp=[]
    #t=0
    for r in range(n+1):
        for  j in range(r+1): 
            temp.append(pow(x_1,r-j)*pow(x_2,j))
            # t=t+1
            # print('#:',t,' a^{} b^{}'.format(r-j,j))
    return np.array(temp)

def mean_normalize(x):
    mu  = x.mean()
    sd = x.std()
    x_nor = (x - mu)/sd
    return x_nor

# read dataset
df = pd.read_csv('ex2data2.txt', sep =',', names= ['Test_1','Test_2','QA_Status'])
df_passed = df[df['QA_Status'] == 1][{'Test_1','Test_2'}]
df_failed = df[df['QA_Status'] == 0][{'Test_1','Test_2'}]
plt.scatter(df_passed['Test_1'],df_passed['Test_2'], color='Green',label='Passed')
plt.scatter(df_failed['Test_1'], df_failed[['Test_2']], color='Red',label='Failed')
plt.xlabel('QA Test 1')
plt.ylabel('QA Test 2')
plt.title('QA Test Results for Microchip')
plt.legend()
plt.show()

# from the data plots it is clear that linear hypothesis will not work
# implement feature selection
test_1 = df['Test_1'].to_numpy()
test_2 = df['Test_2'].to_numpy()
y_arr = df['QA_Status'].to_numpy()

test_1_norm = mean_normalize(test_1)
test_2_norm = mean_normalize(test_2)

feature_selected = np.transpose(map_feature(test_1,test_2,6))
feature_selected_norm = np.transpose(map_feature(test_1_norm,test_2_norm,6))
X_arr = feature_selected_norm
rows,features =  np.shape(feature_selected_norm)[0],np.shape(feature_selected_norm)[1]
theta_in = np.zeros(shape=features, dtype=np.float32)
# cost, grad = cost_f(X_arr,theta_in,y_arr,1000)
iter = 20000
alpha = 0.0001
reg_param= 1
theta_c,cost_arr=grad_desc(X_arr,y_arr,theta_in,alpha,iter,reg_param)
print('Initial cost:',cost_arr[0])
print('Final cost:',cost_arr[-1])

#convergence of cost function
plt.plot(cost_arr)
plt.xlabel('Iterations')
plt.ylabel('Cost function value')
plt.title('Convergence of cost function with mean normalized inputs')
plt.show()

# accuracy of the model
## accuracy
y_model = sigmoid(hypo(X_arr,theta_c))
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
print('Number of iterations:',iter,' rate of gradient descent:',alpha,' regularization:',reg_param)
# # decision boundary

mean_test_1= df['Test_1'].mean()
mean_test_2= df['Test_2'].mean()
std_test_1= df['Test_1'].std()
std_test_2= df['Test_2'].std()

x_coord  = np.linspace(-1,1,100)
y_coord =  np.linspace(-1,1,100)
siz = len(x_coord)
grd = np.zeros((siz,siz))
for i in range(siz):
     for j in range(siz):
         grd[i,j] = np.dot(map_feature((x_coord[i]-mean_test_1)/std_test_1,(y_coord[j]-mean_test_2)/std_test_2,6),theta_c)

# xd = np.array([x_dat_raw['x_1'].min(),x_dat_raw['x_1'].max()])
# yd = slope * sd_Ex_2/sd_Ex_1 * xd + intercept * sd_Ex_2 + mean_Ex_2 - slope * mean_Ex_1/sd_Ex_1 * sd_Ex_2
plt.scatter(df_passed['Test_1'],df_passed['Test_2'], color='Green',label='Passed')
plt.scatter(df_failed['Test_1'], df_failed[['Test_2']], color='Red',label='Failed')
plt.contour(x_coord,y_coord,grd.T,0)
plt.xlabel('QA Test 1')
plt.ylabel('QA Test 2')
plt.title('Decision boundary for QA tests with regularization')
plt.legend()
plt.show()


#     z = zeros(length(u), length(v));
#     % Evaluate z = theta*x over the grid
#     for i = 1:length(u)
#         for j = 1:length(v)
#             z(i,j) = mapFeature(u(i), v(j))*theta;
#         end
#     end
#     z = z'; % important to transpose z before calling contour

#     % Plot z = 0
#     % Notice you need to specify the range [0, 0]
#     contour(u, v, z, [0, 0], 'LineWidth', 2)