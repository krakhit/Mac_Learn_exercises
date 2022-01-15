from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df =pd.read_csv('/Users/karthikinbasekar/Documents/test/test_BMC/FraudData.csv')
# get the column names
col_names = df.columns.values
#['Transaction ID' 'Type' 'Amount' 'Area Code' 'Client']

#create caregorized analysis for client and spending trend 
#in particular I want to know a given client, how often he spends, maximum, minimum and average
#Visualize data set
# df with client and no of transactions performed
total_no_transc = df.groupby('Client')['Amount'].count()
# df with client transactions, mean, max and min.
spec_of_transc = df.groupby('Client')['Amount'].agg([min,max,mean])

# convert into proper data frame with default index
transactions_summary= spec_of_transc.reset_index()
total = total_no_transc.reset_index() 
total_summary= pd.concat([total_no_transc, spec_of_transc], axis=1).reset_index()

#general trends (Level zero analysis)
fig, axes = plt.subplots(nrows=2, ncols=1)
plt.suptitle('General spending pattern per client')
transactions_summary.set_index('Client').plot(ax=axes[0],kind='bar')
total.plot(x='Client', y='Amount', kind= 'bar', color='red',ax=axes[1])
plt.show()

print('1. Based on the plot, we think that there are three high value spendings well above the average spending of all users')
print('2. All of these were executed in a single transaction')
print('3. That is our grounds for suspicion and we flag them below')
print('4. There are two spenders who have made several transactions above 12, we believe they are regular shopaholics.')
# Let us identify the people
top = total_summary[total_summary['Amount'] == 1].sort_values(by='mean', ascending=False, ignore_index=True)['Client']
top_3 = pd.DataFrame(top,columns=['Client'])[0:3]
fraud_transc = df[df['Client'].isin(top_3['Client'])]
print('The Fraudulent transactions appear to be:\n' , fraud_transc)
