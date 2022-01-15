import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data_clean_up
df =pd.read_csv('titanic.csv')
col_names = df.columns.values
#['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
# 'Ticket' 'Fare' 'Cabin' 'Embarked']
print('Before deleting, there are ',len(df.columns),'columns')
del df["Ticket"]
del df["Cabin"]
print('After deleting the columns Ticket and Cabin, there are', len(df.columns), 'columns')
# replace all missing values with mean age
avg =df['Age'].mean().round(decimals=2)
df['Age'] = df['Age'].fillna(avg)
df.to_csv('titanic_clean_up.csv',index = False)




