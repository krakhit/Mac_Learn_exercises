import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
def similarity(pass_a,pass_b):
    res = 0.0
    if pass_a.Pclass == pass_b.Pclass:
        res+=1.0
    if pass_a.Sex == pass_b.Sex:
        res+=3.0
    if pass_a.SibSp == pass_b.SibSp:
        res+=1.0
    if pass_a.Parch == pass_b.Parch:
        res+=1.0
    diff_Age = float(round(2 - abs(pass_a.Age - pass_b.Age)* 0.1,2))
    diff_Fare = float(round(2 - abs(pass_a.Fare - pass_b.Fare)* 0.2,2)) 
    #i hate this numpy.64 object is not callable error...
    if diff_Age>0.0:
        res+=diff_Age
    else:
        res += 0.0
    if diff_Fare>0.0:
        res+=diff_Fare
    else:
        res += 0.0

    return round(res*0.1,2)

df =pd.read_csv('titanic_clean_up.csv')
col_names = df.columns.values
#print(col_names)
#['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
# 'Fare' 'Embarked']

min = df['PassengerId'].min()
max = df['PassengerId'].max()

q = int(input('Enter number of similarity comparisons q, 1 <=q <= 100: '))
for i in range(1,q):
    for j in range(1,q):
        us_a = df[df['PassengerId'] == i].iloc[0]
        us_b = df[df['PassengerId'] == j].iloc[0]
        print('Input',i,j, 'output:', similarity(us_a,us_b))
        

