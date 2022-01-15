import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#function measures similiarity of two rows (pasengers)
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
    #i hate this numpy.64 object is not callable error... so I am forced to dothis
    # max([0.0,diff_Age]) results in this problem. one could remedy this by treating 
    # all variables with numpy, but I am going for a quick fix here
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

q = int(input('Enter the range for maximum similarity 1 <= q <= 100: '))
us_a = df[df['PassengerId'] == q].iloc[0]

#compute all similarity scores as compared to the selected and put in a list
for j in range(1,q+1):
    #the passenger to be compared
    us_a = df[df['PassengerId'] == j].iloc[0]
    sim=[]*q
    ind=[]*q
    for i in range(1,892):
        #of course we are the best match for ourselves or the worst :P
        if i==j:
            continue
        else:
            # getting  passenger B to compare with
            us_b = df[df['PassengerId'] == i].iloc[0]
            # computing similarity of passenger A with B and make a list of all B's similarity score with A
            sim.append(similarity(us_a,us_b))
            # storing Id of B
            ind.append(i)
    data = np.transpose(np.array([ind,sim]))
    columns=['ind','sim_score']
    # this data frame has the similarity scores of A  vs B and the ID of B in the index
    compar = pd.DataFrame(data,columns=columns)
    # identifying the maximum matches 
    max_val = compar['sim_score'].max()
    max_matches= compar[compar['sim_score'] == max_val]['ind'].iloc[0]
    print('Maximum Match for PassengerId ',j, 'is/are PassengerId', int(max_matches))

