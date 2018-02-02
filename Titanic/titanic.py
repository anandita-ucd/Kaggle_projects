# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:48:34 2018

@author: A
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt 

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Handling missing Age
ax = train["Age"].hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
train["Age"].fillna(train["Age"].median(skipna=True), inplace=True)

ax = test["Age"].hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
test["Age"].fillna(test["Age"].median(skipna=True), inplace=True)
#print(train['Age'])


#Handling missing Embarked
train["Embarked"].fillna("S", inplace=True)

#Handling missing Cabin
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

#Dropping PassengerId and Name
drop_columns = ['PassengerId', 'Name']
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

#Combining SibSp and Parch into one column whether or not travelling alone 
train['TravelAlone']= 0 if train["SibSp"]+train["Parch"] > 0 else 1

"""train['TravelAlone'].loc[train['TravelAlone']>0]=0
train['TravelAlone'].loc[train['TravelAlone']==0]=1"""

test['TravelAlone']=1 if (test["SibSp"]+test["Parch"])==0 else 0
"""test['TravelAlone'].loc[test['TravelAlone']>0]=0
test['TravelAlone'].loc[test['TravelAlone']==0]=1"""

#Drop SibSp and Parch
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)

#Handling the missing Fare only in test data
test["Fare"].fillna(test["Fare"].median(skipna=True), inplace=True)




