# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:37:43 2022

@author: rash0007
"""

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv("shakti.csv")
df1=pd.read_csv("HTRAIN.csv")

X=df
Y=df1["SalePrice"]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3)

re=LinearRegression()
re.fit(X,Y)

Y_pred=re.predict(X_test)

sc=r2_score(Y_test,Y_pred)
sc
pickle.dump(re,open('M1','wb'))