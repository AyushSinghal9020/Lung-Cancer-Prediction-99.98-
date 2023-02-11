# **DATA PROCESSING**

import numpy as np # Array Processing
import pandas as pd # Data Processing 

# **DATA ANALYSIS**

import seaborn as sns # Graphs
import matplotlib.pyplot as plt # Plots

# **PRE PROCESSING**

from sklearn.preprocessing import FunctionTransformer # Transforming of Data

# **MODELS**

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# **METRICS REPORT**

from sklearn.metrics import r2_score

data = pd.read_csv("cancer patient data sets.csv")

data.drop(["Patient Id" , "index"], axis = 1 , inplace = True)

data.replace(to_replace = "Low" , value = 0 , inplace = True)
data.replace(to_replace = "Medium" , value = 1 , inplace = True)
data.replace(to_replace = "High" , value = 2 , inplace = True)

a = data.drop("Level" , axis = 1)
b = data["Level"]

right_skew = []
left_skew = []
for i in data.columns:
    if data[i].skew() > 0:
        right_skew.append(i)
    else:
        left_skew.append(i)
        
right_trf = FunctionTransformer(func = np.square)
left_trf = FunctionTransformer(func = np.log1p)
right_trfd = right_trf.fit_transform(data[right_skew])
left_trfd = left_trf.fit_transform(data[left_skew])

data_proc = pd.concat([right_trfd , left_trfd , b] , axis = 1 , join = "inner")

train , test = np.split(data.sample(frac = 1) , [int(0.8 * len(data))])

def pre(dataframe):
    target = ["Level"]
    x = dataframe.drop(target , axis = 1)
    y = dataframe[target]
    
    return x , y
X_train , Y_train = pre(train)
X_test , Y_test = pre(test)

model_0 = KNeighborsRegressor()
model_0.fit(X_train , Y_train)

model_1 = LinearRegression()
model_1.fit(X_train , Y_train)

model_2 = LogisticRegression()
model_2.fit(X_train , Y_train)

model_3 = Ridge()
model_3.fit(X_train , Y_train)

model_4 = Lasso()
model_4.fit(X_train , Y_train)

model_5 = SVC()
model_5.fit(X_train , Y_train)

model_6 = GaussianNB()
model_6.fit(X_train , Y_train)

model_7 = RandomForestClassifier()
model_7.fit(X_train , Y_train)

print("R2 score for " , model_0 , " is : " , r2_score(Y_test , model_0.predict(X_test)))
print("R2 score for " , model_1 , " is : " , r2_score(Y_test , model_1.predict(X_test)))
print("R2 score for " , model_2 , " is : " , r2_score(Y_test , model_2.predict(X_test)))
print("R2 score for " , model_3 , " is : " , r2_score(Y_test , model_3.predict(X_test)))
print("R2 score for " , model_4 , " is : " , r2_score(Y_test , model_4.predict(X_test)))
print("R2 score for " , model_5 , " is : " , r2_score(Y_test , model_5.predict(X_test)))
print("R2 score for " , model_6 , " is : " , r2_score(Y_test , model_6.predict(X_test)))
print("R2 score for " , model_7 , " is : " , r2_score(Y_test , model_7.predict(X_test)))
