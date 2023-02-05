import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("cancer patient data sets.csv")

data.drop("Patient Id", axis = 1 , inplace = True)
data.drop("index" , axis = 1 , inplace = True)

data.replace(to_replace = "Low" , value = 0 , inplace = True)
data.replace(to_replace = "Medium" , value = 1 , inplace = True)
data.replace(to_replace = "High" , value = 2 , inplace = True)

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
print(r2_score(Y_test , model_0.predict(X_test)))
