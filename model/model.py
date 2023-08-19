import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('datascicont.csv')

X = data[[ 'rain', 'humid', 'storage', 'used ', 'flow ']]
y = data[['discharge']]

ohenc1 = OneHotEncoder(sparse=False)
m1=ohenc1.fit_transform(data[['discharge']])

X_train, X_test , y_train, y_test = train_test_split(X,y,
                                                     test_size=0.2,
                                                     random_state=8)

model = GradientBoostingClassifier(n_estimators=100,max_features= None)
model.fit(X_train, y_train)


with open('model.pkl','wb') as file :
    pickle.dump(model, file)