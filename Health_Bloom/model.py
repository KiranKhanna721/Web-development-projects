import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

heart = pd.read_csv('heart.csv')
heart = heart.dropna() # If any null values are present in dataset it will be drop
n_f = heart.select_dtypes(include=[np.number]).columns
c_f = heart.select_dtypes(include=[np.object]).columns
heart = pd.get_dummies(heart)
X = heart.drop(heart.iloc[:,-1:],axis=1)
y = heart.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state =2)
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
predicted_values = rf.predict(x_test)
x = metrics.accuracy_score(y_test, predicted_values)
print("RF Accuracy is: ", x)
pickle.dump(rf, open('heart.pkl','wb'))