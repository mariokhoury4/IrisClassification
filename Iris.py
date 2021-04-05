import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.gaussian_process.kernels import RBF

#list of models to test for classifications:
classifications = [RidgeClassifier(), SGDClassifier(max_iter=1000, tol=1e-3), PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3), SVC(gamma='auto'), NuSVC(), LinearSVC(random_state=0, tol=1e-5), KNeighborsClassifier(n_neighbors=3), GaussianProcessClassifier(kernel=1.0 * RBF(1.0),random_state=0), DecisionTreeClassifier(random_state=0), RandomForestClassifier(max_depth=2, random_state=0), MLPClassifier(random_state=1, max_iter=300)]

#read the data
df = pd.read_csv("IRIS.csv")

#fill the Null values with the previous value
df = df.fillna(method = 'ffill')

#X and y
y = df.pop('species')
X = df

#resuslts
HashMap = {}


#Check every model and see the results
for i in classifications:
    #Train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Title
    Title = i
    #model
    model = Title    
    model = model.fit(X_train, y_train)
    
    #prediction
    y_pred = model.predict(X_test)

    #accuracy_score
    score = accuracy_score(y_test, y_pred)

    
    HashMap[Title] = score

#print 
print(HashMap)










