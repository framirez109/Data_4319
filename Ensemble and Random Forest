import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from sklearn.datasets import make_moons

#METHOD 1 
#you can use this data
from sklearn.datasets import make_hastie_10_2

X,y = make_moons(n_samples=500, noise=0.40,random_state=1 )

colors = ["blue " if label == 1 else "red" for label in y]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y)

train_colors = ["blue" if label == 1 else "red" for label in y_train]
test_colors = ["blue" if label == 1 else "red" for label in y_test]

plt.scatter(x_train[:,0],x_train[:,1], c = train_colors)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

from sklearn.ensemble import VotingClassifier

clf = VotingClassifier([("log_reg",LogisticRegression()),
                       ("tree", DecisionTreeClassifier(max_depth= 2)),
                       ("perceptron",Perceptron())],voting="hard")
learners = [LogisticRegression(),
           DecisionTreeClassifier(max_depth=2),
           Perceptron(),
           clf]

from sklearn.metrics import accuracy_score

for learner in learners:
    learner.fit(x_train, y_train)
    y_pred = learner.predict(x_test)
    print(f"{learner.__class__.__name__},accuracy= {accuracy_score(y_test,y_pred)}")


#Random Forest

from sklearn.ensemble import RandomForestClassifier


#n_estimators = n_samples
#too 
rf_clf = RandomForestClassifier(n_estimators=750, max_leaf_nodes=6)

rf_clf.fit(x_train, y_train)
accuracy_score(y_test, rf_clf.predict(x_test))

from sklearn.ensemble import BaggingClassifier

#You take a whole bunch of models and aggregate them 

bag_clf = BaggingClassifier(Perceptron(),
                           n_estimators=750,
                           bootstrap=True, n_jobs= -1) #-1 sets the model to parallel computing 

bag_clf.fit(x_train, y_train)
accuracy_score(y_test, bag_clf.predict(x_test))


