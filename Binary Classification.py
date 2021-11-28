import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

%matplotlib inline

mnist = fetch_openml('mnist_784',version=1)
mnist.keys()
X,y = mnist['data'],mnist['target']
X.shape
y[0]

plt.imshow(X[0].reshape(28,28), cmap='binary')

X_train,x_test = X[:60_000],X[60_000:]
y_train, y_test = y[:60_000],y[60_000:]

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

y_train_5[1]

#type of class python
sgd_ctf = SGDClassifier(random_state = 42)
sgd_ctf.fit(X_train, y_train_5)

sgd_ctf.predict(X_train[0].reshape(1,-1))

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_ctf,X_train,y_train_5,cv=3,scoring='accuracy')

np.array([0.95035, 0.96035, 0.9604 ]).mean()

#design our own classifier

class Never5Classifier(BaseEstimator):
    
    def fit(self,x,y=None):
        return self
    
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train, cv=3, scoring= 'accuracy')


#count the # of instance that the prediction is made

y_train_predict = cross_val_predict(sgd_ctf,X_train,y_train_5,cv=3)

y_train_predict

#true negatives - 53k
#true positvies - 1891
confusion_matrix(y_train_5, y_train_predict)

from sklearn.metrics import precision_score, recall_score

#when it claims the image is a 5, its only 83% accurate
#if precision increases, recall decreases and vice versa
precision_score(y_train_5,y_train_predict)

#when it isnt a 5, its only 65% accurate
recall_score(y_train_5,y_train_predict)



