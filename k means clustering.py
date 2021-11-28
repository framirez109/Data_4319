import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


data = load_iris()

X = data['data']
x = X[:,2:4]

plt.scatter(x[:, 0], x[:, 1])
k = 3
#part 1 of algorithm
k = 3
c = []
for _ in range(k):
    i = np.random.randint(len(x))
    c.append(x[i, :])
    
#eucledian norm - to measure distance
def distance(p, q): 
    return np.sqrt((p-q)@(p-q))

p = x[0, :]
c = []
L = dict()
for i in range(3):
    L[i] = []
    
for p in x:
    distances = np.array([distance(p, centroid)for centroid in C])
    assigment = np.argmin(distances)
    #print(f"{p} is assgined label {assigment}")
    
    L[assigment].append(p)
    if assigment == 0:
        c.append("red")
    elif assigment == 1:
        c.append("blue")
    else:
        c.append("green")

#the yellow points are the centroids
plt.scatter(X[:, 0], X[:, 1], color = c)
plt.scatter([c[0] for c in C], [c[1] for c in C], color = "yellow")

L[0]

for i, _ in enumerate(C):
    C[i] = sum(L[i])/len(L[i])
