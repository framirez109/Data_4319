from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(train_X,train_Y),(test_x, test_y) = mnist.load_data()

train_X.shape
train_X[0].shape

train_Y[0].shape
plt.imshow(train_X[0])

np.max(train_X)

train_X = train_X/255
test_X = test_x/255
test_x[0].shape

X = []
for x in train_X:
    X.append(x.flatten().reshape(784,1)) #1 since we want a column vector

Y =[]
for y in train_y:
    temp_vec = np.zeros((10,1))
    temp_vec[y][0] = 1.0
    Y.append(temp_vec)

train_data = [p for p in zip(X,Y)]

#activation function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

#derivative of sigmoid function
def d_sigmoid(z):
    return sigmoid(z)*(1.0-sigmoid(z))

def mse(a, y):#output and label 
    return .5*sum((a[i] - y[i])**2 for i in range(10))[0]

#iinitialize_weights function
#centers the weight all together so it can train faster
#this sqrt equations at the end scales down the weights so it can run at a reasonable speed


def initialize_weights(layers = [784, 60,60,10]):
    W = []
    B = []# dummy entries
    
    for i in range(1, len(layers)):
        w_temp = np.random.randn(layers[i],1)* (np.sqrt(2/layers(i-1))) #(60, 784)
        b_temp = np.random.randn(layers[i],1)* (np.sqrt(2/layers(i-1)))
                                               
                                               
        W.append(w_temp)                                     
        B.append(b_temp)
    
    return W,B                                           
                                        
                                            
x, y = train_data[0]            

a0 = x
z1 = W[1]@ a0 + B[1]
a1 = sigmoid(z1)

z2 = W[2]@a1 + B[2]
a2 = sigmoid(z2)

z3 = W[3]@a2 + B[3]
a3 = sigmoid(z3)

W,B = initialize_weights([784,60,60,10])

inti

a0,y = train_data[0]
Z = [[0.0]]
A = [X]
L = 4

for i in range(1,L):
    z = W[i]@ A[i-1] + B[i]
    a = sigmoid(z)
    
    Z.append(z)
    A.append(a)

A[-1].shape

deltas = dict()
delta_last = (A[-1] - y)*sigmoid(z[-1])
deltas[L-1] = delta_last

for i in range(L-2,0 , -1):
    deltas[1] = W[1+1].T@ deltas
    
alpha = 0.04
#checks if all the shape match up
for i in range(1, 4):
    W[i] -= alpha*deltas[i]@AA[i - 1].T
    B[i] -= alpha*deltas

#function to compute the deltas and forward pass


def forward_pass(W,B,p, predict_vector = False):
    Z = [[0.0]]
    A = [p]
    L = 4
    
    for i in range(1,L):
        z = W[i]@A[i -1] - B[i]
        a = sigmoid(z)
        
        
        Z.append(z)
        A.append(a)
        
    if predict_vector == True:
        return A[-1]
    else:
        return Z,A
    
def deltas_dict(W,B, p):
    Z, A = forward_pass(W,B,p)
    L = len(W)
    deltas = dict()
    deltas[L-1] = (A[-1] - p[1])* sigmoid(Z[-1])
    for i in range(L-2,0,-1):
        deltas[i] = (W[-1]).T@deltas[i + 1] * d_sigmoid(Z[1])
        

#mean square error function 
def MSE(W,B,data):
    c = 0.0
    for p in data:
        A = forward_pass(W,B,p, predict_vector=True)
        c += mse(a, p[1])
    return c/len(data)

W,B = initalize_weights()
print(f"Intiall cost = {MSE(W,B,train_data)}")

i = np.random.randint(0,len(test_X))
#our prediction should be the largest
prediction = np.argmax(forward_pass(W,B,test_data[i],predict_vector=True))
print("Predicted Value  ")

alpha = .04
ephos = 3
for p in train_data:
    A,deltas = deltas_dict(W,B,p)
    for i in range(1,L):
        w[i] -= alpha 


