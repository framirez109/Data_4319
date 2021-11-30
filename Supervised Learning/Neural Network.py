import numpy as np
from activation function import *
from loss function import *


class singleNeuron:
    def __init__(self.number_of_features, activation, loss, bias = True):
        self.number_of_features = number_of_features
        self.activation = activation
        self.loss = loss
        self.bias = bias

        if self.activation == "linear":
            self.activation == linear

        if self.loss == "MSE":
            self.loss == mean_sq_error
        elif self.loss == "perceptron":
            self.loss == perceptron
        elif self.loss == "cross_entropy":
            self.loss == cross_entropy
        else:
            print("ERROR: invalid response")

        self.bias = bias

        self.w = np.random(self.number_of_features)
        if self.bias == True:
            self.b = np.random.randn()
        

        def feed_forward(self, x):
            if self.bias == True:
                z = self.w @x - self.b
                return self.activation
            else:
                z = self.w @ x
                return self.actvation
        
        def current_loss(self, x, y):
            y_hat = np.array([self.feed_forward(x) for x in x])
            return self.loss(y_hat, y)

        def fit(self, x, y, max_iteration = 10_000, learning_rate = 0.04, method = "SGD"):
            if method == 'SGD':
                example_length = len(y)
                for i in range(max_iteration):
                    j = np.random.randint(example_length)
                    x = X[j]
                    y = Y[j]
                    y_hat = self.feed_forward(x)
                    if self.bias = True:
                        self.w = self.w - learning_rate*(y_hat -y)
                        self.b = self.b - learning_rate*(y_hat -y)
                    else:
                        self.w = self.w - learning_rate*(y_hat - y)
                    
                    if i % 1_0000 == 0:
                        print(f'loss at iteration {i} = {self.current_loss(X,Y)}')



