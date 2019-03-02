# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:22:35 2019

@author: kgund
"""
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

class neural_network():
    
    def initialize_parameters(self, n_x, n_h, n_y):
        
        #np.random.seed(2)         
        #weight matrix of shape (n_h, n_x)
        self.W1 = np.random.randn(n_h, n_x) * 0.01
        #bias vector of shape (n_h, 1)
        self.b1 = np.zeros(shape=(n_h, 1)) - 1 
        #weight matrix of shape (n_y, n_h)
        self.W2 = np.random.randn(n_y, n_h) * 0.01  
        #bias vector of shape (n_y, 1)
        self.b2 = np.zeros(shape=(n_y, 1)) - 1  
        self.learning_rate = .1
        self.threshold = .5

    #Function to define the size of the layer
    def layer_sizes(self, X, Y):
        # size of input layer
        n_x = X.shape[0]
        # size of hidden layer
        n_h = 13
        # size of output layer
        n_y = Y.shape[0] 
        
        return (n_x, n_h, n_y)

    def forward_propagation(self, X, parameters):
        
        # Implement Forward Propagation to calculate A2 (probability)
        self.Z1 = np.dot(self.W1, X) + self.b1
        #tanh activation function
        self.A1 = np.tanh(self.Z1)  
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
         #sigmoid activation function
        self.A2 = 1/(1+np.exp(-self.Z2)) 
        
        cache = {"Z1": self.Z1,
                 "A1": self.A1,
                 "Z2": self.Z2,
                 "A2": self.A2}
        
        return self.A2, cache
    
    def compute_cost(self, Y):
        # number of training examples
        m = Y.shape[1] 
                
        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(self.A2), Y) + np.multiply((1 - Y), np.log(1 - self.A2))
        cost = - np.sum(logprobs) / m
        
        return cost
    
    def backward_propagation(self, X, Y):
        # Number of training examples
        m = X.shape[1]
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        self.dZ2= self.A2 - Y
        self.dW2 = (1 / m) * np.dot(self.dZ2, self.A1.T)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)
        self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), 1 - np.power(self.A1, 2))
        self.dW1 = (1 / m) * np.dot(self.dZ1, X.T)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)

    def update_parameters(self):

        # Update rule for each parameter
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        return self.W1, self.b1, self.W2, self.b2

    def train(self, X, Y):
        o = self.forward_propagation(X, parameters)
        self.backward_propagation(X, Y)

#def get_accuracy(X,Y):
#    length = X.shape[1]
#    num_correct = 0
#    for i in range(length):
#        if(X[i] == Y[i]):
#            num_correct = num_correct + 1
#    accuracy = num_correct/length
#    return accuracy


def get_data():
    """Load the Iris dataset"""
    iris = load_iris()
    data = iris.data
    target = iris.target
    #normalizing the data
    normal_data = preprocessing.normalize(data)  
    normal_data = normal_data[0:100]
    X = np.array(normal_data)
    target = target[0:100]
    Y = np.array([target]).T   
    return X,Y

def get_data1():
    """Load the wine dataset"""
    wine = load_wine()
    data = wine.data
    target = wine.target
    #normalizing the data
    normal_data = preprocessing.normalize(data)  
    normal_data = normal_data[0:100]
    X = np.array(normal_data)
    target = target[0:100]
    Y = np.array([target]).T
    return X,Y
    
def nn_model(n_h, num_iterations=10000, print_cost=False):
    X,Y = get_data()
    nn = neural_network()
    np.random.seed(3)
    n_x = nn.layer_sizes(X, Y)[0]
    n_y = nn.layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = nn.initialize_parameters(n_x, n_h, n_y)
      
    # Loop (gradient descent)
    for i in range(0, num_iterations):        
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = nn.forward_propagation(X, parameters)            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = nn.compute_cost(Y)     
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            nn.backward_propagation(X, Y)
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = nn.update_parameters()
            #W1, b1, W2, b2 = nn.update_parameters()            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
                
parameters = nn_model(n_h = 6, num_iterations=10000, print_cost=True)

                
def nn_model1(n_h, num_iterations=10000, print_cost=False):
    X,Y = get_data1()
    nn = neural_network()
    np.random.seed(3)
    n_x = nn.layer_sizes(X, Y)[0]
    n_y = nn.layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = nn.initialize_parameters(n_x, n_h, n_y)
      
    # Loop (gradient descent)
    for i in range(0, num_iterations):        
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = nn.forward_propagation(X, parameters)            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = []
            cost = nn.compute_cost(Y)    
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            nn.backward_propagation(X, Y)
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = nn.update_parameters()
            #W1, b1, W2, b2 = nn.update_parameters()            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
            nn.train(X,Y)


                   
    plt.plot(cost)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()            

    return parameters, n_h
        
print()
parameters = nn_model1(n_h = 6, num_iterations=10000, print_cost=True)




