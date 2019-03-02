from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from math import exp
from random import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine


class nueral_network:

    # Forward propagate input to a network output
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    #activation fucntion
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation
    # Backpropagate error and store in neurons
    def backward_propagate(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.sigmoid_prime(neuron['output'])
                
    #signmoid               
    def sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))
    
    # Calculate the derivative of an neuron output
    def sigmoid_prime(self, output):
        return output * (1.0 - output)

    def update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    def fit(self, data, target, n_of_hidden_nodes, n_of_outputs, l_rate, cycles):

        network = []
        if len(data.shape) > 1:
            number_of_inputs = data.shape[1]
        else:
            number_of_inputs = 1

        hidden_layer = [{'weights': [random() for i in range(number_of_inputs + 1)]} for i in range(n_of_hidden_nodes)]
        output_layer = [{'weights': [random() for i in range(n_of_hidden_nodes + 1)]} for i in range(n_of_outputs)]
        network.append(hidden_layer)
        network.append(output_layer)

        graph = []
        for n in range(cycles - 1):
            accuracy = 0
            for x in range(len(data)):
                outputs = self.forward_propagate(network, data[x])
                expected = [0 for i in range(n_of_outputs)]
                expected[target[x]] = 1
                if (np.argmax(expected) == np.argmax(outputs)):
                    accuracy += 1
                self.backward_propagate(network, expected)
                self.update_weights(network, data[x], l_rate)
            accuracy = accuracy / len(data)
            graph = np.append(graph, accuracy)
        return neural_model(network), graph
    
    

class neural_model:
    """creates the model, so you can input data"""
    def __init__(self, network):
        self.network = network
        
    # Forward propagate input to a network output
    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    def sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def predict(self, test):
        outputs_list = []

        for row in test:
            cur_output = self.forward_propagate(self.network, row)
            cur_output =  np.argmax(self.forward_propagate(self.network, row))
            outputs_list.append(cur_output)

        return outputs_list
	
def get_data():
    # Iris dataset
    iris = datasets.load_iris()
    # Wine dataset
    wine = datasets.load_wine()
        
#    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', skipinitialspace=True)
#    
#    data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',
#                    'class_values']
#    #label encoding makes sense here beause of the values
#    cleanup_nums = {"buying": {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}, 
#                    "maint": {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
#                    "doors": {'2': 2, '3': 3, '4': 4, '5more': 5},
#                    "persons": {'2': 2, '4': 4, 'more': 5},
#                    "lug_boot": {'small': 0, 'med': 1, 'big': 2},
#                    "safety": {'low': 0, 'med': 1, 'high': 2},
#                    "class_values": {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}}
#    
#    #setup of the car data and target
#    data.replace(cleanup_nums, inplace=True)
#    # Shuffling
#    data = data.sample(frac=1)
#    
#    std_scaler = preprocessing.StandardScaler().fit(data)
#    data_std = std_scaler.transform(data)
#    
##    std_scale = preprocessing.StandardScaler().fit(data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']])
##    data_std = std_scale.transform(data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']])
#    # Getting target
#    target = np.array(data['class_values'])
    
    return iris, wine
		

def main():
    iris, wine = get_data()
    """Iris dataset run"""
    my_classifier_accuracy = 0
    scikit_classifier_accuracy = 0
    kf = KFold(n_splits=10, shuffle=True)
    
    print("Iris dataset")
    for train, test in kf.split(iris.data):
        data_train, data_test, target_train, target_test = iris.data[train], iris.data[test], iris.target[train], iris.target[test]
    
        classifier = nueral_network()
        model, graph = classifier.fit(data_train, target_train, 4, 3, 0.05, 1000)
        targets = model.predict(data_test)
    
        corrects = 0
    
        for x in range(len(target_test)):
            if (target_test[x] == targets[x]):
                corrects += 1
    
        print("Accuracy: {}".format(corrects / len(target_test)))
    
        my_classifier_accuracy += corrects / len(target_test)
        
        #Sklearn #Sklearn 
        mlp = MLPClassifier(hidden_layer_sizes=(4), learning_rate_init=0.05, max_iter=1000)
        mlp.fit(data_train, target_train)
        predictions = mlp.predict(data_test)
    
        corrects = 0
        for x in range(len(target_test)):
            if (target_test[x] == predictions[x]):
                corrects += 1
    
        scikit_classifier_accuracy += corrects / len(target_test)
    plt.plot(graph)
    plt.ylabel('Accuracy')
    plt.xlabel('Loop')
    plt.title('Iris')
    plt.show()
    print("My accuracy: {}".format(my_classifier_accuracy / 10))
    print("Scikits accuracy: {}".format(scikit_classifier_accuracy / 10))

    """Wine dataset run"""
    my_classifier_accuracy = 0
    scikit_classifier_accuracy = 0
    kf = KFold(n_splits=5, shuffle=True)
    print()
    print("Wine dataset")
    
    # training / test sets
    for train, test in kf.split(wine):
#       data_train, data_test, target_train, target_test = data[train], data[test], target[train], target[test]
        data_train, data_test, target_train, target_test = wine.data[train], wine.data[test], wine.target[train], wine.target[test]
        classifier = nueral_network()
        model, graph = classifier.fit(data_train, target_train, 4, 2, 0.05, 1500)
        targets = model.predict(data_test)
        corrects = 0
        for x in range(len(target_test)):
            if (target_test[x] == targets[x]):
                corrects += 1
    
        print("Accuracy: {}".format(corrects / len(target_test)))
        my_classifier_accuracy += corrects / len(target_test)
    
        #Sklearn 
        mlp = MLPClassifier(hidden_layer_sizes=(4), learning_rate_init=0.05, max_iter=1500)
        mlp.fit(data_train, target_train)
        predictions = mlp.predict(data_test)
        corrects = 0
        for x in range(len(target_test)):
            if (target_test[x] == predictions[x]):
                corrects += 1
    
        scikit_classifier_accuracy += corrects / len(target_test)
    plt.plot(graph)
    plt.ylabel('Accuracy')
    plt.xlabel('Loop')
    plt.title('Wine dataset')
    plt.show()
    print("My classifier accuracy: {}".format(my_classifier_accuracy / 10))
    print("Scikit classifier accuracy: {}".format(scikit_classifier_accuracy / 10))
    
main()