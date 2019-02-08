# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:37:36 2019

@author: kgund
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from anytree import Node, RenderTree, LevelOrderIter, findall
import pandas as pd
import numpy as np
import math 




class Decision_tree:
    
    def __init__(self, data):
        pass
                  
    def calc_entropy(self,data):
        
        #Calculate the length of the data-set
        entries = len(data) 
        labels = {}
        #Read the class labels from the data-set file into the dict object "labels"
        for rec in data:
          label = rec[-1] 
          if label not in labels.keys():
            labels[label] = 0
            labels[label] += 1
        #entropy variable is initialized to zero
        entropy = 0.0
        #For every class label (x) calculate the probability p(x)
        for key in labels:
          prob = float(labels[key])/entries
        #Entropy formula calculation
          entropy -= prob * math.log(prob,2) 
        #print "Entropy -- ",entropy
        #Return the entropy of the data-set
        return entropy, labels
#        #Calc the length of the data set
#        entries = len(data)
#        #print(entries)
#        val, counts = np.unique(data, return_counts=True)
#        #Initalize entropy 
#        entropy = 0.0
#        #weighted_average = 0.0
#        #for every class label calc the probability 
#        for i in range(0, len(val)):
#            prob = float(counts[i]/entries)
#            #Entropy calc
#            entropy -= prob * math.log2(prob)
#            #weighted_average += prob*entropy
#            #print(weighted_average)
#
#        return entropy
        
    def best_attribute(self, data):
        # List of entropies for each column
        #ent = []
        # Get number of columns  
        #num_col = data.shape[1]
        #get the number of freatures for the data set
        features = len(data[0])-1
        #calc entropy of total data set
        base_entropy = self.calc_entropy(data)
        #initialize the info-gain variable to zero
        max_info_gain = 0.0
        best_attribute = -1
        #Iterate through the features
        for i in range(features):
            #store the values of the features in a variable 
            attribute_list = [rec[i] for rec in data]
            unique_values = set(attribute_list)
        #initialize entropy and the attribute entopy 
        new_entropy = 0.0
        attribute_entropy = 0.0
        #iterate throught the list of unique values
        for value in unique_values:
            #fucntion to split data 
            new_data = data.split(data, i, value)
            #probability calculation 
            prob = len(new_data)/float(len(data))
            #entropy calc for the new attributes 
            new_entropy = prob * self.calc_entropy(new_data)
            attribute_entropy += new_entropy
        #calc info gain 
        info_gain = base_entropy - attribute_entropy
        #locate the attriute with the highest info gain 
        if (info_gain > max_info_gain):
            max_info_gain = info_gain 
            best_attribute = i
        return best_attribute 
#            
#        
##        # Get entropies for each column
##        for i in range(0, num_col):
##            e = self.calc_entropy(data[:,i])
##            ent.append(e)
##        
##   
   
    
    def data_split(self, data, arc, val):
        """
        fucntion to split the data based on the attribute that 
        has the highest information gain
        """
        #place to store the newly split data 
        new_data = []
        #iterate through the the data and split it 
        for rec in data: 
            if rec[arc] == rec:
                reduced_set = list(rec[:arc])
                reduced_set.extend(rec[arc+1:])
                new_data.append(reduced_set)
        #return the new data that is split on selected attribute 
        return new_data
    
    def build_tree(self, data, labels):
        """
        This function will build the decision tree 
        """
        #list variable to last nodes of the tree
        class_list = [rec[-1] for rec in data]
        
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        #identify attribute for split 
        max_gain_node = self.best_attribute(data)
        #store class label value
        tree_label = labels[max_gain_node]
        #dictionary object for the nodes in the tree
        the_tree = {tree_label:{}}
        del[labels[max_gain_node]]
        #get the unique values of the attributes
        node_values = [rec[max_gain_node] for rec in data]
        unique_values = set(node_values)
        for value in unique_values:
            sub_labels = labels[:]
            #update the not last nodes of the tree
            the_tree[tree_label][value] = self.decision_tree(self.data_split(data, max_gain_node, value),sub_labels)
        #return the dictonary objecet tree
        return the_tree

def get_and_fix_data():
    
    url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    data = pd.read_csv(url_data, names=['buying','maint','doors','persons','lug_boot','safety','class'])

    #to check stuff about the data you can use data.info()
    
    """
    This is the target variable 
    they are in a string so we want to convert to an int. so factoize does this
    data['class'],class_names = pd.factorize(data['class'])
    data.info()
    print(data['class'].unique())
    """
    #We wan to factorize the whole data set to get rid of strings 
    #data['class'],class_names = pd.factorize(data['class'])
    
    data = data.apply(lambda x: pd.factorize(x)[0])
    
    data = data.iloc[:,:-1]
    target = data.iloc[:,-1]
    #print(data.info())
    return data, target

    
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
    
   
def main():
    
    #data = get_and_fix_data()
    data, target = get_and_fix_data()
     
#    tree = Decision_tree(data)
#    data, labels = tree.calc_entropy(data)
#    tree.build_tree(data, labels)
     #Split the data into training and test set 
#    data_train, data_test, targets_train, targets_test = train_test_split(data, 
#                                                                        target, 
#                                                                        train_size=.7,
#                                                                        test_size=.3)
#    
#    kf = KFold(n_splits=8, shuffle=True)
#    for train, test in kf.split(data):
#        data_train, data_test, target_train, target_test = data.iloc[train], data.iloc[test], target[train], target[test]
#        classifier = tree_fit()
#        model = classifier.fit(data)
#        targets_predicted = model.predict(data_test)
#        corrects = 0
#        for x in range(len(data_test)):
#            if target_test.iloc[x] == targets_predicted[x]:
#                corrects += 1
#
#    print('Accuracy: {}/{} => {:4.2f}'.format(corrects, len(test), corrects / len(test)))
     
    """
    # Build the tree   
    dtree = Decision_tree(data)
    dtree.fit(data)
    # use the model to make predictions with the test data
    targets_pred = dtree.predict(data_test)
    # how did our model perform?
    count_misclassified = (targets_test != targets_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(targets_test, targets_pred)
    print('Accuracy: {:.2f}'.format(accuracy))
    """  

if __name__ == "__main__":
    main()
