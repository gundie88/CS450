# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:36:37 2019

@author: Keegan Gunderson
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


#Read in the iris data
iris = datasets.load_iris()


"""
This is the sklearn kNN algorithm
"""
# Randomize & split into a training set: (70%) and testing set (30%)
data_train, data_test, targets_train, targets_test = train_test_split(iris.data, 
                                                                    iris.target, 
                                                                    train_size=.7,
                                                                    test_size=.3)

classifier = KNeighborsClassifier(n_neighbors = 3)
model = classifier.fit(data_train, targets_train)
targets_predicted = model.predict(data_test)
accuracysk = accuracy_score(targets_predicted, targets_test)
print("Percentage of sklearn kNN correct:\n{:.2f}%".format(accuracysk*100))

"""
Hardcodded basic kNN algorithm
"""
class knnClassifier:

    k = 1
    data = []
    target = []

    def __init__(self, k, data=[], target=[]):
        self.k = k
        self.data = data
        self.target = target
        
    def fit(self, data, target):
        self.data = data
        self.target = target
        return knnClassifier(self.k, self.data, self.target)

    def predict(self, test_data):
        nInputs = np.shape(test_data)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):
            #compute euclidean distance
            distances = np.sum((self.data-test_data[n,:])**2, axis=1)
            #returns the array of inidices of the distnaces
            indices = np.argsort(distances, axis=0)
            #Finds the unique elements of an array
            classes = np.unique(self.target[indices[:self.k]])
            #returns the number of items in an object == 1
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest


hc_Classifier = knnClassifier(4)
hc_Model = hc_Classifier.fit(data_train, targets_train)
hc_Predicted = hc_Model.predict(data_test)
#finds the accuracy 
accuracyhc = accuracy_score(hc_Predicted, targets_test)
print("Percentage of HardCoded kNN correct:\n{:.2f}%".format(accuracyhc*100))



"""
K-Folds Cross Validation
"""
# creating list of K for KNN
k_list = list(range(1,50,2))
# creating list of cross validation scores
cv_scores = []

# perform 10-fold cross validation computing the score 10 consecutive times 
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    #Evaluate a score by cross-validation
    scores = cross_val_score(knn, data_train, targets_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
#Mean squared error
MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()