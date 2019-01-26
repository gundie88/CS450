
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:31:12 2019

@author: kgund
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score

class HardCodedClassifier:
    
    def __init__(self):
        pass
    
    def fit(self, data_train, targets_train):
        pass
        
    def predict(self, data_test):
        prediction = []
        for p in data_test:
            prediction.append(0)            
        return prediction
    
    def prediction_of_name(self, prediction, names):
        prediction_names = []
        for p in prediction:
            prediction_names.append(names[p])
        return prediction_names
        

                        
def main():
    #Loaded the dataset in
    iris = datasets.load_iris()
    
 
    
    # Randomize & split into a training set: (70%) and testing set (30%)
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, 
                                                                    iris.target, 
                                                                    train_size=.7,
                                                                    test_size=.3)
    
    
    
                                                        
    #GaussianNB Model 
    classifier1 = GaussianNB()
    classifier1.fit(data_train, targets_train)
    targets_predicted = classifier1.predict(data_test)
    accuracyg = accuracy_score(targets_predicted, targets_test)                          
    print("\nGaussianNB Model\nPredicted targets to the actual targets {:.2f}%\n" .format(accuracyg*100)) 

        
    
    # HardCodeClassifier Model 
    classifier2 = HardCodedClassifier()
    classifier2.fit(data_train, targets_train)
    predicted = classifier2.predict(data_test)
    prediction_names = classifier2.prediction_of_name(predicted, iris.target_names)
    accuracyh = accuracy_score(predicted, targets_test)                          
    print("\nHardCodeClassifier Model\nPredicted targets to the actual targets {:.2f}%\n" .format(accuracyh*100)) 
    print(prediction_names)
    
if __name__ == "__main__":
    main()