"""
Created on Mon Mar 18 10:50:27 2019

@author: kgund
"""
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np



def get_cancer_data():
    """
    Getting and cleaning data for breast cancer 
        
    """
    url_cancer= 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    cancer = pd.read_csv(url_cancer, skipinitialspace=True)
    cancer.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                      'Uniformity of Cell Shape', 'Marginal Adhesion', 
                    'Single Epithelial Cell Size', 'Bare_Nuclei', 
                    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses',
                    'Class']
    
    cancer[['Bare_Nuclei']] = cancer[['Bare_Nuclei']].replace('?',np.NaN)
    cancer.Bare_Nuclei = pd.to_numeric(cancer.Bare_Nuclei)
    cancer.fillna(cancer.mean().round(0), inplace=True)
    #where 1 is the axis number (0 for rows and 1 for columns.)
    cancer = cancer.drop('Sample code number',axis = 1)
    
    values = cancer.values

    # Now impute it
    imputer = Imputer()
    imputedData = imputer.fit_transform(values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalizedData = scaler.fit_transform(imputedData)
    
    cancer_data = normalizedData[:,0:9]
    cancer_target = normalizedData[:,9]
    
    return cancer_data, cancer_target

def get_diabetes_data():
    """
    Getting and cleaning data for diabetes dataset

    """
    diabetes = datasets.load_diabetes()
    diabetes_target = diabetes.target
    diabetes_data = diabetes.data
#    values = diabetes.values
#
#
#    # Now impute it
#    imputer = Imputer()
#    imputedData = imputer.fit_transform(values)
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    normalizedData = scaler.fit_transform(imputedData)
#    
#    diabetes_data = normalizedData[:,0:9]
#    diabetes_target = normalizedData[:,9]
    
    return diabetes_data, diabetes_target


def get_iris_data():
    """
    Getting and cleaning data for iris iris
    
    """
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_target = iris.target



    return iris_data, iris_target

#Display Regression Data
def displayRegression(score):
    print ("{:.2f}%\n".format(score * 100))
    
    
def knn_testing(name, data_train, data_test, targets_train, targets_test):
    """
    testing data sets on knn
    """
    best_cross_val_score = 10000
    best_k = 1
    for k in range(1,15):
        regressor = KNeighborsRegressor(n_neighbors=k)
        model = regressor.fit(data_train, targets_train)
        predictions = model.predict(data_test)    
        metric = mean_absolute_error(targets_test, predictions)

    
        print("KNN CV score for {} n_neighbors = {}: {:.2f}".format(name, k, metric))
        if metric <= best_cross_val_score:
            best_cross_val_score = metric
            best_k = k   
        # print(" ")
    print("KNN Best CV score {} n_neighbors = {}: {:.2f}\n".format(name, best_k, best_cross_val_score))
#        print(" ")
    iris_classifier = KNeighborsRegressor(n_neighbors=best_k)
    iris_model = iris_classifier.fit(data_train, targets_train)
    score = (iris_model.score(data_test, targets_test))
     
    print ("{} best accuracy:" .format(name))
    displayRegression(score)
    print()

def dtree(name, data_train, data_test, targets_train, targets_test):
#    if name == "iris Data":
#        classifier = DecisionTreeRegressor(random_state=42)
#        model = classifier.fit(data_train, targets_train)
#        predictions = model.predict(data_test)
#        
#        metric = mean_absolute_error(targets_test, predictions)
#        print("Best Accuracy for {} on Decision Tree: {:.2f}".format(name, metric*100))
#        print(" ")
#        
#    else:
    classifier = DecisionTreeClassifier(random_state=42)
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)
    
    metric = accuracy_score(targets_test, predictions)
    print("Best Accuracy for {} on Decision Tree: {:.2f}%".format(name, metric*100))
    print(" ")   
        
def naive_bayes(name, data_train, data_test, targets_train, targets_test):
    """
    Naive Bayes
    """
#    if name == "iris Data":
#        regressor = BayesianRidge()
#        model = regressor.fit(data_train, targets_train)
#        predictions = model.predict(data_test)
#        
#        metric = mean_absolute_error(targets_test, predictions)
#        print("Best Accuracy for {} on Naive Bayes: {:.2f}".format(name, metric*100))
#        print(" ")
#
#    else:
    clf = GaussianNB()
    clf.fit(data_train, targets_train)
    predictions = clf.predict(data_test)
    metric = accuracy_score(targets_test, predictions)
    print("Best Accuracy for {} on Naive Bayes: {:.2f}%".format(name, metric*100))
    print(" ")


def bagging(name, data_train, data_test, targets_train, targets_test):
    """
    bagging
    """
    """
    testing data sets on knn
    """
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    best_cross_val_score = 0
    best_k = 100
    for k in range(100,500,50):
        classifier = BaggingClassifier(n_estimators=k)
        model = classifier.fit(data_train, targets_train)
        
        score = cross_val_score(model, data_train, targets_train, cv=kfold)
        metric = score.mean()
        print("CV score for bagging on {} n_estimators = {}: {:.2f}".format(name,k, metric))

    if metric > best_cross_val_score:
        best_cross_val_score = metric
        best_k = k
    print("{} best CV score for bagging n_estimators = {}: {:.2f}".format(name, best_k, metric))
    print()
    
    classifier = BaggingRegressor(n_estimators=best_k)
    model = classifier.fit(data_train, targets_train)
    score = (model.score(data_test, targets_test))
    print ("{} best accuracy:" .format(name))
    displayRegression(score)
        

def random_for(name, data_train, data_test, targets_train, targets_test):
    """
    random forrrest
    """
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    best_cross_val_score = 0
    best_k = 100

    for k in range(100,500,50):
#        if name == "iris Data":
#            classifier = BaggingRegressor(max_samples=k)
#            model = classifier.fit(data_train, targets_train)
#            score = cross_val_score(model, data_train, targets_train, cv=kfold)
#            metric = score.mean()
#            print("CV score for Random Forest on {} n_estimators = {}: {:.2f}".format(name,k, metric))
#        else:
        classifier = RandomForestClassifier(n_estimators=k)
        model = classifier.fit(data_train, targets_train)
        score = cross_val_score(model, data_train, targets_train, cv=kfold)
        metric = score.mean()
        print("CV score for Random Forest on {} n_estimators = {}: {:.2f}".format(name,k, metric))

    if metric > best_cross_val_score:
        best_cross_val_score = metric
        best_k = k
    print("{} best CV score for Random Foresst n_estimators = {}: {:.2f}".format(name, best_k, metric))
    print()
    
    classifier = RandomForestRegressor(n_estimators=best_k)
    model = classifier.fit(data_train, targets_train)
    score = (model.score(data_test, targets_test))
    print ("{} best accuracy:" .format(name))
    displayRegression(score)

def ada_boost(name, data_train, data_test, targets_train, targets_test):
    """
    AdaBoost
    """
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    best_cross_val_score = 0
    best_k = 100
    for k in range(100, 600, 100):
#        if name == "iris Data":  
#            classifier = RandomForestRegressor(n_estimators=k)
#            model = classifier.fit(data_train, targets_train)
#            score = cross_val_score(model, data_train, targets_train, cv=kfold)
#            metric = score.mean()
#            print("AdaBoost CV score for {} n_estimators = {}: {:.2f}".format(name,k, metric))
#
#        else:
        classifier = AdaBoostClassifier(n_estimators=k)
        model = classifier.fit(data_train, targets_train)
        score = cross_val_score(model, data_train, targets_train, cv=kfold)
        metric = score.mean()
        print("AdaBoost CV score for {} n_estimators = {}: {:.2f}".format(name,k, metric))
        
    if metric > best_cross_val_score:
        best_cross_val_score = metric
        best_k = k
    # print(" ")
    print("Best CV score for AdaBoost on {} n_estimators {}: {:.2f}".format(name, best_k, metric))
    print(" ")
    
    classifier = AdaBoostRegressor(n_estimators=best_k)
    model = classifier.fit(data_train, targets_train)
    score = (model.score(data_test, targets_test))
    print ("{} best accuracy:" .format(name))
    displayRegression(score)
    
    
    
    
    
def main():
    cancer_data, cancer_target = get_cancer_data()
    diabetes_data, diabetes_target = get_diabetes_data()
    iris_data, iris_target = get_iris_data()
    
    
    
    data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer = train_test_split(cancer_data, 
                                                                        cancer_target, 
                                                                        train_size=.7,
                                                                        test_size=.3)
    
    data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes = train_test_split(diabetes_data, 
                                                                    diabetes_target, 
                                                                    train_size=.7,
                                                                    test_size=.3)
    
    data_train_iris, data_test_iris, targets_train_iris, targets_test_iris = train_test_split(iris_data, 
                                                                    iris_target, 
                                                                    train_size=.7,
                                                                    test_size=.3)
    """
    3 different "regular" learning algorithms
    """
    #running KNN on the three dataset
    name = "Cancer Data"
    knn_testing(name, data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer)
    
    name = "Diabetes Data"
    knn_testing(name, data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes)

    name = "iris Data"
    knn_testing(name,data_train_iris, data_test_iris, targets_train_iris, targets_test_iris)
    
    #Decision tree classifer
    name = "Cancer Data"
    dtree(name, data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer)
    
    name = "Diabetes Data"
    dtree(name, data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes)

    name = "iris Data"
    dtree(name,data_train_iris, data_test_iris, targets_train_iris, targets_test_iris)
    
    #Naive Bayes
    name = "Cancer Data"
    naive_bayes(name, data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer)
    
    name = "Diabetes Data"
    naive_bayes(name, data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes)

    name = "iris Data"
    naive_bayes(name,data_train_iris, data_test_iris, targets_train_iris, targets_test_iris)

    
    """
    Bagging 
    """
    name = "Cancer Data"
    bagging(name, data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer)
    
    name = "Diabetes Data"
    bagging(name, data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes)

    name = "iris Data"
    bagging(name,data_train_iris, data_test_iris, targets_train_iris, targets_test_iris)

    """
    AdaBoost
    """
    
    name = "Cancer Data"
    ada_boost(name, data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer)
    
    name = "Diabetes Data"
    ada_boost(name, data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes)

    name = "i   ris Data"
    ada_boost(name,data_train_iris, data_test_iris, targets_train_iris, targets_test_iris)

    
    """
    random forest
    """
    name = "Cancer Data"
    random_for(name, data_train_cancer, data_test_cancer, targets_train_cancer, targets_test_cancer)
    
    name = "Diabetes Data"
    random_for(name, data_train_diabetes, data_test_diabetes, targets_train_diabetes, targets_test_diabetes)

    name = "iris Data"
    random_for(name,data_train_iris, data_test_iris, targets_train_iris, targets_test_iris)
    
    
if __name__ == "__main__":
    main()
