# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:55:45 2019

@author: Keegan Gunderson 
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np


"""
Car data 

    -The target is the last column where you classify 
    -The car as acceptable, unacceptable, good, or very good
    
    -data is read in 
    -columns are given names 
    -categorical data is assinged numbers 
    -setup of the car data and target
    -The data is then randomed
    -The data is then normalized
    -Creates an array for car_data and car_target

"""

car = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', skipinitialspace=True)

car.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',
                'class_values']

cleanup_nums = {"buying": {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}, 
                "maint": {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
                "doors": {'2': 2, '3': 3, '4': 4, '5more': 5},
                "persons": {'2': 2, '4': 4, 'more': 5},
                "lug_boot": {'small': 0, 'med': 1, 'big': 2},
                "safety": {'low': 0, 'med': 1, 'high': 2},
                "class_values": {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}}

#setup of the car data and target
car.replace(cleanup_nums, inplace=True)
target = car.class_values

#Randomizing the data 
car.sample(frac=1)

# Normalize data
std_scaler = preprocessing.StandardScaler().fit(car)
scaled_car = std_scaler.transform(car)

#Creating the array for car_data and car_target 
cars_data = np.array(scaled_car)
cars_target = np.array(target)

"""
Car kNN alorithm 
"""
data_train_car, data_test_car, targets_train_car, targets_test_car = train_test_split(cars_data, 
                                                                    cars_target, 
                                                                    train_size=.7,
                                                                    test_size=.3)

classifier = KNeighborsClassifier(n_neighbors=3)
car_model = classifier.fit(data_train_car, targets_train_car)
targets_predicted = car_model.predict(data_test_car)
accuracy_car = accuracy_score(targets_predicted, targets_test_car)
print("Car Data Results:\n{:.2f}%".format(accuracy_car*100))


"""
Auto MPG 
    -predict the MPG column which is the first column 

    -data is read in 
    -columns are given names 
    -replaces the missing vlaues with NaN, 
    -Makes the horsepower column an int, 
    -fills in the NaNs with the mean of each column, 
    -deletes the 'car_name' column because it is not needed, 
    -randomizes the data, then normalizes it.
    -creates an array for the mpg_data and mpg_target,

"""

auto_mpg = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data', 
                       skipinitialspace=True, delim_whitespace=True)

auto_mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model', 'origin', 'car_name']
#run to see what the data types are
#auto_mpg.dtypes

#replacing the missing vlaues with NaN
auto_mpg[['horsepower']] = auto_mpg[['horsepower']].replace('?',np.NaN )
auto_mpg.horsepower = pd.to_numeric(auto_mpg.horsepower)

#fills NaNs with mean of each column 
auto_mpg.fillna(auto_mpg.mean().round(0) , inplace=True)

##make objec catergory
#for col in ['horsepower']:
#    auto_mpg[col] = auto_mpg[col].astype('category')
#    
##make category incrament
#for col in auto_mpg.columns:
#    auto_mpg[col] = auto_mpg[col].astype("category")
#    auto_mpg[col] = pd.factorize(auto_mpg[col])[0] + 1

#dont need this column
del auto_mpg['car_name']

#setup of the auto_mpg data and target
target = auto_mpg.mpg

#Randomizing the data 
auto_mpg.sample(frac=1)

# Normalize data
std_scaler = preprocessing.StandardScaler().fit(auto_mpg)
scaled_auto_mpg = std_scaler.transform(auto_mpg)

#Creating the array for mpg_data and mpg_target 
mpg_data = np.array(scaled_auto_mpg)
mpg_target = np.array(target)

"""
Regrssion algorithm for auto mpg_data
"""
#Display Regression Data
def displayRegression(score):
    print ("{:.2f}%".format(score * 100))

    
data_train_mpg, data_test_mpg, targets_train_mpg, targets_test_mpg = train_test_split(mpg_data, 
                                                                    mpg_target, 
                                                                    train_size=.7,
                                                                    test_size=.3)

                                                                    
mpg_classifier = KNeighborsRegressor(n_neighbors=3)
mpg_model = mpg_classifier.fit(data_train_mpg, targets_train_mpg)
score = (mpg_model.score(data_test_mpg, targets_test_mpg))
 
print ("MPG Data Results")
displayRegression (score)


"""
Math class
    -predict the final grade which is stored in the final column (G3)
    
    -read in data that is seperated by ;
    -create column names 
    -categorical data that should be binary is assigned to binary
    -setup of the math data and target
    -The data is then randomed
    -The data is then normalized
    -Creates an array for math_data and math_target
    
"""
math = pd.read_csv('student/student-mat.csv', skipinitialspace=True, sep=";")

math.columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu',
                'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
                'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3' ]

cleanup_nums = {"school": {'GP': 0, 'MS': 1},
                "sex": {'F': 0, 'M':1},
                "famsize": {'LE3':3 , 'GT3':4},
                "Pstatus": {'T':0, 'A':1},
                "schoolsup": {'yes': 0, 'no':1},
                "famsup": {'yes': 0, 'no':1},
                "paid": {'yes': 0, 'no':1},
                "activities": {'yes': 0, 'no':1},
                "nursery": {'yes': 0, 'no':1},
                "higher": {'yes': 0, 'no':1}, 
                "internet": {'yes': 0, 'no':1},
                "romantic": {'yes': 0, 'no':1}}

math.replace(cleanup_nums, inplace=True)

#pulls out only the numeric data 
#math = math1._get_numeric_data()

#make objec catergory
for col in ['address', 'Mjob', 'Fjob', 'reason', 'guardian',]:
    math[col] = math[col].astype('category')
    
#make category incrament
for col in math.columns:
    math[col] = math[col].astype("category")
    math[col] = pd.factorize(math[col])[0] + 1
    
target = math.G3

#Randomizing the data 
math.sample(frac=1)

# Normalize data
std_scaler = preprocessing.StandardScaler().fit(math)
scaled_math= std_scaler.transform(math)

#Creating the array for mpg_data and mpg_target 
math_data = np.array(math)
math_target = np.array(target)

"""
regression algorithm for math data
"""
data_train_math, data_test_math, targets_train_math, targets_test_math = train_test_split(mpg_data, 
                                                                    mpg_target, 
                                                                    train_size=.7,
                                                                    test_size=.3)

                                                                    
math_classifier = KNeighborsRegressor(n_neighbors=3)
math_model = mpg_classifier.fit(data_train_math, targets_train_math)
score = (math_model.score(data_test_math, targets_test_math))
print ("Math Data Results")
displayRegression (score)

"""
Cross Validation scores
"""
car_score = cross_val_score(car_model, data_train_car, targets_train_car)
mpg_score = cross_val_score(mpg_model, data_train_mpg, targets_train_mpg, cv=10)
math_score = cross_val_score(math_model,data_train_math, targets_train_math, cv=10)

print ("\nCross Validation\n")
if np.average(car_score) < 1.0 or np.average(car_score) < 1:
    print ("Cars Accuracy: {:.2f}" .format(np.average(car_score) * 100))
else:
    print ("Cars Accuracy: {:.2f}" .format(np.average(car_score)))

if np.average(mpg_score) < 1.0 or np.average(mpg_score) < 1:
    print ("MPG Accuracy: {:.2f}" .format(np.average(mpg_score)* 100))
else:
    print ("MPG Accuracy: {:.2f}" .format(np.average(mpg_score)))
    
if np.average(math_score) < 1.0 or np.average(math_score) < 1:
    print ("Math Accuracy: {:.2f}" .format(np.average(math_score)* 100))
else:
        print ("Math Accuracy: {:.2f}" .format(np.average(math_score)))
        


