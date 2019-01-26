# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:01:41 2019

@author: kgund
"""
from sklearn import datasets 
import numpy as np

iris = datasets.load_iris()
data = iris.data

row1 = data[0]
row2 = data[1]


distances = []

for row in data:
    diff = row1 - row2
    diff_squared = diff ** 2
    dist = sum(diff_squared)
    distances.append(dist)
    
sort = np.argsort(distances)
#dist = 0
#for i in range(len(row1)):
#    diff = row1[i] - row2[1]
#    sq = diff ** 2
#    sum =+ sq
