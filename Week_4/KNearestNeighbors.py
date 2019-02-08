# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 13:25:15 2019

@author: kgund
"""

import numpy as np

class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, data, targets):
        self.data = data
        self.targets = targets

        return self

    def predict(self, inputs):
        predictions = []
        for input in inputs:
            predictions.append(self.__predict_target(input))

        return np.array(predictions)

    def __predict_target(self, input):
        # Find distances between points
        distances = np.sum((self.data - input) ** 2, axis=1)

        # Sort to find neighbors
        closest = np.sort(distances)

        # Translate closest values to closest targets
        indices = []
        size = np.size(closest)
        for i in range(0, self.k):
            if i == size:
                break
            indices.append(np.where(distances == closest[i])[0][0])

        closestTargets = []
        for index in indices:
            closestTargets.append(self.targets[index])

        # Find the most frequent targets depending on k
        counts = np.bincount(np.array(closestTargets))

        return np.argmax(counts)