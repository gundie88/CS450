# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:41:00 2019

@author: kgund
"""

import pandas as pd
import numpy as np

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                   skipinitialspace=True)

data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'wage']

data.replace('?',np.NaN )



for col in ['workclass', 'education', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country', 'wage']:
    data[col] = data[col].astype('category')

for col in data.columns:
    data[col] = data[col].astype("category")
    data[col] = pd.factorize(data[col])[0] + 1



