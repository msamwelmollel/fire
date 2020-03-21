# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:18:49 2020

@author: Team Mandela
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
from matplotlib.pyplot import interactive
interactive(True)

import numpy as np
import pandas as pd
import seaborn as sns


from sklearn import preprocessing

path = "train.csv"

train_dataset = pd.read_csv(path, parse_dates=['date'])
test_dataset = pd.read_csv('test.csv',sep=',', parse_dates=['date'])

#print(train_dataset.head())
#%%
# looking for features to remove 
train_dataset.corr()['burn_area'].sort_values().plot(kind='bar', figsize=(18,6))

value = train_dataset.corr()['burn_area'].sort_values()
#print(value.iloc[:])

value = value[value<0.2]
value = value[value>-0.2]
#print(value)
value = list(value.index)
#print(value)

train_dataset = train_dataset.drop(value, axis=1)
#print(train_dataset.head())
train_dataset.corr()['burn_area'].sort_values().plot(kind='bar', figsize=(18,6))





