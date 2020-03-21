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

value = value[value<0.15]
value = value[value>-0.15]
#print(value)
value = list(value.index)
#print(value)

train_dataset = train_dataset.drop(value, axis=1)
#print(train_dataset.head())
train_dataset.corr()['burn_area'].sort_values().plot(kind='bar', figsize=(18,6))


# Date variables
train_dataset['month'] = train_dataset.date.dt.month
train_dataset['year'] = train_dataset.date.dt.year

# Plotting mean burn_area for each month - very strong mid-year peak (dry season)
train_dataset.groupby('month').mean().reset_index().plot(y='burn_area', x='month', kind='bar')


train_all = train_dataset.copy().dropna()
train = train_all.loc[train_all.date < '2011-01-01']
valid = train_all.loc[train_all.date > '2011-01-01']
print(train.shape, valid.shape)





