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



from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

path = "train.csv"

train_dataset = pd.read_csv(path, parse_dates=['date'])
test = pd.read_csv('test.csv',sep=',', parse_dates=['date'])

#print(train_dataset.head())
#%%
# looking for features to remove 
train_dataset.corr()['burn_area'].sort_values().plot(kind='bar', figsize=(18,6))

value = train_dataset.corr()['burn_area'].sort_values()
#print(value.iloc[:])

value = value[value<0.21]
value = value[value>-0.21]
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
#print(train.shape, valid.shape)

#print(train.columns)

# Define input and output columns
train_c = train.copy()
target_col = train_c.pop('burn_area')

train_column = train_c.columns[3:]
target_column = 'burn_area'

train_column = list(train_column)
print(train_column)



#simple model 
# Get our X and y training and validation sets ready
X_train, y_train = train[train_column], train[target_column]

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train = max_abs_scaler.fit_transform(X_train)

X_valid, y_valid = valid[train_column], valid[target_column]

max_abs_scaler = preprocessing.MaxAbsScaler()
X_valid = max_abs_scaler.fit_transform(X_valid)

# Create and fit the model
model = RidgeCV()
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_valid)

# Score
msep=mean_squared_error(y_valid, preds)**0.5 # RMSE - should match Zindi score. Lower is better
print(msep)

# Look at the sample submission file
ss = pd.read_csv('ss.csv')
#print(ss.head())
#print(test.head())


# So we need to predict the burn area for each row in test. 

# Add the same features to test as we did to train:
test['month'] = test.date.dt.month
test['year'] = test.date.dt.year
test= test[train_column]
print(test.head())
max_abs_scaler = preprocessing.MaxAbsScaler()
test = max_abs_scaler.fit_transform(test)
print(test[1,1:])
# print(test)

# Get predictions
preds = model.predict(test) # fillna(0) here could be improved by examining the missing data and filling more appropriately.

# Add to submission dataframe
ss['Prediction'] = preds

# View
print(ss.head())

# Save ready for submission:
ss.to_csv('starter_submission.csv', index=False)









